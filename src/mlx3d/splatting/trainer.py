"""A reference training loop for Gaussian Splatting.

Implements the optimization recipe of the 3DGS paper: per-parameter learning
rates, an L1 + D-SSIM photometric loss, accumulation of screen-space
positional gradients, and periodic adaptive density control.
"""

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np

from ..cameras import Camera
from ..losses import ssim
from .model import GaussianModel
from .projection import project_gaussians
from .rasterize import rasterize
from .sh import eval_sh
from .tiles import bin_gaussians

__all__ = ["TrainerConfig", "GaussianTrainer"]


@dataclass
class TrainerConfig:
    method: str = "vanilla"
    """Training strategy: ``vanilla``, ``mcmc``, or ``2dgs``."""
    lr_means: float = 1.6e-4
    lr_means_final: float = 1.6e-6
    lr_means_max_steps: int = 30_000
    lr_scales: float = 5e-3
    lr_quats: float = 1e-3
    lr_opacities: float = 5e-2
    lr_sh_dc: float = 2.5e-3
    lr_sh_rest: float = 2.5e-3 / 20.0
    lambda_dssim: float = 0.2
    antialias: bool = False
    """Use Mip-Splatting-style opacity compensation for projection blur."""
    # Adaptive density control.
    densify_from: int = 500
    densify_until: int = 15000
    densify_every: int = 100
    densify_grad_threshold: float = 0.0002
    mcmc_relocate_frac: float = 0.02
    """Max fraction of Gaussians relocated at each MCMC density event."""
    mcmc_min_opacity: float = 0.01
    """Rows below this opacity are considered relocation targets in MCMC mode."""
    mcmc_jitter_scale: float = 0.25
    """Relocated copies are jittered by this multiple of their source scale."""
    mcmc_noise_scale: float = 0.01
    """Per-step SGLD-like xyz noise scale used in MCMC mode."""
    two_d_thickness: float = 1e-4
    """2DGS local-normal thickness as a fraction of scene extent."""
    opacity_reset_every: int = 3000
    sh_increase_every: int = 1000
    white_background: bool = False
    # Low-memory controls.
    max_gaussians: int | None = None
    """Stop clone/split growth above this count (pruning continues)."""
    low_memory: bool = False
    """Cap MLX's buffer cache and clear it after densification events.
    Reduces peak memory noticeably on 8-16 GB machines at a small speed cost."""
    cache_limit_gb: float = 2.0
    """MLX buffer-cache cap used when ``low_memory`` is enabled."""


class GaussianTrainer:
    """Optimizes a :class:`GaussianModel` against posed images."""

    def __init__(
        self, model: GaussianModel, config: TrainerConfig | None = None, scene_extent: float = 1.0
    ):
        self.model = model
        self.config = config or TrainerConfig()
        if self.config.method not in {"vanilla", "mcmc", "2dgs"}:
            raise ValueError("TrainerConfig.method must be 'vanilla', 'mcmc', or '2dgs'")
        self.scene_extent = scene_extent
        self.step_count = 0
        if self.config.low_memory:
            # MLX caches freed GPU buffers for reuse; under shape churn
            # (densification changes N every 100 steps) the cache can grow
            # by gigabytes. Cap it, and clear at densification events below.
            mx.set_cache_limit(int(self.config.cache_limit_gb * (1 << 30)))
        self._apply_method_constraints()
        self._build_optimizers()
        self._reset_grad_accum()

    def _build_optimizers(self) -> None:
        c = self.config
        lr_means = c.lr_means * self.scene_extent
        lr_means_final = c.lr_means_final * self.scene_extent
        lr_means_max_steps = int(c.lr_means_max_steps)
        if lr_means_max_steps > 0 and lr_means > 0 and lr_means_final > 0:
            means_lr = optim.exponential_decay(
                lr_means,
                (lr_means_final / lr_means) ** (1.0 / lr_means_max_steps),
            )
            if self.step_count:
                offset = self.step_count

                def means_lr_with_offset(step, schedule=means_lr, offset=offset):
                    return schedule(step + offset)

                means_lr = means_lr_with_offset
        else:
            means_lr = lr_means
        lrs = {
            "means": means_lr,
            "scales": c.lr_scales,
            "quats": c.lr_quats,
            "opacities": c.lr_opacities,
            "sh_dc": c.lr_sh_dc,
            "sh_rest": c.lr_sh_rest,
        }
        self.optimizers = {k: optim.Adam(learning_rate=lr, eps=1e-15) for k, lr in lrs.items()}

    def learning_rates(self) -> dict[str, float]:
        """Current per-parameter Adam learning rates."""
        return {k: float(opt.learning_rate) for k, opt in self.optimizers.items()}

    def _reset_grad_accum(self) -> None:
        n = self.model.num_gaussians
        self.grad_accum = mx.zeros((n,), dtype=mx.float32)
        self.grad_count = mx.zeros((n,), dtype=mx.float32)

    def _resize_optimizer_states_after_densify(self, result: dict[str, object]) -> None:
        """Preserve Adam moments for surviving Gaussians after ADC changes N."""
        keep_idx_np = result.get("_keep_idx")
        new_count = int(result.get("_new_count", 0))
        if keep_idx_np is None:
            return
        keep_idx = mx.array(keep_idx_np)
        for name, opt in self.optimizers.items():
            state = opt.state.get(name)
            if not isinstance(state, dict):
                continue
            for slot in ("m", "v"):
                old = state.get(slot)
                if old is None:
                    continue
                kept = old[keep_idx]
                if new_count > 0:
                    zeros = mx.zeros((new_count, *old.shape[1:]), dtype=old.dtype)
                    state[slot] = mx.concatenate([kept, zeros], axis=0)
                else:
                    state[slot] = kept

    def _zero_optimizer_state_rows(self, idx_np) -> None:
        if idx_np is None:
            return
        if idx_np.shape[0] == 0:
            return
        idx = (
            idx_np.astype(mx.int32)
            if isinstance(idx_np, mx.array)
            else mx.array(np.asarray(idx_np, dtype=np.int32))
        )
        for name, opt in self.optimizers.items():
            state = opt.state.get(name)
            if not isinstance(state, dict):
                continue
            for slot in ("m", "v"):
                old = state.get(slot)
                if old is None:
                    continue
                state[slot] = old.at[idx].add(-old[idx])

    def _apply_method_constraints(self) -> None:
        if self.config.method == "2dgs":
            thickness = self.config.two_d_thickness * max(float(self.scene_extent), 1e-8)
            self.model.apply_2dgs_constraints(thickness)

    # ------------------------------------------------------------------ losses
    def _render_loss(
        self, params, means2d_probe, camera: Camera, target: mx.array, background: mx.array
    ):
        """Photometric loss; ``means2d_probe`` (zeros) exposes screen-space
        positional gradients for densification."""
        proj = project_gaussians(
            camera,
            params["means"],
            params["quats"],
            mx.exp(params["scales"]),
            antialias=self.config.antialias,
        )
        means2d = proj["means2d"] + means2d_probe
        opacities = mx.sigmoid(params["opacities"]) * proj["compensation"]

        sh = mx.concatenate([params["sh_dc"], params["sh_rest"]], axis=1)
        dirs = params["means"] - camera.camera_center
        dirs = dirs / mx.maximum(mx.linalg.norm(dirs, axis=-1, keepdims=True), 1e-8)
        colors = mx.maximum(eval_sh(self.model.active_sh_degree, sh, mx.stop_gradient(dirs)), 0.0)

        sorted_ids, tile_ranges, tiles_x, tiles_y = bin_gaussians(
            means2d,
            proj["radii"],
            proj["depths"],
            camera.width,
            camera.height,
        )
        out = rasterize(
            means2d,
            proj["conics"],
            colors,
            opacities,
            sorted_ids,
            tile_ranges,
            camera.width,
            camera.height,
            tiles_x,
            tiles_y,
            background=background,
        )
        img = out["image"]
        l1 = mx.abs(img - target).mean()
        c = self.config.lambda_dssim
        loss = (1.0 - c) * l1 + c * (1.0 - ssim(img, target))
        return loss, (img, proj["radii"])

    # -------------------------------------------------------------------- step
    def step(self, camera: Camera, target: mx.array) -> dict[str, object]:
        """One optimization step on a single view. Returns logging info."""
        self.step_count += 1
        cfg = self.config
        bg = mx.ones((3,)) if cfg.white_background else mx.zeros((3,))
        params = self.model.params
        probe = mx.zeros((self.model.num_gaussians, 2))
        densify_stats = None
        opacity_reset = False
        sh_degree_changed = False

        def loss_fn(params, probe):
            loss, aux = self._render_loss(params, probe, camera, target, bg)
            return loss, aux

        (loss, (img, radii)), grads = mx.value_and_grad(loss_fn, argnums=(0, 1))(params, probe)
        param_grads, probe_grad = grads

        for k, opt in self.optimizers.items():
            self.model.params[k] = opt.apply_gradients(
                {k: param_grads[k]}, {k: self.model.params[k]}
            )[k]
        if cfg.method == "mcmc" and cfg.mcmc_noise_scale > 0:
            means_lr = max(float(self.learning_rates()["means"]), 0.0)
            sigma = cfg.mcmc_noise_scale * self.scene_extent * math.sqrt(means_lr)
            if sigma > 0:
                self.model.params["means"] = (
                    self.model.params["means"]
                    + mx.random.normal(self.model.params["means"].shape) * sigma
                )
        self._apply_method_constraints()
        # Release gradient references before evaluating so their buffers can
        # be recycled within the same step (see the MLX performance guide).
        del param_grads, grads
        mx.eval(self.model.params)

        # Accumulate NDC-space positional gradient norms for densification.
        if cfg.densify_from <= self.step_count <= cfg.densify_until:
            g = mx.stop_gradient(probe_grad)
            gx = g[:, 0] * (camera.width / 2.0)
            gy = g[:, 1] * (camera.height / 2.0)
            norms = mx.sqrt(gx * gx + gy * gy)
            visible = (mx.stop_gradient(radii) > 0).astype(mx.float32)
            self.grad_accum += norms * visible
            self.grad_count += visible

            if self.step_count % cfg.densify_every == 0 and self.step_count > cfg.densify_from:
                if cfg.method == "mcmc":
                    densify_result = self.model.relocate_mcmc(
                        self.grad_accum,
                        self.grad_count,
                        relocate_frac=cfg.mcmc_relocate_frac,
                        min_opacity=cfg.mcmc_min_opacity,
                        jitter_scale=cfg.mcmc_jitter_scale,
                    )
                    self._zero_optimizer_state_rows(densify_result.get("_moved_idx"))
                else:
                    # At the cap, disable growth (infinite threshold) but keep
                    # pruning so the count can recover below the cap.
                    at_cap = (
                        cfg.max_gaussians is not None
                        and self.model.num_gaussians >= cfg.max_gaussians
                    )
                    threshold = float("inf") if at_cap else cfg.densify_grad_threshold
                    densify_result = self.model.densify_and_prune(
                        self.grad_accum,
                        self.grad_count,
                        grad_threshold=threshold,
                        scene_extent=self.scene_extent,
                        return_optimizer_state=True,
                    )
                    self._resize_optimizer_states_after_densify(densify_result)
                self._apply_method_constraints()
                densify_stats = {
                    k: v for k, v in densify_result.items() if not str(k).startswith("_")
                }
                self._reset_grad_accum()
                if cfg.low_memory:
                    mx.eval(self.model.params)
                    mx.clear_cache()

        if self.step_count % cfg.opacity_reset_every == 0 and self.step_count <= cfg.densify_until:
            self.model.reset_opacities()
            opacity_reset = True

        if self.step_count % cfg.sh_increase_every == 0:
            self.model.one_up_sh_degree()
            sh_degree_changed = True

        return {
            "loss": float(loss),
            "num_gaussians": self.model.num_gaussians,
            "step": self.step_count,
            "active_sh_degree": self.model.active_sh_degree,
            "lr_means": self.learning_rates()["means"],
            "method": cfg.method,
            "densify": densify_stats,
            "opacity_reset": opacity_reset,
            "sh_degree_changed": sh_degree_changed,
        }
