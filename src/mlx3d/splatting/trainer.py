"""A reference training loop for Gaussian Splatting.

Implements the optimization recipe of the 3DGS paper: per-parameter learning
rates, an L1 + D-SSIM photometric loss, accumulation of screen-space
positional gradients, and periodic adaptive density control.
"""

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
    lr_means: float = 1.6e-4
    lr_scales: float = 5e-3
    lr_quats: float = 1e-3
    lr_opacities: float = 5e-2
    lr_sh_dc: float = 2.5e-3
    lr_sh_rest: float = 2.5e-3 / 20.0
    lambda_dssim: float = 0.2
    # Adaptive density control.
    densify_from: int = 500
    densify_until: int = 15000
    densify_every: int = 100
    densify_grad_threshold: float = 0.0002
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

    def __init__(self, model: GaussianModel, config: TrainerConfig | None = None,
                 scene_extent: float = 1.0):
        self.model = model
        self.config = config or TrainerConfig()
        self.scene_extent = scene_extent
        self.step_count = 0
        if self.config.low_memory:
            # MLX caches freed GPU buffers for reuse; under shape churn
            # (densification changes N every 100 steps) the cache can grow
            # by gigabytes. Cap it, and clear at densification events below.
            mx.set_cache_limit(int(self.config.cache_limit_gb * (1 << 30)))
        self._build_optimizers()
        self._reset_grad_accum()

    def _build_optimizers(self) -> None:
        c = self.config
        lrs = {
            "means": c.lr_means * self.scene_extent,
            "scales": c.lr_scales,
            "quats": c.lr_quats,
            "opacities": c.lr_opacities,
            "sh_dc": c.lr_sh_dc,
            "sh_rest": c.lr_sh_rest,
        }
        self.optimizers = {k: optim.Adam(learning_rate=lr, eps=1e-15) for k, lr in lrs.items()}

    def _reset_grad_accum(self) -> None:
        n = self.model.num_gaussians
        self.grad_accum = np.zeros(n, dtype=np.float64)
        self.grad_count = np.zeros(n, dtype=np.float64)

    # ------------------------------------------------------------------ losses
    def _render_loss(self, params, means2d_probe, camera: Camera, target: mx.array,
                     background: mx.array):
        """Photometric loss; ``means2d_probe`` (zeros) exposes screen-space
        positional gradients for densification."""
        proj = project_gaussians(
            camera, params["means"], params["quats"], mx.exp(params["scales"])
        )
        means2d = proj["means2d"] + means2d_probe

        sh = mx.concatenate([params["sh_dc"], params["sh_rest"]], axis=1)
        dirs = params["means"] - camera.camera_center
        dirs = dirs / mx.maximum(mx.linalg.norm(dirs, axis=-1, keepdims=True), 1e-8)
        colors = mx.maximum(
            eval_sh(self.model.active_sh_degree, sh, mx.stop_gradient(dirs)), 0.0
        )

        sorted_ids, tile_ranges, tiles_x, tiles_y = bin_gaussians(
            means2d, proj["radii"], proj["depths"], camera.width, camera.height
        )
        out = rasterize(
            means2d, proj["conics"], colors, mx.sigmoid(params["opacities"]),
            sorted_ids, tile_ranges, camera.width, camera.height, tiles_x, tiles_y,
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

        (loss, (img, radii)), grads = mx.value_and_grad(loss_fn, argnums=(0, 1))(
            params, probe
        )
        param_grads, probe_grad = grads

        for k, opt in self.optimizers.items():
            self.model.params[k] = opt.apply_gradients(
                {k: param_grads[k]}, {k: self.model.params[k]}
            )[k]
        # Release gradient references before evaluating so their buffers can
        # be recycled within the same step (see the MLX performance guide).
        del param_grads, grads
        mx.eval(self.model.params)

        # Accumulate NDC-space positional gradient norms for densification.
        if cfg.densify_from <= self.step_count <= cfg.densify_until:
            g = np.array(probe_grad)
            g[:, 0] *= camera.width / 2.0
            g[:, 1] *= camera.height / 2.0
            norms = np.linalg.norm(g, axis=1)
            visible = np.array(radii) > 0
            self.grad_accum += norms * visible
            self.grad_count += visible

            if (
                self.step_count % cfg.densify_every == 0
                and self.step_count > cfg.densify_from
            ):
                # At the cap, disable growth (infinite threshold) but keep
                # pruning so the count can recover below the cap.
                at_cap = (
                    cfg.max_gaussians is not None
                    and self.model.num_gaussians >= cfg.max_gaussians
                )
                threshold = float("inf") if at_cap else cfg.densify_grad_threshold
                densify_stats = self.model.densify_and_prune(
                    mx.array(self.grad_accum.astype(np.float32)),
                    mx.array(self.grad_count.astype(np.float32)),
                    grad_threshold=threshold,
                    scene_extent=self.scene_extent,
                )
                self._build_optimizers()  # parameter shapes changed
                self._reset_grad_accum()
                if cfg.low_memory:
                    mx.eval(self.model.params)
                    mx.clear_cache()

        if self.step_count % cfg.opacity_reset_every == 0 and self.step_count <= cfg.densify_until:
            self.model.reset_opacities()
            self._build_optimizers()
            opacity_reset = True

        if self.step_count % cfg.sh_increase_every == 0:
            self.model.one_up_sh_degree()
            sh_degree_changed = True

        return {
            "loss": float(loss),
            "num_gaussians": self.model.num_gaussians,
            "step": self.step_count,
            "active_sh_degree": self.model.active_sh_degree,
            "densify": densify_stats,
            "opacity_reset": opacity_reset,
            "sh_degree_changed": sh_degree_changed,
        }
