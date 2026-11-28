"""A reference training loop for Gaussian Splatting.

Implements the optimization recipe of the 3DGS paper: per-parameter learning
rates, an L1 + D-SSIM photometric loss, accumulation of screen-space
positional gradients, and periodic adaptive density control.

Optional, per-step extras for real-world captures (all off unless used):

- **foreground masks** (``step(..., mask=)``): ``masked`` mode ignores pixels
  outside the mask; ``transparent`` mode supervises rendered opacity to get a
  clean cut-out;
- **appearance compensation** (``TrainerConfig.appearance``): a learnable
  per-view affine color transform absorbs exposure / white-balance changes;
- **geometry priors** (``step(..., depth_prior=)``): a scale-invariant depth
  correlation loss and a normal-consistency loss against e.g. Depth Anything 3
  depth maps.
"""

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np

from ..cameras import Camera, refine_camera
from ..losses import masked_l1, masked_ssim, ssim
from ..transforms import quaternion_to_matrix
from .model import GaussianModel
from .projection import project_gaussians, project_gaussians_ut
from .rasterize import rasterize, rasterize_features
from .sh import eval_sh
from .tiles import bin_gaussians

__all__ = ["TrainerConfig", "GaussianTrainer"]

_ALPHA_MODES = ("masked", "transparent")


def _normalize(v: mx.array) -> mx.array:
    """Unit vectors with a finite gradient at zero (``linalg.norm``'s is NaN
    there, which poisons the whole step even under a zero mask weight)."""
    return v * mx.rsqrt(mx.maximum(mx.sum(v * v, axis=-1, keepdims=True), 1e-16))


def _depth_to_normals(camera: Camera, depth: mx.array) -> mx.array:
    """World-space unit normals ``(H-1, W-1, 3)`` of a z-depth map, from cross
    products of the differences between neighboring unprojected pixels."""
    u = mx.arange(camera.width, dtype=mx.float32) + 0.5
    v = mx.arange(camera.height, dtype=mx.float32) + 0.5
    uu = mx.broadcast_to(u[None, :], (camera.height, camera.width))
    vv = mx.broadcast_to(v[:, None], (camera.height, camera.width))
    points = camera.unproject_points(mx.stack([uu, vv], axis=-1), depth)
    tangent_x = points[:-1, 1:] - points[:-1, :-1]
    tangent_y = points[1:, :-1] - points[:-1, :-1]
    return _normalize(mx.linalg.cross(tangent_x, tangent_y))


def _quad_valid(valid: mx.array) -> mx.array:
    """Pixels whose 2x2 neighborhood is fully valid (for finite differences)."""
    return valid[:-1, :-1] * valid[:-1, 1:] * valid[1:, :-1] * valid[1:, 1:]


def _weighted_pearson(a: mx.array, b: mx.array, w: mx.array) -> mx.array:
    """Weighted Pearson correlation of two maps (scale- and shift-invariant)."""
    total = mx.maximum(w.sum(), 1e-8)
    a0 = a - (a * w).sum() / total
    b0 = b - (b * w).sum() / total
    cov = (a0 * b0 * w).sum()
    return cov / mx.sqrt(mx.maximum((a0 * a0 * w).sum() * (b0 * b0 * w).sum(), 1e-12))


def _apply_appearance(image: mx.array, alpha: mx.array, delta: mx.array, bg: mx.array):
    """Per-view affine color correction of the foreground radiance.

    ``delta`` is ``(3, 4)``: the deviation from the identity transform
    ``[I | 0]``. The background is excluded so it stays exactly ``bg``.
    """
    a = alpha[..., None]
    fg = image - (1.0 - a) * bg
    corrected = fg @ (mx.eye(3) + delta[:, :3]).T + a * delta[:, 3]
    return corrected + (1.0 - a) * bg


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
    projection: str = "ewa"
    """Gaussian projection method: ``ewa`` or 3DGUT-style ``ut``."""
    lambda_2d_depth_variance: float = 0.0
    """2DGS-only ray-depth variance penalty; encourages one surface per pixel."""
    lambda_2d_normal_consistency: float = 0.0
    """2DGS-only loss between rendered surfel normals and depth-map normals."""
    geometry_min_alpha: float = 0.05
    """Minimum alpha for pixels included in 2DGS geometry losses."""
    # Foreground masks (``step(..., mask=...)``).
    alpha_mode: str = "masked"
    """How a mask is used. ``masked``: it weights the photometric loss and
    pixels outside it are ignored entirely. ``transparent``: it is the target
    opacity; the target is composited over the render background and the
    rendered alpha is supervised, for a clean cut-out."""
    lambda_alpha: float = 1.0
    """Weight of the alpha L1 loss in ``transparent`` mode."""
    random_background: bool = False
    """``transparent`` mode: draw a random background color every step, which
    discourages background color from leaking into the silhouette."""
    # Per-view appearance compensation.
    appearance: bool = False
    """Learn a per-view affine color transform (exposure / white balance).
    Needs ``num_views`` and ``step(..., view_index=...)``; renders of the
    model itself stay uncorrected."""
    lr_appearance: float = 1e-3
    # Geometry priors (``step(..., depth_prior=...)``).
    lambda_depth_prior: float = 0.0
    """Weight of ``1 - Pearson(rendered depth, prior depth)``; scale- and
    shift-invariant, so relative (e.g. monocular or DA3) depth works."""
    lambda_normal_prior: float = 0.0
    """Weight of ``1 - cos`` between normals of the rendered and prior depth."""
    prior_min_alpha: float = 0.5
    """Minimum rendered alpha for pixels included in the prior losses."""
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
        self,
        model: GaussianModel,
        config: TrainerConfig | None = None,
        scene_extent: float = 1.0,
        num_views: int | None = None,
    ):
        self.model = model
        self.config = config or TrainerConfig()
        if self.config.method not in {"vanilla", "mcmc", "2dgs"}:
            raise ValueError("TrainerConfig.method must be 'vanilla', 'mcmc', or '2dgs'")
        if self.config.projection not in {"ewa", "ut"}:
            raise ValueError("TrainerConfig.projection must be 'ewa' or 'ut'")
        if self.config.alpha_mode not in _ALPHA_MODES:
            raise ValueError(f"TrainerConfig.alpha_mode must be one of {_ALPHA_MODES}")
        self.appearance: mx.array | None = None
        if self.config.appearance:
            if not num_views:
                raise ValueError("TrainerConfig.appearance needs num_views.")
            self.appearance = mx.zeros((num_views, 3, 4))
            self._appearance_opt = optim.Adam(learning_rate=self.config.lr_appearance)
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
    def _geometry_loss_2dgs(
        self,
        params: dict[str, mx.array],
        camera: Camera,
        proj: dict[str, mx.array],
        opacities: mx.array,
        sorted_ids: mx.array,
        tile_ranges: mx.array,
        tiles_x: int,
        tiles_y: int,
        support: mx.array | None = None,
    ) -> tuple[mx.array, dict[str, mx.array]]:
        """2DGS depth-variance / normal-consistency losses, restricted to
        ``support`` (the mask region in masked mode) when given."""
        cfg = self.config
        zero = mx.array(0.0, dtype=mx.float32)
        metrics = {"depth_variance": zero, "normal_consistency": zero}
        if cfg.method != "2dgs" or (
            cfg.lambda_2d_depth_variance <= 0 and cfg.lambda_2d_normal_consistency <= 0
        ):
            return zero, metrics

        depths = proj["depths"]
        normals = quaternion_to_matrix(params["quats"])[:, :, 2]
        view = camera.camera_center - params["means"]
        normals = normals / mx.maximum(mx.linalg.norm(normals, axis=-1, keepdims=True), 1e-8)
        normals = mx.where(mx.sum(normals * view, axis=-1, keepdims=True) < 0, -normals, normals)
        features = mx.concatenate([depths[:, None], (depths * depths)[:, None], normals], axis=1)
        geom = rasterize_features(
            proj["means2d"],
            proj["conics"],
            features,
            opacities,
            sorted_ids,
            tile_ranges,
            camera.width,
            camera.height,
            tiles_x,
            tiles_y,
            normalize=True,
        )
        alpha = geom["alpha"]
        valid = (alpha > cfg.geometry_min_alpha).astype(mx.float32)
        if support is not None:
            valid = valid * support
        denom = mx.maximum(valid.sum(), 1.0)

        expected_depth = geom["features"][..., 0]
        expected_depth2 = geom["features"][..., 1]
        depth_var = mx.maximum(expected_depth2 - expected_depth * expected_depth, 0.0)
        loss = zero
        if cfg.lambda_2d_depth_variance > 0:
            metrics["depth_variance"] = (depth_var * valid).sum() / denom
            loss = loss + float(cfg.lambda_2d_depth_variance) * metrics["depth_variance"]

        if cfg.lambda_2d_normal_consistency > 0 and camera.width > 1 and camera.height > 1:
            depth_normals = _depth_to_normals(camera, expected_depth)
            rendered_normals = _normalize(geom["features"][:-1, :-1, 2:5])
            valid4 = _quad_valid(valid)
            denom4 = mx.maximum(valid4.sum(), 1.0)
            cos = mx.sum(rendered_normals * depth_normals, axis=-1)
            metrics["normal_consistency"] = ((1.0 - mx.abs(cos)) * valid4).sum() / denom4
            loss = loss + float(cfg.lambda_2d_normal_consistency) * metrics["normal_consistency"]

        return loss, metrics

    def _prior_losses(
        self,
        camera: Camera,
        proj: dict[str, mx.array],
        opacities: mx.array,
        binning: tuple,
        depth_prior: mx.array,
        support: mx.array | None,
    ) -> tuple[mx.array, dict[str, mx.array]]:
        """Depth-correlation and normal losses against a prior depth map."""
        cfg = self.config
        sorted_ids, tile_ranges, tiles_x, tiles_y = binning
        geom = rasterize_features(
            proj["means2d"],
            proj["conics"],
            proj["depths"][:, None],
            opacities,
            sorted_ids,
            tile_ranges,
            camera.width,
            camera.height,
            tiles_x,
            tiles_y,
            normalize=True,
        )
        depth = geom["features"][..., 0]
        prior = mx.stop_gradient(depth_prior.astype(mx.float32))
        valid = ((geom["alpha"] > cfg.prior_min_alpha) & (prior > 0)).astype(mx.float32)
        if support is not None:
            valid = valid * support
        valid = mx.stop_gradient(valid)

        loss = mx.array(0.0, dtype=mx.float32)
        metrics = {}
        if cfg.lambda_depth_prior > 0:
            metrics["depth_prior"] = 1.0 - _weighted_pearson(depth, prior, valid)
            loss = loss + cfg.lambda_depth_prior * metrics["depth_prior"]
        if cfg.lambda_normal_prior > 0 and camera.width > 1 and camera.height > 1:
            prior_normals = mx.stop_gradient(_depth_to_normals(camera, prior))
            cos = mx.sum(_depth_to_normals(camera, depth) * prior_normals, axis=-1)
            valid4 = _quad_valid(valid)
            metrics["normal_prior"] = ((1.0 - cos) * valid4).sum() / mx.maximum(valid4.sum(), 1.0)
            loss = loss + cfg.lambda_normal_prior * metrics["normal_prior"]
        return loss, metrics

    def _photometric_loss(
        self,
        image: mx.array,
        alpha: mx.array,
        target: mx.array,
        background: mx.array,
        mask: mx.array | None,
    ) -> tuple[mx.array, dict[str, mx.array]]:
        """L1 + D-SSIM, honoring the mask semantics of ``alpha_mode``."""
        cfg = self.config
        c = cfg.lambda_dssim
        if mask is None:
            return (1.0 - c) * mx.abs(image - target).mean() + c * (1.0 - ssim(image, target)), {}
        if cfg.alpha_mode == "masked":
            l1 = masked_l1(image, target, mask)
            return (1.0 - c) * l1 + c * (1.0 - masked_ssim(image, target, mask)), {}
        # transparent: the mask is the target opacity over the render background.
        m = mask[..., None]
        composite = target * m + background * (1.0 - m)
        loss = (1.0 - c) * mx.abs(image - composite).mean() + c * (1.0 - ssim(image, composite))
        alpha_l1 = mx.abs(alpha - mask).mean()
        return loss + cfg.lambda_alpha * alpha_l1, {"alpha_l1": alpha_l1}

    def _render_loss(
        self,
        params,
        means2d_probe,
        camera: Camera,
        target: mx.array,
        background: mx.array,
        *,
        extras: dict[str, mx.array] | None = None,
        mask: mx.array | None = None,
        depth_prior: mx.array | None = None,
    ):
        """Training loss for one view.

        ``means2d_probe`` (zeros) exposes screen-space positional gradients for
        densification; ``extras`` holds other differentiable per-view inputs
        (``twist`` for pose refinement, ``appearance`` for color correction).
        """
        cfg = self.config
        extras = extras or {}
        if "twist" in extras:
            camera = refine_camera(camera, extras["twist"])
        project = project_gaussians_ut if cfg.projection == "ut" else project_gaussians
        proj = project(
            camera,
            params["means"],
            params["quats"],
            mx.exp(params["scales"]),
            antialias=cfg.antialias,
        )
        means2d = proj["means2d"] + means2d_probe
        opacities = mx.sigmoid(params["opacities"]) * proj["compensation"]

        sh = mx.concatenate([params["sh_dc"], params["sh_rest"]], axis=1)
        dirs = _normalize(params["means"] - camera.camera_center)
        colors = mx.maximum(eval_sh(self.model.active_sh_degree, sh, mx.stop_gradient(dirs)), 0.0)

        binning = bin_gaussians(
            means2d,
            proj["radii"],
            proj["depths"],
            camera.width,
            camera.height,
        )
        sorted_ids, tile_ranges, tiles_x, tiles_y = binning
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
        img, alpha = out["image"], out["alpha"]
        if "appearance" in extras:
            img = _apply_appearance(img, alpha, extras["appearance"], background)

        loss, metrics = self._photometric_loss(img, alpha, target, background, mask)
        # Geometry losses only ever see the foreground in masked mode.
        support = None
        if mask is not None and cfg.alpha_mode == "masked":
            support = mx.stop_gradient((mask > 0).astype(mx.float32))
        # Auxiliary losses use the probe-free projection so they do not feed the
        # screen-space gradient statistics that drive densification.
        geom_loss, geom_metrics = self._geometry_loss_2dgs(
            params, camera, proj, opacities, *binning, support=support
        )
        loss = loss + geom_loss
        metrics.update(geom_metrics)
        if depth_prior is not None and (cfg.lambda_depth_prior > 0 or cfg.lambda_normal_prior > 0):
            prior_loss, prior_metrics = self._prior_losses(
                camera, proj, opacities, binning, depth_prior, support
            )
            loss = loss + prior_loss
            metrics.update(prior_metrics)
        return loss, (img, proj["radii"], metrics)

    # -------------------------------------------------------------------- step
    def step(
        self,
        camera: Camera,
        target: mx.array,
        twist: mx.array | None = None,
        *,
        mask: mx.array | None = None,
        depth_prior: mx.array | None = None,
        view_index: int | None = None,
    ) -> dict[str, object]:
        """One optimization step on a single view. Returns logging info.

        Args:
            camera: the view to render.
            target: its ``(H, W, 3)`` image.
            twist: optional 6D SE(3) Lie-algebra vector. The view is rendered
                through ``refine_camera(camera, twist)`` and the loss gradient
                with respect to the twist is returned under
                ``info["twist_grad"]``. Callers own the twists and their
                optimizer, enabling BARF-style joint pose refinement.
            mask: optional ``(H, W)`` foreground mask in ``[0, 1]``, used per
                ``TrainerConfig.alpha_mode``.
            depth_prior: optional ``(H, W)`` prior z-depth (``<= 0`` = unknown)
                for the ``lambda_depth_prior`` / ``lambda_normal_prior`` losses.
            view_index: index of the view, required with
                ``TrainerConfig.appearance``.
        """
        self.step_count += 1
        cfg = self.config
        if mask is not None and cfg.alpha_mode == "transparent" and cfg.random_background:
            bg = mx.random.uniform(shape=(3,))
        else:
            bg = mx.ones((3,)) if cfg.white_background else mx.zeros((3,))
        if mask is not None:
            mask = mask.reshape(target.shape[:2]).astype(mx.float32)
        params = self.model.params
        probe = mx.zeros((self.model.num_gaussians, 2))
        densify_stats = None
        opacity_reset = False
        sh_degree_changed = False

        extras: dict[str, mx.array] = {}
        if twist is not None:
            extras["twist"] = twist
        if self.appearance is not None:
            if view_index is None:
                raise ValueError("TrainerConfig.appearance needs step(..., view_index=...).")
            extras["appearance"] = self.appearance[view_index]

        def loss_fn(params, probe, extras):
            return self._render_loss(
                params,
                probe,
                camera,
                target,
                bg,
                extras=extras,
                mask=mask,
                depth_prior=depth_prior,
            )

        (loss, (img, radii, geom_metrics)), grads = mx.value_and_grad(loss_fn, argnums=(0, 1, 2))(
            params, probe, extras
        )
        param_grads, probe_grad, extra_grads = grads
        twist_grad = extra_grads.get("twist")
        if "appearance" in extra_grads:
            # One dense (V, 3, 4) parameter so every view keeps its own Adam state.
            grad = mx.zeros_like(self.appearance).at[view_index].add(extra_grads["appearance"])
            self.appearance = self._appearance_opt.apply_gradients(
                {"a": grad}, {"a": self.appearance}
            )["a"]

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
        del param_grads, grads, extra_grads
        mx.eval(self.model.params, self.appearance if self.appearance is not None else [])

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
            "geometry": {k: float(v) for k, v in geom_metrics.items()},
            "twist_grad": twist_grad,
        }
