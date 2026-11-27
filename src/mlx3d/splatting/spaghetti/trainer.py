"""Training loop for spaghetti splats (an MCMC-style densification strategy).

Follows the recipe of the Spaghetti Splatting paper, built on 3DGS-MCMC:

- loss: ``(1 - l) L1 + l D-SSIM`` plus the MCMC opacity / scale regularizers
  and, late in training, an endpoint *attraction* term that pulls nearby
  segment ends together so segments chain into curves;
- position noise on low-opacity segments (shaped by each kernel's covariance)
  to keep exploring;
- periodic **relocation** of dead (near-transparent) segments and **growth**
  up to a budget. Both duplicate sampled live segments with render-preserving
  moves: long segments are split at their midpoint, short ones cloned with
  half the optical depth each (exact, because optical depth adds along rays).
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np

from ...cameras import Camera
from ...losses import ssim
from ...transforms import quaternion_to_matrix
from .model import SpaghettiModel

__all__ = ["SpaghettiTrainer", "SpaghettiTrainerConfig", "default_config_for"]


@dataclass
class SpaghettiTrainerConfig:
    lr_means: float = 1.6e-4
    lr_means_final: float = 1.6e-6
    lr_means_max_steps: int = 30_000
    lr_halves: float = 1.6e-4
    """Half-vector learning rate (scaled by the scene extent, like means)."""
    lr_scales: float = 5e-3
    lr_quats: float = 1e-3
    lr_opacities: float = 5e-2
    lr_sh_dc: float = 2.5e-3
    lr_sh_rest: float = 2.5e-3 / 20.0
    lambda_dssim: float = 0.2
    lambda_opacity: float = 0.01
    """MCMC regularizer: mean peak opacity (encourages few, opaque segments)."""
    lambda_scale: float = 0.01
    """MCMC regularizer: mean kernel scale (encourages thin kernels)."""
    # Density control.
    max_segments: int = 200_000
    densify_from: int = 500
    densify_until: int = 25_000
    densify_every: int = 100
    grow_rate: float = 0.05
    min_opacity: float = 0.005
    split_ratio: float = 2.0
    """Duplicated segments longer than this many kernel widths are split."""
    noise_lr: float = 5e5
    # Polyline formation.
    lambda_connect: float = 0.0
    """Endpoint attraction weight (applied from ``connect_from``)."""
    connect_from: int = 25_000
    connect_radius: float = 3.0
    """Endpoints closer than this many kernel widths attract each other."""
    connect_every: int = 100
    sh_increase_every: int = 1000
    white_background: bool = False


class SpaghettiTrainer:
    """Optimizes a :class:`SpaghettiModel` against posed images."""

    def __init__(
        self,
        model: SpaghettiModel,
        config: SpaghettiTrainerConfig | None = None,
        scene_extent: float = 1.0,
    ):
        self.model = model
        self.config = config or SpaghettiTrainerConfig()
        self.scene_extent = float(scene_extent)
        self.step_count = 0
        self._pairs: mx.array | None = None
        c = self.config
        means_lr = optim.exponential_decay(
            c.lr_means * self.scene_extent,
            (c.lr_means_final / c.lr_means) ** (1.0 / max(c.lr_means_max_steps, 1)),
        )
        lrs = {
            "means": means_lr,
            "halves": c.lr_halves * self.scene_extent,
            "scales": c.lr_scales,
            "quats": c.lr_quats,
            "opacities": c.lr_opacities,
            "sh_dc": c.lr_sh_dc,
            "sh_rest": c.lr_sh_rest,
        }
        self.optimizers = {k: optim.Adam(learning_rate=lr, eps=1e-15) for k, lr in lrs.items()}

    # ------------------------------------------------------------------ loss
    def _attraction(self, params: dict[str, mx.array]) -> mx.array:
        """Mean squared distance between paired nearby endpoints of different
        segments. Pairs are ``(segment, side, segment, side)`` rows with side
        -1 / +1 for ``m - h`` / ``m + h``; they are refreshed every
        ``connect_every`` steps and dropped whenever densification changes
        the rows."""
        pairs = self._pairs
        if pairs is None or pairs.shape[0] == 0:
            return mx.array(0.0)
        m, h = params["means"], params["halves"]

        def end(seg, side):
            return m[seg] + side[:, None].astype(m.dtype) * h[seg]

        diff = end(pairs[:, 0], pairs[:, 1]) - end(pairs[:, 2], pairs[:, 3])
        return mx.mean(mx.sum(diff * diff, axis=-1))

    def _refresh_pairs(self) -> None:
        from .model import nearest_other_within

        n = self.model.num_segments
        a, b = self.model.endpoints
        pts = np.concatenate([np.array(a), np.array(b)])
        owner = np.concatenate([np.arange(n), np.arange(n)])
        width = np.array(mx.mean(self.model.scales_act, axis=-1))
        radius = self.config.connect_radius * np.concatenate([width, width])
        nbr = nearest_other_within(pts, owner, radius)
        i = np.nonzero(nbr >= 0)[0]
        j = nbr[i]
        side = np.where(np.arange(2 * n) < n, -1, 1)
        rows = np.stack([i % n, side[i], j % n, side[j]], axis=1)
        self._pairs = mx.array(rows.astype(np.int32))

    def loss(self, params, camera: Camera, target: mx.array, background: mx.array):
        c = self.config
        out = self.model.render_params(params, camera, background)
        img = out["image"]
        loss = (1.0 - c.lambda_dssim) * mx.abs(img - target).mean()
        loss = loss + c.lambda_dssim * (1.0 - ssim(img, target))
        if c.lambda_opacity > 0:
            loss = loss + c.lambda_opacity * mx.mean(mx.sigmoid(params["opacities"]))
        if c.lambda_scale > 0:
            loss = loss + c.lambda_scale * mx.mean(mx.exp(params["scales"]))
        if c.lambda_connect > 0 and self.step_count >= c.connect_from:
            loss = loss + c.lambda_connect * self._attraction(params) / self.scene_extent**2
        return loss, img

    # ------------------------------------------------------------------ step
    def step(self, camera: Camera, target: mx.array) -> dict[str, object]:
        """One optimization step on one view; returns logging info."""
        self.step_count += 1
        c = self.config
        bg = mx.ones((3,)) if c.white_background else mx.zeros((3,))
        if c.lambda_connect > 0 and self.step_count >= c.connect_from:
            if self._pairs is None or self.step_count % c.connect_every == 0:
                self._refresh_pairs()

        (loss, _), grads = mx.value_and_grad(lambda p: self.loss(p, camera, target, bg))(
            self.model.params
        )
        for k, opt in self.optimizers.items():
            self.model.params[k] = opt.apply_gradients({k: grads[k]}, {k: self.model.params[k]})[k]
        del grads
        self._add_noise()
        mx.eval(self.model.params)

        density = None
        if (
            c.densify_from < self.step_count <= c.densify_until
            and self.step_count % c.densify_every == 0
        ):
            density = self._relocate_and_grow()
            self._pairs = None  # rows moved: endpoint pairs are stale

        if self.step_count % c.sh_increase_every == 0:
            self.model.one_up_sh_degree()
        return {
            "loss": float(loss),
            "num_segments": self.model.num_segments,
            "step": self.step_count,
            "density": density,
        }

    def _add_noise(self) -> None:
        """MCMC exploration: Sigma-shaped position noise, gated by 3DGS-MCMC's
        ``sigmoid(-100 (o - min_opacity))``: strong for near-dead segments,
        negligible for ones that are contributing."""
        c = self.config
        if c.noise_lr <= 0:
            return
        p = self.model.params
        lr = float(self.optimizers["means"].learning_rate)
        o = mx.sigmoid(p["opacities"])
        gate = mx.sigmoid(-100.0 * (o - c.min_opacity))
        # As in 3DGS-MCMC, the noise is multiplied by the kernel covariance
        # Sigma = R S^2 R^T itself (not its square root).
        R = quaternion_to_matrix(p["quats"] / mx.linalg.norm(p["quats"], axis=-1, keepdims=True))
        eps = mx.random.normal(p["means"].shape)
        local = (mx.swapaxes(R, -1, -2) @ eps[..., None])[..., 0] * mx.exp(2.0 * p["scales"])
        noise = (R @ local[..., None])[..., 0] * (gate * c.noise_lr * lr)[:, None]
        p["means"] = p["means"] + mx.stop_gradient(noise)

    # ------------------------------------------------------- density control
    def _sample_sources(self, k: int, exclude: mx.array | None = None) -> mx.array:
        """``k`` distinct live segments sampled proportionally to opacity
        (Gumbel top-k)."""
        o = self.model.opacities_act
        logits = mx.log(mx.maximum(o, 1e-12))
        if exclude is not None:
            logits = mx.where(exclude, -1e30, logits)
        gumbel = -mx.log(-mx.log(mx.random.uniform(shape=o.shape, low=1e-12, high=1.0)))
        return mx.argpartition(-(logits + gumbel), kth=k - 1)[:k]

    def _relocate_and_grow(self) -> dict[str, int]:
        c = self.config
        model = self.model
        n = model.num_segments
        width = mx.mean(model.scales_act, axis=-1)
        dead = model.opacities_act < c.min_opacity
        n_dead = int(dead.sum())
        relocated = 0
        if 0 < n_dead < n:
            dst = mx.argpartition(-dead.astype(mx.float32), kth=n_dead - 1)[:n_dead]
            src = self._sample_sources(n_dead, exclude=dead)
            new = model.split_or_clone(src, c.split_ratio * width[src])
            for k, v in model.params.items():
                model.params[k] = v.at[dst].add(new[k] - v[dst])
            self._reset_moments(mx.concatenate([src, dst]))
            relocated = n_dead

        n_add = min(c.max_segments - n, int(c.grow_rate * n))
        if n_add > 0:
            src = self._sample_sources(n_add)
            width = mx.mean(model.scales_act, axis=-1)
            new = model.split_or_clone(src, c.split_ratio * width[src])
            model.append(new)
            self._grow_moments(n_add)
            self._reset_moments(src)
        mx.eval(model.params)
        return {"relocated": relocated, "added": max(n_add, 0)}

    def _reset_moments(self, rows: mx.array) -> None:
        for name, opt in self.optimizers.items():
            state = opt.state.get(name)
            if isinstance(state, dict):
                for slot in ("m", "v"):
                    if slot in state:
                        state[slot] = state[slot].at[rows].multiply(0.0)

    def _grow_moments(self, count: int) -> None:
        for name, opt in self.optimizers.items():
            state = opt.state.get(name)
            if isinstance(state, dict):
                for slot in ("m", "v"):
                    if slot in state:
                        old = state[slot]
                        pad = mx.zeros((count, *old.shape[1:]), dtype=old.dtype)
                        state[slot] = mx.concatenate([old, pad], axis=0)

    def finalize_polylines(self, radius_widths: float | None = None) -> int:
        """Snap nearby endpoints into shared joints (the paper's merging step);
        returns the number of joints created."""
        r = self.config.connect_radius if radius_widths is None else radius_widths
        width = np.array(mx.mean(self.model.scales_act, axis=-1))
        return self.model.snap_endpoints(r * width)


def default_config_for(iterations: int, **overrides) -> SpaghettiTrainerConfig:
    """Schedule scaled to ``iterations``: densify until 5/6 of the run, attract
    endpoints in the final sixth (the paper's 25k / 30k split)."""
    stop = int(iterations * 5 / 6)
    base = dict(
        lr_means_max_steps=iterations,
        densify_until=stop,
        connect_from=stop,
        densify_from=min(500, max(iterations // 20, 1)),
    )
    base.update(overrides)
    return SpaghettiTrainerConfig(**base)
