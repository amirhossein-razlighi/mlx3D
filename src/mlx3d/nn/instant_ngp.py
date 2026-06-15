"""Instant-NGP-style NeRF: multi-resolution hash-grid encoding + a tiny MLP.

Trades the large 8-layer MLP of the original NeRF for a trainable hash grid
(:class:`HashGridEncoding`) feeding two small MLPs. It converges roughly an
order of magnitude faster — minutes instead of hours — at similar quality, and
drops straight into :func:`mlx3d.nn.render_rays` since it exposes the same
``model(points, directions) -> (density, rgb)`` call.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .hashgrid import HashGridEncoding
from .nerf import PositionalEncoding

__all__ = ["HashGridNeRF"]


def _trunc_exp(x: mx.array) -> mx.array:
    """Exponential density activation with a clamped argument for stability."""
    return mx.exp(mx.clip(x, -15.0, 8.0))


class HashGridNeRF(nn.Module):
    """A compact hash-grid NeRF (Instant-NGP style).

    The hash-grid hyperparameters (``num_levels``, ``features_per_level``,
    ``log2_hashmap_size``, ``base_resolution``, ``finest_resolution``) are
    forwarded to :class:`~mlx3d.nn.HashGridEncoding`.

    Args:
        bounds: axis-aligned scene bounds the hash grid covers; sample points
            should lie within this cube (density is zeroed outside it).
        geo_feat_dim: size of the geometry feature passed to the color MLP.
        hidden_dim: width of both small MLPs.
        dir_freqs: positional-encoding frequencies for the view direction.
    """

    def __init__(
        self,
        bounds: tuple[float, float] = (-1.5, 1.5),
        num_levels: int = 16,
        features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        finest_resolution: int = 1024,
        geo_feat_dim: int = 15,
        hidden_dim: int = 64,
        dir_freqs: int = 4,
    ):
        super().__init__()
        self.bounds = bounds
        self.encoding = HashGridEncoding(
            num_levels=num_levels,
            features_per_level=features_per_level,
            log2_hashmap_size=log2_hashmap_size,
            base_resolution=base_resolution,
            finest_resolution=finest_resolution,
            bounds=bounds,
        )
        self.dir_enc = PositionalEncoding(dir_freqs)
        self.geo_feat_dim = geo_feat_dim

        enc_dim = self.encoding.output_dim
        dir_dim = 3 * self.dir_enc.output_dim_multiplier

        # Density MLP: hash features -> [density, geometry features].
        self.sigma_l1 = nn.Linear(enc_dim, hidden_dim)
        self.sigma_l2 = nn.Linear(hidden_dim, 1 + geo_feat_dim)
        # Color MLP: [geometry features, encoded direction] -> RGB.
        self.color_l1 = nn.Linear(geo_feat_dim + dir_dim, hidden_dim)
        self.color_l2 = nn.Linear(hidden_dim, hidden_dim)
        self.color_l3 = nn.Linear(hidden_dim, 3)

    def __call__(self, points: mx.array, directions: mx.array) -> tuple[mx.array, mx.array]:
        h = self.encoding(points)
        h = nn.relu(self.sigma_l1(h))
        h = self.sigma_l2(h)
        density = _trunc_exp(h[..., 0])
        geo = h[..., 1:]

        # Outside the scene AABB there is no geometry. Zeroing density there is
        # both physically correct and essential for the hash grid: points beyond
        # the bounds get clamped onto the cube face and would otherwise share
        # garbage features, preventing the field from localizing the object.
        lo, hi = self.bounds
        inside = mx.all((points >= lo) & (points <= hi), axis=-1)
        density = density * inside

        d = self.dir_enc(directions)
        c = mx.concatenate([geo, d], axis=-1)
        c = nn.relu(self.color_l1(c))
        c = nn.relu(self.color_l2(c))
        rgb = mx.sigmoid(self.color_l3(c))
        return density, rgb
