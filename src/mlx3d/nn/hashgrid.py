"""Multi-resolution hash-grid encoding for NeRF-style fields."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

__all__ = ["HashGridEncoding"]


class HashGridEncoding(nn.Module):
    """Trainable multi-resolution 3D hash-grid encoder.

    This follows the Instant-NGP idea: points are normalized to a unit cube,
    each level performs trilinear interpolation over hashed grid vertices, and
    all level features are concatenated.
    """

    def __init__(
        self,
        num_levels: int = 12,
        features_per_level: int = 2,
        log2_hashmap_size: int = 15,
        base_resolution: int = 16,
        finest_resolution: int = 512,
        bounds: tuple[float, float] = (-1.0, 1.0),
    ):
        super().__init__()
        self.num_levels = int(num_levels)
        self.features_per_level = int(features_per_level)
        self.hashmap_size = 1 << int(log2_hashmap_size)
        self.base_resolution = int(base_resolution)
        self.finest_resolution = int(finest_resolution)
        self.bounds = bounds
        self.per_level_scale = (
            math.exp(math.log(finest_resolution / base_resolution) / max(num_levels - 1, 1))
            if num_levels > 1
            else 1.0
        )
        self.tables = [
            mx.random.uniform(
                low=-1e-4,
                high=1e-4,
                shape=(self.hashmap_size, self.features_per_level),
            )
            for _ in range(self.num_levels)
        ]

    @property
    def output_dim(self) -> int:
        return self.num_levels * self.features_per_level

    def _hash(self, coords: mx.array) -> mx.array:
        coords = coords.astype(mx.uint32)
        x, y, z = coords[..., 0], coords[..., 1], coords[..., 2]
        h = x * mx.array(1, dtype=mx.uint32)
        h = h ^ (y * mx.array(2654435761, dtype=mx.uint32))
        h = h ^ (z * mx.array(805459861, dtype=mx.uint32))
        return (h % self.hashmap_size).astype(mx.int32)

    def __call__(self, x: mx.array) -> mx.array:
        lo, hi = self.bounds
        x = (x - lo) / max(float(hi - lo), 1e-8)
        x = mx.clip(x, 0.0, 1.0)
        outs = []
        offsets = mx.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [0, 1, 1],
                [1, 1, 1],
            ],
            dtype=mx.int32,
        )
        for level, table in enumerate(self.tables):
            res = int(math.floor(self.base_resolution * (self.per_level_scale**level)))
            res = max(res, 2)
            p = x * (res - 1)
            p0 = mx.floor(p).astype(mx.int32)
            frac = p - p0.astype(mx.float32)
            corner = mx.minimum(p0[..., None, :] + offsets, res - 1)
            idx = self._hash(corner)
            feat = table[idx]
            wx = mx.where(offsets[:, 0] == 1, frac[..., 0:1], 1.0 - frac[..., 0:1])
            wy = mx.where(offsets[:, 1] == 1, frac[..., 1:2], 1.0 - frac[..., 1:2])
            wz = mx.where(offsets[:, 2] == 1, frac[..., 2:3], 1.0 - frac[..., 2:3])
            weight = (wx * wy * wz)[..., None]
            outs.append(mx.sum(feat * weight, axis=-2))
        return mx.concatenate(outs, axis=-1)
