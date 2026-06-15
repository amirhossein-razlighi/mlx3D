"""Occupancy grid for empty-space skipping in NeRF-style volume rendering.

A coarse 3D grid caches where the density field is non-empty. Periodically
refreshed from the field during training, it lets the renderer ignore samples
in empty space — concentrating capacity (and, with compaction, compute) on the
occupied region. This is the core of the Instant-NGP acceleration.
"""

from __future__ import annotations

from typing import Callable

import mlx.core as mx

__all__ = ["OccupancyGrid"]


class OccupancyGrid:
    """A dense ``res^3`` occupancy cache over an axis-aligned box.

    Args:
        resolution: cells per axis.
        bounds: ``(lo, hi)`` world extent of the grid (same on every axis).
    """

    def __init__(self, resolution: int = 128, bounds: tuple[float, float] = (-1.5, 1.5)):
        self.resolution = int(resolution)
        self.bounds = bounds
        self.occupancy = mx.zeros((resolution, resolution, resolution), dtype=mx.bool_)

    def _cell_centers(self) -> mx.array:
        lo, hi = self.bounds
        r = self.resolution
        c = (mx.arange(r, dtype=mx.float32) + 0.5) / r  # cell centers in [0, 1)
        coords = lo + c * (hi - lo)
        gz, gy, gx = mx.meshgrid(coords, coords, coords, indexing="ij")
        return mx.stack([gx, gy, gz], axis=-1)  # (r, r, r, 3)

    def update(
        self,
        density_fn: Callable[[mx.array], mx.array],
        threshold: float = 0.01,
        chunk: int = 1 << 18,
    ) -> None:
        """Refresh occupancy by thresholding the density field at cell centers.

        Args:
            density_fn: callable mapping ``(P, 3)`` points to ``(P,)`` densities.
            threshold: cells with density above this are marked occupied.
            chunk: points evaluated per batch (bounds memory for fine grids).
        """
        centers = self._cell_centers().reshape(-1, 3)
        out = []
        for s in range(0, centers.shape[0], chunk):
            out.append(mx.stop_gradient(density_fn(centers[s : s + chunk])))
        density = mx.concatenate(out) if len(out) > 1 else out[0]
        r = self.resolution
        self.occupancy = (density > threshold).reshape(r, r, r)
        mx.eval(self.occupancy)

    def query(self, points: mx.array) -> mx.array:
        """Return a boolean mask of which ``(..., 3)`` points fall in occupied cells.

        Points outside the grid bounds are reported empty.
        """
        lo, hi = self.bounds
        r = self.resolution
        norm = (points - lo) / (hi - lo)  # -> [0, 1]
        idx = mx.floor(norm * r).astype(mx.int32)
        inside = mx.all((idx >= 0) & (idx < r), axis=-1)
        ci = mx.clip(idx, 0, r - 1)
        flat = (ci[..., 0] * r + ci[..., 1]) * r + ci[..., 2]
        occ = self.occupancy.reshape(-1)[flat]
        return occ & inside

    @property
    def occupied_fraction(self) -> float:
        """Fraction of cells currently marked occupied (useful for diagnostics)."""
        return float(self.occupancy.astype(mx.float32).mean())
