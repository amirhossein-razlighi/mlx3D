"""Occupancy-accelerated ray rendering for NeRF fields.

Empty space dominates most scenes, yet a dense NeRF still evaluates its MLP at
every sample. Given an :class:`OccupancyGrid`, this renderer evaluates the field
**only** at samples in occupied cells: it compacts the occupied samples into a
fixed-size buffer (MLX has no boolean indexing, so compaction is done with an
``argsort``), runs the network once on that buffer, and scatters the results
back. Empty samples contribute zero density, exactly as they should.

This is the Instant-NGP empty-space-skipping idea expressed in pure MLX. The
win grows with how empty the scene is and how heavy the field MLP is.
"""

from __future__ import annotations

from typing import Callable

import mlx.core as mx

from ..renderer.rays import sample_along_rays, volume_render
from .occupancy import OccupancyGrid

__all__ = ["render_rays_occupancy"]

Field = Callable[[mx.array, mx.array], tuple[mx.array, mx.array]]


def render_rays_occupancy(
    model: Field,
    origins: mx.array,
    directions: mx.array,
    near: float,
    far: float,
    grid: OccupancyGrid,
    num_samples: int = 128,
    eval_fraction: float = 1.0,
    stratified: bool = False,
    white_background: bool = False,
) -> dict[str, mx.array]:
    """Render rays, evaluating ``model`` only at occupied samples.

    Args:
        model: a field ``model(points, directions) -> (density, rgb)`` (e.g.
            :class:`HashGridNeRF`).
        origins: ``(R, 3)`` ray origins.
        directions: ``(R, 3)`` ray directions.
        near: near sampling bound.
        far: far sampling bound.
        grid: occupancy cache identifying non-empty space (kept fixed / detached).
        num_samples: samples per ray.
        eval_fraction: fraction of all ``R * num_samples`` samples to actually
            evaluate (the compaction budget). With sparse occupancy a small
            fraction covers every occupied sample; the rest are forced empty.
        stratified: jitter samples (training) vs. deterministic (eval).
        white_background: composite onto white.

    Returns:
        Same dict as :func:`~mlx3d.renderer.volume_render`.
    """
    r = origins.shape[0]
    points, t_vals = sample_along_rays(origins, directions, near, far, num_samples, stratified)
    view = mx.broadcast_to(directions[:, None, :], points.shape)

    m = r * num_samples
    pts = points.reshape(m, 3)
    views = view.reshape(m, 3)
    occupied = grid.query(pts)  # (M,) bool, detached (grid is constant)

    budget = max(1, min(m, int(m * eval_fraction)))
    # Bring occupied samples to the front, then keep the first `budget`.
    order = mx.argsort((~occupied).astype(mx.int32))
    idx = order[:budget]
    keep = occupied[idx].astype(mx.float32)[:, None]  # zero out any empty stragglers

    density_c, rgb_c = model(pts[idx], views[idx])
    density_c = density_c[:, None] * keep
    rgb_c = rgb_c * keep

    density = mx.zeros((m, 1)).at[idx].add(density_c).reshape(r, num_samples)
    rgb = mx.zeros((m, 3)).at[idx].add(rgb_c).reshape(r, num_samples, 3)
    return volume_render(density, rgb, t_vals, directions, white_background)
