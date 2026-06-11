"""Tile binning for the Gaussian Splatting rasterizer.

Each Gaussian is duplicated once per 16x16 screen tile its 3-sigma extent
touches, then all duplicates are sorted by ``(tile, depth)``. The Metal
rasterizer consumes the resulting per-tile contiguous ranges.

None of this carries gradients, but it runs every frame, so it is built
entirely from vectorized MLX ops (cumsum-scatter expansion instead of a
``searchsorted``).
"""

import mlx.core as mx

TILE_SIZE = 16

__all__ = ["TILE_SIZE", "bin_gaussians"]


def bin_gaussians(
    means2d: mx.array,
    radii: mx.array,
    depths: mx.array,
    width: int,
    height: int,
) -> tuple[mx.array, mx.array, int, int]:
    """Assign Gaussians to screen tiles and sort by (tile, depth).

    Args:
        means2d: (N, 2) pixel-space centers.
        radii: (N,) pixel radii; 0 means culled.
        depths: (N,) camera-space depths.

    Returns:
        ``(sorted_ids, tile_ranges, tiles_x, tiles_y)`` where ``sorted_ids``
        is (D,) int32 Gaussian indices for all duplicates in render order and
        ``tile_ranges`` is (tiles_x * tiles_y, 2) int32 [start, end) ranges
        into ``sorted_ids``.
    """
    means2d = mx.stop_gradient(means2d)
    radii = mx.stop_gradient(radii)
    depths = mx.stop_gradient(depths)

    N = means2d.shape[0]
    tiles_x = (width + TILE_SIZE - 1) // TILE_SIZE
    tiles_y = (height + TILE_SIZE - 1) // TILE_SIZE
    num_tiles = tiles_x * tiles_y

    # Inclusive tile bounding box of each Gaussian's 3-sigma square.
    x, y = means2d[:, 0], means2d[:, 1]
    r = radii
    xmin = mx.clip(mx.floor((x - r) / TILE_SIZE), 0, tiles_x - 1).astype(mx.int32)
    xmax = mx.clip(mx.floor((x + r) / TILE_SIZE), 0, tiles_x - 1).astype(mx.int32)
    ymin = mx.clip(mx.floor((y - r) / TILE_SIZE), 0, tiles_y - 1).astype(mx.int32)
    ymax = mx.clip(mx.floor((y + r) / TILE_SIZE), 0, tiles_y - 1).astype(mx.int32)

    on_screen = (
        (radii > 0)
        & (x + r >= 0) & (x - r < width)
        & (y + r >= 0) & (y - r < height)
    )
    w_tiles = (xmax - xmin + 1) * on_screen
    h_tiles = (ymax - ymin + 1) * on_screen
    counts = (w_tiles * h_tiles).astype(mx.int32)  # (N,)

    offsets = mx.cumsum(counts) - counts  # exclusive prefix sum
    total = int(counts.sum().item())
    if total == 0:
        return (
            mx.zeros((0,), dtype=mx.int32),
            mx.zeros((num_tiles, 2), dtype=mx.int32),
            tiles_x,
            tiles_y,
        )

    # Expand: duplicate j belongs to Gaussian g(j). Scatter a 1 at each
    # Gaussian's first duplicate slot, cumsum, subtract 1. Zero-count
    # Gaussians scatter onto the same slot as their successor, and the
    # cumulative sum skips them correctly.
    marker = mx.zeros((total,), dtype=mx.int32)
    marker = marker.at[offsets].add(mx.ones((N,), dtype=mx.int32))
    gauss_id = mx.cumsum(marker) - 1  # (D,)

    local = mx.arange(total, dtype=mx.int32) - offsets[gauss_id]  # rank within bbox
    gw = mx.maximum(w_tiles[gauss_id], 1)
    tile_x = xmin[gauss_id] + local % gw
    tile_y = ymin[gauss_id] + local // gw
    tile_id = (tile_y * tiles_x + tile_x).astype(mx.int64)

    # Depth ranks (dense, < N) make an exact composite sort key.
    order = mx.argsort(depths)
    ranks = mx.zeros((N,), dtype=mx.int32).at[order].add(mx.arange(N, dtype=mx.int32))
    key = tile_id * N + ranks[gauss_id].astype(mx.int64)
    sort_idx = mx.argsort(key)

    sorted_ids = gauss_id[sort_idx].astype(mx.int32)
    sorted_tiles = tile_id[sort_idx].astype(mx.int32)

    # Per-tile [start, end) ranges via scatter-min / scatter-max.
    positions = mx.arange(total, dtype=mx.int32)
    starts = mx.full((num_tiles,), total, dtype=mx.int32).at[sorted_tiles].minimum(positions)
    ends = mx.zeros((num_tiles,), dtype=mx.int32).at[sorted_tiles].maximum(positions + 1)
    starts = mx.minimum(starts, ends)  # empty tiles -> start == end == 0
    tile_ranges = mx.stack([starts, ends], axis=-1)

    return sorted_ids, tile_ranges, tiles_x, tiles_y
