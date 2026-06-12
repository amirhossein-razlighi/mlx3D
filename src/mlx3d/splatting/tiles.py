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
    conics: mx.array | None = None,
) -> tuple[mx.array, mx.array, int, int]:
    """Assign Gaussians to screen tiles and sort by (tile, depth).

    Args:
        means2d: (N, 2) pixel-space centers.
        radii: (N,) pixel radii; 0 means culled.
        depths: (N,) camera-space depths.
        conics: optional (N, 3) inverse 2D covariance upper-triangular
            coefficients ``(a, b, c)``. When provided, duplicate tiles whose
            rectangle cannot intersect the Gaussian's 3-sigma ellipse are
            conservatively rejected after the radius-bbox pass.

    Returns:
        ``(sorted_ids, tile_ranges, tiles_x, tiles_y)`` where ``sorted_ids``
        is (D,) int32 Gaussian indices for all duplicates in render order and
        ``tile_ranges`` is (tiles_x * tiles_y, 2) int32 [start, end) ranges
        into ``sorted_ids``.
    """
    means2d = mx.stop_gradient(means2d)
    radii = mx.stop_gradient(radii)
    depths = mx.stop_gradient(depths)
    conics = None if conics is None else mx.stop_gradient(conics)

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

    on_screen = (radii > 0) & (x + r >= 0) & (x - r < width) & (y + r >= 0) & (y - r < height)
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
    tile_id = (tile_y * tiles_x + tile_x).astype(mx.int32)

    if conics is not None:
        # Conservative AccuTile-style refinement: for each duplicate generated
        # by the square 3-sigma bbox, find the minimum conic distance over the
        # continuous tile rectangle. The renderer skips alpha < 1/255, so keep
        # a margin beyond 3 sigma to preserve faint high-opacity tails.
        x0 = (tile_x * TILE_SIZE).astype(mx.float32)
        y0 = (tile_y * TILE_SIZE).astype(mx.float32)
        x1 = mx.minimum((tile_x + 1) * TILE_SIZE, width).astype(mx.float32)
        y1 = mx.minimum((tile_y + 1) * TILE_SIZE, height).astype(mx.float32)

        gx = x[gauss_id]
        gy = y[gauss_id]
        dx_min = x0 - gx
        dx_max = x1 - gx
        dy_min = y0 - gy
        dy_max = y1 - gy

        co = conics[gauss_id]
        a = co[:, 0]
        b = co[:, 1]
        c = co[:, 2]

        def q(dx, dy):
            return a * dx * dx + 2.0 * b * dx * dy + c * dy * dy

        zero = mx.zeros_like(dx_min)
        inside = (dx_min <= 0) & (dx_max >= 0) & (dy_min <= 0) & (dy_max >= 0)

        # Check corners plus the four edge-wise minimizers of the positive
        # definite quadratic form. This gives the exact minimum over the box.
        dx_left = dx_min
        dx_right = dx_max
        dy_bottom = dy_min
        dy_top = dy_max

        dy_at_left = mx.clip(-(b / c) * dx_left, dy_bottom, dy_top)
        dy_at_right = mx.clip(-(b / c) * dx_right, dy_bottom, dy_top)
        dx_at_bottom = mx.clip(-(b / a) * dy_bottom, dx_left, dx_right)
        dx_at_top = mx.clip(-(b / a) * dy_top, dx_left, dx_right)

        qmin = q(dx_left, dy_bottom)
        qmin = mx.minimum(qmin, q(dx_left, dy_top))
        qmin = mx.minimum(qmin, q(dx_right, dy_bottom))
        qmin = mx.minimum(qmin, q(dx_right, dy_top))
        qmin = mx.minimum(qmin, q(dx_left, dy_at_left))
        qmin = mx.minimum(qmin, q(dx_right, dy_at_right))
        qmin = mx.minimum(qmin, q(dx_at_bottom, dy_bottom))
        qmin = mx.minimum(qmin, q(dx_at_top, dy_top))
        qmin = mx.where(inside, zero, qmin)

        keep = qmin <= 12.0
        keep_i = keep.astype(mx.int32)
        active_total = int(keep_i.sum().item())
        if active_total == 0:
            return (
                mx.zeros((0,), dtype=mx.int32),
                mx.zeros((num_tiles, 2), dtype=mx.int32),
                tiles_x,
                tiles_y,
            )

        active_pos = mx.cumsum(keep_i) - 1
        safe_pos = mx.where(keep, active_pos, mx.zeros_like(active_pos))
        gauss_id = (
            mx.zeros((active_total,), dtype=mx.int32)
            .at[safe_pos]
            .add(mx.where(keep, gauss_id, mx.zeros_like(gauss_id)))
        )
        tile_id = (
            mx.zeros((active_total,), dtype=mx.int32)
            .at[safe_pos]
            .add(mx.where(keep, tile_id, mx.zeros_like(tile_id)))
        )
        total = active_total

    # Depth ranks (dense, < N) make an exact composite sort key.
    order = mx.argsort(depths)
    ranks = mx.zeros((N,), dtype=mx.int32).at[order].add(mx.arange(N, dtype=mx.int32))
    key = tile_id.astype(mx.int64) * N + ranks[gauss_id].astype(mx.int64)
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
