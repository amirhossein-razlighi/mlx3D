"""A simple differentiable point-cloud renderer.

Each point is splatted as an isotropic Gaussian over a small pixel window and
accumulated with depth-aware soft weights (SoftRas-style aggregation). The
result is differentiable with respect to point positions, colors and radii,
which is enough for silhouette/color-based point cloud optimization.
"""

import mlx.core as mx

from ..cameras import Camera

__all__ = ["render_points"]


def render_points(
    camera: Camera,
    points: mx.array,
    colors: mx.array | None = None,
    radius: float = 2.0,
    window: int = 5,
    depth_temperature: float = 10.0,
    background: float = 0.0,
    eps: float = 1e-8,
) -> dict[str, mx.array]:
    """Render a point cloud with soft Gaussian splats.

    Args:
        camera: the :class:`Camera` to render from.
        points: (P, 3) world-space positions.
        colors: (P, 3) per-point colors in [0, 1] (defaults to white).
        radius: Gaussian sigma in pixels.
        window: splat window size in pixels (odd).
        depth_temperature: sharpness of the soft depth weighting; larger
            values approach hard z-ordering.
        background: scalar background intensity.

    Returns:
        dict with ``image`` (H, W, 3), ``alpha`` (H, W) coverage, and
        ``depth`` (H, W) soft depth.
    """
    H, W = camera.height, camera.width
    P = points.shape[0]
    if colors is None:
        colors = mx.ones((P, 3))

    xy, z = camera.project_points(points)  # (P, 2), (P,)
    in_front = z > camera.znear

    # Window pixel offsets around each point's containing pixel.
    half = window // 2
    base = mx.stop_gradient(mx.floor(xy))  # (P, 2) gradient flows via the Gaussian
    offs = mx.arange(-half, half + 1, dtype=mx.float32)
    ox = mx.broadcast_to(offs[None, :], (window, window)).reshape(-1)
    oy = mx.broadcast_to(offs[:, None], (window, window)).reshape(-1)
    px = base[:, 0:1] + ox[None, :]  # (P, K)
    py = base[:, 1:2] + oy[None, :]

    # Gaussian weight from the continuous projected position to pixel centers.
    dx = px + 0.5 - xy[:, 0:1]
    dy = py + 0.5 - xy[:, 1:2]
    w = mx.exp(-(dx * dx + dy * dy) / (2.0 * radius * radius))  # (P, K)

    # Depth-aware weighting: nearer points dominate (soft z-buffer).
    inv_depth = 1.0 / mx.maximum(z, camera.znear)
    depth_w = mx.exp(depth_temperature * (inv_depth - inv_depth.max()))
    w = w * depth_w[:, None] * in_front[:, None]

    # Mask out-of-bounds pixels and flatten indices.
    valid = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    w = w * valid
    idx = (mx.clip(py, 0, H - 1) * W + mx.clip(px, 0, W - 1)).astype(mx.int32)  # (P, K)
    flat_idx = idx.reshape(-1)
    flat_w = w.reshape(-1)

    wc = (w[..., None] * colors[:, None, :]).reshape(-1, 3)  # (P*K, 3)
    wz = (w * z[:, None]).reshape(-1)

    sum_wc = mx.zeros((H * W, 3)).at[flat_idx].add(wc)
    sum_w = mx.zeros((H * W,)).at[flat_idx].add(flat_w)
    sum_wz = mx.zeros((H * W,)).at[flat_idx].add(wz)

    norm = mx.maximum(sum_w, eps)[:, None]
    image = sum_wc / norm
    alpha = 1.0 - mx.exp(-sum_w)  # soft coverage
    image = image * alpha[:, None] + background * (1.0 - alpha[:, None])

    return {
        "image": image.reshape(H, W, 3),
        "alpha": alpha.reshape(H, W),
        "depth": (sum_wz / mx.maximum(sum_w, eps)).reshape(H, W),
    }
