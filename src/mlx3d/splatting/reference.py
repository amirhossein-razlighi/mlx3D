"""Naive pure-MLX Gaussian rasterizer used as a correctness oracle in tests.

Evaluates every Gaussian at every pixel (O(H * W * N)) with the same math as
the Metal kernels, relying on MLX autodiff for gradients. Only use it for
small scenes.
"""

import mlx.core as mx

from ..cameras import Camera
from .projection import project_gaussians

__all__ = ["render_gaussians_reference"]


def render_gaussians_reference(
    camera: Camera,
    means: mx.array,
    quats: mx.array,
    scales: mx.array,
    opacities: mx.array,
    colors: mx.array,
    background: mx.array | None = None,
) -> dict[str, mx.array]:
    """Reference renderer matching :func:`mlx3d.splatting.render_gaussians`."""
    if background is None:
        background = mx.zeros((3,))
    H, W = camera.height, camera.width

    proj = project_gaussians(camera, means, quats, scales)
    means2d, conics, depths, radii = (
        proj["means2d"], proj["conics"], proj["depths"], proj["radii"],
    )

    # Depth-sort all Gaussians (global front-to-back order).
    order = mx.argsort(mx.stop_gradient(depths))
    means2d = means2d[order]
    conics = conics[order]
    colors = colors[order]
    opac = opacities[order]
    radii = radii[order]

    xs = mx.arange(W, dtype=mx.float32) + 0.5
    ys = mx.arange(H, dtype=mx.float32) + 0.5
    px = mx.broadcast_to(xs[None, :], (H, W))
    py = mx.broadcast_to(ys[:, None], (H, W))

    dx = means2d[:, 0][:, None, None] - px[None]  # (N, H, W)
    dy = means2d[:, 1][:, None, None] - py[None]
    a = conics[:, 0][:, None, None]
    b = conics[:, 1][:, None, None]
    c = conics[:, 2][:, None, None]
    power = -0.5 * (a * dx * dx + c * dy * dy) - b * dx * dy
    alpha = mx.minimum(opac[:, None, None] * mx.exp(power), 0.99)
    # Match the kernel cutoffs exactly.
    alpha = mx.where(power > 0.0, mx.zeros_like(alpha), alpha)
    alpha = mx.where(alpha < 1.0 / 255.0, mx.zeros_like(alpha), alpha)
    alpha = alpha * (radii > 0)[:, None, None]

    one_minus = 1.0 - alpha
    trans = mx.cumprod(one_minus, axis=0)
    trans = mx.concatenate([mx.ones_like(trans[:1]), trans[:-1]], axis=0)
    # Early-termination threshold of the kernel: stop once T drops below 1e-4.
    keep = trans > 1e-4
    weights = alpha * trans * keep  # (N, H, W)

    image = mx.sum(weights[..., None] * colors[:, None, None, :], axis=0)
    final_T = mx.prod(mx.where(keep, one_minus, mx.ones_like(one_minus)), axis=0)
    image = image + final_T[..., None] * background
    return {"image": image, "alpha": 1.0 - final_T, "final_T": final_T}
