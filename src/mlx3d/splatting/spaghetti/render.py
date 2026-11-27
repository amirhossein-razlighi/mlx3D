"""Differentiable rendering of spaghetti primitives."""

from __future__ import annotations

import mlx.core as mx

from ...cameras import Camera
from ..sh import eval_sh
from ..tiles import bin_gaussians
from .projection import project_segments, segment_optical_depth
from .rasterize import pack_segments, rasterize_segments

__all__ = ["render_spaghetti", "render_spaghetti_reference", "segment_colors"]


def segment_colors(camera: Camera, means: mx.array, sh: mx.array, sh_degree: int) -> mx.array:
    """View-dependent RGB per segment: SH evaluated toward the segment midpoint."""
    dirs = means - camera.camera_center
    dirs = dirs * mx.rsqrt(mx.maximum(mx.sum(dirs * dirs, axis=-1, keepdims=True), 1e-16))
    return mx.maximum(eval_sh(sh_degree, sh, mx.stop_gradient(dirs)), 0.0)


def render_spaghetti(
    camera: Camera,
    means: mx.array,
    halves: mx.array,
    quats: mx.array,
    scales: mx.array,
    rho: mx.array,
    colors: mx.array,
    background: mx.array | None = None,
) -> dict[str, mx.array]:
    """Render Gaussian-blurred segments with the tiled Metal rasterizer.

    Differentiable with respect to every per-segment input and, through
    ``camera``'s extrinsics, the pose (e.g. via :func:`~mlx3d.cameras.refine_camera`).

    Args:
        means, halves: (N, 3) midpoints and half-vectors; segment = ``m ± h``.
        quats, scales: (N, 4) / (N, 3) kernel rotation and standard deviations.
        rho: (N,) densities. colors: (N, 3) RGB.

    Returns:
        dict with ``image`` (H, W, 3), ``alpha`` (H, W) and the projection
        (``radii``, ``means2d``, ``depths``) used for binning.
    """
    proj = project_segments(camera, means, halves, quats, scales, rho)
    sorted_ids, tile_ranges, tiles_x, tiles_y = bin_gaussians(
        proj["means2d"], proj["radii"], proj["depths"], camera.width, camera.height
    )
    out = rasterize_segments(
        pack_segments(proj["W"], proj["a"], proj["v"], proj["rho"]),
        colors,
        proj["means2d"],
        proj["conics"],
        sorted_ids,
        tile_ranges,
        (camera.fx, camera.fy, camera.cx, camera.cy),
        camera.width,
        camera.height,
        tiles_x,
        tiles_y,
        background=background,
    )
    out.update(radii=proj["radii"], means2d=proj["means2d"], depths=proj["depths"])
    return out


def pixel_rays(camera: Camera) -> mx.array:
    """(H, W, 3) unit camera-space ray directions through pixel centers."""
    u = (mx.arange(camera.width, dtype=mx.float32) + 0.5 - camera.cx) / camera.fx
    v = (mx.arange(camera.height, dtype=mx.float32) + 0.5 - camera.cy) / camera.fy
    d = mx.stack(
        [
            mx.broadcast_to(u[None, :], (camera.height, camera.width)),
            mx.broadcast_to(v[:, None], (camera.height, camera.width)),
            mx.ones((camera.height, camera.width)),
        ],
        axis=-1,
    )
    return d / mx.linalg.norm(d, axis=-1, keepdims=True)


def render_spaghetti_reference(
    camera: Camera,
    means: mx.array,
    halves: mx.array,
    quats: mx.array,
    scales: mx.array,
    rho: mx.array,
    colors: mx.array,
    background: mx.array | None = None,
) -> dict[str, mx.array]:
    """Dense pure-MLX renderer with the same semantics (tests / small scenes).

    Evaluates every segment at every pixel (``O(H W N)`` memory) with the
    rasterizer's depth order, opacity clamp and cut-off, but no early
    termination. Differentiable via MLX autodiff.
    """
    proj = project_segments(camera, means, halves, quats, scales, rho)
    keep = mx.array(proj["radii"] > 0)
    order = mx.argsort(proj["depths"])
    d = pixel_rays(camera)  # (H, W, 3)
    e = mx.einsum("nij,hwj->hwni", proj["W"][order], d)  # (H, W, N, 3)
    tau = segment_optical_depth(e, proj["a"][order], proj["v"][order], proj["rho"][order])
    alpha = mx.minimum(0.99, 1.0 - mx.exp(-tau))
    alpha = mx.where((alpha >= 1.0 / 255.0) & keep[order], alpha, 0.0)
    T = mx.cumprod(1.0 - alpha, axis=-1)
    T_before = mx.concatenate([mx.ones_like(T[..., :1]), T[..., :-1]], axis=-1)
    weights = alpha * T_before
    image = mx.einsum("hwn,nc->hwc", weights, colors[order])
    bg = mx.zeros((3,)) if background is None else background
    final_T = T[..., -1]
    return {"image": image + final_T[..., None] * bg, "alpha": 1.0 - final_T}
