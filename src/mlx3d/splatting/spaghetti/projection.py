"""Segment primitives in camera space, and their closed-form optical depth.

A spaghetti primitive is a line segment ``x(u) = a + u v`` (``u`` in [0, 1])
convolved with an anisotropic Gaussian kernel ``N(0, Sigma)`` and scaled by a
density ``rho`` (Shramko et al., "Spaghetti Splatting", 2026). Its optical
depth along a camera ray ``r(t) = t d`` has a closed form (their Eq. 19).

We evaluate it in the kernel's *whitened* space: with ``Q = Sigma^-1 / 2 =
W^T W``, every Q-norm becomes a Euclidean norm after mapping by ``W``. Per
segment a pixel then needs only ``W`` (3x3), the whitened start point ``a'``,
the whitened direction ``v'`` and ``rho`` — 16 numbers — and the formula
avoids the float32 cancellation of ``a^T Q a - (a^T Q d)^2 / d^T Q d`` for thin
kernels far from the camera.

For a pixel ray with whitened direction ``e = W d`` (``d`` unit):

    P   = I - e e^T / |e|^2          (projector orthogonal to the ray)
    y   = -P a'                      (ray-to-start offset, ray removed)
    z   = P v'                       (segment direction, ray removed)
    k   = |z|,  ub = (z . y) / k^2   (closest line parameter)
    E   = |y - ub z|^2               (squared whitened ray-to-line distance)
    I   = sqrt(pi) / (2 k) * (erf(k (1 - ub)) + erf(k ub))
    tau = rho * sqrt(pi / |e|^2) * exp(-E) * I

For ``k -> 0`` (a point, or a segment parallel to the ray) ``I -> 1`` and
``E -> |y|^2``: a zero-length segment is exactly a 3D Gaussian.
"""

from __future__ import annotations

import math

import mlx.core as mx

from ...cameras import Camera
from ...transforms import quaternion_to_matrix

__all__ = ["SegmentProjection", "project_segments", "segment_optical_depth"]

# Below this whitened length a segment is treated as a point (I = 1).
POINT_EPS = 1e-4
# Footprint radius bounds, in kernel standard deviations.
MIN_FOOTPRINT_SIGMAS, MAX_FOOTPRINT_SIGMAS = 2.0, 6.0

_SQRT_PI = math.sqrt(math.pi)


def segment_optical_depth(e: mx.array, a: mx.array, v: mx.array, rho: mx.array) -> mx.array:
    """Closed-form optical depth ``tau`` (pure MLX, differentiable).

    Args:
        e: ``(..., 3)`` whitened ray directions ``W d``.
        a: ``(..., 3)`` whitened segment start.
        v: ``(..., 3)`` whitened segment direction (end minus start).
        rho: ``(...)`` densities.
    """
    n = mx.sum(e * e, axis=-1, keepdims=True)
    y = -(a - e * mx.sum(e * a, axis=-1, keepdims=True) / n)
    z = v - e * mx.sum(e * v, axis=-1, keepdims=True) / n
    k2 = mx.sum(z * z, axis=-1)
    is_line = k2 > POINT_EPS * POINT_EPS
    k2_safe = mx.where(is_line, k2, 1.0)
    k = mx.sqrt(k2_safe)
    ub = mx.where(is_line, mx.sum(z * y, axis=-1) / k2_safe, 0.0)
    r = y - ub[..., None] * z
    E = mx.sum(r * r, axis=-1)
    integral = _SQRT_PI / (2.0 * k) * (mx.erf(k * (1.0 - ub)) + mx.erf(k * ub))
    integral = mx.where(is_line, integral, 1.0)
    return rho * mx.sqrt(math.pi / n[..., 0]) * mx.exp(-E) * integral


class SegmentProjection(dict):
    """Per-segment camera-space quantities (a dict of ``(N, ...)`` arrays).

    Keys: ``W`` (N, 3, 3) whitening map, ``a`` / ``v`` (N, 3) whitened start /
    direction, ``rho`` (N,); for tile binning ``means2d`` (N, 2) projected
    midpoints, ``radii`` (N,) pixel radii bounding the footprint (0 = culled)
    and ``depths`` (N,) midpoint depths; and ``conics`` (N, 3), the
    ``(a, b, c)`` of a pixel-space ellipse ``x^T C x <= 1`` around the
    projected midpoint that contains the whole footprint.
    """


def project_segments(
    camera: Camera,
    means: mx.array,
    halves: mx.array,
    quats: mx.array,
    scales: mx.array,
    rho: mx.array,
) -> SegmentProjection:
    """Map world-space segments into whitened camera space (differentiable).

    Args:
        means: (N, 3) segment midpoints.
        halves: (N, 3) half-vectors: the segment runs from ``m - h`` to ``m + h``.
        quats: (N, 4) kernel rotations (w, x, y, z).
        scales: (N, 3) kernel standard deviations (activated).
        rho: (N,) densities (activated).
    """
    if camera.distortion is not None or camera.orthographic:
        raise ValueError("Spaghetti splatting supports undistorted pinhole cameras.")
    R_cam = mx.array(camera.R).astype(mx.float32)
    t_cam = mx.array(camera.t).astype(mx.float32)

    start = (means - halves) @ R_cam.T + t_cam  # camera space
    end = (means + halves) @ R_cam.T + t_cam
    mid = (start + end) * 0.5

    # Q = Sigma^-1 / 2 = W^T W with W = S^-1 Rc^T / sqrt(2), Rc = R_cam R_kernel.
    R_kernel = quaternion_to_matrix(quats / mx.linalg.norm(quats, axis=-1, keepdims=True))
    Rc = R_cam[None] @ R_kernel
    W = mx.swapaxes(Rc, -1, -2) / (math.sqrt(2.0) * scales[..., None])
    a_w = (W @ start[..., None])[..., 0]
    v_w = (W @ (end - start)[..., None])[..., 0]

    # Screen footprint: the projected segment swept by the kernel's cut-off
    # radius (a capsule), bounded by an ellipse. The cut-off is opacity-aware:
    # where the peak optical depth falls below 1/255 (the compositing
    # threshold), so dense thin strands keep their visible tails.
    fx, fy, cx, cy = camera.fx, camera.fy, camera.cx, camera.cy

    def to_pixels(p):
        z = mx.maximum(p[:, 2], 1e-6)
        return mx.stack([fx * p[:, 0] / z + cx, fy * p[:, 1] / z + cy], axis=-1)

    sg = mx.stop_gradient
    s_max = sg(mx.max(scales, axis=-1))
    # Peak optical depth over all rays: a ray along the segment also crosses
    # its full length.
    seg_len = mx.linalg.norm(sg(end - start), axis=-1)
    tau_peak = sg(rho) * (math.sqrt(2.0 * math.pi) * s_max + seg_len)
    sigmas = mx.sqrt(2.0 * mx.log(mx.maximum(255.0 * tau_peak, math.e)))
    sigmas = mx.clip(sigmas, MIN_FOOTPRINT_SIGMAS, MAX_FOOTPRINT_SIGMAS)
    p0, p1 = to_pixels(sg(start)), to_pixels(sg(end))
    mid_px = (p0 + p1) * 0.5
    axis = p1 - p0
    length = mx.linalg.norm(axis, axis=-1)
    half_len = length * 0.5
    z_min = mx.maximum(mx.minimum(sg(start[:, 2]), sg(end[:, 2])), 1e-6)
    kernel_px = sigmas * max(fx, fy) * s_max / z_min
    radius = mx.ceil(half_len + kernel_px)

    # Ellipse containing the capsule: semi-axes sqrt(2) (h + r) along the
    # segment and sqrt(2) r across it.
    u = mx.where(
        length[:, None] > 1e-6, axis / mx.maximum(length, 1e-6)[:, None], mx.array([1.0, 0.0])
    )
    inv_a2 = 1.0 / (2.0 * (half_len + kernel_px) ** 2)
    inv_b2 = 1.0 / (2.0 * kernel_px**2)
    ux, uy = u[:, 0], u[:, 1]
    conics = mx.stack(
        [
            ux * ux * inv_a2 + uy * uy * inv_b2,
            ux * uy * (inv_a2 - inv_b2),
            uy * uy * inv_a2 + ux * ux * inv_b2,
        ],
        axis=-1,
    )

    near = camera.znear
    visible = (start[:, 2] > near) & (end[:, 2] > near)
    on_screen = (
        (mid_px[:, 0] + radius > 0)
        & (mid_px[:, 0] - radius < camera.width)
        & (mid_px[:, 1] + radius > 0)
        & (mid_px[:, 1] - radius < camera.height)
    )
    radii = mx.where(visible & on_screen, radius, 0.0).astype(mx.int32)

    return SegmentProjection(
        W=W,
        a=a_w,
        v=v_w,
        rho=rho,
        means2d=mid_px,
        radii=radii,
        depths=mx.stop_gradient(mid[:, 2]),
        conics=conics,
    )
