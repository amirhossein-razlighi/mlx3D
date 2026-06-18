"""Differentiable projection of 3D Gaussians to screen space (EWA splatting).

This is the half of the 3DGS rasterizer that pure MLX autodiff handles well:
building 3D covariances from quaternion + scale, transforming to camera
space, and the EWA first-order projection to 2D covariances/conics. The
per-pixel compositing lives in the Metal kernels (see ``rasterize.py``).
"""

import mlx.core as mx

from ..cameras import Camera
from ..transforms import quaternion_to_matrix

__all__ = ["quat_scale_to_cov3d", "project_gaussians"]


def quat_scale_to_cov3d(quats: mx.array, scales: mx.array) -> mx.array:
    """Build 3D covariances ``R S S^T R^T`` from quaternions (N, 4) and scales (N, 3)."""
    R = quaternion_to_matrix(quats)
    M = R * scales[:, None, :]  # scale the columns: R @ diag(s)
    return M @ M.swapaxes(-1, -2)


def project_gaussians(
    camera: Camera,
    means: mx.array,
    quats: mx.array,
    scales: mx.array,
    blur: float = 0.3,
    antialias: bool = False,
) -> dict[str, mx.array]:
    """Project 3D Gaussians into screen space.

    Args:
        camera: the viewing :class:`~mlx3d.cameras.Camera`.
        means: (N, 3) Gaussian centers in world space.
        quats: (N, 4) rotations (w, x, y, z), need not be normalized.
        scales: (N, 3) per-axis standard deviations.
        blur: screen-space dilation added to the diagonal (0.3 px as in 3DGS,
            which guarantees splats cover at least about one pixel).
        antialias: if ``True``, also return Mip-Splatting-style opacity
            compensation for the added screen-space blur. The conic still uses
            the blurred covariance, while ``compensation`` scales opacity by
            ``sqrt(det(cov) / det(cov + blur I))`` so subpixel Gaussians do not
            gain energy from the low-pass filter.

    Returns:
        dict with:
            - ``means2d`` (N, 2): pixel-space centers.
            - ``conics`` (N, 3): upper-triangular inverse 2D covariance (a, b, c).
            - ``depths`` (N,): camera-space z.
            - ``radii`` (N,): conservative pixel radii (0 for culled Gaussians).
            - ``compensation`` (N,): opacity multiplier for anti-aliased mode,
              otherwise all ones.
    """
    R, t = camera.R, camera.t
    mx_, my_, mz_ = means[:, 0], means[:, 1], means[:, 2]
    x = mx_ * R[0, 0] + my_ * R[0, 1] + mz_ * R[0, 2] + t[0]
    y = mx_ * R[1, 0] + my_ * R[1, 1] + mz_ * R[1, 2] + t[1]
    z = mx_ * R[2, 0] + my_ * R[2, 1] + mz_ * R[2, 2] + t[2]
    z_safe = mx.maximum(z, 1e-6)

    # Project centers.
    u = camera.fx * x / z_safe + camera.cx
    v = camera.fy * y / z_safe + camera.cy
    means2d = mx.stack([u, v], axis=-1)

    # EWA Jacobian, with x/z, y/z clamped to a slightly padded frustum for
    # stability of far off-screen Gaussians (as in the reference CUDA code).
    tan_fov_x = 0.5 * camera.width / camera.fx
    tan_fov_y = 0.5 * camera.height / camera.fy
    tx = mx.clip(x / z_safe, -1.3 * tan_fov_x, 1.3 * tan_fov_x) * z_safe
    ty = mx.clip(y / z_safe, -1.3 * tan_fov_y, 1.3 * tan_fov_y) * z_safe

    inv_z = 1.0 / z_safe
    inv_z2 = inv_z * inv_z

    # Project the rotated/scaled covariance without materializing per-Gaussian
    # 3x3 and 2x3 matrices. Let T = J @ camera.R and M = R_quat @ diag(scales).
    # The screen covariance is (T @ M) @ (T @ M)^T, so only six scalar dot
    # products are needed per Gaussian.
    j00 = camera.fx * inv_z
    j02 = -camera.fx * tx * inv_z2
    j11 = camera.fy * inv_z
    j12 = -camera.fy * ty * inv_z2

    t00 = j00 * R[0, 0] + j02 * R[2, 0]
    t01 = j00 * R[0, 1] + j02 * R[2, 1]
    t02 = j00 * R[0, 2] + j02 * R[2, 2]
    t10 = j11 * R[1, 0] + j12 * R[2, 0]
    t11 = j11 * R[1, 1] + j12 * R[2, 1]
    t12 = j11 * R[1, 2] + j12 * R[2, 2]

    q = quats / mx.linalg.norm(quats, axis=-1, keepdims=True)
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    two = 2.0
    r00 = 1.0 - two * (qy * qy + qz * qz)
    r01 = two * (qx * qy - qw * qz)
    r02 = two * (qx * qz + qw * qy)
    r10 = two * (qx * qy + qw * qz)
    r11 = 1.0 - two * (qx * qx + qz * qz)
    r12 = two * (qy * qz - qw * qx)
    r20 = two * (qx * qz - qw * qy)
    r21 = two * (qy * qz + qw * qx)
    r22 = 1.0 - two * (qx * qx + qy * qy)

    sx, sy, sz = scales[:, 0], scales[:, 1], scales[:, 2]
    m00 = (t00 * r00 + t01 * r10 + t02 * r20) * sx
    m01 = (t00 * r01 + t01 * r11 + t02 * r21) * sy
    m02 = (t00 * r02 + t01 * r12 + t02 * r22) * sz
    m10 = (t10 * r00 + t11 * r10 + t12 * r20) * sx
    m11 = (t10 * r01 + t11 * r11 + t12 * r21) * sy
    m12 = (t10 * r02 + t11 * r12 + t12 * r22) * sz

    a0 = m00 * m00 + m01 * m01 + m02 * m02
    b = m00 * m10 + m01 * m11 + m02 * m12
    c0 = m10 * m10 + m11 * m11 + m12 * m12

    a = a0 + blur
    c = c0 + blur

    det = a * c - b * b
    det_safe = mx.maximum(det, 1e-12)
    conics = mx.stack([c / det_safe, -b / det_safe, a / det_safe], axis=-1)
    if antialias and blur > 0:
        det0 = mx.maximum(a0 * c0 - b * b, 0.0)
        compensation = mx.sqrt(det0 / det_safe)
    else:
        compensation = mx.ones_like(det)

    # Conservative radius: 3 sigma of the larger eigenvalue.
    mid = 0.5 * (a + c)
    lam1 = mid + mx.sqrt(mx.maximum(mid * mid - det, 0.01))
    radii = mx.ceil(3.0 * mx.sqrt(mx.maximum(lam1, 0.0)))

    valid = (z > camera.znear) & (det > 0)
    radii = mx.where(valid, radii, mx.zeros_like(radii))
    compensation = mx.where(valid, compensation, mx.zeros_like(compensation))

    return {
        "means2d": means2d,
        "conics": conics,
        "depths": z,
        "radii": radii,
        "compensation": compensation,
    }
