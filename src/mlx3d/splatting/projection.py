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
) -> dict[str, mx.array]:
    """Project 3D Gaussians into screen space.

    Args:
        camera: the viewing :class:`~mlx3d.cameras.Camera`.
        means: (N, 3) Gaussian centers in world space.
        quats: (N, 4) rotations (w, x, y, z), need not be normalized.
        scales: (N, 3) per-axis standard deviations.
        blur: screen-space dilation added to the diagonal (0.3 px as in 3DGS,
            which guarantees splats cover at least about one pixel).

    Returns:
        dict with:
            - ``means2d`` (N, 2): pixel-space centers.
            - ``conics`` (N, 3): upper-triangular inverse 2D covariance (a, b, c).
            - ``depths`` (N,): camera-space z.
            - ``radii`` (N,): conservative pixel radii (0 for culled Gaussians).
    """
    N = means.shape[0]
    R, t = camera.R, camera.t
    p_cam = means @ R.T + t  # (N, 3)
    x, y, z = p_cam[:, 0], p_cam[:, 1], p_cam[:, 2]
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

    zero = mx.zeros_like(z_safe)
    inv_z = 1.0 / z_safe
    inv_z2 = inv_z * inv_z
    J = mx.stack(
        [
            mx.stack([camera.fx * inv_z, zero, -camera.fx * tx * inv_z2], axis=-1),
            mx.stack([zero, camera.fy * inv_z, -camera.fy * ty * inv_z2], axis=-1),
        ],
        axis=-2,
    )  # (N, 2, 3)

    cov3d = quat_scale_to_cov3d(quats, scales)
    T = J @ R  # (N, 2, 3)
    cov2d = T @ cov3d @ T.swapaxes(-1, -2)  # (N, 2, 2)
    a = cov2d[:, 0, 0] + blur
    b = cov2d[:, 0, 1]
    c = cov2d[:, 1, 1] + blur

    det = a * c - b * b
    det_safe = mx.maximum(det, 1e-12)
    conics = mx.stack([c / det_safe, -b / det_safe, a / det_safe], axis=-1)

    # Conservative radius: 3 sigma of the larger eigenvalue.
    mid = 0.5 * (a + c)
    lam1 = mid + mx.sqrt(mx.maximum(mid * mid - det, 0.01))
    radii = mx.ceil(3.0 * mx.sqrt(mx.maximum(lam1, 0.0)))

    valid = (z > camera.znear) & (det > 0)
    radii = mx.where(valid, radii, mx.zeros_like(radii))

    return {"means2d": means2d, "conics": conics, "depths": z, "radii": radii}
