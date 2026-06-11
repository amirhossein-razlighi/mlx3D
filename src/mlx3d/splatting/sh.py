"""Real spherical harmonics evaluation (degrees 0-3), as used by 3DGS."""

import mlx.core as mx

__all__ = ["eval_sh", "rgb_to_sh", "sh_to_rgb", "num_sh_bases"]

_C0 = 0.28209479177387814
_C1 = 0.4886025119029199
_C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
       -1.0925484305920792, 0.5462742152960396)
_C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658,
       0.3731763325901154, -0.4570457994644658, 1.445305721320277,
       -0.5900435899266435)


def num_sh_bases(degree: int) -> int:
    return (degree + 1) ** 2


def rgb_to_sh(rgb: mx.array) -> mx.array:
    """Convert RGB in [0, 1] to the DC spherical-harmonic coefficient."""
    return (rgb - 0.5) / _C0


def sh_to_rgb(sh_dc: mx.array) -> mx.array:
    """Convert the DC coefficient back to RGB."""
    return sh_dc * _C0 + 0.5


def eval_sh(degree: int, sh: mx.array, dirs: mx.array) -> mx.array:
    """Evaluate SH at unit directions.

    Args:
        degree: SH degree in [0, 3].
        sh: (N, K, C) coefficients with K >= (degree+1)^2.
        dirs: (N, 3) unit view directions.

    Returns:
        (N, C) colors, offset by +0.5 (callers should clamp to >= 0).
    """
    result = _C0 * sh[:, 0]
    if degree > 0:
        x, y, z = dirs[:, 0:1], dirs[:, 1:2], dirs[:, 2:3]
        result = result - _C1 * y * sh[:, 1] + _C1 * z * sh[:, 2] - _C1 * x * sh[:, 3]
        if degree > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (
                result
                + _C2[0] * xy * sh[:, 4]
                + _C2[1] * yz * sh[:, 5]
                + _C2[2] * (2.0 * zz - xx - yy) * sh[:, 6]
                + _C2[3] * xz * sh[:, 7]
                + _C2[4] * (xx - yy) * sh[:, 8]
            )
            if degree > 2:
                result = (
                    result
                    + _C3[0] * y * (3.0 * xx - yy) * sh[:, 9]
                    + _C3[1] * xy * z * sh[:, 10]
                    + _C3[2] * y * (4.0 * zz - xx - yy) * sh[:, 11]
                    + _C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh[:, 12]
                    + _C3[4] * x * (4.0 * zz - xx - yy) * sh[:, 13]
                    + _C3[5] * z * (xx - yy) * sh[:, 14]
                    + _C3[6] * x * (xx - 3.0 * yy) * sh[:, 15]
                )
    return result + 0.5
