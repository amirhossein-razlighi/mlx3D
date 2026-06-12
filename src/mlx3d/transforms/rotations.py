"""Conversions between 3D rotation representations.

All functions operate on batched MLX arrays and are fully differentiable.
Quaternions use the ``(w, x, y, z)`` (scalar-first) convention.
"""

import mlx.core as mx

__all__ = [
    "quaternion_to_matrix",
    "matrix_to_quaternion",
    "axis_angle_to_quaternion",
    "quaternion_to_axis_angle",
    "axis_angle_to_matrix",
    "matrix_to_axis_angle",
    "euler_angles_to_matrix",
    "rotation_6d_to_matrix",
    "matrix_to_rotation_6d",
    "quaternion_multiply",
    "quaternion_invert",
    "quaternion_apply",
    "standardize_quaternion",
    "random_quaternions",
    "random_rotations",
]

_NORM_EPS = 1e-12


def quaternion_to_matrix(quaternions: mx.array) -> mx.array:
    """Convert quaternions ``(..., 4)`` in (w, x, y, z) order to rotation matrices ``(..., 3, 3)``."""
    q = quaternions / mx.linalg.norm(quaternions, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    two = 2.0
    m = mx.stack(
        [
            1 - two * (y * y + z * z),
            two * (x * y - w * z),
            two * (x * z + w * y),
            two * (x * y + w * z),
            1 - two * (x * x + z * z),
            two * (y * z - w * x),
            two * (x * z - w * y),
            two * (y * z + w * x),
            1 - two * (x * x + y * y),
        ],
        axis=-1,
    )
    return m.reshape(*quaternions.shape[:-1], 3, 3)


def matrix_to_quaternion(matrix: mx.array) -> mx.array:
    """Convert rotation matrices ``(..., 3, 3)`` to quaternions ``(..., 4)`` in (w, x, y, z) order.

    Uses the numerically stable branch selection of Shepperd's method.
    """
    m = matrix
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]

    # Four candidate quaternions, one per branch of Shepperd's method. Each
    # uses 2*sqrt(t_*) as the denominator, where t_* is the (always positive
    # in its branch) diagonal combination.
    t_w = 1 + m00 + m11 + m22
    t_x = 1 + m00 - m11 - m22
    t_y = 1 - m00 + m11 - m22
    t_z = 1 - m00 - m11 + m22

    def _denom(t):
        return 2.0 * mx.sqrt(mx.maximum(t, 1e-12))[..., None]

    q_w = mx.stack([t_w, m21 - m12, m02 - m20, m10 - m01], axis=-1) / _denom(t_w)
    q_x = mx.stack([m21 - m12, t_x, m01 + m10, m02 + m20], axis=-1) / _denom(t_x)
    q_y = mx.stack([m02 - m20, m01 + m10, t_y, m12 + m21], axis=-1) / _denom(t_y)
    q_z = mx.stack([m10 - m01, m02 + m20, m12 + m21, t_z], axis=-1) / _denom(t_z)

    trace = m00 + m11 + m22
    cond_w = (trace > 0)[..., None]
    cond_x = ((m00 >= m11) & (m00 >= m22))[..., None]
    cond_y = (m11 >= m22)[..., None]

    q = mx.where(cond_w, q_w, mx.where(cond_x, q_x, mx.where(cond_y, q_y, q_z)))
    return standardize_quaternion(q / mx.linalg.norm(q, axis=-1, keepdims=True))


def standardize_quaternion(quaternions: mx.array) -> mx.array:
    """Flip quaternions so the real part is non-negative."""
    return mx.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def axis_angle_to_quaternion(axis_angle: mx.array) -> mx.array:
    """Convert axis-angle vectors ``(..., 3)`` to quaternions ``(..., 4)``."""
    angles = mx.linalg.norm(axis_angle, axis=-1, keepdims=True)
    half = angles * 0.5
    eps = 1e-6
    small = angles < eps
    # sin(x/2)/x ~= 1/2 - x^2/48 for small x
    sin_half_over_angle = mx.where(
        small, 0.5 - (angles * angles) / 48.0, mx.sin(half) / mx.maximum(angles, eps)
    )
    return mx.concatenate([mx.cos(half), axis_angle * sin_half_over_angle], axis=-1)


def quaternion_to_axis_angle(quaternions: mx.array) -> mx.array:
    """Convert quaternions ``(..., 4)`` to axis-angle vectors ``(..., 3)``."""
    q = standardize_quaternion(quaternions / mx.linalg.norm(quaternions, axis=-1, keepdims=True))
    norms = mx.linalg.norm(q[..., 1:], axis=-1, keepdims=True)
    half_angles = mx.arctan2(norms, q[..., 0:1])
    angles = 2.0 * half_angles
    eps = 1e-6
    small = mx.abs(angles) < eps
    sin_half_over_angle = mx.where(
        small,
        0.5 - (angles * angles) / 48.0,
        mx.sin(half_angles) / mx.where(small, mx.ones_like(angles), angles),
    )
    return q[..., 1:] / sin_half_over_angle


def axis_angle_to_matrix(axis_angle: mx.array) -> mx.array:
    """Convert axis-angle vectors ``(..., 3)`` to rotation matrices ``(..., 3, 3)``."""
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


def matrix_to_axis_angle(matrix: mx.array) -> mx.array:
    """Convert rotation matrices ``(..., 3, 3)`` to axis-angle vectors ``(..., 3)``."""
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


def _axis_rotation(axis: str, angle: mx.array) -> mx.array:
    cos, sin = mx.cos(angle), mx.sin(angle)
    one, zero = mx.ones_like(angle), mx.zeros_like(angle)
    if axis == "X":
        flat = [one, zero, zero, zero, cos, -sin, zero, sin, cos]
    elif axis == "Y":
        flat = [cos, zero, sin, zero, one, zero, -sin, zero, cos]
    elif axis == "Z":
        flat = [cos, -sin, zero, sin, cos, zero, zero, zero, one]
    else:
        raise ValueError(f"Invalid axis {axis!r}; expected 'X', 'Y' or 'Z'.")
    return mx.stack(flat, axis=-1).reshape(*angle.shape, 3, 3)


def euler_angles_to_matrix(euler_angles: mx.array, convention: str = "XYZ") -> mx.array:
    """Convert Euler angles ``(..., 3)`` (radians) to rotation matrices ``(..., 3, 3)``.

    ``convention`` is a 3-letter string of axes, e.g. ``"XYZ"`` applies
    R = R_X(a0) @ R_Y(a1) @ R_Z(a2).
    """
    if len(convention) != 3 or any(c not in "XYZ" for c in convention):
        raise ValueError(f"Invalid convention {convention!r}.")
    matrices = [_axis_rotation(axis, euler_angles[..., i]) for i, axis in enumerate(convention)]
    return matrices[0] @ matrices[1] @ matrices[2]


def rotation_6d_to_matrix(d6: mx.array) -> mx.array:
    """Convert 6D rotation representation ``(..., 6)`` to matrices via Gram-Schmidt.

    Reference: Zhou et al., "On the Continuity of Rotation Representations in
    Neural Networks" (CVPR 2019).
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / mx.maximum(mx.linalg.norm(a1, axis=-1, keepdims=True), _NORM_EPS)
    a2 = a2 - mx.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 / mx.maximum(mx.linalg.norm(a2, axis=-1, keepdims=True), _NORM_EPS)
    b3 = mx.linalg.cross(b1, b2)
    return mx.stack([b1, b2, b3], axis=-2)


def matrix_to_rotation_6d(matrix: mx.array) -> mx.array:
    """Convert rotation matrices ``(..., 3, 3)`` to the 6D representation ``(..., 6)``."""
    return matrix[..., :2, :].reshape(*matrix.shape[:-2], 6)


def quaternion_multiply(a: mx.array, b: mx.array) -> mx.array:
    """Hamilton product of two quaternion arrays ``(..., 4)``."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return mx.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def quaternion_invert(quaternion: mx.array) -> mx.array:
    """Inverse of unit quaternions: the conjugate."""
    return quaternion * mx.array([1.0, -1.0, -1.0, -1.0], dtype=quaternion.dtype)


def quaternion_apply(quaternion: mx.array, point: mx.array) -> mx.array:
    """Rotate points ``(..., 3)`` by unit quaternions ``(..., 4)``."""
    zeros = mx.zeros_like(point[..., :1])
    p = mx.concatenate([zeros, point], axis=-1)
    out = quaternion_multiply(quaternion_multiply(quaternion, p), quaternion_invert(quaternion))
    return out[..., 1:]


def random_quaternions(n: int, key: mx.array | None = None) -> mx.array:
    """Sample ``n`` uniform random unit quaternions, shape ``(n, 4)``."""
    if key is None:
        q = mx.random.normal((n, 4))
    else:
        q = mx.random.normal((n, 4), key=key)
    return standardize_quaternion(q / mx.linalg.norm(q, axis=-1, keepdims=True))


def random_rotations(n: int, key: mx.array | None = None) -> mx.array:
    """Sample ``n`` uniform random rotation matrices, shape ``(n, 3, 3)``."""
    return quaternion_to_matrix(random_quaternions(n, key=key))
