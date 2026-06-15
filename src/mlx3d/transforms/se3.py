"""Composable rigid/affine transforms and the SO(3)/SE(3) Lie-group maps.

Conventions match the rest of MLX3D: points are row vectors ``(..., 3)`` and a
rotation acts as ``R @ x`` (column-vector math), so a transform maps
``x -> R @ x + t``.  This is exactly :meth:`mlx3d.cameras.Camera.world_to_camera`.

The exponential/log maps make camera poses optimizable: parameterize a pose by a
6D twist in the Lie algebra, ``exp`` it to a transform, and let gradients flow.
"""

from __future__ import annotations

import mlx.core as mx

__all__ = [
    "Transform3d",
    "hat",
    "vee",
    "so3_exp_map",
    "so3_log_map",
    "se3_exp_map",
    "se3_log_map",
]

_EPS = 1e-8


def hat(v: mx.array) -> mx.array:
    """Map ``(..., 3)`` vectors to skew-symmetric matrices ``(..., 3, 3)``."""
    zero = mx.zeros_like(v[..., 0])
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    row0 = mx.stack([zero, -z, y], axis=-1)
    row1 = mx.stack([z, zero, -x], axis=-1)
    row2 = mx.stack([-y, x, zero], axis=-1)
    return mx.stack([row0, row1, row2], axis=-2)


def vee(m: mx.array) -> mx.array:
    """Inverse of :func:`hat`: extract ``(..., 3)`` from a skew matrix ``(..., 3, 3)``."""
    return mx.stack([m[..., 2, 1], m[..., 0, 2], m[..., 1, 0]], axis=-1)


def _safe_theta(omega: mx.array) -> tuple[mx.array, mx.array]:
    """Return ``(theta, theta^2)`` with a gradient-safe norm at the origin.

    ``theta^2`` is the smooth ``sum(omega^2)``; ``theta`` clamps the radicand so
    its derivative stays finite at ``omega = 0`` (a plain norm gives 0/0 = NaN,
    which then poisons the ``where`` branches via their VJP).
    """
    theta2 = mx.sum(omega * omega, axis=-1, keepdims=True)
    theta = mx.sqrt(mx.maximum(theta2, 1e-24))
    return theta, theta2


def so3_exp_map(omega: mx.array) -> mx.array:
    """Rotation matrices ``(..., 3, 3)`` from axis-angle vectors ``(..., 3)`` (Rodrigues)."""
    theta, theta2 = _safe_theta(omega)  # (..., 1)
    small = theta < 1e-4
    # A = sin(theta)/theta, B = (1-cos theta)/theta^2, with Taylor fallbacks.
    a = mx.where(
        small, 1.0 - theta2 / 6.0, mx.sin(theta) / mx.where(small, mx.ones_like(theta), theta)
    )
    b = mx.where(
        small,
        0.5 - theta2 / 24.0,
        (1.0 - mx.cos(theta)) / mx.where(small, mx.ones_like(theta2), theta2),
    )
    a = a[..., None]  # (..., 1, 1)
    b = b[..., None]
    k = hat(omega)
    eye = mx.broadcast_to(mx.eye(3), k.shape)
    return eye + a * k + b * (k @ k)


def so3_log_map(r: mx.array) -> mx.array:
    """Axis-angle vectors ``(..., 3)`` from rotation matrices ``(..., 3, 3)``.

    This is the inverse of :func:`so3_exp_map`. It delegates to the quaternion
    based conversion, which stays accurate near ``theta = pi`` where the naive
    ``theta / (2 sin theta)`` form is singular.
    """
    from .rotations import matrix_to_axis_angle

    return matrix_to_axis_angle(r)


def se3_exp_map(xi: mx.array) -> "Transform3d":
    """Exponential map from twists ``(..., 6)`` = ``[v(3), omega(3)]`` to transforms.

    ``omega`` is the rotation part and ``v`` the translation part of the twist
    (PyTorch3D ordering). Returns a :class:`Transform3d`.
    """
    v, omega = xi[..., :3], xi[..., 3:]
    r = so3_exp_map(omega)
    theta, theta2 = _safe_theta(omega)
    small = theta < 1e-4
    b = mx.where(
        small,
        0.5 - theta2 / 24.0,
        (1.0 - mx.cos(theta)) / mx.where(small, mx.ones_like(theta2), theta2),
    )
    c = mx.where(
        small,
        1.0 / 6.0 - theta2 / 120.0,
        (theta - mx.sin(theta)) / mx.where(small, mx.ones_like(theta), theta * theta2),
    )
    k = hat(omega)
    eye = mx.broadcast_to(mx.eye(3), k.shape)
    vmat = eye + b[..., None] * k + c[..., None] * (k @ k)  # left Jacobian
    t = (vmat @ v[..., None])[..., 0]
    return Transform3d.from_rot_trans(r, t)


def se3_log_map(transform: "Transform3d") -> mx.array:
    """Inverse of :func:`se3_exp_map`: twists ``(..., 6)`` = ``[v, omega]`` from a transform."""
    r, t = transform.rot, transform.trans
    omega = so3_log_map(r)
    theta = mx.linalg.norm(omega, axis=-1, keepdims=True)
    theta2 = theta * theta
    small = theta < 1e-4
    k = hat(omega)
    eye = mx.broadcast_to(mx.eye(3), k.shape)
    # V^{-1} = I - 0.5 K + (1/theta^2)(1 - theta sin / (2(1-cos))) K^2.
    half_theta = 0.5 * theta
    coeff = mx.where(
        small,
        1.0 / 12.0 + theta2 / 720.0,
        (
            1.0
            - half_theta
            * mx.cos(half_theta)
            / mx.sin(mx.where(small, mx.ones_like(half_theta), half_theta))
        )
        / mx.where(small, mx.ones_like(theta2), theta2),
    )
    vinv = eye - 0.5 * k + coeff[..., None] * (k @ k)
    v = (vinv @ t[..., None])[..., 0]
    return mx.concatenate([v, omega], axis=-1)


class Transform3d:
    """A batched rigid/affine 3D transform: ``x -> R @ x + t``.

    Stores a rotation/scale-shear block ``rot`` ``(..., 3, 3)`` and a translation
    ``trans`` ``(..., 3)``. Transforms compose, invert, and apply to points and
    normals, and are fully differentiable (handy for pose optimization).

    Build one with the constructors :meth:`from_rot_trans`, :meth:`translate`,
    :meth:`rotate`, :meth:`scale`, or compose with :meth:`compose` / ``@``.
    """

    def __init__(self, rot: mx.array | None = None, trans: mx.array | None = None):
        if rot is None:
            rot = mx.eye(3)
        if trans is None:
            trans = mx.zeros(rot.shape[:-2] + (3,))
        self.rot = rot
        self.trans = trans

    # ------------------------------------------------------------ constructors
    @classmethod
    def from_rot_trans(cls, rot: mx.array, trans: mx.array) -> "Transform3d":
        return cls(rot, trans)

    @classmethod
    def translate(cls, t: mx.array) -> "Transform3d":
        t = mx.array(t)
        return cls(mx.broadcast_to(mx.eye(3), t.shape[:-1] + (3, 3)), t)

    @classmethod
    def rotate(cls, rot: mx.array) -> "Transform3d":
        return cls(rot, mx.zeros(rot.shape[:-2] + (3,)))

    @classmethod
    def scale(cls, s: mx.array | float) -> "Transform3d":
        s = mx.array(s, dtype=mx.float32)
        if s.ndim == 0:
            s = mx.broadcast_to(s, (3,))
        diag = s[..., :, None] * mx.eye(3)
        return cls(diag, mx.zeros(diag.shape[:-2] + (3,)))

    # ---------------------------------------------------------------- algebra
    def compose(self, other: "Transform3d") -> "Transform3d":
        """Return the transform that applies ``self`` first, then ``other``."""
        rot = other.rot @ self.rot
        trans = (other.rot @ self.trans[..., None])[..., 0] + other.trans
        return Transform3d(rot, trans)

    def __matmul__(self, other: "Transform3d") -> "Transform3d":
        # (self @ other) applies `other` first, then `self` — matrix-style.
        return other.compose(self)

    def inverse(self) -> "Transform3d":
        rinv = mx.linalg.inv(self.rot, stream=mx.cpu)
        return Transform3d(rinv, -(rinv @ self.trans[..., None])[..., 0])

    # ---------------------------------------------------------------- apply
    def transform_points(self, points: mx.array) -> mx.array:
        """Apply to points ``(..., 3)``: ``R @ x + t`` (row-vector form, broadcasts)."""
        return points @ mx.swapaxes(self.rot, -1, -2) + self.trans[..., None, :]

    def transform_normals(self, normals: mx.array) -> mx.array:
        """Apply to normals with the inverse transform (ignores translation)."""
        return normals @ mx.linalg.inv(self.rot, stream=mx.cpu)

    def get_matrix(self) -> mx.array:
        """Return the homogeneous ``(..., 4, 4)`` matrix form."""
        top = mx.concatenate([self.rot, self.trans[..., :, None]], axis=-1)  # (...,3,4)
        bottom = mx.broadcast_to(mx.array([0.0, 0.0, 0.0, 1.0]), top.shape[:-2] + (1, 4))
        return mx.concatenate([top, bottom], axis=-2)

    def __repr__(self) -> str:
        return f"Transform3d(rot={self.rot.shape}, trans={self.trans.shape})"
