"""Unit tests for SO(3)/SE(3) maps and Transform3d."""

import mlx.core as mx
import numpy as np

from mlx3d.transforms import (
    Transform3d,
    axis_angle_to_matrix,
    hat,
    se3_exp_map,
    se3_log_map,
    so3_exp_map,
    so3_log_map,
    vee,
)


def test_hat_vee_roundtrip():
    v = mx.random.normal((6, 3))
    np.testing.assert_allclose(np.array(vee(hat(v))), np.array(v), atol=1e-6)
    # hat is skew-symmetric.
    H = hat(v)
    np.testing.assert_allclose(np.array(H + mx.swapaxes(H, -1, -2)), 0.0, atol=1e-6)


def test_so3_exp_is_rotation_and_matches_axis_angle():
    omega = mx.random.normal((8, 3)) * 1.3
    R = so3_exp_map(omega)
    # Orthonormal with det +1.
    np.testing.assert_allclose(
        np.array(R @ mx.swapaxes(R, -1, -2)), np.broadcast_to(np.eye(3), (8, 3, 3)), atol=1e-5
    )
    # Independent implementation (quaternion path) agrees.
    np.testing.assert_allclose(np.array(R), np.array(axis_angle_to_matrix(omega)), atol=1e-5)


def test_so3_log_exp_roundtrip():
    omega = mx.random.normal((8, 3)) * 1.1
    R = so3_exp_map(omega)
    np.testing.assert_allclose(np.array(so3_exp_map(so3_log_map(R))), np.array(R), atol=1e-5)


def test_so3_small_angle_is_finite():
    omega = mx.array([[1e-8, 0.0, 0.0], [0.0, 2e-9, -1e-9]])
    assert bool(mx.isfinite(so3_exp_map(omega)).all())
    assert bool(mx.isfinite(so3_log_map(so3_exp_map(omega))).all())


def test_se3_exp_log_roundtrip():
    xi = mx.random.normal((5, 6)) * 0.7
    T = se3_exp_map(xi)
    np.testing.assert_allclose(np.array(se3_log_map(T)), np.array(xi), atol=1e-4)


def test_se3_pure_translation():
    xi = mx.array([[0.3, -1.2, 2.0, 0.0, 0.0, 0.0]])
    T = se3_exp_map(xi)
    np.testing.assert_allclose(np.array(T.trans), np.array(xi[:, :3]), atol=1e-6)
    np.testing.assert_allclose(np.array(T.rot[0]), np.eye(3), atol=1e-6)


def test_transform_compose_and_inverse():
    T1 = Transform3d.from_rot_trans(
        so3_exp_map(mx.array([0.1, 0.2, 0.3])), mx.array([1.0, 2.0, 3.0])
    )
    T2 = Transform3d.from_rot_trans(
        so3_exp_map(mx.array([-0.2, 0.05, 0.1])), mx.array([-1.0, 0.5, 0.2])
    )
    pts = mx.random.normal((7, 3))
    # compose applies self first, then other.
    np.testing.assert_allclose(
        np.array(T1.compose(T2).transform_points(pts)),
        np.array(T2.transform_points(T1.transform_points(pts))),
        atol=1e-5,
    )
    # @ is matrix-style: (T2 @ T1) applies T1 then T2.
    np.testing.assert_allclose(
        np.array((T2 @ T1).transform_points(pts)),
        np.array(T2.transform_points(T1.transform_points(pts))),
        atol=1e-5,
    )
    # inverse undoes.
    np.testing.assert_allclose(
        np.array(T1.inverse().transform_points(T1.transform_points(pts))), np.array(pts), atol=1e-5
    )


def test_transform_scale_inverse_and_matrix():
    pts = mx.random.normal((5, 3))
    S = Transform3d.scale(mx.array([2.0, 3.0, 0.5]))
    np.testing.assert_allclose(
        np.array(S.inverse().transform_points(S.transform_points(pts))), np.array(pts), atol=1e-5
    )
    M = Transform3d.translate(mx.array([1.0, 2.0, 3.0])).get_matrix()
    assert M.shape == (4, 4)
    np.testing.assert_allclose(np.array(M[3]), [0, 0, 0, 1], atol=1e-6)


def test_grad_through_twist_is_finite_at_origin():
    pts = mx.random.normal((7, 3))

    def loss(xi):
        return mx.sum(se3_exp_map(xi).transform_points(pts) ** 2)

    for x0 in (mx.zeros((6,)), mx.random.normal((6,)) * 0.5):
        g = mx.grad(loss)(x0)
        assert bool(mx.isfinite(g).all())
