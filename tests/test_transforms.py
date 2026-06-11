import mlx.core as mx
import numpy as np

from mlx3d.transforms import (
    axis_angle_to_matrix,
    axis_angle_to_quaternion,
    euler_angles_to_matrix,
    matrix_to_axis_angle,
    matrix_to_quaternion,
    matrix_to_rotation_6d,
    quaternion_apply,
    quaternion_to_matrix,
    random_quaternions,
    random_rotations,
    rotation_6d_to_matrix,
)


def assert_close(a, b, atol=1e-5):
    np.testing.assert_allclose(np.array(a), np.array(b), atol=atol)


def test_quaternion_matrix_roundtrip():
    q = random_quaternions(64)
    m = quaternion_to_matrix(q)
    q2 = matrix_to_quaternion(m)
    # q and -q are the same rotation; both are standardized to w >= 0.
    assert_close(q, q2, atol=1e-4)


def test_rotation_matrices_are_valid():
    m = random_rotations(32)
    eye = mx.broadcast_to(mx.eye(3), m.shape)
    assert_close(m @ m.transpose(0, 2, 1), eye, atol=1e-5)
    dets = np.linalg.det(np.array(m))
    np.testing.assert_allclose(dets, np.ones(32), atol=1e-5)


def test_axis_angle_roundtrip():
    aa = mx.random.normal((50, 3)) * 0.8
    m = axis_angle_to_matrix(aa)
    aa2 = matrix_to_axis_angle(m)
    assert_close(aa, aa2, atol=1e-4)


def test_axis_angle_small_angles_stable():
    aa = mx.array([[1e-9, 0.0, 0.0], [0.0, 0.0, 0.0]])
    q = axis_angle_to_quaternion(aa)
    assert not bool(mx.isnan(q).any())
    assert_close(q[1], mx.array([1.0, 0.0, 0.0, 0.0]))


def test_euler_z_90deg():
    m = euler_angles_to_matrix(mx.array([[0.0, 0.0, np.pi / 2]]), "XYZ")
    p = mx.array([1.0, 0.0, 0.0])
    assert_close(m[0] @ p, mx.array([0.0, 1.0, 0.0]), atol=1e-6)


def test_rotation_6d_roundtrip():
    m = random_rotations(16)
    d6 = matrix_to_rotation_6d(m)
    m2 = rotation_6d_to_matrix(d6)
    assert_close(m, m2, atol=1e-5)


def test_quaternion_apply_matches_matrix():
    q = random_quaternions(8)
    pts = mx.random.normal((8, 3))
    out_q = quaternion_apply(q, pts)
    out_m = mx.squeeze(quaternion_to_matrix(q) @ pts[..., None], axis=-1)
    assert_close(out_q, out_m, atol=1e-5)


def test_gradients_flow():
    def f(aa):
        m = axis_angle_to_matrix(aa)
        return mx.sum(m * m)

    g = mx.grad(f)(mx.array([[0.3, -0.2, 0.5]]))
    assert not bool(mx.isnan(g).any())
