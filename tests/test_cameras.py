import math

import mlx.core as mx
import numpy as np

from mlx3d.cameras import Camera, look_at, look_at_view_transform


def assert_close(a, b, atol=1e-5):
    np.testing.assert_allclose(np.array(a), np.array(b), atol=atol)


def test_look_at_center_projects_to_principal_point():
    cam = Camera.look_at(eye=(0.0, 0.0, -3.0), at=(0, 0, 0), width=640, height=480)
    xy, z = cam.project_points(mx.array([[0.0, 0.0, 0.0]]))
    assert_close(xy[0], mx.array([320.0, 240.0]), atol=1e-3)
    assert float(z[0]) > 0  # in front of the camera


def test_look_at_rotation_is_valid():
    R, t = look_at(mx.array([1.0, 2.0, 3.0]), mx.array([0.0, 0.0, 0.0]), mx.array([0.0, 1.0, 0.0]))
    assert_close(R @ R.T, mx.eye(3), atol=1e-5)
    assert abs(np.linalg.det(np.array(R)) - 1.0) < 1e-5


def test_camera_center():
    eye = (1.5, -0.5, -4.0)
    cam = Camera.look_at(eye=eye, width=100, height=100)
    assert_close(cam.camera_center, mx.array(eye), atol=1e-5)


def test_world_camera_roundtrip():
    cam = Camera.look_at(eye=(2.0, 1.0, -3.0), width=64, height=64)
    pts = mx.random.normal((10, 3))
    assert_close(cam.camera_to_world(cam.world_to_camera(pts)), pts, atol=1e-5)


def test_project_unproject_roundtrip():
    cam = Camera.look_at(eye=(0.0, 0.0, -4.0), width=128, height=128)
    pts = mx.random.normal((20, 3)) * 0.5
    xy, z = cam.project_points(pts)
    back = cam.unproject_points(xy, z)
    assert_close(back, pts, atol=1e-4)


def test_rays_through_center_hit_target():
    cam = Camera.look_at(eye=(0.0, 0.0, -3.0), at=(0, 0, 0), width=64, height=64)
    origins, dirs = cam.generate_rays()
    assert origins.shape == (64, 64, 3)
    # Central rays point roughly toward the origin.
    d = dirs[32, 32]
    assert_close(d, mx.array([0.0, 0.0, 1.0]), atol=0.05)


def test_look_at_view_transform_distance():
    R, t = look_at_view_transform(dist=2.5, elev=30.0, azim=45.0)
    center = -(R.T @ t)
    assert abs(float(mx.linalg.norm(center)) - 2.5) < 1e-5


def test_fov_focal():
    cam = Camera.from_fov(90.0, width=200, height=100)
    assert abs(cam.fy - 50.0) < 1e-4
    assert abs(math.degrees(cam.fov_y) - 90.0) < 1e-4


def test_orthographic_projection_is_parallel():
    R, t = look_at(mx.array([0.0, 0.0, -3.0]), mx.array([0.0, 0.0, 0.0]), mx.array([0.0, 1.0, 0.0]))
    cam = Camera.orthographic_camera(scale=1.0, width=64, height=64, R=R, t=t)
    assert cam.orthographic
    # Points with the same camera-space xy project to the same pixel regardless of depth.
    pts = mx.array([[0.3, 0.2, 0.0], [0.3, 0.2, 0.8], [0.3, 0.2, -0.5]])
    xy, z = cam.project_points(pts)
    assert_close(xy[0], xy[1], atol=1e-4)
    assert_close(xy[0], xy[2], atol=1e-4)
    # The principal point images the camera-space origin to the image center.
    xy0, _ = cam.project_points(mx.array([[0.0, 0.0, 1.0]]))
    assert_close(xy0[0], mx.array([32.0, 32.0]), atol=1e-4)


def test_orthographic_rays_are_parallel_with_varying_origins():
    R, t = look_at(mx.array([0.0, 0.0, -3.0]), mx.array([0.0, 0.0, 0.0]), mx.array([0.0, 1.0, 0.0]))
    cam = Camera.orthographic_camera(scale=1.0, width=32, height=32, R=R, t=t)
    o, d = cam.generate_rays()
    assert_close(d[0, 0], d[31, 31], atol=1e-5)  # parallel
    assert float(mx.abs(o[0, 0] - o[31, 31]).max()) > 0.1  # but origins differ


def test_orthographic_unproject_roundtrip():
    R, t = look_at(mx.array([0.0, 0.0, -3.0]), mx.array([0.0, 0.0, 0.0]), mx.array([0.0, 1.0, 0.0]))
    cam = Camera.orthographic_camera(scale=1.5, width=48, height=48, R=R, t=t)
    pts = mx.array([[0.4, -0.3, 0.2]])
    xy, z = cam.project_points(pts)
    back = cam.unproject_points(xy[0], z[0])
    assert_close(back, pts[0], atol=1e-4)
