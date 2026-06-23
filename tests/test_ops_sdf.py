import mlx.core as mx
import numpy as np
import pytest

from mlx3d.ops import (
    sample_sdf_grid,
    sdf_box,
    sdf_difference,
    sdf_intersection,
    sdf_smooth_union,
    sdf_sphere,
    sdf_to_mesh,
    sdf_torus,
    sdf_union,
)

pytestmark = pytest.mark.unit


def test_sdf_sphere_distances():
    pts = mx.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    d = sdf_sphere(pts, radius=1.0)
    np.testing.assert_allclose(np.array(d), [-1.0, 0.0, 1.0], atol=1e-5)


def test_sdf_box_distances():
    pts = mx.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    d = sdf_box(pts, half_extents=(1.0, 1.0, 1.0))
    # center is fully inside (-1), face is on the surface (0), outside is +1.
    np.testing.assert_allclose(np.array(d), [-1.0, 0.0, 1.0], atol=1e-5)


def test_sdf_torus_on_tube_center_is_negative_minor():
    # A point on the tube center circle is exactly minor_radius inside.
    pts = mx.array([[1.0, 0.0, 0.0]])
    d = sdf_torus(pts, major_radius=1.0, minor_radius=0.25)
    np.testing.assert_allclose(np.array(d), [-0.25], atol=1e-5)


def test_csg_union_intersection_difference():
    a = sdf_sphere(mx.array([[0.4, 0.0, 0.0]]), radius=1.0, center=(-0.5, 0, 0))
    b = sdf_sphere(mx.array([[0.4, 0.0, 0.0]]), radius=1.0, center=(0.5, 0, 0))
    assert float(sdf_union(a, b)) == pytest.approx(min(float(a), float(b)))
    assert float(sdf_intersection(a, b)) == pytest.approx(max(float(a), float(b)))
    assert float(sdf_difference(a, b)) == pytest.approx(max(float(a), -float(b)))


def test_smooth_union_is_below_hard_union():
    a = sdf_sphere(mx.array([[0.0, 0.0, 0.0]]), radius=0.6, center=(-0.4, 0, 0))
    b = sdf_sphere(mx.array([[0.0, 0.0, 0.0]]), radius=0.6, center=(0.4, 0, 0))
    hard = float(sdf_union(a, b))
    soft = float(sdf_smooth_union(a, b, k=0.3))
    # The smooth min rounds the join, never exceeding the hard min.
    assert soft <= hard + 1e-6


def test_sample_sdf_grid_shapes_and_metadata():
    vol, spacing, origin = sample_sdf_grid(lambda p: sdf_sphere(p, 0.5), resolution=16, bounds=1.0)
    assert vol.shape == (16, 16, 16)
    np.testing.assert_allclose(spacing, [2.0 / 15] * 3, atol=1e-6)
    np.testing.assert_allclose(origin, [-1.0, -1.0, -1.0], atol=1e-6)


def test_sdf_to_mesh_recovers_sphere_radius():
    mesh = sdf_to_mesh(lambda p: sdf_sphere(p, radius=0.7), resolution=48, bounds=1.0)
    v = mesh.verts_list()[0]
    f = mesh.faces_list()[0]
    assert v.shape[0] > 0 and f.shape[0] > 0
    r = mx.linalg.norm(v, axis=-1)
    assert abs(float(r.mean()) - 0.7) < 0.02


def test_sdf_is_differentiable_wrt_parameters():
    pts = mx.array([[0.9, 0.0, 0.0], [0.0, 0.8, 0.0]])

    def loss(radius):
        return mx.sum(sdf_sphere(pts, radius=radius) ** 2)

    g = mx.grad(loss)(mx.array(0.5))
    mx.eval(g)
    assert abs(float(g)) > 0.0
