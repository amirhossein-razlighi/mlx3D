import mlx.core as mx
import numpy as np
import pytest

from mlx3d.structures import Meshes, Pointclouds, join_meshes_as_batch


def assert_close(a, b, atol=1e-5):
    np.testing.assert_allclose(np.array(a), np.array(b), atol=atol)


def two_triangles():
    verts = [
        mx.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        mx.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [2.0, 2.0, 0.0]]),
    ]
    faces = [
        mx.array([[0, 1, 2]]),
        mx.array([[0, 1, 2], [1, 3, 2]]),
    ]
    return Meshes(verts, faces)


def test_packed_views():
    m = two_triangles()
    assert len(m) == 2
    assert m.verts_packed().shape == (7, 3)
    fp = m.faces_packed()
    assert fp.shape == (3, 3)
    # Second mesh face indices offset by 3 vertices.
    assert fp[1].tolist() == [3, 4, 5]
    assert m.num_verts_per_mesh.tolist() == [3, 4]
    assert m.num_faces_per_mesh.tolist() == [1, 2]
    assert m.mesh_to_verts_packed_first_idx().tolist() == [0, 3]
    assert m.verts_packed_to_mesh_idx().tolist() == [0, 0, 0, 1, 1, 1, 1]


def test_padded_views_and_roundtrip():
    m = two_triangles()
    vp = m.verts_padded()
    fp = m.faces_padded()
    assert vp.shape == (2, 4, 3)
    assert fp.shape == (2, 2, 3)
    assert fp[0, 1].tolist() == [-1, -1, -1]
    # Rebuild from padded and compare.
    m2 = Meshes(vp, fp)
    # Note: padded verts keep zero rows; counts come from faces for padded init.
    assert m2.num_faces_per_mesh.tolist() == [1, 2]


def test_face_areas_and_normals():
    m = two_triangles()
    areas = m.faces_areas_packed()
    assert_close(areas, mx.array([0.5, 2.0, 2.0]))
    n = m.verts_normals_packed()
    # Flat meshes in the z=0 plane: every normal is +/- z.
    assert_close(mx.abs(n[:, 2]), mx.ones((7,)), atol=1e-6)


def test_edges_packed():
    m = two_triangles()
    e = m.edges_packed()
    # Triangle has 3 edges; two-face quad has 5 unique edges.
    assert e.shape == (8, 2)


def test_offset_and_scale():
    m = two_triangles()
    m2 = m.offset_verts(mx.array([1.0, 0.0, 0.0]))
    assert_close(m2.verts_list()[0][0], mx.array([1.0, 0.0, 0.0]))
    m3 = m.scale_verts(2.0)
    assert_close(m3.verts_list()[1][1], mx.array([4.0, 0.0, 0.0]))


def test_getitem_and_join():
    m = two_triangles()
    assert len(m[0]) == 1
    assert m[0].verts_packed().shape == (3, 3)
    j = join_meshes_as_batch([m, m[0]])
    assert len(j) == 3


def test_gradients_through_normals():
    m = two_triangles()
    faces = m.faces_list()

    def loss(verts_packed):
        mm = Meshes([verts_packed[:3], verts_packed[3:]], faces)
        return mx.sum(mm.verts_normals_packed() ** 2) + mx.sum(mm.faces_areas_packed())

    g = mx.grad(loss)(m.verts_packed())
    assert g.shape == (7, 3)
    assert not bool(mx.isnan(g).any())


def test_pointclouds_basic():
    pts = [mx.random.normal((10, 3)), mx.random.normal((5, 3))]
    feats = [mx.random.normal((10, 4)), mx.random.normal((5, 4))]
    pc = Pointclouds(pts, features=feats)
    assert len(pc) == 2
    assert pc.points_packed().shape == (15, 3)
    assert pc.features_packed().shape == (15, 4)
    assert pc.points_padded().shape == (2, 10, 3)
    mask = pc.padded_mask()
    assert mask.sum().tolist() == 15
    assert pc.num_points_per_cloud.tolist() == [10, 5]
    pc2 = pc.offset_points(mx.array([1.0, 0.0, 0.0]))
    assert_close(pc2.points_packed()[:, 0], pc.points_packed()[:, 0] + 1.0)


def test_pointclouds_validation():
    with pytest.raises(ValueError):
        Pointclouds([mx.zeros((3, 2))])
    with pytest.raises(ValueError):
        Pointclouds([mx.zeros((3, 3))], features=[mx.zeros((2, 4))])
