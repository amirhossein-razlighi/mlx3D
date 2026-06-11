import mlx.core as mx
import numpy as np
import pytest

from mlx3d.io import load_obj, load_ply, save_obj, save_ply


def assert_close(a, b, atol=1e-5):
    np.testing.assert_allclose(np.array(a), np.array(b), atol=atol)


@pytest.fixture
def cube():
    verts = mx.array(
        [
            [-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0],
        ]
    )
    faces = mx.array(
        [
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
            [1, 2, 6], [1, 6, 5], [3, 0, 4], [3, 4, 7],
        ]
    )
    return verts, faces


def test_obj_roundtrip(tmp_path, cube):
    verts, faces = cube
    p = str(tmp_path / "cube.obj")
    save_obj(p, verts, faces)
    data = load_obj(p)
    assert_close(data.verts, verts)
    assert data.faces.tolist() == faces.tolist()


def test_obj_with_colors(tmp_path, cube):
    verts, faces = cube
    colors = mx.random.uniform(shape=(8, 3))
    p = str(tmp_path / "cube_color.obj")
    save_obj(p, verts, faces, verts_colors=colors)
    data = load_obj(p)
    assert data.verts_colors is not None
    assert_close(data.verts_colors, colors, atol=1e-5)


def test_obj_quads_and_indices(tmp_path):
    p = str(tmp_path / "quad.obj")
    with open(p, "w") as f:
        f.write("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n")
        f.write("f 1 2 3 4\n")  # quad -> two triangles
        f.write("f -4 -3 -2\n")  # negative indices
    data = load_obj(p)
    assert data.faces.shape == (3, 3)
    assert data.faces[0].tolist() == [0, 1, 2]
    assert data.faces[1].tolist() == [0, 2, 3]
    assert data.faces[2].tolist() == [0, 1, 2]


def test_obj_with_normals_texcoords(tmp_path):
    p = str(tmp_path / "tex.obj")
    with open(p, "w") as f:
        f.write("v 0 0 0\nv 1 0 0\nv 0 1 0\n")
        f.write("vt 0 0\nvt 1 0\nvt 0 1\n")
        f.write("vn 0 0 1\n")
        f.write("f 1/1/1 2/2/1 3/3/1\n")
    data = load_obj(p)
    assert data.texcoords.shape == (3, 2)
    assert data.normals.shape == (1, 3)
    assert data.faces_texcoords_idx[0].tolist() == [0, 1, 2]
    assert data.faces_normals_idx[0].tolist() == [0, 0, 0]


@pytest.mark.parametrize("binary", [True, False])
def test_ply_roundtrip_mesh(tmp_path, cube, binary):
    verts, faces = cube
    p = str(tmp_path / f"cube_{binary}.ply")
    save_ply(p, verts, faces=faces, binary=binary)
    data = load_ply(p)
    assert_close(data.verts, verts)
    assert data.faces.tolist() == faces.tolist()


@pytest.mark.parametrize("binary", [True, False])
def test_ply_pointcloud_with_attrs(tmp_path, binary):
    pts = mx.random.normal((50, 3))
    normals = mx.random.normal((50, 3))
    colors = mx.random.uniform(shape=(50, 3))
    extra = {"opacity": mx.random.normal((50,)), "scale_0": mx.random.normal((50,))}
    p = str(tmp_path / f"pc_{binary}.ply")
    save_ply(p, pts, normals=normals, colors=colors, extra=extra, binary=binary)
    data = load_ply(p)
    assert_close(data.verts, pts, atol=1e-5)
    assert_close(data.normals, normals, atol=1e-5)
    # Colors quantized to uint8.
    assert_close(data.colors, colors, atol=1.0 / 255.0 + 1e-6)
    assert data.faces is None
    assert set(data.extra.keys()) == {"opacity", "scale_0"}
    assert_close(data.extra["opacity"], extra["opacity"], atol=1e-5)
