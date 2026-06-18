import base64
import io
import json

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

from mlx3d.io import load_gltf, load_obj, load_ply, save_gltf, save_obj, save_ply
from mlx3d.utils import ico_sphere


def assert_close(a, b, atol=1e-5):
    np.testing.assert_allclose(np.array(a), np.array(b), atol=atol)


def test_gltf_glb_roundtrip(tmp_path):
    mesh = ico_sphere(level=2, radius=1.0)
    v, f, n = mesh.verts_packed(), mesh.faces_packed(), mesh.verts_normals_packed()
    path = str(tmp_path / "sphere.glb")
    save_gltf(path, v, f, normals=n)
    g = load_gltf(path)
    assert g.verts.shape == v.shape and g.faces.shape == f.shape
    np.testing.assert_allclose(np.array(g.verts), np.array(v), atol=1e-6)
    assert int(mx.abs(g.faces - f).max()) == 0
    np.testing.assert_allclose(np.array(g.normals), np.array(n), atol=1e-6)


def test_gltf_save_without_normals(tmp_path):
    mesh = ico_sphere(level=1, radius=1.0)
    path = str(tmp_path / "m.glb")
    save_gltf(path, mesh.verts_packed(), mesh.faces_packed())
    g = load_gltf(path)
    assert g.normals is None
    assert g.faces.shape == mesh.faces_packed().shape


def test_gltf_roundtrip_uvs_and_material(tmp_path):
    verts = mx.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces = mx.array([[0, 1, 2]])
    uvs = mx.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    path = str(tmp_path / "textured.glb")
    save_gltf(path, verts, faces, uvs=uvs, material_base_color=(0.2, 0.4, 0.6, 1.0))

    g = load_gltf(path)

    assert_close(g.uvs, uvs)
    assert g.material_ids.tolist() == [0]
    assert g.materials is not None
    assert g.materials[0].base_color == (0.2, 0.4, 0.6, 1.0)


def _png_data_uri(rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgb.astype(np.uint8)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _pack_gltf_buffer(arrays):
    blob = b""
    views = []
    accessors = []
    for arr, component_type, typ, target in arrays:
        arr = np.asarray(arr)
        offset = len(blob)
        raw = arr.tobytes()
        blob += raw + b"\x00" * ((4 - len(raw) % 4) % 4)
        views.append(
            {
                "buffer": 0,
                "byteOffset": offset,
                "byteLength": len(raw),
                "target": target,
            }
        )
        accessors.append(
            {
                "bufferView": len(views) - 1,
                "componentType": component_type,
                "count": int(arr.shape[0]),
                "type": typ,
            }
        )
    return blob, views, accessors


def test_gltf_loads_embedded_base_color_texture(tmp_path):
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    uv = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
    idx = np.array([0, 1, 2], dtype=np.uint16)
    blob, views, accessors = _pack_gltf_buffer(
        [
            (pos, 5126, "VEC3", 34962),
            (uv, 5126, "VEC2", 34962),
            (idx, 5123, "SCALAR", 34963),
        ]
    )
    tex = np.array([[[255, 0, 0], [0, 255, 0]]], dtype=np.uint8)
    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "TEXCOORD_0": 1},
                        "indices": 2,
                        "material": 0,
                    }
                ]
            }
        ],
        "materials": [
            {
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": 0},
                    "metallicFactor": 0.25,
                    "roughnessFactor": 0.8,
                }
            }
        ],
        "textures": [{"source": 0}],
        "images": [{"uri": _png_data_uri(tex)}],
        "buffers": [
            {
                "byteLength": len(blob),
                "uri": "data:application/octet-stream;base64,"
                + base64.b64encode(blob).decode("ascii"),
            }
        ],
        "bufferViews": views,
        "accessors": accessors,
    }
    path = tmp_path / "textured.gltf"
    path.write_text(json.dumps(gltf))

    g = load_gltf(str(path))

    assert g.texture_image is not None
    assert g.texture_image.shape == (1, 2, 3)
    np.testing.assert_allclose(np.array(g.texture_image), tex.astype(np.float32) / 255.0)
    assert g.materials is not None
    assert g.materials[0].base_color_texture == 0
    assert g.materials[0].metallic_factor == 0.25
    assert g.materials[0].roughness_factor == 0.8
    assert_close(g.uvs, uv)


def test_gltf_loads_scene_nodes_primitives_transforms_and_material_ids(tmp_path):
    p0 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    f0 = np.array([0, 1, 2], dtype=np.uint16)
    p1 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    f1 = np.array([0, 1, 2], dtype=np.uint16)
    blob, views, accessors = _pack_gltf_buffer(
        [
            (p0, 5126, "VEC3", 34962),
            (f0, 5123, "SCALAR", 34963),
            (p1, 5126, "VEC3", 34962),
            (f1, 5123, "SCALAR", 34963),
        ]
    )
    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0, 1]}],
        "nodes": [
            {"mesh": 0},
            {"mesh": 1, "translation": [10.0, 0.0, 0.0]},
        ],
        "meshes": [
            {"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "material": 0}]},
            {"primitives": [{"attributes": {"POSITION": 2}, "indices": 3, "material": 1}]},
        ],
        "materials": [
            {"name": "red", "pbrMetallicRoughness": {"baseColorFactor": [1, 0, 0, 1]}},
            {"name": "green", "pbrMetallicRoughness": {"baseColorFactor": [0, 1, 0, 1]}},
        ],
        "buffers": [
            {
                "byteLength": len(blob),
                "uri": "data:application/octet-stream;base64,"
                + base64.b64encode(blob).decode("ascii"),
            }
        ],
        "bufferViews": views,
        "accessors": accessors,
    }
    path = tmp_path / "scene.gltf"
    path.write_text(json.dumps(gltf))

    g = load_gltf(str(path))

    assert g.verts.shape == (6, 3)
    assert g.faces.tolist() == [[0, 1, 2], [3, 4, 5]]
    np.testing.assert_allclose(np.array(g.verts[:3]), p0, atol=1e-6)
    np.testing.assert_allclose(np.array(g.verts[3:]), p1 + np.array([10, 0, 0]), atol=1e-6)
    assert g.material_ids.tolist() == [0, 1]
    assert g.materials is not None
    assert [m.name for m in g.materials] == ["red", "green"]


@pytest.fixture
def cube():
    verts = mx.array(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ]
    )
    faces = mx.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [2, 3, 7],
            [2, 7, 6],
            [1, 2, 6],
            [1, 6, 5],
            [3, 0, 4],
            [3, 4, 7],
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


def test_obj_loads_mtl_diffuse_texture(tmp_path):
    Image.fromarray(np.array([[[255, 0, 0], [0, 255, 0]]], dtype=np.uint8)).save(
        tmp_path / "albedo.png"
    )
    (tmp_path / "mat.mtl").write_text("newmtl mat\nmap_Kd albedo.png\n")
    p = tmp_path / "textured.obj"
    p.write_text(
        "mtllib mat.mtl\n"
        "v 0 0 0\nv 1 0 0\nv 0 1 0\n"
        "vt 0 0\nvt 1 0\nvt 0 1\n"
        "usemtl mat\n"
        "f 1/1 2/2 3/3\n"
    )
    data = load_obj(str(p))
    assert data.texture_path is not None
    assert data.texture_image is not None
    assert data.texture_image.shape == (1, 2, 3)
    assert data.faces_texcoords_idx[0].tolist() == [0, 1, 2]


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
