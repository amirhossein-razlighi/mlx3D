import base64
import io
import json

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

import mlx3d.cli.render as render_cli
from mlx3d.cli.render import main
from mlx3d.io import load_image, save_gltf
from mlx3d.splatting import GaussianModel


def test_render_cli_autodetects_gaussian_ply(tmp_path):
    model = GaussianModel.from_points(
        mx.array([[0.0, 0.0, 0.0]]),
        mx.array([[1.0, 0.1, 0.0]]),
        sh_degree=0,
        initial_opacity=0.95,
    )
    model.params["scales"] = mx.log(mx.full((1, 3), 0.2))
    ckpt = tmp_path / "point_cloud.ply"
    out = tmp_path / "render.png"
    model.save_ply(str(ckpt))

    main([str(ckpt), "--out", str(out), "--width", "32", "--height", "24"])

    img = load_image(str(out))
    assert img.shape == (24, 32, 3)
    assert float(img.max()) > 0.05


def test_render_cli_renders_gltf_mesh_depth(tmp_path):
    verts = mx.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]])
    faces = mx.array([[0, 1, 2]], dtype=mx.int32)
    mesh_path = tmp_path / "tri.glb"
    out = tmp_path / "depth.png"
    save_gltf(str(mesh_path), verts, faces, material_base_color=(0.1, 0.7, 0.2, 1.0))

    main(
        [
            str(mesh_path),
            "--type",
            "mesh",
            "--mode",
            "depth",
            "--out",
            str(out),
            "--width",
            "32",
            "--height",
            "32",
            "--eye",
            "0",
            "0",
            "-3",
            "--at",
            "0",
            "0",
            "0",
        ]
    )

    img = np.array(load_image(str(out)))
    assert img.shape == (32, 32, 3)
    assert img.max() > 0.0


def _textured_gltf(path, material=None):
    pos = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype=np.float32)
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    idx = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint16)
    blob = b""
    views = []
    accessors = []
    for arr, component_type, typ, target in [
        (pos, 5126, "VEC3", 34962),
        (uv, 5126, "VEC2", 34962),
        (idx, 5123, "SCALAR", 34963),
    ]:
        offset = len(blob)
        raw = arr.tobytes()
        blob += raw + b"\x00" * ((4 - len(raw) % 4) % 4)
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(raw), "target": target})
        accessors.append(
            {
                "bufferView": len(views) - 1,
                "componentType": component_type,
                "count": int(arr.shape[0]),
                "type": typ,
            }
        )
    tex = np.array(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 0]],
        ],
        dtype=np.uint8,
    )
    img_buf = io.BytesIO()
    Image.fromarray(tex).save(img_buf, format="PNG")
    if material is None:
        material = {"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}
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
        "materials": [material],
        "textures": [{"source": 0}],
        "images": [
            {"uri": "data:image/png;base64," + base64.b64encode(img_buf.getvalue()).decode("ascii")}
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
    path.write_text(json.dumps(gltf))


def test_render_cli_uses_gltf_base_color_texture(tmp_path):
    mesh_path = tmp_path / "textured.gltf"
    out = tmp_path / "textured.png"
    _textured_gltf(mesh_path)

    main(
        [
            str(mesh_path),
            "--type",
            "mesh",
            "--out",
            str(out),
            "--width",
            "48",
            "--height",
            "48",
            "--unlit",
            "--eye",
            "0",
            "0",
            "-3",
            "--at",
            "0",
            "0",
            "0",
        ]
    )

    img = np.array(load_image(str(out)))
    visible = img[img.sum(axis=-1) > 0.05]
    assert visible.shape[0] > 0
    assert visible.std(axis=0).max() > 0.1


def test_render_cli_forwards_gltf_pbr_factors(tmp_path, monkeypatch):
    mesh_path = tmp_path / "textured.gltf"
    out = tmp_path / "pbr.png"
    _textured_gltf(
        mesh_path,
        {
            "pbrMetallicRoughness": {
                "baseColorTexture": {"index": 0},
                "metallicFactor": 0.65,
                "roughnessFactor": 0.2,
            }
        },
    )
    captured = {}

    def fake_render_mesh(camera, mesh, **kwargs):
        captured.update(kwargs)
        return {"image": mx.ones((camera.height, camera.width, 3), dtype=mx.float32)}

    monkeypatch.setattr(render_cli, "render_mesh", fake_render_mesh)

    main(
        [
            str(mesh_path),
            "--type",
            "mesh",
            "--out",
            str(out),
            "--width",
            "16",
            "--height",
            "16",
            "--shading",
            "pbr",
        ]
    )

    assert captured["shading"] == "pbr"
    assert captured["metallic"] == 0.65
    assert captured["roughness"] == 0.2
    assert captured["texture"] is not None


def test_render_cli_rejects_ambiguous_shading_flags(tmp_path):
    mesh_path = tmp_path / "textured.gltf"
    _textured_gltf(mesh_path)

    with pytest.raises(SystemExit):
        main(
            [
                str(mesh_path),
                "--type",
                "mesh",
                "--out",
                str(tmp_path / "x.png"),
                "--unlit",
                "--shading",
                "pbr",
            ]
        )


def test_render_cli_validates_dimensions(tmp_path):
    with pytest.raises(SystemExit):
        main(["missing.ply", "--out", str(tmp_path / "x.png"), "--width", "0"])
