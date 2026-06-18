import mlx.core as mx
import numpy as np
import pytest

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


def test_render_cli_validates_dimensions(tmp_path):
    with pytest.raises(SystemExit):
        main(["missing.ply", "--out", str(tmp_path / "x.png"), "--width", "0"])
