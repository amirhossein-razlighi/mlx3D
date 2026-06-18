import json

import mlx.core as mx
import numpy as np
import pytest

from mlx3d.cameras import Camera
from mlx3d.cli.eval import EvalConfig, _view_indices, evaluate_gaussian_checkpoint, main
from mlx3d.io import save_image
from mlx3d.splatting import GaussianModel


def _write_blender_scene(root, model: GaussianModel, n: int = 3) -> None:
    frames = []
    # Identity Blender c2w converts to an OpenCV camera looking down world -Z.
    cam = Camera(
        R=mx.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]),
        t=mx.zeros((3,)),
        fx=27.71281292110204,
        fy=27.71281292110204,
        cx=16.0,
        cy=16.0,
        width=32,
        height=32,
    )
    for i in range(n):
        path = root / f"r_{i}.png"
        save_image(str(path), model.render(cam, background=mx.zeros((3,)))["image"])
        frames.append(
            {
                "file_path": f"r_{i}.png",
                "transform_matrix": np.eye(4, dtype=float).tolist(),
            }
        )
    (root / "transforms_train.json").write_text(
        json.dumps({"camera_angle_x": 1.0471975511965976, "frames": frames})
    )


def _tiny_checkpoint(tmp_path) -> tuple[str, GaussianModel]:
    model = GaussianModel.from_points(
        mx.array([[0.0, 0.0, -3.0], [0.35, 0.0, -3.2]]),
        mx.array([[1.0, 0.0, 0.0], [0.0, 0.5, 1.0]]),
        sh_degree=0,
        initial_opacity=0.9,
    )
    model.params["scales"] = mx.log(mx.full((2, 3), 0.15))
    path = tmp_path / "point_cloud.ply"
    model.save_ply(str(path))
    return str(path), model


def test_view_indices_evenly_spaced_and_all_views():
    assert _view_indices(5, None).tolist() == [0, 1, 2, 3, 4]
    assert _view_indices(5, 0).tolist() == [0, 1, 2, 3, 4]
    assert _view_indices(5, 3).tolist() == [0, 2, 4]
    with pytest.raises(ValueError, match="no views"):
        _view_indices(0, 1)


def test_eval_gaussian_checkpoint_on_blender_scene(tmp_path):
    ckpt, model = _tiny_checkpoint(tmp_path)
    _write_blender_scene(tmp_path, model, n=3)

    result = evaluate_gaussian_checkpoint(
        EvalConfig(
            checkpoint=ckpt,
            data=str(tmp_path),
            data_format="blender",
            image_cache="uint8",
            views=2,
            white_background=False,
        )
    )

    assert result["num_views"] == 2
    assert result["indices"] == [0, 2]
    assert result["psnr"] > 45.0
    assert result["ssim"] > 0.99
    assert result["l1"] < 0.01
    assert len(result["per_view"]) == 2


def test_eval_cli_writes_json(tmp_path, capsys):
    ckpt, model = _tiny_checkpoint(tmp_path)
    _write_blender_scene(tmp_path, model, n=1)
    out = tmp_path / "metrics.json"

    main(
        [
            ckpt,
            "--data",
            str(tmp_path),
            "--format",
            "blender",
            "--black-background",
            "--views",
            "1",
            "--json-out",
            str(out),
        ]
    )

    printed = json.loads(capsys.readouterr().out)
    saved = json.loads(out.read_text())
    assert printed["num_views"] == 1
    assert saved == printed
    assert saved["psnr"] > 45.0


def test_eval_cli_validates_args(tmp_path):
    with pytest.raises(SystemExit):
        main(["missing.ply", "--data", str(tmp_path), "--downscale", "0"])
