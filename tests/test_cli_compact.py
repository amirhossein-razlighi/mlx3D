import json

import mlx.core as mx
import pytest

from mlx3d.cli.compact import compact_checkpoint, main
from mlx3d.splatting import GaussianModel


def _checkpoint(tmp_path) -> str:
    pts = mx.stack([mx.array([float(i), 0.0, 0.0]) for i in range(6)], axis=0)
    model = GaussianModel.from_points(pts, sh_degree=2)
    model.active_sh_degree = 2
    model.params["opacities"] = mx.array([-8.0, 2.0, -4.0, 4.0, 1.0, -7.0])
    model.params["scales"] = mx.log(
        mx.array(
            [
                [0.1, 0.1, 0.1],
                [0.2, 0.2, 0.2],
                [5.0, 5.0, 5.0],
                [1.0, 1.0, 1.0],
                [0.4, 0.4, 0.4],
                [10.0, 10.0, 10.0],
            ]
        )
    )
    path = tmp_path / "in.ply"
    model.save_ply(str(path))
    return str(path)


def test_compact_checkpoint_writes_smaller_checkpoint(tmp_path):
    inp = _checkpoint(tmp_path)
    out = tmp_path / "compact.ply"

    summary = compact_checkpoint(inp, str(out), min_opacity=0.01, max_gaussians=2, sh_degree=1)
    loaded = GaussianModel.load_ply(str(out))

    assert summary["gaussians_before"] == 6
    assert summary["gaussians_after"] == 2
    assert summary["sh_degree_after"] == 1
    assert loaded.num_gaussians == 2
    assert loaded.sh_degree == 1
    assert loaded.sh.shape == (2, 4, 3)


def test_compact_cli_prints_and_writes_json(tmp_path, capsys):
    inp = _checkpoint(tmp_path)
    out = tmp_path / "compact.ply"
    metrics = tmp_path / "summary.json"

    main(
        [
            inp,
            "--out",
            str(out),
            "--min-opacity",
            "0.01",
            "--max-gaussians",
            "3",
            "--json-out",
            str(metrics),
        ]
    )

    printed = json.loads(capsys.readouterr().out)
    saved = json.loads(metrics.read_text())
    assert saved == printed
    assert printed["gaussians_after"] == 3
    assert out.exists()


def test_compact_cli_validates_args(tmp_path):
    with pytest.raises(SystemExit):
        main(["missing.ply", "--out", str(tmp_path / "x.ply"), "--min-opacity", "1.5"])
    with pytest.raises(SystemExit):
        main(["missing.ply", "--out", str(tmp_path / "x.ply"), "--max-gaussians", "0"])
