"""Thin wrapper around the COLMAP CLI for automatic pose estimation.

Runs the standard feature_extractor -> matcher -> mapper chain into a
workspace, then selects the largest reconstructed model as ``sparse/0`` so the
result loads directly with :func:`mlx3d.datasets.load_colmap`.
"""

import os
import shutil
import struct
import subprocess

__all__ = ["has_colmap", "run_colmap"]


def has_colmap() -> bool:
    """Whether the ``colmap`` binary is available on PATH."""
    return shutil.which("colmap") is not None


def _run(cmd: list[str], log_path: str, log=print) -> None:
    log(f"  $ {' '.join(cmd[:2])} ...")
    with open(log_path, "a") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        f.flush()
        try:
            subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"COLMAP step failed ({cmd[1]}, exit {e.returncode}). "
                f"See the full log at {log_path}."
            ) from e


def _model_image_count(model_dir: str) -> int:
    path = os.path.join(model_dir, "images.bin")
    if not os.path.exists(path):
        return 0
    with open(path, "rb") as f:
        return struct.unpack("<Q", f.read(8))[0]


def run_colmap(
    images_dir: str,
    workspace: str,
    sequential: bool = False,
    camera_model: str = "OPENCV",
    single_camera: bool = True,
    log=print,
) -> str:
    """Run COLMAP SfM on ``images_dir`` and return the sparse model directory.

    Args:
        images_dir: directory of input images.
        workspace: output directory; receives ``database.db``, ``sparse/`` and
            ``colmap.log``.
        sequential: use the sequential matcher (video / ordered captures)
            instead of exhaustive matching.
        camera_model: COLMAP camera model for feature extraction.
        single_camera: share one camera across all images (same device).
    """
    colmap = shutil.which("colmap")
    if colmap is None:
        raise RuntimeError("COLMAP binary not found on PATH.")
    os.makedirs(workspace, exist_ok=True)
    db = os.path.join(workspace, "database.db")
    sparse = os.path.join(workspace, "sparse")
    log_path = os.path.join(workspace, "colmap.log")
    os.makedirs(sparse, exist_ok=True)

    _run(
        [
            colmap,
            "feature_extractor",
            "--database_path",
            db,
            "--image_path",
            images_dir,
            "--ImageReader.camera_model",
            camera_model,
            "--ImageReader.single_camera",
            "1" if single_camera else "0",
        ],
        log_path,
        log,
    )
    if sequential:
        matcher = [
            colmap,
            "sequential_matcher",
            "--database_path",
            db,
            "--SequentialMatching.overlap",
            "15",
        ]
    else:
        matcher = [colmap, "exhaustive_matcher", "--database_path", db]
    _run(matcher, log_path, log)
    _run(
        [
            colmap,
            "mapper",
            "--database_path",
            db,
            "--image_path",
            images_dir,
            "--output_path",
            sparse,
        ],
        log_path,
        log,
    )

    models = [
        os.path.join(sparse, d)
        for d in sorted(os.listdir(sparse))
        if os.path.isdir(os.path.join(sparse, d))
    ]
    if not models:
        raise RuntimeError(
            "COLMAP produced no reconstruction. The capture likely lacks "
            "overlap or texture; try more/sharper images with generous overlap."
        )
    best = max(models, key=_model_image_count)
    target = os.path.join(sparse, "0")
    if best != target:
        # Promote the largest model to sparse/0 (swap directories).
        tmp = os.path.join(sparse, "_tmp_swap")
        if os.path.exists(target):
            os.rename(target, tmp)
        os.rename(best, target)
        if os.path.exists(tmp):
            os.rename(tmp, best)
    n = _model_image_count(target)
    log(f"  COLMAP registered {n} images (model {os.path.basename(best)}).")
    return target
