"""Train Spaghetti Splatting on a COLMAP scene (optionally against 3DGS).

Uses the Mip-NeRF 360 protocol: every 8th image is held out for testing.
Both methods start from the scene's SfM points with the same primitive cap,
schedule and resolution, so ``--compare-3dgs`` is a like-for-like comparison.

    python examples/train_spaghetti_splatting.py --data path/to/bicycle \\
        --downscale 2 --iters 7000 --budget 300000 --compare-3dgs --out outputs/bicycle

Writes ``metrics.json``, ``spaghetti.ply`` (and ``gaussians.ply``), and, for
each held-out view, ``renders/<name>_{gt,spaghetti,3dgs}.png``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from mlx3d.datasets import load_colmap
from mlx3d.losses import psnr, ssim
from mlx3d.splatting import GaussianModel, GaussianTrainer, TrainerConfig
from mlx3d.splatting.spaghetti import SpaghettiModel, SpaghettiTrainer, default_config_for


def to_uint8(image: mx.array) -> np.ndarray:
    return (np.clip(np.array(image), 0, 1) * 255).astype(np.uint8)


def evaluate(render, ds, test_ids, out_dir: Path | None, tag: str) -> dict[str, float]:
    psnrs, ssims = [], []
    for i in test_ids:
        cam, gt = ds[i]
        pred = mx.clip(render(cam), 0.0, 1.0)
        psnrs.append(float(psnr(pred, gt)))
        ssims.append(float(ssim(pred, gt)))
        if out_dir is not None:
            stem = Path(ds.image_names[i]).stem
            Image.fromarray(to_uint8(pred)).save(out_dir / f"{stem}_{tag}.png")
            gt_path = out_dir / f"{stem}_gt.png"
            if not gt_path.exists():
                Image.fromarray(to_uint8(gt)).save(gt_path)
    return {"psnr": float(np.mean(psnrs)), "ssim": float(np.mean(ssims))}


def train(trainer, ds, train_ids, iters: int, label: str, seed: int, checkpoint=None) -> float:
    """Run ``iters`` steps on random training views; ``checkpoint()`` every 1000."""
    rng = np.random.default_rng(seed)
    start = time.perf_counter()
    for it in range(1, iters + 1):
        i = int(train_ids[rng.integers(len(train_ids))])
        cam, img = ds[i]
        info = trainer.step(cam, img)
        if it % 500 == 0 or it == 1:
            count = info.get("num_segments", info.get("num_gaussians"))
            rate = it / (time.perf_counter() - start)
            print(
                f"{label:9s} {it:6d}/{iters}  loss {info['loss']:.4f}  n {count}  {rate:.2f} it/s",
                flush=True,
            )
        if checkpoint is not None and it % 1000 == 0:
            checkpoint()
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", required=True, help="COLMAP scene root (images/ + sparse/0)")
    parser.add_argument("--out", default="outputs/spaghetti_scene")
    parser.add_argument("--downscale", type=int, default=2)
    parser.add_argument("--iters", type=int, default=7000)
    parser.add_argument("--budget", type=int, default=300_000, help="max primitives (both methods)")
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--compare-3dgs", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out = Path(args.out)
    renders = out / "renders"
    renders.mkdir(parents=True, exist_ok=True)
    mx.random.seed(args.seed)

    ds = load_colmap(args.data, downscale=args.downscale, cache="uint8")
    test_ids = list(range(0, len(ds), 8))
    train_ids = [i for i in range(len(ds)) if i % 8]
    extent = ds.scene_extent
    cam0, _ = ds[0]
    print(
        f"{len(ds)} views ({len(train_ids)} train / {len(test_ids)} test) at "
        f"{cam0.width}x{cam0.height}, {ds.points.shape[0]} SfM points, extent {extent:.2f}"
    )
    metrics: dict[str, dict] = {}

    model = SpaghettiModel.from_points(
        ds.points, ds.point_colors, sh_degree=args.sh_degree, max_scale=0.01 * extent
    )
    config = default_config_for(args.iters, max_segments=args.budget, lambda_connect=0.5)
    trainer = SpaghettiTrainer(model, config, scene_extent=extent)
    seconds = train(
        trainer,
        ds,
        train_ids,
        args.iters,
        "spaghetti",
        args.seed,
        checkpoint=lambda: model.save_ply(str(out / "spaghetti.ply")),
    )
    joints = trainer.finalize_polylines()
    metrics["spaghetti"] = {
        **evaluate(lambda c: model.render(c)["image"], ds, test_ids, renders, "spaghetti"),
        "primitives": model.num_segments,
        "joints": joints,
        "train_seconds": seconds,
    }
    model.save_ply(str(out / "spaghetti.ply"))
    print("spaghetti", metrics["spaghetti"], flush=True)
    del trainer
    mx.clear_cache()

    if args.compare_3dgs:
        gaussians = GaussianModel.from_points(
            ds.points,
            ds.point_colors,
            sh_degree=args.sh_degree,
            scale_init_max_scale=0.01 * extent,
            scale_init_max_ref=min(int(ds.points.shape[0]), 100_000),
        )
        gconfig = TrainerConfig(
            densify_from=500,
            densify_until=int(args.iters * 5 / 6),
            max_gaussians=args.budget,
            lr_means_max_steps=args.iters,
        )
        gtrainer = GaussianTrainer(gaussians, gconfig, scene_extent=extent)
        seconds = train(
            gtrainer,
            ds,
            train_ids,
            args.iters,
            "3dgs",
            args.seed,
            checkpoint=lambda: gaussians.save_ply(str(out / "gaussians.ply")),
        )
        metrics["3dgs"] = {
            **evaluate(lambda c: gaussians.render(c)["image"], ds, test_ids, renders, "3dgs"),
            "primitives": gaussians.num_gaussians,
            "train_seconds": seconds,
        }
        gaussians.save_ply(str(out / "gaussians.ply"))
        print("3dgs", metrics["3dgs"], flush=True)

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
