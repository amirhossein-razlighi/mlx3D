"""Evaluate Gaussian Splatting checkpoints on posed image datasets."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from ..datasets import load_blender, load_colmap
from ..losses import l1_loss, psnr, ssim
from ..splatting import GaussianModel


@dataclass
class EvalConfig:
    checkpoint: str
    data: str
    data_format: str = "colmap"
    split: str = "train"
    downscale: int = 1
    image_cache: str = "uint8"
    views: int | None = None
    antialias: bool = False
    white_background: bool | None = None


def _view_indices(n: int, views: int | None) -> np.ndarray:
    if n <= 0:
        raise ValueError("Dataset contains no views.")
    if views is None or views <= 0 or views >= n:
        return np.arange(n, dtype=np.int32)
    return np.linspace(0, n - 1, int(views), dtype=np.int32)


def _load_dataset(config: EvalConfig):
    if config.data_format == "colmap":
        return load_colmap(config.data, downscale=config.downscale, cache=config.image_cache), False
    if config.data_format == "blender":
        white = True if config.white_background is None else bool(config.white_background)
        return (
            load_blender(
                config.data,
                split=config.split,
                downscale=config.downscale,
                white_background=white,
                cache=config.image_cache,
            ),
            white,
        )
    raise ValueError("data_format must be 'colmap' or 'blender'.")


def evaluate_gaussian_checkpoint(config: EvalConfig) -> dict[str, object]:
    """Evaluate a 3DGS checkpoint and return aggregate/per-view metrics."""
    ds, default_white = _load_dataset(config)
    white = default_white if config.white_background is None else bool(config.white_background)
    bg = mx.ones((3,), dtype=mx.float32) if white else mx.zeros((3,), dtype=mx.float32)
    model = GaussianModel.load_ply(config.checkpoint)
    indices = _view_indices(len(ds), config.views)

    per_view = []
    for idx in indices:
        cam, target = ds[int(idx)]
        pred = model.render(cam, background=bg, antialias=config.antialias)["image"]
        metrics = {
            "index": int(idx),
            "psnr": float(psnr(pred, target)),
            "ssim": float(ssim(pred, target)),
            "l1": float(l1_loss(pred, target)),
        }
        per_view.append(metrics)

    summary = {
        "checkpoint": config.checkpoint,
        "data": config.data,
        "format": config.data_format,
        "split": config.split,
        "num_views": int(len(indices)),
        "indices": [int(i) for i in indices],
        "psnr": float(np.mean([m["psnr"] for m in per_view])),
        "ssim": float(np.mean([m["ssim"] for m in per_view])),
        "l1": float(np.mean([m["l1"] for m in per_view])),
        "per_view": per_view,
    }
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="3DGS PLY checkpoint")
    parser.add_argument("--data", required=True, help="COLMAP or Blender scene root")
    parser.add_argument("--format", choices=["colmap", "blender"], default="colmap")
    parser.add_argument("--split", default="train", help="Blender split name")
    parser.add_argument("--downscale", type=int, default=1)
    parser.add_argument("--image-cache", choices=["ram", "uint8", "disk"], default="uint8")
    parser.add_argument("--views", type=int, default=None, help="number of evenly spaced views")
    parser.add_argument("--antialias", action="store_true")
    bg = parser.add_mutually_exclusive_group()
    bg.add_argument("--white-background", action="store_true", dest="white_background")
    bg.add_argument("--black-background", action="store_false", dest="white_background")
    parser.set_defaults(white_background=None)
    parser.add_argument("--json-out", help="optional path for JSON metrics")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.downscale <= 0:
        parser.error("--downscale must be positive.")
    if args.views is not None and args.views < 0:
        parser.error("--views must be non-negative.")

    result = evaluate_gaussian_checkpoint(
        EvalConfig(
            checkpoint=args.checkpoint,
            data=args.data,
            data_format=args.format,
            split=args.split,
            downscale=args.downscale,
            image_cache=args.image_cache,
            views=args.views,
            antialias=args.antialias,
            white_background=args.white_background,
        )
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        with open(args.json_out, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
