"""Benchmark the fast forward-only splat renderer against the training rasterizer.

Renders an orbiting camera with both paths and reports per-frame latency, FPS,
speedup, and PSNR of the fast path against the reference output. Works on a
synthetic scene out of the box, or any 3DGS ``.ply`` checkpoint:

    python examples/benchmark_fast_rasterization.py
    python examples/benchmark_fast_rasterization.py --ply point_cloud.ply
    python examples/benchmark_fast_rasterization.py --ply scene.ply --width 1920 --height 1080
"""

import argparse
import math
import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.exists():
    sys.path.insert(0, str(_SRC))

import mlx.core as mx
import numpy as np

from mlx3d.cameras import Camera
from mlx3d.splatting import FastGaussianRenderer, GaussianModel


def synthetic_scene(n: int, big_splats: bool = False) -> GaussianModel:
    mx.random.seed(0)
    model = GaussianModel.from_points(mx.random.normal((n, 3)) * 1.5, sh_degree=3)
    model.params["sh_rest"] = mx.random.normal(model.params["sh_rest"].shape) * 0.3
    shift = 0.3 if big_splats else -1.2
    model.params["scales"] = (
        model.params["scales"] + shift
        if big_splats
        else mx.clip(model.params["scales"] + shift, -8.0, -3.2)
    )
    mx.eval(model.params)
    return model


def scene_orbit(model: GaussianModel, width: int, height: int):
    means = np.array(model.params["means"])
    center = means.mean(axis=0)
    radius = float(np.percentile(np.linalg.norm(means - center, axis=1), 90)) * 2.5 + 1e-3

    def cam(i: int) -> Camera:
        th = 0.12 * i
        eye = center + radius * np.array([math.sin(th), 0.25, -math.cos(th)])
        return Camera.look_at(
            eye=tuple(float(c) for c in eye),
            at=tuple(float(c) for c in center),
            width=width,
            height=height,
        )

    return cam


def bench(frame_fn, warmup: int = 3, iters: int = 20) -> float:
    for i in range(warmup):
        frame_fn(i)
    t0 = time.perf_counter()
    for i in range(iters):
        frame_fn(warmup + i)
    return (time.perf_counter() - t0) / iters * 1000.0


def psnr(a: mx.array, b: mx.array) -> float:
    mse = float(mx.mean((a - b) ** 2).item())
    return 10.0 * math.log10(1.0 / max(mse, 1e-12))


def run(model: GaussianModel, width: int, height: int, label: str) -> None:
    cam = scene_orbit(model, width, height)
    bg = mx.zeros((3,))
    fast = FastGaussianRenderer(model)

    t_ref = bench(lambda i: mx.eval(model.render(cam(i), background=bg)["image"]))
    t_fast = bench(lambda i: mx.eval(fast.render(cam(i), background=bg)["image"]))
    quality = psnr(
        model.render(cam(0), background=bg)["image"],
        fast.render(cam(0), background=bg)["image"],
    )
    print(
        f"| {label} | {width}x{height} | {t_ref:8.2f} ms ({1000 / t_ref:5.1f} fps) "
        f"| {t_fast:8.2f} ms ({1000 / t_fast:5.1f} fps) "
        f"| **{t_ref / t_fast:.2f}x** | {quality:.0f} dB |"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", type=str, default=None, help="3DGS checkpoint to benchmark")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    args = parser.parse_args()

    print("| scene | resolution | reference | fast | speedup | PSNR |")
    print("|---|---|---|---|---|---|")
    if args.ply:
        model = GaussianModel.load_ply(args.ply)
        name = Path(args.ply).stem + f" ({model.num_gaussians:,})"
        sizes = (
            [(args.width, args.height)]
            if args.width and args.height
            else [(1280, 720), (1920, 1080)]
        )
        for w, h in sizes:
            run(model, w, h, name)
    else:
        for n in (50_000, 200_000, 500_000):
            model = synthetic_scene(n)
            for w, h in ((1280, 720), (1920, 1080)):
                run(model, w, h, f"synthetic {n:,}")
        big = synthetic_scene(20_000, big_splats=True)
        run(big, 1920, 1080, "synthetic 20,000 (large splats)")


if __name__ == "__main__":
    main()
