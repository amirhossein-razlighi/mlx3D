"""Benchmark Gaussian Splatting startup, training, and rendering.

This is intentionally small and dependency-light so it can be run while
optimizing MLX/Metal kernels:

    uv run python examples/benchmark_gaussian_splatting.py \
        --data data/tandt/truck --downscale 4 --steps 20 --save-render /tmp/bench.png
"""

import argparse
import json
import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.exists():
    sys.path.insert(0, str(_SRC))

import mlx.core as mx
import numpy as np

from mlx3d.losses import psnr
from mlx3d.splatting import (
    GaussianModel,
    GaussianTrainer,
    TrainerConfig,
    bin_gaussians,
    eval_sh,
    project_gaussians,
    rasterize,
)


def _gb(nbytes: int | float) -> float:
    return float(nbytes) / float(1 << 30)


def _memory() -> dict[str, float]:
    return {
        "active_gb": _gb(mx.get_active_memory()),
        "peak_gb": _gb(mx.get_peak_memory()),
        "cache_gb": _gb(mx.get_cache_memory()),
    }


def _image_stats(img: mx.array) -> dict[str, float]:
    arr = np.array(img)
    return {
        "min": float(arr.min()),
        "mean": float(arr.mean()),
        "max": float(arr.max()),
        "std": float(arr.std()),
    }


def _projection_stats(model: GaussianModel, cam) -> dict[str, float | int]:
    proj = project_gaussians(
        cam,
        model.params["means"],
        model.params["quats"],
        mx.exp(model.params["scales"]),
    )
    mx.eval(proj["means2d"], proj["radii"], proj["depths"])
    radii = np.array(proj["radii"])
    xy = np.array(proj["means2d"])
    visible = (
        (radii > 0)
        & (xy[:, 0] + radii >= 0)
        & (xy[:, 0] - radii < cam.width)
        & (xy[:, 1] + radii >= 0)
        & (xy[:, 1] - radii < cam.height)
    )
    vis_radii = radii[visible]
    if vis_radii.size == 0:
        return {"visible": 0}
    p50, p90, p99 = np.percentile(vis_radii, [50, 90, 99])
    return {
        "visible": int(visible.sum()),
        "visible_frac": float(visible.mean()),
        "radius_p50": float(p50),
        "radius_p90": float(p90),
        "radius_p99": float(p99),
        "radius_max": float(vis_radii.max()),
    }


def _profile_render(model: GaussianModel, cam, bg: mx.array) -> dict[str, float | int]:
    timings = {}

    t0 = time.perf_counter()
    proj = project_gaussians(
        cam,
        model.params["means"],
        model.params["quats"],
        mx.exp(model.params["scales"]),
    )
    mx.eval(proj["means2d"], proj["conics"], proj["radii"], proj["depths"])
    timings["projection_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    sh = mx.concatenate([model.params["sh_dc"], model.params["sh_rest"]], axis=1)
    dirs = model.params["means"] - cam.camera_center
    dirs = dirs / mx.maximum(mx.linalg.norm(dirs, axis=-1, keepdims=True), 1e-8)
    colors = mx.maximum(eval_sh(model.active_sh_degree, sh, mx.stop_gradient(dirs)), 0.0)
    opacities = mx.sigmoid(model.params["opacities"])
    mx.eval(colors, opacities)
    timings["color_opacity_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    sorted_ids, tile_ranges, tiles_x, tiles_y = bin_gaussians(
        proj["means2d"], proj["radii"], proj["depths"], cam.width, cam.height
    )
    mx.eval(sorted_ids, tile_ranges)
    timings["binning_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    out = rasterize(
        proj["means2d"],
        proj["conics"],
        colors,
        opacities,
        sorted_ids,
        tile_ranges,
        cam.width,
        cam.height,
        tiles_x,
        tiles_y,
        background=bg,
    )
    mx.eval(out["image"], out["alpha"])
    timings["rasterize_s"] = time.perf_counter() - t0

    timings["total_s"] = sum(v for k, v in timings.items() if k.endswith("_s"))
    radii = np.array(proj["radii"])
    xy = np.array(proj["means2d"])
    visible = (
        (radii > 0)
        & (xy[:, 0] + radii >= 0)
        & (xy[:, 0] - radii < cam.width)
        & (xy[:, 1] + radii >= 0)
        & (xy[:, 1] - radii < cam.height)
    )
    timings["duplicates"] = int(sorted_ids.shape[0])
    timings["tiles"] = int(tiles_x * tiles_y)
    timings["visible"] = int(visible.sum())
    timings["visible_frac"] = float(visible.mean())
    return timings


def _save_render(path: str, img: mx.array) -> None:
    from PIL import Image

    arr = (np.clip(np.array(img), 0.0, 1.0) * 255).astype(np.uint8)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(out)


def _eval_psnr(model: GaussianModel, ds, white_bg: bool, n_views: int) -> dict[str, object] | None:
    if n_views <= 0:
        return None
    bg = mx.ones((3,)) if white_bg else mx.zeros((3,))
    n_eval = max(1, min(int(n_views), len(ds)))
    view_ids = np.linspace(0, len(ds) - 1, n_eval, dtype=np.int32)
    values = []
    for view_id in view_ids:
        cam, img = ds[int(view_id)]
        out = model.render(cam, background=bg)
        values.append(float(psnr(out["image"], img)))
    return {
        "views": [int(v) for v in view_ids],
        "mean_psnr": float(np.mean(values)),
        "values": values,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--format", choices=["colmap", "blender"], default="colmap")
    parser.add_argument("--downscale", type=int, default=4)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-views", type=int, default=3)
    parser.add_argument("--image-cache", choices=["ram", "uint8", "disk"], default=None)
    parser.add_argument("--low-mem", action="store_true")
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--init-points", type=int, default=30000)
    parser.add_argument("--scale-init-max-ref", type=int, default=10_000)
    parser.add_argument("--scale-init-chunk-size", type=int, default=1024)
    parser.add_argument("--init-scale-max-frac", type=float, default=0.01)
    parser.add_argument("--position-lr-final", type=float, default=1.6e-6)
    parser.add_argument("--position-lr-max-steps", type=int, default=30_000)
    parser.add_argument("--method", choices=["vanilla", "mcmc", "2dgs"], default="vanilla")
    parser.add_argument("--densify-from", type=int, default=500)
    parser.add_argument("--densify-until", type=int, default=None)
    parser.add_argument("--densify-every", type=int, default=100)
    parser.add_argument("--densify-grad-threshold", type=float, default=0.0002)
    parser.add_argument("--mcmc-relocate-frac", type=float, default=0.02)
    parser.add_argument("--mcmc-min-opacity", type=float, default=0.01)
    parser.add_argument("--mcmc-jitter-scale", type=float, default=0.25)
    parser.add_argument("--mcmc-noise-scale", type=float, default=0.01)
    parser.add_argument("--2d-thickness", dest="two_d_thickness", type=float, default=1e-4)
    parser.add_argument("--max-gaussians", type=int, default=None)
    parser.add_argument("--save-render", type=str, default=None)
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()
    if args.seed >= 0:
        np.random.seed(args.seed)
        mx.random.seed(args.seed)

    timings: dict[str, float] = {}
    image_cache = args.image_cache or ("uint8" if args.low_mem else "ram")

    mx.clear_cache()
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()

    t0 = time.perf_counter()
    if args.format == "colmap":
        from mlx3d.datasets import load_colmap

        ds = load_colmap(args.data, downscale=args.downscale, cache=image_cache)
        init_points, init_colors = ds.points, ds.point_colors
        scene_extent = ds.scene_extent
        white_bg = False
    else:
        from mlx3d.datasets import load_blender

        ds = load_blender(args.data, "train", downscale=args.downscale, cache=image_cache)
        init_points = mx.random.uniform(low=-1.5, high=1.5, shape=(args.init_points, 3))
        init_colors = mx.random.uniform(shape=(args.init_points, 3))
        scene_extent = 3.0
        white_bg = True
    cam0, img0 = ds[0]
    mx.eval(init_points, init_colors, img0)
    timings["load_s"] = time.perf_counter() - t0

    scale_cap = None if args.init_scale_max_frac <= 0 else args.init_scale_max_frac * scene_extent
    t0 = time.perf_counter()
    model = GaussianModel.from_points(
        init_points,
        init_colors,
        sh_degree=args.sh_degree,
        scale_init_max_ref=args.scale_init_max_ref,
        scale_init_chunk_size=args.scale_init_chunk_size,
        scale_init_max_scale=scale_cap,
    )
    if args.method == "2dgs":
        model.apply_2dgs_constraints(args.two_d_thickness * scene_extent)
    mx.eval(model.params)
    timings["init_s"] = time.perf_counter() - t0
    init_memory = _memory()

    bg = mx.ones((3,)) if white_bg else mx.zeros((3,))
    t0 = time.perf_counter()
    initial = model.render(cam0, background=bg)
    mx.eval(initial["image"], initial["alpha"])
    timings["initial_render_s"] = time.perf_counter() - t0
    initial_stats = {
        "image": _image_stats(initial["image"]),
        "alpha": _image_stats(initial["alpha"]),
        "projection": _projection_stats(model, cam0),
    }

    config = TrainerConfig(
        method=args.method,
        white_background=white_bg,
        densify_from=args.densify_from,
        densify_until=args.densify_until
        if args.densify_until is not None
        else max(args.steps // 2, 0),
        densify_every=args.densify_every,
        densify_grad_threshold=args.densify_grad_threshold,
        mcmc_relocate_frac=args.mcmc_relocate_frac,
        mcmc_min_opacity=args.mcmc_min_opacity,
        mcmc_jitter_scale=args.mcmc_jitter_scale,
        mcmc_noise_scale=args.mcmc_noise_scale,
        two_d_thickness=args.two_d_thickness,
        max_gaussians=args.max_gaussians,
        low_memory=args.low_mem,
        lr_means_final=args.position_lr_final,
        lr_means_max_steps=args.position_lr_max_steps,
    )
    trainer = GaussianTrainer(model, config, scene_extent=scene_extent)
    order = np.random.permutation(len(ds))
    all_step_times = []
    warmup_step_times = []
    step_times = []
    losses = []
    densify_events = []
    total_steps = args.warmup + args.steps
    for i in range(total_steps):
        cam, target = ds[int(order[i % len(ds)])]
        t0 = time.perf_counter()
        info = trainer.step(cam, target)
        mx.eval(model.params)
        dt = time.perf_counter() - t0
        all_step_times.append(dt)
        if i >= args.warmup:
            step_times.append(dt)
            losses.append(float(info["loss"]))
            if info["densify"] is not None:
                densify_events.append(
                    {
                        "step": int(info["step"]),
                        "wall_time_s": float(dt),
                        **info["densify"],
                    }
                )
        else:
            warmup_step_times.append(dt)

    t0 = time.perf_counter()
    final = model.render(cam0, background=bg)
    mx.eval(final["image"], final["alpha"])
    timings["final_render_s"] = time.perf_counter() - t0
    render_profile = _profile_render(model, cam0, bg)
    eval_stats = _eval_psnr(model, ds, white_bg, args.eval_views)
    if args.save_render:
        _save_render(args.save_render, final["image"])

    result = {
        "dataset": {
            "views": len(ds),
            "points": int(init_points.shape[0]),
            "resolution": [int(cam0.width), int(cam0.height)],
            "scene_extent": float(scene_extent),
            "image_cache": image_cache,
        },
        "config": {
            "steps": args.steps,
            "warmup": args.warmup,
            "seed": args.seed,
            "eval_views": args.eval_views,
            "low_mem": args.low_mem,
            "scale_init_max_ref": args.scale_init_max_ref,
            "scale_init_chunk_size": args.scale_init_chunk_size,
            "init_scale_max_frac": args.init_scale_max_frac,
            "scale_init_max_scale": scale_cap,
            "position_lr_final": args.position_lr_final,
            "position_lr_max_steps": args.position_lr_max_steps,
            "method": args.method,
            "densify_from": args.densify_from,
            "densify_until": config.densify_until,
            "densify_every": args.densify_every,
            "densify_grad_threshold": args.densify_grad_threshold,
            "mcmc_relocate_frac": args.mcmc_relocate_frac,
            "mcmc_min_opacity": args.mcmc_min_opacity,
            "mcmc_jitter_scale": args.mcmc_jitter_scale,
            "mcmc_noise_scale": args.mcmc_noise_scale,
            "two_d_thickness": args.two_d_thickness,
        },
        "timings": timings,
        "train": {
            "first_step_s": all_step_times[0] if all_step_times else None,
            "warmup_step_mean_s": (
                float(np.mean(warmup_step_times)) if warmup_step_times else None
            ),
            "step_mean_s": float(np.mean(step_times)) if step_times else None,
            "step_median_s": float(np.median(step_times)) if step_times else None,
            "step_p90_s": float(np.percentile(step_times, 90)) if step_times else None,
            "step_p95_s": float(np.percentile(step_times, 95)) if step_times else None,
            "step_max_s": float(np.max(step_times)) if step_times else None,
            "steps_per_s": float(1.0 / np.mean(step_times)) if step_times else None,
            "loss_first": losses[0] if losses else None,
            "loss_last": losses[-1] if losses else None,
            "lr_means": trainer.learning_rates()["means"],
            "gaussians": int(model.num_gaussians),
            "densify_events": densify_events,
        },
        "initial": initial_stats,
        "final": {
            "image": _image_stats(final["image"]),
            "alpha": _image_stats(final["alpha"]),
            "projection": _projection_stats(model, cam0),
        },
        "eval": eval_stats,
        "render_profile": render_profile,
        "memory": {
            "after_init": init_memory,
            "final": _memory(),
        },
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
