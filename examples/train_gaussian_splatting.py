"""Train 3D Gaussian Splatting on a COLMAP or Blender-synthetic scene.

COLMAP scene (the standard 3DGS input — a directory with sparse/0 and images):

    python examples/train_gaussian_splatting.py --data /path/to/scene --iters 7000

Blender-synthetic scene:

    python examples/train_gaussian_splatting.py --data /path/to/nerf_synthetic/lego \\
        --format blender --iters 7000

Saves a standard 3DGS PLY checkpoint (viewable in any splat viewer) and
periodic renders.
"""

import argparse
import os
import time

import mlx.core as mx
import numpy as np

from mlx3d.losses import psnr
from mlx3d.splatting import GaussianModel, GaussianTrainer, TrainerConfig


def _gb(nbytes: int | float) -> float:
    return float(nbytes) / float(1 << 30)


def _memory_summary() -> str:
    try:
        active = mx.get_active_memory()
        peak = mx.get_peak_memory()
        cache = mx.get_cache_memory()
    except AttributeError:
        return ""
    return f"mem active {_gb(active):.2f}G peak {_gb(peak):.2f}G cache {_gb(cache):.2f}G"


def _make_progress(total: int, disabled: bool):
    if disabled:
        return None
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return None
    return tqdm(total=total, unit="it", dynamic_ncols=True)


def _write_progress(pbar, message: str) -> None:
    if pbar is not None:
        pbar.write(message)
    else:
        print(message, flush=True)


def _format_train_info(it: int, total: int, info: dict[str, object], elapsed: float) -> str:
    rate = it / max(elapsed, 1e-9)
    mem = _memory_summary()
    msg = (
        f"iter {it:6d}/{total}  loss {float(info['loss']):.5f}  "
        f"gaussians {int(info['num_gaussians'])}  "
        f"sh {int(info['active_sh_degree'])}  {rate:.2f} it/s"
    )
    return f"{msg}  {mem}" if mem else msg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--format", choices=["colmap", "blender"], default="colmap")
    parser.add_argument("--iters", type=int, default=7000)
    parser.add_argument("--downscale", type=int, default=1)
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--init-points", type=int, default=30000,
                        help="random init size for blender scenes (no SfM points)")
    parser.add_argument("--out", type=str, default="outputs/gs")
    parser.add_argument("--low-mem", action="store_true",
                        help="low-memory mode for 8-16 GB machines: uint8 image "
                             "cache, capped Gaussian count, capped MLX buffer cache")
    parser.add_argument("--image-cache", choices=["ram", "uint8", "disk"], default=None,
                        help="image storage policy (default: ram, or uint8 with --low-mem)")
    parser.add_argument("--max-gaussians", type=int, default=None,
                        help="cap on the Gaussian count (default: 1.2M with --low-mem)")
    parser.add_argument("--scale-init-max-ref", type=int, default=None,
                        help="max reference points for initial scale KNN "
                             "(default: 10000)")
    parser.add_argument("--scale-init-chunk-size", type=int, default=None,
                        help="query chunk size for initial scale KNN "
                             "(default: 1024; fallback path only)")
    parser.add_argument("--cache-limit-gb", type=float, default=2.0,
                        help="MLX buffer-cache cap used with --low-mem")
    parser.add_argument("--log-every", type=int, default=10,
                        help="print fallback/log event frequency")
    parser.add_argument("--save-every", type=int, default=1000,
                        help="save render/checkpoint frequency; <=0 disables periodic saves")
    parser.add_argument("--no-progress", action="store_true",
                        help="disable tqdm progress bar even when tqdm is installed")
    args = parser.parse_args()

    image_cache = args.image_cache or ("uint8" if args.low_mem else "ram")
    max_gaussians = args.max_gaussians or (1_200_000 if args.low_mem else None)
    scale_init_max_ref = args.scale_init_max_ref or 10_000
    scale_init_chunk_size = args.scale_init_chunk_size or 1024
    log_every = max(1, args.log_every)

    print("Loading dataset...", flush=True)
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
        # Random init inside the synthetic scenes' bounding box.
        init_points = mx.random.uniform(low=-1.5, high=1.5, shape=(args.init_points, 3))
        init_colors = mx.random.uniform(shape=(args.init_points, 3))
        scene_extent = 3.0
        white_bg = True

    cam0, img0 = ds[0]
    if cam0.width != img0.shape[1] or cam0.height != img0.shape[0]:
        raise RuntimeError(
            "Camera/image shape mismatch after dataset load: "
            f"camera={cam0.width}x{cam0.height}, image={img0.shape[1]}x{img0.shape[0]}. "
            "Check the COLMAP camera metadata, image files, and --downscale."
        )
    resident_gb = getattr(ds.images, "nbytes_resident", 0) / float(1 << 30)
    print(f"{len(ds)} views, init with {init_points.shape[0]} points, "
          f"extent {scene_extent:.2f}", flush=True)
    print(
        f"Training resolution {cam0.width}x{cam0.height}, image cache {image_cache} "
        f"({resident_gb:.2f}G resident), dataset loaded in {time.perf_counter() - t0:.1f}s",
        flush=True,
    )

    print(
        "Initializing Gaussian model "
        f"(scale KNN ref={scale_init_max_ref}, chunk={scale_init_chunk_size})...",
        flush=True,
    )
    t0 = time.perf_counter()
    model = GaussianModel.from_points(
        init_points,
        init_colors,
        sh_degree=args.sh_degree,
        scale_init_max_ref=scale_init_max_ref,
        scale_init_chunk_size=scale_init_chunk_size,
    )
    mx.eval(model.params)
    if args.low_mem:
        mx.clear_cache()
    print(
        f"Initialized {model.num_gaussians} Gaussians in {time.perf_counter() - t0:.1f}s "
        f"{_memory_summary()}",
        flush=True,
    )
    config = TrainerConfig(
        white_background=white_bg,
        densify_until=args.iters // 2,
        max_gaussians=max_gaussians,
        low_memory=args.low_mem,
        cache_limit_gb=args.cache_limit_gb,
    )
    trainer = GaussianTrainer(model, config, scene_extent=scene_extent)

    os.makedirs(args.out, exist_ok=True)
    order = np.random.permutation(len(ds))
    cursor = 0
    pbar = _make_progress(args.iters, args.no_progress)
    if pbar is None:
        print(f"Training {args.iters} iterations (log every {log_every})...", flush=True)
    train_start = time.perf_counter()
    for it in range(1, args.iters + 1):
        if cursor >= len(ds):
            order = np.random.permutation(len(ds))
            cursor = 0
        cam, img = ds[int(order[cursor])]
        cursor += 1

        info = trainer.step(cam, img)
        elapsed = time.perf_counter() - train_start
        if pbar is not None:
            pbar.update(1)
            if it == 1 or it % log_every == 0:
                pbar.set_postfix(
                    loss=f"{float(info['loss']):.5f}",
                    N=int(info["num_gaussians"]),
                    sh=int(info["active_sh_degree"]),
                )
        elif it == 1 or it % log_every == 0 or it == args.iters:
            print(_format_train_info(it, args.iters, info, elapsed), flush=True)

        if info["densify"] is not None:
            stats = info["densify"]
            _write_progress(
                pbar,
                "densify "
                f"iter {it}: cloned {stats['cloned']}, split {stats['split']}, "
                f"pruned {stats['pruned']}, gaussians {info['num_gaussians']}",
            )
        if info["opacity_reset"]:
            _write_progress(pbar, f"opacity reset at iter {it}")
        if info["sh_degree_changed"]:
            _write_progress(pbar, f"active SH degree -> {info['active_sh_degree']} at iter {it}")

        should_save = args.save_every > 0 and (it % args.save_every == 0 or it == args.iters)
        if should_save:
            save_view(
                model,
                ds,
                white_bg,
                os.path.join(args.out, f"render_{it:06d}.png"),
                log=lambda msg: _write_progress(pbar, msg),
            )
            model.save_ply(os.path.join(args.out, "point_cloud.ply"))

    if pbar is not None:
        pbar.close()
    print(f"Done. Checkpoint: {args.out}/point_cloud.ply", flush=True)


def save_view(model, ds, white_bg, path, log=print):
    from PIL import Image

    cam, img = ds[0]
    bg = mx.ones((3,)) if white_bg else mx.zeros((3,))
    out = model.render(cam, background=bg)
    log(f"  eval view PSNR: {float(psnr(out['image'], img)):.2f} dB")
    arr = (np.clip(np.array(out["image"]), 0, 1) * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


if __name__ == "__main__":
    main()
