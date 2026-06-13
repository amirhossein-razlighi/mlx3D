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
import sys
import threading
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.exists():
    sys.path.insert(0, str(_SRC))

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
        f"sh {int(info['active_sh_degree'])}  "
        f"lr_xyz {float(info['lr_means']):.2e}  {rate:.2f} it/s"
    )
    return f"{msg}  {mem}" if mem else msg


def _format_density_stats(stats: dict[str, object]) -> str:
    return ", ".join(f"{k} {v}" for k, v in stats.items() if not str(k).startswith("_"))


def _viewer_initial_frame(ds, scene_extent: float) -> tuple[float, tuple[float, float, float]]:
    cameras = getattr(ds, "cameras", None)
    if not cameras:
        return scene_extent, (0.0, 0.0, 0.0)
    centers = np.stack([np.array(c.camera_center) for c in cameras])
    target = tuple(float(v) for v in centers.mean(axis=0))
    return max(scene_extent, 1e-3), target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--format", choices=["colmap", "blender"], default="colmap")
    parser.add_argument("--iters", type=int, default=7000)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed for reproducible init/order; <0 disables seeding",
    )
    parser.add_argument("--downscale", type=int, default=1)
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument(
        "--init-points",
        type=int,
        default=30000,
        help="random init size for blender scenes (no SfM points)",
    )
    parser.add_argument("--out", type=str, default="outputs/gs")
    parser.add_argument(
        "--low-mem",
        action="store_true",
        help="low-memory mode for 8-16 GB machines: uint8 image "
        "cache, capped Gaussian count, capped MLX buffer cache",
    )
    parser.add_argument(
        "--image-cache",
        choices=["ram", "uint8", "disk"],
        default=None,
        help="image storage policy (default: ram, or uint8 with --low-mem)",
    )
    parser.add_argument(
        "--max-gaussians",
        type=int,
        default=None,
        help="cap on the Gaussian count (default: 1.2M with --low-mem)",
    )
    parser.add_argument(
        "--scale-init-max-ref",
        type=int,
        default=None,
        help="max reference points for initial scale KNN (default: 10000)",
    )
    parser.add_argument(
        "--scale-init-chunk-size",
        type=int,
        default=None,
        help="query chunk size for initial scale KNN (default: 1024; fallback path only)",
    )
    parser.add_argument(
        "--init-scale-max-frac",
        type=float,
        default=0.01,
        help="cap initial Gaussian scale as this fraction of scene extent; <=0 disables the cap",
    )
    parser.add_argument(
        "--position-lr-final",
        type=float,
        default=1.6e-6,
        help="final unscaled xyz learning rate for exponential decay",
    )
    parser.add_argument(
        "--position-lr-max-steps",
        type=int,
        default=30_000,
        help="steps over which xyz learning rate decays",
    )
    parser.add_argument(
        "--method",
        choices=["vanilla", "mcmc", "2dgs"],
        default="vanilla",
        help="training strategy; vanilla 3DGS is the default",
    )
    parser.add_argument(
        "--densify-from",
        type=int,
        default=500,
        help="first iteration that accumulates densification stats",
    )
    parser.add_argument(
        "--densify-until",
        type=int,
        default=None,
        help="last iteration for densification (default: iters // 2)",
    )
    parser.add_argument(
        "--densify-every", type=int, default=100, help="run clone/split/prune every N iterations"
    )
    parser.add_argument(
        "--densify-grad-threshold",
        type=float,
        default=0.0002,
        help="screen-space gradient threshold for clone/split",
    )
    parser.add_argument(
        "--mcmc-relocate-frac",
        type=float,
        default=0.02,
        help="MCMC mode: max fraction of Gaussians relocated per density event",
    )
    parser.add_argument(
        "--mcmc-min-opacity",
        type=float,
        default=0.01,
        help="MCMC mode: opacity threshold for relocation targets",
    )
    parser.add_argument(
        "--mcmc-jitter-scale",
        type=float,
        default=0.25,
        help="MCMC mode: relocation jitter as a multiple of source scale",
    )
    parser.add_argument(
        "--mcmc-noise-scale",
        type=float,
        default=0.01,
        help="MCMC mode: per-step SGLD-like xyz noise scale",
    )
    parser.add_argument(
        "--2d-thickness",
        dest="two_d_thickness",
        type=float,
        default=1e-4,
        help="2DGS mode: local-normal thickness as a fraction of scene extent",
    )
    parser.add_argument(
        "--cache-limit-gb", type=float, default=2.0, help="MLX buffer-cache cap used with --low-mem"
    )
    parser.add_argument(
        "--log-every", type=int, default=10, help="print fallback/log event frequency"
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1000,
        help="save render/checkpoint frequency; <=0 disables periodic saves",
    )
    parser.add_argument(
        "--eval-views",
        type=int,
        default=1,
        help="number of deterministic training views to average for save-time PSNR",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="disable tqdm progress bar even when tqdm is installed",
    )
    parser.add_argument(
        "--viewer", action="store_true", help="start a live browser viewer while training"
    )
    parser.add_argument("--viewer-host", type=str, default="127.0.0.1")
    parser.add_argument("--viewer-port", type=int, default=8090)
    parser.add_argument(
        "--viewer-no-browser",
        action="store_true",
        help="start the live viewer without opening a browser",
    )
    parser.add_argument(
        "--viewer-update-every",
        type=int,
        default=25,
        help="publish a fresh live-viewer snapshot every N steps",
    )
    parser.add_argument(
        "--viewer-max-scale",
        type=float,
        default=0.5,
        help="max browser render scale for live preview",
    )
    parser.add_argument(
        "--viewer-poll-ms",
        type=int,
        default=750,
        help="browser metadata polling interval for live preview",
    )
    parser.add_argument(
        "--viewer-keep-open",
        action="store_true",
        help="keep the process alive after training so the viewer stays open",
    )
    args = parser.parse_args()
    if args.seed >= 0:
        np.random.seed(args.seed)
        mx.random.seed(args.seed)

    image_cache = args.image_cache or ("uint8" if args.low_mem else "ram")
    max_gaussians = args.max_gaussians or (1_200_000 if args.low_mem else None)
    scale_init_max_ref = args.scale_init_max_ref or 10_000
    scale_init_chunk_size = args.scale_init_chunk_size or 1024
    log_every = max(1, args.log_every)
    viewer_update_every = max(1, args.viewer_update_every)

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

    scale_init_max_scale = (
        None if args.init_scale_max_frac <= 0 else args.init_scale_max_frac * scene_extent
    )

    cam0, img0 = ds[0]
    if cam0.width != img0.shape[1] or cam0.height != img0.shape[0]:
        raise RuntimeError(
            "Camera/image shape mismatch after dataset load: "
            f"camera={cam0.width}x{cam0.height}, image={img0.shape[1]}x{img0.shape[0]}. "
            "Check the COLMAP camera metadata, image files, and --downscale."
        )
    resident_gb = getattr(ds.images, "nbytes_resident", 0) / float(1 << 30)
    print(
        f"{len(ds)} views, init with {init_points.shape[0]} points, extent {scene_extent:.2f}",
        flush=True,
    )
    print(
        f"Training resolution {cam0.width}x{cam0.height}, image cache {image_cache} "
        f"({resident_gb:.2f}G resident), dataset loaded in {time.perf_counter() - t0:.1f}s",
        flush=True,
    )

    print(
        "Initializing Gaussian model "
        f"(scale KNN ref={scale_init_max_ref}, chunk={scale_init_chunk_size}, "
        f"max_scale={scale_init_max_scale})...",
        flush=True,
    )
    t0 = time.perf_counter()
    model = GaussianModel.from_points(
        init_points,
        init_colors,
        sh_degree=args.sh_degree,
        scale_init_max_ref=scale_init_max_ref,
        scale_init_chunk_size=scale_init_chunk_size,
        scale_init_max_scale=scale_init_max_scale,
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
        method=args.method,
        white_background=white_bg,
        densify_from=args.densify_from,
        densify_until=args.densify_until if args.densify_until is not None else args.iters // 2,
        densify_every=args.densify_every,
        densify_grad_threshold=args.densify_grad_threshold,
        mcmc_relocate_frac=args.mcmc_relocate_frac,
        mcmc_min_opacity=args.mcmc_min_opacity,
        mcmc_jitter_scale=args.mcmc_jitter_scale,
        mcmc_noise_scale=args.mcmc_noise_scale,
        two_d_thickness=args.two_d_thickness,
        max_gaussians=max_gaussians,
        low_memory=args.low_mem,
        cache_limit_gb=args.cache_limit_gb,
        lr_means_final=args.position_lr_final,
        lr_means_max_steps=args.position_lr_max_steps,
    )
    trainer = GaussianTrainer(model, config, scene_extent=scene_extent)
    live_viewer = None
    viewer_thread = None
    if args.viewer:
        from mlx3d.viewer import view_live_gaussians

        bg = (1.0, 1.0, 1.0) if white_bg else (0.0, 0.0, 0.0)
        viewer_radius, viewer_target = _viewer_initial_frame(ds, scene_extent)
        live_viewer = view_live_gaussians(
            model,
            background=bg,
            serve=False,
            max_scale=args.viewer_max_scale,
            poll_ms=args.viewer_poll_ms,
            initial_radius=viewer_radius,
            initial_target=viewer_target,
        )

        def serve_viewer() -> None:
            try:
                live_viewer.serve(
                    host=args.viewer_host,
                    port=args.viewer_port,
                    open_browser=not args.viewer_no_browser,
                )
            except OSError as e:
                print(f"Live viewer failed to start: {e}", flush=True)

        viewer_thread = threading.Thread(target=serve_viewer, daemon=True)
        viewer_thread.start()
        print(
            f"Live viewer: http://{args.viewer_host}:{args.viewer_port} "
            f"(snapshot every {viewer_update_every} iters, max scale {args.viewer_max_scale:.2f})",
            flush=True,
        )

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
                    lr=f"{float(info['lr_means']):.1e}",
                )
        elif it == 1 or it % log_every == 0 or it == args.iters:
            print(_format_train_info(it, args.iters, info, elapsed), flush=True)

        if info["densify"] is not None:
            stats = info["densify"]
            _write_progress(
                pbar,
                f"density iter {it}: {_format_density_stats(stats)}, "
                f"gaussians {info['num_gaussians']}",
            )
        if info["opacity_reset"]:
            _write_progress(pbar, f"opacity reset at iter {it}")
        if info["sh_degree_changed"]:
            _write_progress(pbar, f"active SH degree -> {info['active_sh_degree']} at iter {it}")

        if live_viewer is not None and (
            it == 1
            or it % viewer_update_every == 0
            or info["densify"] is not None
            or info["sh_degree_changed"]
            or it == args.iters
        ):
            live_viewer.publish(
                model,
                step=it,
                loss=float(info["loss"]),
                lr_means=float(info["lr_means"]),
            )

        should_save = args.save_every > 0 and (it % args.save_every == 0 or it == args.iters)
        if should_save:
            save_view(
                model,
                ds,
                white_bg,
                os.path.join(args.out, f"render_{it:06d}.png"),
                eval_views=args.eval_views,
                log=lambda msg: _write_progress(pbar, msg),
            )
            model.save_ply(os.path.join(args.out, "point_cloud.ply"))

    if pbar is not None:
        pbar.close()
    if live_viewer is not None:
        live_viewer.mark_done()
    if args.save_every > 0:
        print(f"Done. Checkpoint: {args.out}/point_cloud.ply", flush=True)
    else:
        print("Done. Checkpoint saving disabled (--save-every <= 0).", flush=True)
    if viewer_thread is not None and args.viewer_keep_open:
        print("Training done; live viewer remains open. Press Ctrl-C to stop.", flush=True)
        try:
            while viewer_thread.is_alive():
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass


def save_view(model, ds, white_bg, path, eval_views: int = 1, log=print):
    from PIL import Image

    bg = mx.ones((3,)) if white_bg else mx.zeros((3,))
    n_eval = max(1, min(int(eval_views), len(ds)))
    view_ids = np.linspace(0, len(ds) - 1, n_eval, dtype=np.int32)
    psnrs = []
    first = None
    for i, view_id in enumerate(view_ids):
        cam, img = ds[int(view_id)]
        out = model.render(cam, background=bg)
        psnrs.append(float(psnr(out["image"], img)))
        if i == 0:
            first = out["image"]
    mean_psnr = float(np.mean(psnrs))
    msg = f"  eval PSNR mean({n_eval} views): {mean_psnr:.2f} dB"
    if n_eval > 1:
        msg += f"  first {psnrs[0]:.2f}  last {psnrs[-1]:.2f}"
    log(msg)
    arr = (np.clip(np.array(first), 0, 1) * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


if __name__ == "__main__":
    main()
