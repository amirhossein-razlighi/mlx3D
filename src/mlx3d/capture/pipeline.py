"""One-command capture pipeline: photos or video -> poses -> 3DGS -> PLY.

Orchestrates the pieces that already exist in mlx3d (frame ingestion, COLMAP
or built-in SfM, the Gaussian Splatting trainer, the live viewer, compaction)
into a single resumable flow with sensible defaults:

    from mlx3d.capture import run_capture
    run_capture("my_photos/", "captures/scene")

or, from the shell, ``mlx3d-capture my_photos/``. Stages write their outputs
under the capture directory and are skipped on re-runs when already complete:

    <out>/images/       extracted / linked input frames
    <out>/sparse/0      COLMAP-format poses + sparse points
    <out>/splat.ply     compacted splat (also point_cloud.ply, renders/)
    <out>/capture.json  stage summary
"""

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from .colmap_wrap import has_colmap, run_colmap
from .frames import extract_video_frames, is_video, list_images

__all__ = ["CaptureConfig", "QUALITY_PRESETS", "run_capture"]


@dataclass
class QualityPreset:
    iters: int
    train_max_dim: int
    num_frames: int
    sh_degree: int


QUALITY_PRESETS: dict[str, QualityPreset] = {
    "fast": QualityPreset(iters=3000, train_max_dim=960, num_frames=100, sh_degree=2),
    "balanced": QualityPreset(iters=7000, train_max_dim=1280, num_frames=150, sh_degree=3),
    "best": QualityPreset(iters=30_000, train_max_dim=1920, num_frames=250, sh_degree=3),
}


@dataclass
class CaptureConfig:
    quality: str = "balanced"
    """Preset for iterations / resolution / frame count: fast, balanced, best."""
    poses: str = "auto"
    """Pose source: ``auto`` (COLMAP if installed, else built-in SfM),
    ``colmap``, ``builtin``, or ``existing`` (reuse ``<out>/sparse/0``)."""
    refine_poses: str = "auto"
    """Joint pose refinement during training: ``auto`` (on for built-in SfM
    poses), ``on``, or ``off``."""
    iters: int | None = None
    num_frames: int | None = None
    train_max_dim: int | None = None
    sh_degree: int | None = None
    method: str = "vanilla"
    """Trainer strategy: vanilla, mcmc, or 2dgs."""
    viewer: bool = True
    viewer_port: int = 8090
    viewer_open_browser: bool = True
    keep_open: bool = False
    """Keep the live viewer alive after training finishes."""
    low_memory: bool = False
    image_cache: str = "uint8"
    save_every: int = 1000
    compact_min_opacity: float = 0.005
    seed: int = 0
    overwrite: bool = False
    """Re-run all stages even when outputs already exist."""

    def resolved(self) -> "CaptureConfig":
        if self.quality not in QUALITY_PRESETS:
            raise ValueError(f"quality must be one of {sorted(QUALITY_PRESETS)}")
        preset = QUALITY_PRESETS[self.quality]
        cfg = CaptureConfig(**asdict(self))
        cfg.iters = self.iters or preset.iters
        cfg.num_frames = self.num_frames or preset.num_frames
        cfg.train_max_dim = self.train_max_dim or preset.train_max_dim
        cfg.sh_degree = preset.sh_degree if self.sh_degree is None else self.sh_degree
        return cfg


@dataclass
class _Stage:
    name: str
    seconds: float = 0.0
    info: dict = field(default_factory=dict)


def _banner(log, text: str) -> None:
    log(f"\n=== {text} ===")


def _pose_twist_lr(step: int, total: int, lr0: float = 1e-4, lr1: float = 1e-6) -> float:
    frac = min(max(step / max(total, 1), 0.0), 1.0)
    return lr0 * (lr1 / lr0) ** frac


# ------------------------------------------------------------------- stages
def _stage_frames(input_path: str, out: str, cfg: CaptureConfig, log) -> tuple[str, dict]:
    """Return (images_dir, info). Video frames land in <out>/images."""
    input_path = os.path.abspath(os.path.expanduser(input_path))
    if os.path.isdir(input_path):
        images = list_images(input_path)
        if len(images) < 3:
            raise RuntimeError(
                f"{input_path} contains {len(images)} usable images; need at least 3. "
                f"Supported extensions: jpg/png/tiff/webp."
            )
        log(f"Using {len(images)} photos from {input_path}")
        return input_path, {"source": "photos", "count": len(images)}
    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)
    if not is_video(input_path):
        raise RuntimeError(
            f"{input_path} is neither a directory of photos nor a supported video file."
        )
    frames_dir = os.path.join(out, "images")
    existing = list_images(frames_dir) if os.path.isdir(frames_dir) else []
    if existing and not cfg.overwrite:
        log(f"Reusing {len(existing)} previously extracted frames in {frames_dir}")
        return frames_dir, {"source": "video", "count": len(existing), "reused": True}
    frames = extract_video_frames(input_path, frames_dir, num_frames=cfg.num_frames, log=log)
    return frames_dir, {"source": "video", "count": len(frames)}


def _stage_poses(images_dir: str, out: str, cfg: CaptureConfig, source: str, log) -> dict:
    sparse = os.path.join(out, "sparse", "0")
    complete = all(
        os.path.exists(os.path.join(sparse, f))
        for f in ("cameras.bin", "images.bin", "points3D.bin")
    )
    if cfg.poses == "existing":
        if not complete:
            raise RuntimeError(f"--poses existing, but no sparse model at {sparse}.")
        log(f"Using existing poses at {sparse}")
        return {"method": "existing"}
    if complete and not cfg.overwrite:
        log(f"Reusing poses at {sparse} (pass --overwrite to recompute)")
        return {"method": "existing"}

    method = cfg.poses
    if method == "auto":
        method = "colmap" if has_colmap() else "builtin"
        log(
            f"Pose method: {method} ({'COLMAP found' if method == 'colmap' else 'COLMAP not found'})"
        )
    if method == "colmap":
        if not has_colmap():
            raise RuntimeError(
                "--poses colmap, but the COLMAP binary is not on PATH. "
                "Install it (`brew install colmap`) or use --poses builtin."
            )
        n = len(list_images(images_dir))
        sequential = source == "video" and n > 60
        run_colmap(images_dir, out, sequential=sequential, log=log)
        return {"method": "colmap", "sequential": sequential}
    if method == "builtin":
        from .sfm import run_sfm

        result = run_sfm(list_images(images_dir), out, log=log)
        return {
            "method": "builtin",
            "registered": len(result.registered),
            "points": result.num_points,
            "mean_reproj_px": result.mean_reproj_px,
        }
    raise ValueError(f"Unknown pose method {cfg.poses!r}.")


def _auto_downscale(images_dir: str, max_dim: int) -> int:
    from PIL import Image

    first = list_images(images_dir)[0]
    with Image.open(first) as img:
        d = max(img.size)
    return max(1, math.ceil(d / max_dim))


def _stage_train(out: str, images_dir: str, cfg: CaptureConfig, pose_info: dict, log) -> dict:
    import mlx.core as mx

    from ..cameras import refine_camera
    from ..datasets import load_colmap, save_colmap
    from ..losses import psnr
    from ..splatting import GaussianModel, GaussianTrainer, TrainerConfig

    if cfg.seed >= 0:
        np.random.seed(cfg.seed)
        mx.random.seed(cfg.seed)

    downscale = _auto_downscale(images_dir, cfg.train_max_dim)
    ds = load_colmap(out, images_dir=images_dir, downscale=downscale, cache=cfg.image_cache)
    if len(ds) < 3:
        raise RuntimeError(f"Only {len(ds)} posed views available; need at least 3.")
    scene_extent = ds.scene_extent
    cam0, _ = ds[0]
    log(
        f"{len(ds)} posed views at {cam0.width}x{cam0.height} "
        f"(downscale {downscale}), {ds.points.shape[0]} SfM points, "
        f"extent {scene_extent:.2f}"
    )

    if ds.points.shape[0] < 100:
        raise RuntimeError(
            f"Sparse model has only {ds.points.shape[0]} points; too few to "
            "initialize splats. The pose stage likely failed on this capture."
        )

    refine = cfg.refine_poses == "on" or (
        cfg.refine_poses == "auto" and pose_info.get("method") == "builtin"
    )
    log(f"Pose refinement during training: {'on' if refine else 'off'}")

    model = GaussianModel.from_points(
        ds.points,
        ds.point_colors,
        sh_degree=cfg.sh_degree,
        scale_init_max_scale=0.01 * scene_extent,
    )
    mx.eval(model.params)
    trainer_cfg = TrainerConfig(
        method=cfg.method,
        densify_until=cfg.iters // 2,
        max_gaussians=1_200_000 if cfg.low_memory else None,
        low_memory=cfg.low_memory,
        lr_means_max_steps=max(cfg.iters, 1),
    )
    trainer = GaussianTrainer(model, trainer_cfg, scene_extent=scene_extent)
    log(f"Initialized {model.num_gaussians} Gaussians, training {cfg.iters} iterations")

    # Optional live viewer.
    live_viewer = None
    if cfg.viewer:
        import threading

        from ..viewer import view_live_gaussians

        centers = np.stack([np.array(c.camera_center) for c in ds.cameras])
        live_viewer = view_live_gaussians(
            model,
            serve=False,
            initial_radius=max(scene_extent, 1e-3),
            initial_target=tuple(float(v) for v in centers.mean(axis=0)),
        )

        def _serve():
            try:
                live_viewer.serve(
                    host="127.0.0.1",
                    port=cfg.viewer_port,
                    open_browser=cfg.viewer_open_browser,
                )
            except OSError as e:
                log(f"Live viewer failed to start: {e}")

        viewer_thread = threading.Thread(target=_serve, daemon=True)
        viewer_thread.start()
        log(f"Live viewer: http://127.0.0.1:{cfg.viewer_port}")

    # Per-view learnable pose twists (BARF-style) when refinement is on.
    import mlx.optimizers as optim

    twists = [mx.zeros((6,)) for _ in range(len(ds))] if refine else None
    twist_opt = optim.Adam(learning_rate=1e-4) if refine else None

    try:
        from tqdm.auto import tqdm

        pbar = tqdm(total=cfg.iters, unit="it", dynamic_ncols=True)
    except ImportError:
        pbar = None

    renders_dir = os.path.join(out, "renders")
    os.makedirs(renders_dir, exist_ok=True)
    order = np.random.permutation(len(ds))
    cursor = 0
    train_start = time.perf_counter()
    for it in range(1, cfg.iters + 1):
        if cursor >= len(ds):
            order = np.random.permutation(len(ds))
            cursor = 0
        view_id = int(order[cursor])
        cursor += 1
        cam, img = ds[view_id]

        if twists is not None:
            twist_opt.learning_rate = _pose_twist_lr(it, cfg.iters)
            info = trainer.step(cam, img, twist=twists[view_id])
            twists[view_id] = twist_opt.apply_gradients(
                {"t": info["twist_grad"]}, {"t": twists[view_id]}
            )["t"]
            mx.eval(twists[view_id])
        else:
            info = trainer.step(cam, img)

        if pbar is not None:
            pbar.update(1)
            if it == 1 or it % 10 == 0:
                pbar.set_postfix(
                    loss=f"{float(info['loss']):.4f}",
                    N=int(info["num_gaussians"]),
                    sh=int(info["active_sh_degree"]),
                )
        elif it == 1 or it % 100 == 0 or it == cfg.iters:
            rate = it / max(time.perf_counter() - train_start, 1e-9)
            log(
                f"iter {it:6d}/{cfg.iters}  loss {float(info['loss']):.5f}  "
                f"gaussians {int(info['num_gaussians'])}  {rate:.2f} it/s"
            )

        if live_viewer is not None and (it == 1 or it % 25 == 0 or it == cfg.iters):
            live_viewer.publish(model, step=it, loss=float(info["loss"]))

        if cfg.save_every > 0 and (it % cfg.save_every == 0 or it == cfg.iters):
            model.save_ply(os.path.join(out, "point_cloud.ply"))

    if pbar is not None:
        pbar.close()
    train_seconds = time.perf_counter() - train_start

    # Fold refined twists back into the cameras for eval and export.
    refined_cams = list(ds.cameras)
    if twists is not None:
        refined_cams = [refine_camera(c, tw) for c, tw in zip(ds.cameras, twists)]
        save_colmap(
            os.path.join(out, "refined"),
            refined_cams,
            ds.image_names,
            ds.points,
            ds.point_colors,
        )
        log(f"Refined poses written to {os.path.join(out, 'refined', 'sparse', '0')}")

    # Final eval renders on a few evenly spaced views.
    from PIL import Image

    bg = mx.zeros((3,))
    n_eval = min(3, len(ds))
    psnrs = []
    for k, view_id in enumerate(np.linspace(0, len(ds) - 1, n_eval, dtype=int)):
        cam, img = refined_cams[int(view_id)], ds.images[int(view_id)]
        outp = model.render(cam, background=bg)
        psnrs.append(float(psnr(outp["image"], img)))
        arr = (np.clip(np.array(outp["image"]), 0, 1) * 255).astype(np.uint8)
        Image.fromarray(arr).save(os.path.join(renders_dir, f"view_{int(view_id):04d}.png"))
    mean_psnr = float(np.mean(psnrs))
    log(f"Training PSNR (mean of {n_eval} views): {mean_psnr:.2f} dB")

    model.save_ply(os.path.join(out, "point_cloud.ply"))
    if live_viewer is not None:
        live_viewer.publish(model, step=cfg.iters)
        live_viewer.mark_done()

    return {
        "iters": cfg.iters,
        "downscale": downscale,
        "views": len(ds),
        "gaussians": model.num_gaussians,
        "psnr_mean": mean_psnr,
        "pose_refined": bool(refine),
        "seconds": train_seconds,
        "checkpoint": os.path.join(out, "point_cloud.ply"),
    }


def _stage_export(out: str, cfg: CaptureConfig, log) -> dict:
    from ..cli.compact import compact_checkpoint

    raw = os.path.join(out, "point_cloud.ply")
    final = os.path.join(out, "splat.ply")
    stats = compact_checkpoint(raw, final, min_opacity=cfg.compact_min_opacity)
    log(
        f"Compacted {stats['gaussians_before']} -> {stats['gaussians_after']} Gaussians "
        f"({os.path.getsize(final) / 1e6:.1f} MB): {final}"
    )
    return stats


# -------------------------------------------------------------------- driver
def run_capture(input_path: str, out: str, config: CaptureConfig | None = None, log=print):
    """Run the full photos/video -> splat pipeline. Returns a summary dict."""
    cfg = (config or CaptureConfig()).resolved()
    out = os.path.abspath(os.path.expanduser(out))
    os.makedirs(out, exist_ok=True)
    summary: dict[str, object] = {"input": os.path.abspath(input_path), "out": out}
    stages: list[_Stage] = []
    t_all = time.perf_counter()

    _banner(log, "1/4 Frames")
    t0 = time.perf_counter()
    images_dir, frames_info = _stage_frames(input_path, out, cfg, log)
    stages.append(_Stage("frames", time.perf_counter() - t0, frames_info))

    _banner(log, "2/4 Camera poses")
    t0 = time.perf_counter()
    pose_info = _stage_poses(images_dir, out, cfg, str(frames_info.get("source")), log)
    stages.append(_Stage("poses", time.perf_counter() - t0, pose_info))

    _banner(log, "3/4 Train 3D Gaussian Splatting")
    t0 = time.perf_counter()
    train_info = _stage_train(out, images_dir, cfg, pose_info, log)
    stages.append(_Stage("train", time.perf_counter() - t0, train_info))

    _banner(log, "4/4 Export")
    t0 = time.perf_counter()
    export_info = _stage_export(out, cfg, log)
    stages.append(_Stage("export", time.perf_counter() - t0, export_info))

    summary["stages"] = {s.name: {"seconds": round(s.seconds, 2), **s.info} for s in stages}
    summary["total_seconds"] = round(time.perf_counter() - t_all, 2)
    summary["splat"] = os.path.join(out, "splat.ply")
    with open(os.path.join(out, "capture.json"), "w") as f:
        json.dump(summary, f, indent=2)

    total_min = summary["total_seconds"] / 60.0
    _banner(log, "Done")
    log(f"Splat: {summary['splat']}  ({total_min:.1f} min total)")
    log(f"View it anytime:  mlx3d-view {summary['splat']}")

    if cfg.viewer and cfg.keep_open:
        log("Live viewer stays open. Press Ctrl-C to exit.")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
    return summary
