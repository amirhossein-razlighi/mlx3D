"""Turn photos or a video into a 3D Gaussian Splat, in one command.

    mlx3d-capture ./my_photos/
    mlx3d-capture walkaround.mp4 --quality fast

Runs frames -> camera poses (COLMAP if installed, otherwise the built-in
SfM) -> 3DGS training with a live browser preview -> compacted splat.ply.
Stages are resumable: re-running skips work whose outputs already exist.
"""

from __future__ import annotations

import argparse
import os

from ..capture.pipeline import QUALITY_PRESETS, CaptureConfig, run_capture


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlx3d-capture",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="directory of photos, or a video file")
    parser.add_argument(
        "--out",
        default=None,
        help="capture directory (default: captures/<input name>)",
    )
    parser.add_argument(
        "--quality",
        choices=sorted(QUALITY_PRESETS),
        default="balanced",
        help="preset for iterations/resolution/frame count (default: balanced)",
    )
    parser.add_argument(
        "--poses",
        choices=["auto", "colmap", "builtin", "existing"],
        default="auto",
        help="pose source: auto uses COLMAP when installed, else the built-in SfM (default: auto)",
    )
    parser.add_argument(
        "--refine-poses",
        choices=["auto", "on", "off"],
        default="auto",
        help="jointly refine camera poses while training (auto: on for built-in SfM poses)",
    )
    parser.add_argument("--iters", type=int, default=None, help="override preset iterations")
    parser.add_argument(
        "--frames", type=int, default=None, help="frames to keep from a video (preset default)"
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=None,
        help="max training image dimension; larger inputs are downscaled",
    )
    parser.add_argument("--sh-degree", type=int, default=None)
    parser.add_argument(
        "--method",
        choices=["vanilla", "mcmc", "2dgs"],
        default="vanilla",
        help="training strategy (default: vanilla 3DGS)",
    )
    parser.add_argument("--no-viewer", action="store_true", help="disable the live browser viewer")
    parser.add_argument("--viewer-port", type=int, default=8090)
    parser.add_argument(
        "--no-browser", action="store_true", help="run the viewer without opening a browser"
    )
    parser.add_argument(
        "--keep-open",
        action="store_true",
        help="keep the live viewer running after training finishes",
    )
    parser.add_argument(
        "--low-mem", action="store_true", help="low-memory mode for 8-16 GB machines"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="recompute all stages, ignoring cached outputs"
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed; <0 disables seeding")
    return parser


def _default_out(input_path: str) -> str:
    base = os.path.basename(os.path.normpath(os.path.expanduser(input_path)))
    name = os.path.splitext(base)[0] or "scene"
    return os.path.join("captures", name)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = CaptureConfig(
        quality=args.quality,
        poses=args.poses,
        refine_poses=args.refine_poses,
        iters=args.iters,
        num_frames=args.frames,
        train_max_dim=args.max_dim,
        sh_degree=args.sh_degree,
        method=args.method,
        viewer=not args.no_viewer,
        viewer_port=args.viewer_port,
        viewer_open_browser=not args.no_browser,
        keep_open=args.keep_open,
        low_memory=args.low_mem,
        overwrite=args.overwrite,
        seed=args.seed,
    )
    run_capture(args.input, args.out or _default_out(args.input), config)


if __name__ == "__main__":
    main()
