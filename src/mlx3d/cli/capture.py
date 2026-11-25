"""Turn photos or a video into a 3D Gaussian Splat, in one command.

    mlx3d-capture ./my_photos/
    mlx3d-capture walkaround.mp4 --quality fast

Runs frames -> camera poses (feed-forward Depth Anything 3 with the
"models" extra, otherwise COLMAP if installed, otherwise the built-in SfM)
-> 3DGS training with a live browser preview -> compacted splat.ply.
Stages are resumable: re-running skips work whose outputs already exist.
"""

from __future__ import annotations

import argparse
import os

from ..capture.pipeline import QUALITY_PRESETS, CaptureConfig, run_capture
from ..models.depth_anything3.config import PRESETS as DA3_PRESETS


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
        choices=["auto", "colmap", "builtin", "da3", "existing"],
        default="auto",
        help="pose source: auto uses Depth Anything 3 when mlx3d[models] is installed, "
        "else COLMAP when installed, else the built-in SfM (default: auto)",
    )
    parser.add_argument(
        "--pose-model",
        choices=sorted(DA3_PRESETS),
        default="da3-base",
        help="Depth Anything 3 preset for --poses da3 (default: da3-base)",
    )
    parser.add_argument(
        "--refine-poses",
        choices=["auto", "on", "off"],
        default="auto",
        help="jointly refine camera poses while training (auto: on for built-in SfM and DA3)",
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
    masks = parser.add_argument_group("foreground masks")
    source = masks.add_mutually_exclusive_group()
    source.add_argument(
        "--masks", default=None, help="foreground masks directory (<image name>.png, white = keep)"
    )
    source.add_argument(
        "--masks-from-alpha", action="store_true", help="use the input images' alpha as masks"
    )
    masks.add_argument(
        "--alpha-mode",
        choices=["masked", "transparent"],
        default="masked",
        help="masked: ignore the background; transparent: train a clean cut-out",
    )
    masks.add_argument(
        "--mask-sfm", action="store_true", help="also mask SfM feature extraction (COLMAP/builtin)"
    )
    parser.add_argument(
        "--appearance",
        action="store_true",
        help="learn per-image exposure / white-balance compensation",
    )
    parser.add_argument(
        "--depth-prior",
        action="store_true",
        help="depth + normal prior losses from the DA3 pose stage's depth maps",
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
        pose_model=args.pose_model,
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
        masks=args.masks,
        masks_from_alpha=args.masks_from_alpha,
        alpha_mode=args.alpha_mode,
        mask_sfm=args.mask_sfm,
        appearance=args.appearance,
        depth_prior=args.depth_prior,
        overwrite=args.overwrite,
        seed=args.seed,
    )
    run_capture(args.input, args.out or _default_out(args.input), config)


if __name__ == "__main__":
    main()
