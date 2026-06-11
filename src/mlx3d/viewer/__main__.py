"""Command-line entry point: view a Gaussian Splatting checkpoint.

    python -m mlx3d.viewer point_cloud.ply
    mlx3d-view point_cloud.ply --port 8090 --background 1 1 1
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="mlx3d-view",
        description="Interactive viewer for 3DGS .ply checkpoints (Metal-accelerated).",
    )
    parser.add_argument("ply", type=str, help="path to a 3DGS-format .ply checkpoint")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--background", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                        metavar=("R", "G", "B"), help="background color in [0, 1]")
    parser.add_argument("--no-browser", action="store_true",
                        help="don't open the browser automatically")
    args = parser.parse_args()

    from ..splatting import GaussianModel
    from .server import view_gaussians

    print(f"Loading {args.ply} ...")
    model = GaussianModel.load_ply(args.ply)
    print(f"{model.num_gaussians:,} gaussians, SH degree {model.sh_degree}")
    view_gaussians(
        model,
        background=tuple(args.background),
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
    )


if __name__ == "__main__":
    main()
