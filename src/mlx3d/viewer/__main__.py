"""Command-line entry point: view any MLX3D scene file.

The scene type is auto-detected from the file:

    mlx3d-view point_cloud.ply        # 3DGS checkpoint, mesh, or point cloud
    mlx3d-view bunny.obj              # mesh (OBJ / PLY / glTF / GLB)
    mlx3d-view scene.ply --kind points --background 1 1 1
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="mlx3d-view",
        description="Interactive viewer for 3DGS checkpoints, meshes, and point clouds "
        "(Metal-accelerated).",
    )
    parser.add_argument("path", type=str, help="path to a .ply / .obj / .gltf / .glb scene")
    parser.add_argument(
        "--kind",
        choices=["gaussians", "mesh", "points"],
        default=None,
        help="override the auto-detected scene type",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument(
        "--background",
        type=float,
        nargs=3,
        default=None,
        metavar=("R", "G", "B"),
        help="background color in [0, 1]",
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="don't open the browser automatically"
    )
    args = parser.parse_args()

    from .server import view_file

    print(f"Loading {args.path} ...")
    view_file(
        args.path,
        kind=args.kind,
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
        background=tuple(args.background) if args.background is not None else None,
    )


if __name__ == "__main__":
    main()
