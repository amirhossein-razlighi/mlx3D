"""Open the interactive viewer on a scene file, or a built-in demo mesh.

The viewer auto-detects the scene type (3DGS ``.ply`` checkpoint, mesh, or
point cloud) and serves a browser page rendered on the Apple GPU with a live
control panel, display-mode switching, an orbit gizmo, and PNG export.

Usage:
    python examples/view_scene.py path/to/scene.ply      # any .ply/.obj/.gltf/.glb
    python examples/view_scene.py                        # demo: a procedural torus
    python examples/view_scene.py --kind points cloud.ply
"""

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.exists():
    sys.path.insert(0, str(_SRC))

import mlx.core as mx

from mlx3d.utils.primitives import torus
from mlx3d.viewer import view_file, view_mesh


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default=None, help="scene file to view")
    parser.add_argument("--kind", choices=["gaussians", "mesh", "points"], default=None)
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()

    if args.path is None:
        # No file given: show a procedural mesh so the viewer is explorable
        # without any assets on disk. Try every display mode and the controls.
        mesh = torus(r=0.35, R=1.0, sides=32, rings=64)
        verts = mesh.verts_packed()
        colors = 0.5 * mesh.verts_normals_packed() + 0.5  # normal-colored albedo
        mx.eval(verts, colors)
        print("No file given — showing a demo torus. Pass a path to view your own scene.")
        view_mesh(mesh, verts_colors=colors, port=args.port)
    else:
        view_file(args.path, kind=args.kind, port=args.port)


if __name__ == "__main__":
    main()
