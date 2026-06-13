"""Rasterize a mesh with the differentiable soft renderer (no data needed).

Builds an icosphere, colors its vertices by position, and renders it from a
look-at camera with :func:`mlx3d.renderer.render_mesh_soft`. The same call is
differentiable w.r.t. vertices, colors, and textures.

Usage:
    python examples/render_mesh.py [--out outputs/mesh.png] [--size 256]
"""

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.exists():
    sys.path.insert(0, str(_SRC))

import mlx.core as mx

from mlx3d.cameras import Camera
from mlx3d.io import save_image
from mlx3d.renderer import render_mesh_soft
from mlx3d.utils import ico_sphere


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="outputs/mesh_render.png")
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()

    mesh = ico_sphere(level=3, radius=1.0)
    verts = mesh.verts_packed()
    # Per-vertex color from position: maps the unit sphere to a smooth RGB field.
    verts_colors = (
        0.5 * (verts / mx.maximum(mx.linalg.norm(verts, axis=-1, keepdims=True), 1e-6)) + 0.5
    )

    camera = Camera.look_at(
        eye=(2.2, 1.6, 2.2),
        at=(0, 0, 0),
        up=(0, 1, 0),
        fov=45.0,
        width=args.size,
        height=args.size,
    )

    out = render_mesh_soft(
        camera, mesh, verts_colors=verts_colors, sigma=3e-3, background=(0.05, 0.05, 0.08)
    )
    mx.eval(out["image"])
    save_image(args.out, out["image"])
    print(f"Saved {args.out}  (alpha coverage: {float(out['alpha'].mean()):.3f})")


if __name__ == "__main__":
    main()
