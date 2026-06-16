"""Render a mesh with the hard rasterizer and Blinn-Phong lighting (no data).

The hard z-buffer rasterizer (``rasterize_meshes``, backed by a Metal kernel)
resolves visibility, then ``render_mesh`` shades it with point/directional/
ambient lights. Much faster than the soft rasterizer at high resolution; use
the soft one (``render_mesh_soft``) when you need silhouette gradients.

Usage:
    python examples/render_mesh_lit.py [--out outputs/lit.png] [--size 512]
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
from mlx3d.renderer import AmbientLights, PointLights, render_mesh
from mlx3d.utils import ico_sphere


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="outputs/lit_mesh.png")
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()

    mesh = ico_sphere(level=5, radius=1.0)
    n = mesh.verts_packed().shape[0]
    camera = Camera.look_at(
        eye=(2.4, 1.8, 2.4),
        at=(0, 0, 0),
        up=(0, 1, 0),
        fov=45.0,
        width=args.size,
        height=args.size,
    )
    lights = [
        PointLights(location=(3.0, 3.0, -2.0), color=(1.0, 0.95, 0.9)),
        AmbientLights(color=(0.15, 0.16, 0.2)),
    ]
    out = render_mesh(
        camera,
        mesh,
        verts_colors=mx.full((n, 3), mx.array([0.85, 0.3, 0.25])),
        lights=lights,
        shininess=48.0,
        specular_strength=0.4,
        background=(0.05, 0.06, 0.09),
    )
    mx.eval(out["image"])
    save_image(args.out, out["image"])
    # Also dump a normal-map visualization next to it.
    normals_vis = (out["normals"] * 0.5 + 0.5) * out["alpha"][..., None]
    save_image(args.out.replace(".png", "_normals.png"), normals_vis)
    print(f"Saved {args.out} (alpha {float(out['alpha'].mean()):.3f})")


if __name__ == "__main__":
    main()
