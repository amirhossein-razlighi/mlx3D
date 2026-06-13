"""Extract a mesh from an implicit field with marching cubes (no data needed).

Samples a smooth-union signed-distance field of two spheres on a grid, runs
:func:`mlx3d.ops.marching_cubes` to recover a triangle mesh, saves it as OBJ,
and renders it so you can see what came out.

Usage:
    python examples/extract_mesh.py [--res 96] [--out outputs/extracted.obj]
"""

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.exists():
    sys.path.insert(0, str(_SRC))

import mlx.core as mx

from mlx3d.cameras import Camera
from mlx3d.io import save_image, save_obj
from mlx3d.ops import marching_cubes
from mlx3d.renderer import render_mesh_soft


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--res", type=int, default=48)
    parser.add_argument("--out", type=str, default="outputs/extracted_mesh.obj")
    parser.add_argument("--render", type=str, default="outputs/extracted_mesh.png")
    args = parser.parse_args()

    # Sample an SDF on a [-1.5, 1.5]^3 grid: smooth union of two spheres.
    n = args.res
    lin = mx.linspace(-1.5, 1.5, n)
    z, y, x = mx.meshgrid(lin, lin, lin, indexing="ij")
    p = mx.stack([x, y, z], axis=-1)
    d1 = mx.linalg.norm(p - mx.array([-0.45, 0.0, 0.0]), axis=-1) - 0.7
    d2 = mx.linalg.norm(p - mx.array([0.45, 0.0, 0.0]), axis=-1) - 0.7
    k = 0.3
    sdf = -k * mx.log(mx.exp(-d1 / k) + mx.exp(-d2 / k))  # smooth min

    spacing = 3.0 / (n - 1)
    mesh = marching_cubes(
        sdf, level=0.0, spacing=(spacing, spacing, spacing), origin=(-1.5, -1.5, -1.5)
    )
    print(f"Extracted {mesh.faces_packed().shape[0]} faces, {mesh.verts_packed().shape[0]} verts")

    save_obj(args.out, mesh.verts_packed(), mesh.faces_packed())
    print(f"Saved {args.out}")

    camera = Camera.look_at(
        eye=(2.6, 1.8, 2.6), at=(0, 0, 0), up=(0, 1, 0), fov=45.0, width=160, height=160
    )
    normals = mesh.verts_normals_packed()
    verts_colors = 0.5 * normals + 0.5
    out = render_mesh_soft(
        camera, mesh, verts_colors=verts_colors, sigma=3e-3, background=(0.05, 0.05, 0.08)
    )
    mx.eval(out["image"])
    save_image(args.render, out["image"])
    print(f"Saved {args.render}")


if __name__ == "__main__":
    main()
