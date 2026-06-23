"""Build a shape with SDF constructive solid geometry, then mesh and render it.

Composes analytic signed-distance primitives with smooth CSG operators
(:mod:`mlx3d.ops` ``sdf_*`` helpers), extracts a triangle mesh from the field
with :func:`mlx3d.ops.sdf_to_mesh` (marching cubes under the hood), and renders
it with the fast O(H*W) hard rasterizer.

Usage:
    python examples/sdf_csg.py [--res 96] [--blend 0.15] [--size 256]
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
from mlx3d.ops import (
    sdf_box,
    sdf_smooth_difference,
    sdf_smooth_union,
    sdf_sphere,
    sdf_to_mesh,
    sdf_torus,
)
from mlx3d.renderer import interpolate_face_attributes, rasterize_meshes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--res", type=int, default=96, help="marching-cubes grid resolution")
    parser.add_argument("--blend", type=float, default=0.15, help="CSG smooth-blend radius")
    parser.add_argument("--size", type=int, default=256, help="render resolution")
    parser.add_argument("--out", type=str, default="outputs/sdf_csg.obj")
    parser.add_argument("--render", type=str, default="outputs/sdf_csg.png")
    args = parser.parse_args()

    k = args.blend

    def model(p: mx.array) -> mx.array:
        # A rounded cube with a spherical bite taken out, fused with a ring.
        cube = sdf_box(p, half_extents=(0.6, 0.6, 0.6))
        bite = sdf_sphere(p, radius=0.78, center=(0.55, 0.55, 0.55))
        carved = sdf_smooth_difference(cube, bite, k)
        ring = sdf_torus(p, major_radius=0.95, minor_radius=0.18)
        return sdf_smooth_union(carved, ring, k)

    mesh = sdf_to_mesh(model, resolution=args.res, bounds=1.4, level=0.0)
    n_faces = mesh.faces_packed().shape[0]
    n_verts = mesh.verts_packed().shape[0]
    print(f"Extracted {n_faces} faces, {n_verts} verts")

    save_obj(args.out, mesh.verts_packed(), mesh.faces_packed())
    print(f"Saved {args.out}")

    camera = Camera.look_at(
        eye=(2.4, 1.8, 2.4),
        at=(0, 0, 0),
        up=(0, 1, 0),
        fov=45.0,
        width=args.size,
        height=args.size,
    )
    verts_colors = 0.5 * mesh.verts_normals_packed() + 0.5
    frag = rasterize_meshes(camera, mesh)
    color = interpolate_face_attributes(frag, verts_colors)
    background = mx.array([0.05, 0.05, 0.08])
    image = mx.where(frag.valid[..., None], color, background)
    mx.eval(image)
    save_image(args.render, image)
    print(f"Saved {args.render}")


if __name__ == "__main__":
    main()
