"""Deform an icosphere to fit a target mesh.

Optimizes per-vertex offsets with a chamfer loss on sampled surface points
plus the standard mesh regularizers (edge length, Laplacian, normal
consistency) — the classic PyTorch3D "fit mesh" workflow, on MLX.

Usage:
    python examples/fit_mesh.py [--target path/to/mesh.obj] [--iters 500]
"""

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.exists():
    sys.path.insert(0, str(_SRC))

import mlx.core as mx
import mlx.optimizers as optim

from mlx3d.io import load_obj, save_obj
from mlx3d.losses import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)
from mlx3d.ops import sample_points_from_meshes
from mlx3d.structures import Meshes
from mlx3d.utils import ico_sphere, torus


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default=None, help="OBJ file (default: torus)")
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--samples", type=int, default=3000)
    parser.add_argument("--out", type=str, default="outputs/fitted_mesh.obj")
    args = parser.parse_args()

    if args.target:
        data = load_obj(args.target)
        verts = data.verts
        # Normalize to the unit sphere.
        verts = verts - verts.mean(axis=0)
        verts = verts / mx.linalg.norm(verts, axis=-1).max()
        target = Meshes([verts], [data.faces])
    else:
        target = torus(r=0.4, R=1.0, sides=24, rings=48)

    src = ico_sphere(level=3)
    faces = src.faces_list()
    verts0 = src.verts_packed()
    offsets = mx.zeros(verts0.shape)

    optimizer = optim.Adam(learning_rate=1e-2)

    def loss_fn(offsets):
        mesh = Meshes([verts0 + offsets], faces)
        pts_src = sample_points_from_meshes(mesh, args.samples)
        pts_tgt = sample_points_from_meshes(target, args.samples)
        cham, _ = chamfer_distance(pts_src, pts_tgt)
        return (
            cham
            + 0.1 * mesh_edge_loss(mesh)
            + 0.1 * mesh_laplacian_smoothing(mesh)
            + 0.01 * mesh_normal_consistency(mesh)
        )

    state = {"offsets": offsets}
    for it in range(args.iters):
        loss, grads = mx.value_and_grad(loss_fn)(state["offsets"])
        state = optimizer.apply_gradients({"offsets": grads}, state)
        mx.eval(state["offsets"])
        if it % 50 == 0 or it == args.iters - 1:
            print(f"iter {it:4d}  loss {float(loss):.5f}")

    fitted = Meshes([verts0 + state["offsets"]], faces)
    save_obj(args.out, fitted.verts_packed(), fitted.faces_packed())
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
