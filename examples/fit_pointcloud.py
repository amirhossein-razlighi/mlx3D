"""Fit a noisy point cloud to a target shape with chamfer distance.

Usage:
    python examples/fit_pointcloud.py [--iters 300]
"""

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.exists():
    sys.path.insert(0, str(_SRC))

import mlx.core as mx
import mlx.optimizers as optim

from mlx3d.io import save_ply
from mlx3d.losses import chamfer_distance
from mlx3d.ops import sample_points_from_meshes
from mlx3d.utils import torus


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--points", type=int, default=5000)
    parser.add_argument("--out", type=str, default="outputs/fitted_points.ply")
    args = parser.parse_args()

    target = sample_points_from_meshes(torus(r=0.4, R=1.0), args.points)[0]
    points = mx.random.normal((args.points, 3)) * 0.5  # random init

    optimizer = optim.Adam(learning_rate=2e-2)

    def loss_fn(points):
        loss, _ = chamfer_distance(points, target)
        return loss

    state = {"points": points}
    for it in range(args.iters):
        loss, grads = mx.value_and_grad(loss_fn)(state["points"])
        state = optimizer.apply_gradients({"points": grads}, state)
        mx.eval(state["points"])
        if it % 50 == 0 or it == args.iters - 1:
            print(f"iter {it:4d}  chamfer {float(loss):.6f}")

    save_ply(args.out, state["points"])
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
