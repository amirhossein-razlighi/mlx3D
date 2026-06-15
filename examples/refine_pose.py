"""Recover a camera pose by optimizing an SE(3) twist (no data needed).

Demonstrates the differentiable-pose machinery behind pose-free NeRF / 3DGS
(BARF-style): given 2D-3D correspondences from a true camera, start from a
wrongly-posed camera and optimize a 6D twist through ``refine_camera`` until the
reprojection error vanishes. The same pattern lets you refine noisy COLMAP poses
jointly with a scene.

Usage:
    python examples/refine_pose.py [--iters 300]
"""

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.exists():
    sys.path.insert(0, str(_SRC))

import mlx.core as mx
import mlx.optimizers as optim

from mlx3d.cameras import Camera, refine_camera


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.05)
    args = parser.parse_args()

    mx.random.seed(0)
    true_cam = Camera.look_at(eye=(0.4, 0.3, -3.0), at=(0, 0, 0), fov=50.0, width=256, height=256)
    points = mx.random.normal((60, 3)) * 0.6
    target_xy, _ = true_cam.project_points(points)

    # A deliberately wrong starting pose.
    perturbed = refine_camera(true_cam, mx.array([0.15, -0.1, 0.2, 0.08, -0.05, 0.06]))

    def loss_fn(twist):
        xy, _ = refine_camera(perturbed, twist).project_points(points)
        return mx.mean((xy - target_xy) ** 2)

    twist = mx.zeros((6,))
    optimizer = optim.Adam(learning_rate=args.lr)
    loss_and_grad = mx.value_and_grad(loss_fn)
    for it in range(args.iters):
        loss, grad = loss_and_grad(twist)
        twist = optimizer.apply_gradients({"x": grad}, {"x": twist})["x"]
        mx.eval(twist)
        if it % 50 == 0 or it == args.iters - 1:
            print(f"iter {it:4d}  reprojection MSE {float(loss):.5f}")

    print(f"Recovered twist: {[round(float(v), 3) for v in twist]}")


if __name__ == "__main__":
    main()
