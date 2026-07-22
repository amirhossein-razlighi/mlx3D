"""Validate the Metal Gaussian rasterizer's analytic gradients numerically.

The 3DGS backward pass is a hand-written Metal kernel, not autodiff. This script
checks it against central finite differences and prints a report, so you can
trust the training gradients (and catch regressions if you touch the kernel).

For each parameter it reports the symmetric relative error between the analytic
directional derivative ``grad . v`` and the finite-difference estimate
``(L(x + eps v) - L(x - eps v)) / (2 eps)`` along random directions ``v``.

    python examples/validate_gaussian_gradients.py
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.exists():
    sys.path.insert(0, str(_SRC))

import mlx.core as mx
import numpy as np

from mlx3d.cameras import Camera
from mlx3d.splatting import render_gaussians


def loss_fn(cam, target, bg, sh_degree=0):
    def loss(means, quats, log_scales, logit_op, sh):
        out = render_gaussians(
            cam,
            means,
            quats,
            mx.exp(log_scales),
            mx.sigmoid(logit_op),
            sh=sh,
            sh_degree=sh_degree,
            background=bg,
        )
        return mx.mean((out["image"] - target) ** 2)

    return loss


def directional_report(loss, args, names, indices=None, eps=2e-3, n_dirs=6):
    _, grads = mx.value_and_grad(loss, argnums=tuple(range(len(args))))(*args)
    mx.eval(grads)
    indices = range(len(args)) if indices is None else indices
    print(
        f"  {'param':12s} {'analytic':>13s} {'numeric':>13s} {'median rel':>12s} {'max rel':>10s}"
    )
    worst = 0.0
    for i in indices:
        name = names[i]
        errs, a_last, n_last = [], 0.0, 0.0
        for s in range(n_dirs):
            mx.random.seed(500 + s)
            v = mx.random.normal(args[i].shape)
            a = float(mx.sum(grads[i] * v).item())
            plus = list(args)
            plus[i] = args[i] + eps * v
            minus = list(args)
            minus[i] = args[i] - eps * v
            num = (float(loss(*plus).item()) - float(loss(*minus).item())) / (2 * eps)
            errs.append(abs(a - num) / (abs(a) + abs(num) + 1e-8))
            a_last, n_last = a, num
        med, mx_err = float(np.median(errs)), float(np.max(errs))
        worst = max(worst, med)
        print(f"  {name:12s} {a_last:+13.3e} {n_last:+13.3e} {med:12.2e} {mx_err:10.2e}")
    return worst


def main():
    cam = Camera.look_at(eye=(0, 0, -4.0), at=(0, 0, 0), width=64, height=64)
    mx.random.seed(7)
    target = mx.random.uniform(shape=(64, 64, 3))

    print("Single isolated Gaussian (per-Gaussian gradient formulas):")
    args = [
        mx.array([[0.05, -0.03, 0.0]]),
        mx.array([[1.0, 0.2, 0.1, 0.0]]),
        mx.array([[-1.2, -1.0, -1.3]]),
        mx.array([0.4]),
        mx.array([[[0.3, 0.1, -0.2]]]),
    ]
    names = ["means", "quats", "log_scales", "opacity", "color"]
    w1 = directional_report(loss_fn(cam, target, mx.zeros((3,))), args, names)

    print("\nFour depth-stacked Gaussians (compositing colour gradient):")
    # Only colour is finite-difference-clean under overlap; opacity/geometry are
    # validated on the isolated Gaussian above (their FD crosses the discrete
    # n_contrib / tile boundaries once Gaussians overlap).
    mx.random.seed(5)
    target2 = mx.random.uniform(shape=(64, 64, 3))
    args2 = [
        mx.array([[0.0, 0.0, 0.0], [0.05, 0.02, 0.3], [-0.03, 0.04, 0.6], [0.02, -0.05, 0.9]]),
        mx.tile(mx.array([[1.0, 0.0, 0.0, 0.0]]), (4, 1)),
        mx.full((4, 3), -1.0),
        mx.array([0.2, -0.1, 0.3, 0.0]),
        mx.random.normal((4, 1, 3)) * 0.5,
    ]
    w2 = directional_report(
        loss_fn(cam, target2, mx.array([0.2, 0.1, 0.3])), args2, names, indices=[4]
    )

    worst = max(w1, w2)
    print(
        f"\nWorst median relative error: {worst:.2e}  "
        f"({'PASS' if worst < 0.10 else 'CHECK'} — geometry ~1e-2 is finite-difference "
        "truncation error, not a wrong gradient; per-direction max is noisier because "
        "some directions cross discrete tile/sort boundaries)."
    )


if __name__ == "__main__":
    main()
