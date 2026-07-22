"""Finite-difference gradient check for the Metal Gaussian rasterizer.

The rasterizer's backward pass is a hand-written Metal kernel (see
``rasterize.py``), so its gradients are not produced by autodiff and deserve an
independent correctness check. Here we compare the analytic gradient (autodiff
through the custom VJP) against central finite differences along random
directions, using the symmetric relative error
``|analytic - numeric| / (|analytic| + |numeric|)`` which is robust when a
directional derivative is near zero.

Finite differences are only meaningful where the loss is smooth, so the scenes
are built to avoid the rasterizer's genuinely non-differentiable boundaries
(tile assignment, depth-sort order, the alpha 1/255 cutoff, transmittance
early-termination): a single large centered Gaussian for the per-Gaussian
geometry/appearance formulas, and a few depth-separated Gaussians for the
multi-Gaussian compositing colour gradient.
"""

import mlx.core as mx
import numpy as np
import pytest

from mlx3d.cameras import Camera
from mlx3d.splatting import render_gaussians

pytestmark = pytest.mark.unit


def _loss_fn(cam, target, bg):
    def loss(means, quats, log_scales, logit_op, sh_dc):
        out = render_gaussians(
            cam,
            means,
            quats,
            mx.exp(log_scales),
            mx.sigmoid(logit_op),
            sh=sh_dc,
            sh_degree=0,
            background=bg,
        )
        return mx.mean((out["image"] - target) ** 2)

    return loss


def _directional_errors(loss, args, arg_index, grad, eps=2e-3, n_dirs=5, seed=200):
    """Symmetric relative error between analytic and FD directional derivatives."""
    errors = []
    for s in range(n_dirs):
        mx.random.seed(seed + s)
        v = mx.random.normal(args[arg_index].shape)
        analytic = float(mx.sum(grad * v).item())
        plus = list(args)
        plus[arg_index] = args[arg_index] + eps * v
        minus = list(args)
        minus[arg_index] = args[arg_index] - eps * v
        numeric = (float(loss(*plus).item()) - float(loss(*minus).item())) / (2 * eps)
        errors.append(abs(analytic - numeric) / (abs(analytic) + abs(numeric) + 1e-8))
    return float(np.median(errors)), float(np.max(errors))


def test_gradcheck_single_gaussian_all_params():
    """Each per-Gaussian gradient formula (means, conics via quats/scales,
    opacity, colour) matches finite differences for an isolated Gaussian."""
    cam = Camera.look_at(eye=(0, 0, -4.0), at=(0, 0, 0), width=64, height=64)
    mx.random.seed(7)
    target = mx.random.uniform(shape=(64, 64, 3))
    bg = mx.zeros((3,))

    args = [
        mx.array([[0.05, -0.03, 0.0]]),  # means
        mx.array([[1.0, 0.2, 0.1, 0.0]]),  # quats
        mx.array([[-1.2, -1.0, -1.3]]),  # log_scales
        mx.array([0.4]),  # logit opacity
        mx.array([[[0.3, 0.1, -0.2]]]),  # sh dc (colour)
    ]
    loss = _loss_fn(cam, target, bg)
    _, grads = mx.value_and_grad(loss, argnums=(0, 1, 2, 3, 4))(*args)
    mx.eval(grads)

    # The median over random directions is the robust statistic: an occasional
    # direction crosses a discrete tile/sort kink and inflates the max. Geometry
    # gradients also carry finite-difference truncation error (the image is
    # smooth but curved); appearance gradients are near-linear and much tighter.
    median_tol = {0: 0.08, 1: 0.08, 2: 0.08, 3: 0.02, 4: 0.02}
    names = ["means", "quats", "log_scales", "opacity", "color"]
    for i, name in enumerate(names):
        median, mx_err = _directional_errors(loss, args, i, grads[i], n_dirs=8)
        assert median < median_tol[i], f"{name}: median rel err {median:.3f} (max {mx_err:.3f})"
        assert mx_err < 1.5, f"{name}: catastrophic max rel err {mx_err:.3f}"


def test_gradcheck_compositing_color():
    """The multi-Gaussian compositing backward (front-to-back colour blending)
    matches finite differences for depth-stacked Gaussians."""
    cam = Camera.look_at(eye=(0, 0, -4.0), at=(0, 0, 0), width=64, height=64)
    mx.random.seed(5)
    target = mx.random.uniform(shape=(64, 64, 3))
    bg = mx.array([0.2, 0.1, 0.3])

    args = [
        mx.array([[0.0, 0.0, 0.0], [0.05, 0.02, 0.3], [-0.03, 0.04, 0.6], [0.02, -0.05, 0.9]]),
        mx.tile(mx.array([[1.0, 0.0, 0.0, 0.0]]), (4, 1)),
        mx.full((4, 3), -1.0),
        mx.array([0.2, -0.1, 0.3, 0.0]),
        mx.random.normal((4, 1, 3)) * 0.5,
    ]
    loss = _loss_fn(cam, target, bg)
    _, grads = mx.value_and_grad(loss, argnums=(0, 1, 2, 3, 4))(*args)
    mx.eval(grads)

    median, mx_err = _directional_errors(loss, args, 4, grads[4], n_dirs=8, seed=300)
    assert median < 0.03, f"color: median rel err {median:.3f} (max {mx_err:.3f})"
