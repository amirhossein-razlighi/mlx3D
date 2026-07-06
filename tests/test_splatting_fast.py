"""Tests for the fast forward-only Gaussian rasterization path."""

import mlx.core as mx
import numpy as np

from mlx3d.cameras import Camera
from mlx3d.splatting import FastGaussianRenderer, GaussianModel, render_gaussians_fast


def _scene(n=5000, sh_degree=3, seed=0):
    mx.random.seed(seed)
    model = GaussianModel.from_points(mx.random.normal((n, 3)) * 1.2, sh_degree=sh_degree)
    model.params["sh_rest"] = mx.random.normal(model.params["sh_rest"].shape) * 0.3
    model.params["scales"] = mx.clip(model.params["scales"] - 0.8, -8.0, -2.5)
    mx.eval(model.params)
    return model


def _cam(w=320, h=240, eye=(0, 0, -5.0)):
    return Camera.look_at(eye=eye, at=(0, 0, 0), width=w, height=h)


def _psnr(a: mx.array, b: mx.array) -> float:
    mse = float(mx.mean((a - b) ** 2).item())
    return 10.0 * np.log10(1.0 / max(mse, 1e-12))


def test_fast_matches_reference():
    model = _scene()
    cam = _cam()
    bg = mx.array([0.1, 0.2, 0.3])
    ref = model.render(cam, background=bg)
    out = FastGaussianRenderer(model).render(cam, background=bg)
    mx.eval(ref["image"], out["image"])
    assert out["image"].shape == (240, 320, 3)
    # Identical compositing math; only fp16 payload quantization differs.
    assert _psnr(ref["image"], out["image"]) > 60.0
    assert float(mx.abs(ref["alpha"] - out["alpha"]).mean().item()) < 1e-3


def test_functional_one_shot_matches_class():
    model = _scene(n=2000)
    cam = _cam()
    a = FastGaussianRenderer(model).render(cam)["image"]
    b = render_gaussians_fast(
        cam,
        model.params["means"],
        model.params["quats"],
        model.scales_act,
        model.opacities_act,
        sh=model.sh,
        sh_degree=model.active_sh_degree,
    )["image"]
    mx.eval(a, b)
    assert _psnr(a, b) > 80.0


def test_argsort_is_stable():
    # The fast path relies on MLX's radix argsort being stable so that tile
    # duplicates emitted in depth order stay depth-ordered after the tile sort.
    mx.random.seed(3)
    keys = mx.random.randint(0, 50, (200_000,)).astype(mx.uint32)
    idx = np.array(mx.argsort(keys))
    kk = np.array(keys)[idx]
    same_key = kk[1:] == kk[:-1]
    assert np.all(idx[1:][same_key] > idx[:-1][same_key]), "mx.argsort is no longer stable"


def test_update_supports_dynamic_scenes():
    model = _scene(n=2000)
    cam = _cam()
    fast = FastGaussianRenderer(model)
    frame0 = fast.render(cam)["image"]

    # Move the scene (a "timestep") and verify the fast path tracks the
    # reference renderer for the updated state.
    shifted = model.params["means"] + mx.array([0.5, 0.0, 0.0])
    fast.update(
        means=shifted,
        quats=model.params["quats"],
        scales=model.scales_act,
        opacities=model.opacities_act,
        sh=model.sh,
        sh_degree=model.active_sh_degree,
    )
    frame1 = fast.render(cam)["image"]
    mx.eval(frame0, frame1)
    assert float(mx.abs(frame1 - frame0).max().item()) > 0.01  # actually moved

    from mlx3d.splatting import render_gaussians

    ref = render_gaussians(
        cam,
        shifted,
        model.params["quats"],
        model.scales_act,
        model.opacities_act,
        sh=model.sh,
        sh_degree=model.active_sh_degree,
    )["image"]
    mx.eval(ref)
    assert _psnr(ref, frame1) > 60.0


def test_capacity_grows_when_zooming_in():
    model = _scene(n=3000)
    fast = FastGaussianRenderer(model)
    far = fast.render(_cam(eye=(0, 0, -12.0)))["image"]
    mx.eval(far)
    cap_before = fast._capacity
    near = fast.render(_cam(eye=(0, 0, -1.5)))["image"]
    mx.eval(near)
    assert fast._capacity >= cap_before
    # The re-rendered close-up must still match the reference.
    ref = model.render(_cam(eye=(0, 0, -1.5)))["image"]
    mx.eval(ref)
    assert _psnr(ref, near) > 60.0


def test_empty_and_culled_scenes():
    model = _scene(n=100)
    fast = FastGaussianRenderer(model)
    # Camera looking away: everything culled behind znear.
    out = fast.render(_cam(eye=(0, 0, -50.0), w=64, h=48))
    mx.eval(out["image"])
    assert out["image"].shape == (48, 64, 3)

    empty = FastGaussianRenderer(
        means=mx.zeros((0, 3)),
        quats=mx.zeros((0, 4)),
        scales=mx.zeros((0, 3)),
        opacities=mx.zeros((0,)),
        colors=mx.zeros((0, 3)),
    )
    out = empty.render(_cam(w=64, h=48), background=mx.array([1.0, 0.0, 0.0]))
    mx.eval(out["image"])
    assert float(out["image"][0, 0, 0].item()) == 1.0
    assert float(out["alpha"].sum().item()) == 0.0


def test_color_cache_refresh_modes():
    model = _scene(n=2000)
    always = FastGaussianRenderer(model, color_refresh=0.0)
    cached = FastGaussianRenderer(model, color_refresh=0.5)  # very sticky cache
    cam_a = _cam(eye=(0, 0, -5.0))
    cam_b = _cam(eye=(0.2, 0, -5.0))  # small move: cached colors reused
    a0 = always.render(cam_a)["image"]
    c0 = cached.render(cam_a)["image"]
    mx.eval(a0, c0)
    assert _psnr(a0, c0) > 80.0
    a1 = always.render(cam_b)["image"]
    c1 = cached.render(cam_b)["image"]
    mx.eval(a1, c1)
    # Stale view-dependent colors are an approximation, but a close one.
    assert _psnr(a1, c1) > 35.0


def test_sh_degree_cap_runs():
    model = _scene(n=1000)
    out = FastGaussianRenderer(model, sh_degree=0).render(_cam(w=64, h=48))
    mx.eval(out["image"])
    assert out["image"].shape == (48, 64, 3)
