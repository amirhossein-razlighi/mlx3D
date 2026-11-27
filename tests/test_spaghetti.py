"""Spaghetti splatting: closed form, Metal rasterizer, model ops and training."""

import math

import mlx.core as mx
import numpy as np
import pytest

from mlx3d.cameras import Camera
from mlx3d.splatting import GaussianModel
from mlx3d.splatting.spaghetti import (
    SpaghettiModel,
    SpaghettiTrainer,
    default_config_for,
    render_spaghetti,
    render_spaghetti_reference,
    segment_optical_depth,
)
from mlx3d.transforms import quaternion_to_matrix

W_PX, H_PX = 40, 32


def _camera():
    return Camera.look_at(eye=(0.3, -0.2, -3.0), at=(0, 0, 0), width=W_PX, height=H_PX)


def _random_segments(n=40, seed=0, density=(0.0, 2.5)):
    mx.random.seed(seed)
    kinds = mx.random.randint(0, 3, (n, 1))  # point, short, long
    return {
        "means": mx.random.normal((n, 3)) * 0.5,
        "halves": mx.random.normal((n, 3)) * mx.array([0.0, 0.02, 0.3])[kinds],
        "quats": mx.random.normal((n, 4)),
        "scales": mx.exp(mx.random.uniform(-3.5, -2.0, (n, 3))),
        "rho": mx.exp(mx.random.uniform(*density, (n,))),
        "colors": mx.random.uniform(shape=(n, 3)),
    }


# ------------------------------------------------------------- closed form


def test_closed_form_matches_quadrature_for_points_and_segments():
    rng = np.random.default_rng(0)
    for kind, length in enumerate((0.0, 0.05, 0.6)):
        for _ in range(4):
            s = np.exp(rng.uniform(-3, -1.5, 3))
            q = rng.normal(size=4)
            R = np.array(quaternion_to_matrix(mx.array(q / np.linalg.norm(q))))
            Q = 0.5 * np.linalg.inv(R @ np.diag(s**2) @ R.T)
            a = rng.normal(size=3) * 0.1 + np.array([0.0, 0.0, 3.0])
            v = rng.normal(size=3) * length
            d = a + rng.uniform() * v + rng.normal(size=3) * s.max()
            d /= np.linalg.norm(d)
            us, ts = np.linspace(0, 1, 401), np.linspace(0.5, 7.0, 8001)
            X = ts[None, :, None] * d - (a + us[:, None] * v)[:, None, :]
            f = np.exp(-np.einsum("utc,cd,utd->ut", X, Q, X))
            expected = 2.0 * np.trapezoid(np.trapezoid(f, ts, axis=1), us)
            W = R.T / (math.sqrt(2) * s[:, None])
            tau = segment_optical_depth(
                *(mx.array(x, dtype=mx.float32) for x in (W @ d, W @ a, W @ v)), mx.array(2.0)
            )
            assert float(tau) == pytest.approx(expected, rel=2e-3, abs=1e-6), kind


# --------------------------------------------------------------- rasterizer


@pytest.mark.parametrize("density", [(0.0, 2.5), (2.0, 4.0)])  # sparse / heavy occlusion
def test_metal_rasterizer_matches_reference(density):
    cam, p = _camera(), _random_segments(density=density)
    bg = mx.array([0.2, 0.3, 0.4])
    fast = render_spaghetti(cam, **p, background=bg)
    ref = render_spaghetti_reference(cam, **p, background=bg)
    assert float(mx.abs(fast["image"] - ref["image"]).max()) < 2e-3
    assert float(mx.abs(fast["alpha"] - ref["alpha"]).max()) < 2e-3
    assert 0.02 < float(ref["alpha"].mean())


def test_metal_gradients_match_autodiff_of_reference():
    cam, p = _camera(), _random_segments()
    target = mx.random.uniform(shape=(H_PX, W_PX, 3))
    bg = mx.array([0.1, 0.2, 0.3])

    def loss(render):
        def f(params):
            out = render(cam, **params, background=bg)
            return ((out["image"] - target) ** 2).sum() + 0.3 * (out["alpha"] ** 2).sum()

        return f

    g_fast = mx.grad(loss(render_spaghetti))(p)
    g_ref = mx.grad(loss(render_spaghetti_reference))(p)
    for k in p:
        a, b = np.array(g_fast[k]), np.array(g_ref[k])
        assert np.abs(a - b).max() <= 1e-3 * np.abs(b).max() + 1e-6, k


def test_pose_gradient_flows_through_camera():
    from mlx3d.cameras import refine_camera

    cam, p = _camera(), _random_segments()

    def f(twist):
        return render_spaghetti(refine_camera(cam, twist), **p)["image"].sum()

    g = mx.grad(f)(mx.zeros((6,)))
    assert float(mx.abs(g).sum()) > 0


# -------------------------------------------------------------------- model


def _model(n=30, seed=0):
    rng = np.random.default_rng(seed)
    pts = mx.array(rng.normal(size=(n, 3)) * 0.4, dtype=mx.float32)
    cols = mx.array(rng.uniform(size=(n, 3)), dtype=mx.float32)
    return SpaghettiModel.from_points(pts, cols, sh_degree=1, initial_opacity=0.4, seed=seed)


def test_split_and_clone_preserve_the_render():
    """Exact for a segment in isolation: optical depth adds along every ray.
    (With other segments sorted between the two halves, alpha compositing
    order makes it approximate — the same caveat as in the paper.)"""
    cam = _camera()
    single = _model(n=1)
    single.params["halves"] = mx.array([[0.3, 0.1, 0.05]])
    before = single.render(cam)["image"]
    for threshold in (0.0, 1e9):  # split / clone
        trial = single.copy()
        trial.append(trial.split_or_clone(mx.arange(1), mx.full((1,), threshold)))
        assert trial.num_segments == 2
        after = trial.render(cam)["image"]
        assert float(mx.abs(after - before).max()) < 1e-4


def test_point_segments_are_gaussians_and_lift_from_3dgs():
    gs = GaussianModel.from_points(mx.random.normal((20, 3)) * 0.3, sh_degree=0)
    model = SpaghettiModel.from_gaussians(gs)
    assert model.num_segments == 20
    assert float(model.lengths.max()) < 1e-2 * float(mx.exp(gs.params["scales"]).max())
    out = model.render(_camera())
    assert out["image"].shape == (H_PX, W_PX, 3) and float(out["alpha"].max()) > 0


def test_ply_roundtrip(tmp_path):
    model = _model()
    path = str(tmp_path / "spaghetti.ply")
    model.save_ply(path)
    loaded = SpaghettiModel.load_ply(path)
    for k, v in model.params.items():
        np.testing.assert_allclose(np.array(loaded.params[k]), np.array(v), atol=1e-6, err_msg=k)


def test_snap_endpoints_and_polylines():
    gap = 1e-3
    ends = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([1.0, 1.0, 0.0])]
    pts = [(ends[0], ends[1] - gap), (ends[1] + gap, ends[2] - gap), (ends[2] + gap, ends[2] + 1)]
    a = np.array([p[0] for p in pts])
    b = np.array([p[1] for p in pts])
    model = _model(n=3)
    model.params["means"] = mx.array(0.5 * (a + b), dtype=mx.float32)
    model.params["halves"] = mx.array(0.5 * (b - a), dtype=mx.float32)
    assert model.snap_endpoints(0.05) == 2
    lines = model.polylines(tolerance=1e-4)
    assert len(lines) == 1 and lines[0].shape == (4, 3)


# ------------------------------------------------------------------ training


def test_trainer_reduces_loss_and_grows_with_consistent_state():
    cam = _camera()
    target_model = _model(seed=1)
    target = mx.stop_gradient(target_model.render(cam)["image"])
    model = _model(seed=2)
    cfg = default_config_for(
        60, max_segments=45, densify_every=20, densify_from=10, lambda_connect=0.1, connect_from=40
    )
    trainer = SpaghettiTrainer(model, cfg, scene_extent=1.0)
    first = trainer.step(cam, target)["loss"]
    for _ in range(59):
        info = trainer.step(cam, target)
    assert info["loss"] < first
    assert 30 < model.num_segments <= 45
    for name, opt in trainer.optimizers.items():  # moments follow the row count
        assert opt.state[name]["m"].shape[0] == model.num_segments
    assert trainer.finalize_polylines() >= 0
