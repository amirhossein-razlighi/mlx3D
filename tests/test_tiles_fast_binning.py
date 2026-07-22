"""The fused Metal tile-binning path must match the reference pure-MLX path.

``bin_gaussians`` gained a fused-kernel fast path (depth-ordered expansion +
32-bit tile sort + boundary-detection ranges). It is non-differentiable and
must be a drop-in for the composite-key reference: identical rendered images
and (up to atomic-add ordering noise) identical gradients.
"""

import mlx.core as mx
import numpy as np
import pytest

import mlx3d.splatting.tiles as tiles
from mlx3d.cameras import Camera
from mlx3d.splatting import GaussianModel, render_gaussians
from mlx3d.splatting.projection import project_gaussians
from mlx3d.splatting.tiles import bin_gaussians

pytestmark = pytest.mark.unit


def _scene(n, seed=0):
    mx.random.seed(seed)
    m = GaussianModel.from_points(mx.random.normal((n, 3)) * 1.5, sh_degree=3)
    m.params["sh_rest"] = mx.random.normal(m.params["sh_rest"].shape) * 0.3
    m.params["scales"] = mx.clip(m.params["scales"] - 1.2, -8.0, -3.2)
    mx.eval(m.params)
    return m


def _render(m, cam, fast):
    tiles._USE_FAST_BIN = fast
    try:
        out = render_gaussians(
            cam,
            m.params["means"],
            m.params["quats"],
            m.scales_act,
            m.opacities_act,
            sh=m.sh,
            sh_degree=3,
            background=mx.zeros((3,)),
        )
        mx.eval(out["image"], out["alpha"])
        return np.array(out["image"]), np.array(out["alpha"])
    finally:
        tiles._USE_FAST_BIN = True


@pytest.mark.parametrize("n,w,h", [(2000, 160, 120), (30000, 320, 240), (150000, 640, 360)])
def test_fast_binning_image_is_identical(n, w, h):
    m = _scene(n)
    cam = Camera.look_at(eye=(0.3, 0.1, -6.0), at=(0, 0, 0), width=w, height=h)
    fast_img, fast_a = _render(m, cam, True)
    ref_img, ref_a = _render(m, cam, False)
    # Same set + same depth order per tile => bit-identical compositing.
    assert np.array_equal(fast_img, ref_img)
    assert np.array_equal(fast_a, ref_a)


def test_fast_binning_gradients_match():
    m = _scene(40000)
    cam = Camera.look_at(eye=(0.2, 0.0, -6.0), at=(0, 0, 0), width=400, height=300)
    target = mx.random.uniform(shape=(300, 400, 3))
    bg = mx.zeros((3,))

    def loss(p):
        out = render_gaussians(
            cam,
            p["means"],
            p["quats"],
            mx.exp(p["scales"]),
            mx.sigmoid(p["opacities"]),
            sh=mx.concatenate([p["sh_dc"], p["sh_rest"]], axis=1),
            sh_degree=3,
            background=bg,
        )
        return mx.mean((out["image"] - target) ** 2)

    gf = mx.value_and_grad(loss)
    tiles._USE_FAST_BIN = True
    lf, gfast = gf(m.params)
    tiles._USE_FAST_BIN = False
    lr, gref = gf(m.params)
    mx.eval(lf, lr, gfast, gref)
    tiles._USE_FAST_BIN = True

    assert abs(float(lf) - float(lr)) < 1e-6
    for k in gfast:
        # Only atomic-add ordering differs; tolerance is loose relative noise.
        assert float(mx.abs(gfast[k] - gref[k]).max()) < 1e-5, k


def test_fast_binning_matches_reference_ranges():
    """Directly compare bin_gaussians outputs, not just the rendered image."""
    m = _scene(20000)
    cam = Camera.look_at(eye=(0.1, 0.0, -6.0), at=(0, 0, 0), width=320, height=240)
    proj = project_gaussians(cam, m.params["means"], m.params["quats"], m.scales_act)
    mx.eval(proj["means2d"], proj["radii"], proj["depths"])

    tiles._USE_FAST_BIN = True
    f_ids, f_ranges, tx, ty = bin_gaussians(
        proj["means2d"], proj["radii"], proj["depths"], 320, 240
    )
    tiles._USE_FAST_BIN = False
    r_ids, r_ranges, *_ = bin_gaussians(proj["means2d"], proj["radii"], proj["depths"], 320, 240)
    tiles._USE_FAST_BIN = True
    mx.eval(f_ids, f_ranges, r_ids, r_ranges)

    # Per-tile ranges must be identical (same counts + boundaries).
    assert np.array_equal(np.array(f_ranges), np.array(r_ranges))
    # The multiset of (tile, gaussian) pairs must match; within a tile the
    # order is the same depth order, so the full sorted id arrays match too.
    assert np.array_equal(np.array(f_ids), np.array(r_ids))


def test_fast_binning_empty_scene():
    # Camera looking away: everything culled -> total 0 branch.
    m = _scene(200)
    cam = Camera.look_at(eye=(0, 0, 40.0), at=(0, 0, 50.0), width=64, height=48)
    proj = project_gaussians(cam, m.params["means"], m.params["quats"], m.scales_act)
    sorted_ids, tile_ranges, tx, ty = bin_gaussians(
        proj["means2d"], proj["radii"], proj["depths"], 64, 48
    )
    mx.eval(sorted_ids, tile_ranges)
    assert sorted_ids.shape[0] == 0
    assert int(mx.sum(tile_ranges).item()) == 0
