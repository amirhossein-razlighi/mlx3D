import mlx.core as mx
import numpy as np
import pytest

from mlx3d.cameras import Camera
from mlx3d.splatting import (
    GaussianModel,
    GaussianTrainer,
    TrainerConfig,
    bin_gaussians,
    eval_sh,
    num_sh_bases,
    project_gaussians,
    render_gaussians,
    render_gaussians_reference,
    rgb_to_sh,
    sh_to_rgb,
)


def assert_close(a, b, atol=1e-5):
    np.testing.assert_allclose(np.array(a), np.array(b), atol=atol)


@pytest.fixture
def random_scene():
    mx.random.seed(0)
    N = 300
    means = mx.random.normal((N, 3)) * 0.5
    quats = mx.random.normal((N, 4))
    scales = mx.exp(mx.random.normal((N, 3)) * 0.3 - 2.5)
    opac = mx.sigmoid(mx.random.normal((N,)))
    colors = mx.random.uniform(shape=(N, 3))
    cam = Camera.look_at(eye=(0, 0, -4.0), at=(0, 0, 0), width=96, height=64, fov=50.0)
    return cam, means, quats, scales, opac, colors


def test_sh_dc_roundtrip():
    rgb = mx.random.uniform(shape=(10, 3))
    assert_close(sh_to_rgb(rgb_to_sh(rgb)), rgb)
    assert num_sh_bases(3) == 16


def test_eval_sh_degree0_is_color():
    rgb = mx.random.uniform(shape=(5, 3))
    sh = rgb_to_sh(rgb)[:, None, :]
    dirs = mx.random.normal((5, 3))
    dirs = dirs / mx.linalg.norm(dirs, axis=-1, keepdims=True)
    assert_close(eval_sh(0, sh, dirs), rgb, atol=1e-5)


def test_projection_center_gaussian():
    cam = Camera.look_at(eye=(0, 0, -3.0), at=(0, 0, 0), width=64, height=64, fov=60.0)
    proj = project_gaussians(
        cam, mx.zeros((1, 3)), mx.array([[1.0, 0, 0, 0]]), mx.full((1, 3), 0.1)
    )
    assert_close(proj["means2d"][0], mx.array([32.0, 32.0]), atol=1e-3)
    assert abs(float(proj["depths"][0]) - 3.0) < 1e-5
    assert float(proj["radii"][0]) > 0


def test_projection_culls_behind_camera():
    cam = Camera.look_at(eye=(0, 0, -3.0), at=(0, 0, 0), width=64, height=64)
    proj = project_gaussians(
        cam, mx.array([[0.0, 0.0, -10.0]]), mx.array([[1.0, 0, 0, 0]]), mx.full((1, 3), 0.1)
    )
    assert float(proj["radii"][0]) == 0.0


def test_binning_covers_visible_gaussians(random_scene):
    cam, means, quats, scales, opac, colors = random_scene
    proj = project_gaussians(cam, means, quats, scales)
    sorted_ids, tile_ranges, tiles_x, tiles_y = bin_gaussians(
        proj["means2d"], proj["radii"], proj["depths"], cam.width, cam.height
    )
    assert tiles_x == 6 and tiles_y == 4
    ranges = np.array(tile_ranges)
    assert (ranges[:, 1] >= ranges[:, 0]).all()
    # Total duplicates match sum over tiles.
    assert int((ranges[:, 1] - ranges[:, 0]).sum()) == sorted_ids.shape[0]
    # All ids valid.
    ids = np.array(sorted_ids)
    assert ids.min() >= 0 and ids.max() < means.shape[0]


def test_kernel_matches_reference_forward(random_scene):
    cam, means, quats, scales, opac, colors = random_scene
    bg = mx.array([0.1, 0.2, 0.3])
    out_k = render_gaussians(cam, means, quats, scales, opac, colors=colors, background=bg)
    out_r = render_gaussians_reference(cam, means, quats, scales, opac, colors, background=bg)
    # Tolerance covers the early-termination boundary: the kernel stops just
    # before transmittance crosses 1e-4, the reference composites that splat.
    assert_close(out_k["image"], out_r["image"], atol=1e-3)
    assert_close(out_k["alpha"], out_r["alpha"], atol=1e-3)


def test_kernel_matches_reference_gradients(random_scene):
    cam, means, quats, scales, opac, colors = random_scene
    bg = mx.zeros((3,))
    target = mx.random.uniform(shape=(cam.height, cam.width, 3))

    def make_loss(render_fn, use_kw):
        def loss(means, quats, scales, opac, colors):
            if use_kw:
                o = render_fn(cam, means, quats, scales, opac, colors=colors, background=bg)
            else:
                o = render_fn(cam, means, quats, scales, opac, colors, background=bg)
            return ((o["image"] - target) ** 2).mean() + 0.05 * o["alpha"].mean()
        return loss

    gk = mx.grad(make_loss(render_gaussians, True), argnums=(0, 1, 2, 3, 4))(
        means, quats, scales, opac, colors
    )
    gr = mx.grad(make_loss(render_gaussians_reference, False), argnums=(0, 1, 2, 3, 4))(
        means, quats, scales, opac, colors
    )
    for name, a, b in zip(["means", "quats", "scales", "opac", "colors"], gk, gr):
        a, b = np.array(a), np.array(b)
        scale = np.abs(b).max() + 1e-12
        assert np.abs(a - b).max() / scale < 1e-3, f"gradient mismatch for {name}"


def test_render_with_sh(random_scene):
    cam, means, quats, scales, opac, _ = random_scene
    N = means.shape[0]
    sh = mx.random.normal((N, 16, 3)) * 0.1
    out = render_gaussians(cam, means, quats, scales, opac, sh=sh, sh_degree=3)
    assert out["image"].shape == (64, 96, 3)
    assert not bool(mx.isnan(out["image"]).any())


def test_model_from_points_and_render():
    mx.random.seed(1)
    pts = mx.random.normal((100, 3)) * 0.3
    colors = mx.random.uniform(shape=(100, 3))
    model = GaussianModel.from_points(pts, colors, sh_degree=2)
    assert model.num_gaussians == 100
    assert model.sh.shape == (100, 9, 3)
    cam = Camera.look_at(eye=(0, 0, -3.0), at=(0, 0, 0), width=48, height=48)
    out = model.render(cam)
    assert out["image"].shape == (48, 48, 3)
    assert float(out["alpha"].max()) > 0.01


def test_model_ply_roundtrip(tmp_path):
    pts = mx.random.normal((20, 3))
    model = GaussianModel.from_points(pts, sh_degree=2)
    model.params["sh_rest"] = mx.random.normal(model.params["sh_rest"].shape) * 0.1
    p = str(tmp_path / "ckpt.ply")
    model.save_ply(p)
    loaded = GaussianModel.load_ply(p)
    assert loaded.sh_degree == 2
    for k in model.params:
        assert_close(loaded.params[k], model.params[k], atol=1e-5)


def test_densify_and_prune():
    pts = mx.random.normal((50, 3))
    model = GaussianModel.from_points(pts, sh_degree=1)
    n0 = model.num_gaussians
    # High gradient on the first 10, low opacity on the last 5.
    grads = mx.concatenate([mx.full((10,), 1.0), mx.zeros((n0 - 10,))])
    counts = mx.ones((n0,))
    model.params["opacities"] = model.params["opacities"].at[-5:].add(-100.0)
    stats = model.densify_and_prune(
        grads, counts, grad_threshold=0.5, scene_extent=10.0, percent_dense=0.5
    )
    assert stats["pruned"] >= 5
    assert stats["cloned"] + stats["split"] == 10
    expected = n0 - stats["pruned"] + stats["cloned"] + 2 * stats["split"]
    assert model.num_gaussians == expected


def test_trainer_converges_single_view():
    """Fit a tiny scene to one rendered target; the loss must drop sharply."""
    mx.random.seed(3)
    cam = Camera.look_at(eye=(0, 0, -3.0), at=(0, 0, 0), width=48, height=48, fov=60.0)

    # Ground-truth scene: 30 gaussians; target = its own render.
    gt = GaussianModel.from_points(
        mx.random.normal((30, 3)) * 0.4, mx.random.uniform(shape=(30, 3)), sh_degree=0
    )
    target = mx.stop_gradient(gt.render(cam)["image"])

    model = GaussianModel.from_points(
        mx.random.normal((60, 3)) * 0.4, mx.random.uniform(shape=(60, 3)), sh_degree=0
    )
    config = TrainerConfig(densify_from=10_000)  # disable densification here
    trainer = GaussianTrainer(model, config)
    losses = [trainer.step(cam, target)["loss"] for _ in range(60)]
    assert losses[-1] < losses[0] * 0.6, f"no convergence: {losses[0]:.4f} -> {losses[-1]:.4f}"


def test_trainer_densification_runs():
    mx.random.seed(4)
    cam = Camera.look_at(eye=(0, 0, -3.0), at=(0, 0, 0), width=32, height=32)
    target = mx.random.uniform(shape=(32, 32, 3))
    model = GaussianModel.from_points(mx.random.normal((40, 3)) * 0.4, sh_degree=0)
    config = TrainerConfig(densify_from=2, densify_every=5, densify_grad_threshold=1e-9)
    trainer = GaussianTrainer(model, config)
    for _ in range(12):
        info = trainer.step(cam, target)
    # Densification triggered at least once and the model kept training.
    assert np.isfinite(info["loss"])


def test_trainer_max_gaussians_cap():
    mx.random.seed(5)
    cam = Camera.look_at(eye=(0.0, 0.0, -3.0), at=(0, 0, 0), width=32, height=32)
    target = mx.random.uniform(shape=(32, 32, 3))
    model = GaussianModel.from_points(mx.random.normal((40, 3)) * 0.4, sh_degree=0)
    config = TrainerConfig(
        densify_from=2, densify_every=4, densify_grad_threshold=1e-9,
        max_gaussians=45, low_memory=True, cache_limit_gb=0.5,
    )
    trainer = GaussianTrainer(model, config)
    counts = [trainer.step(cam, target)["num_gaussians"] for _ in range(20)]
    # Growth happens but never runs far past the cap (one growth round may
    # land above it; afterwards growth is disabled).
    assert max(counts) > 40
    assert counts[-1] <= 45 * 2  # bounded, not exponential
    # After hitting the cap, count must not keep increasing.
    capped = [c for c in counts if c >= 45]
    if len(capped) >= 2:
        assert capped[-1] <= capped[0]
