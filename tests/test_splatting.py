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
    render_gaussian_depth,
    render_gaussians,
    render_gaussians_reference,
    rgb_to_sh,
    sh_to_rgb,
)
from mlx3d.transforms import quaternion_to_matrix


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


def test_projection_matches_matrix_formula():
    mx.random.seed(11)
    cam = Camera.look_at(eye=(0.3, -0.2, -3.0), at=(0, 0, 0), width=80, height=48, fov=55.0)
    means = mx.random.normal((32, 3)) * 0.4
    quats = mx.random.normal((32, 4))
    scales = mx.exp(mx.random.normal((32, 3)) * 0.2 - 2.0)

    out = project_gaussians(cam, means, quats, scales)

    R, t = cam.R, cam.t
    p_cam = means @ R.T + t
    x, y, z = p_cam[:, 0], p_cam[:, 1], p_cam[:, 2]
    z_safe = mx.maximum(z, 1e-6)
    u = cam.fx * x / z_safe + cam.cx
    v = cam.fy * y / z_safe + cam.cy
    means2d = mx.stack([u, v], axis=-1)

    tan_fov_x = 0.5 * cam.width / cam.fx
    tan_fov_y = 0.5 * cam.height / cam.fy
    tx = mx.clip(x / z_safe, -1.3 * tan_fov_x, 1.3 * tan_fov_x) * z_safe
    ty = mx.clip(y / z_safe, -1.3 * tan_fov_y, 1.3 * tan_fov_y) * z_safe
    zero = mx.zeros_like(z_safe)
    inv_z = 1.0 / z_safe
    inv_z2 = inv_z * inv_z
    J = mx.stack(
        [
            mx.stack([cam.fx * inv_z, zero, -cam.fx * tx * inv_z2], axis=-1),
            mx.stack([zero, cam.fy * inv_z, -cam.fy * ty * inv_z2], axis=-1),
        ],
        axis=-2,
    )
    qR = quaternion_to_matrix(quats)
    M = qR * scales[:, None, :]
    cov3d = M @ M.swapaxes(-1, -2)
    T = J @ R
    cov2d = T @ cov3d @ T.swapaxes(-1, -2)
    a = cov2d[:, 0, 0] + 0.3
    b = cov2d[:, 0, 1]
    c = cov2d[:, 1, 1] + 0.3
    det = a * c - b * b
    det_safe = mx.maximum(det, 1e-12)
    conics = mx.stack([c / det_safe, -b / det_safe, a / det_safe], axis=-1)
    mid = 0.5 * (a + c)
    lam1 = mid + mx.sqrt(mx.maximum(mid * mid - det, 0.01))
    radii = mx.ceil(3.0 * mx.sqrt(mx.maximum(lam1, 0.0)))
    radii = mx.where((z > cam.znear) & (det > 0), radii, mx.zeros_like(radii))

    assert_close(out["means2d"], means2d, atol=1e-5)
    assert_close(out["conics"], conics, atol=1e-4)
    assert_close(out["depths"], z, atol=1e-5)
    assert_close(out["radii"], radii, atol=0)


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


def test_kernel_matches_reference_forward_refined_tiles(random_scene):
    cam, means, quats, scales, opac, colors = random_scene
    bg = mx.array([0.1, 0.2, 0.3])
    out_k = render_gaussians(
        cam,
        means,
        quats,
        scales,
        opac,
        colors=colors,
        background=bg,
        refine_tiles=True,
    )
    out_r = render_gaussians_reference(cam, means, quats, scales, opac, colors, background=bg)
    assert_close(out_k["image"], out_r["image"], atol=1e-3)
    assert_close(out_k["alpha"], out_r["alpha"], atol=1e-3)


def test_depth_rasterization_outputs_valid_depth(random_scene):
    cam, means, quats, scales, opac, _ = random_scene
    out = render_gaussian_depth(cam, means, quats, scales, opac)
    assert out["depth"].shape == (cam.height, cam.width)
    assert out["alpha"].shape == (cam.height, cam.width)
    valid = np.array(out["alpha"]) > 1e-3
    assert valid.any()
    depth = np.array(out["depth"])
    assert np.isfinite(depth[valid]).all()
    assert depth[valid].min() > 0.0


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


def test_model_from_points_caps_initial_scale():
    pts = mx.array(
        [
            [0.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
            [0.0, 100.0, 0.0],
            [0.0, 0.0, 100.0],
        ]
    )
    model = GaussianModel.from_points(
        pts,
        sh_degree=0,
        scale_init_max_scale=0.25,
    )
    mx.eval(model.params["scales"])
    assert float(mx.exp(model.params["scales"]).max()) <= 0.250001


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


def test_mcmc_relocate_keeps_fixed_count():
    mx.random.seed(12)
    pts = mx.random.normal((30, 3)) * 0.2
    model = GaussianModel.from_points(pts, sh_degree=0)
    n0 = model.num_gaussians
    model.params["opacities"] = model.params["opacities"].at[:5].add(-100.0)
    before = np.array(model.params["means"])
    grads = mx.concatenate([mx.zeros((10,)), mx.linspace(0.0, 1.0, 20)])
    counts = mx.ones((n0,))

    stats = model.relocate_mcmc(
        grads,
        counts,
        relocate_frac=0.2,
        min_opacity=0.05,
        jitter_scale=0.0,
    )

    assert model.num_gaussians == n0
    assert stats["relocated"] == 5
    after = np.array(model.params["means"])
    assert not np.allclose(after[:5], before[:5])


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


def test_trainer_position_lr_decays():
    mx.random.seed(7)
    cam = Camera.look_at(eye=(0, 0, -3.0), at=(0, 0, 0), width=24, height=24, fov=60.0)
    target = mx.random.uniform(shape=(24, 24, 3))
    model = GaussianModel.from_points(
        mx.random.normal((20, 3)) * 0.3,
        mx.random.uniform(shape=(20, 3)),
        sh_degree=0,
    )
    trainer = GaussianTrainer(
        model,
        TrainerConfig(
            lr_means=1e-2,
            lr_means_final=1e-4,
            lr_means_max_steps=2,
            densify_from=10_000,
            sh_increase_every=10_000,
        ),
        scene_extent=2.0,
    )

    lr0 = trainer.learning_rates()["means"]
    info1 = trainer.step(cam, target)
    info2 = trainer.step(cam, target)
    info3 = trainer.step(cam, target)

    assert abs(lr0 - 2e-2) < 1e-6
    assert abs(info1["lr_means"] - lr0) < 1e-8
    assert info2["lr_means"] < info1["lr_means"]
    assert info3["lr_means"] < info2["lr_means"]
    assert abs(info3["lr_means"] - 2e-4) < 1e-7


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
    assert trainer.grad_accum.shape == (model.num_gaussians,)
    assert trainer.grad_count.shape == (model.num_gaussians,)


def test_trainer_preserves_optimizer_state_after_densify_resize():
    mx.random.seed(8)
    model = GaussianModel.from_points(mx.random.normal((8, 3)) * 0.2, sh_degree=0)
    trainer = GaussianTrainer(model, TrainerConfig(densify_from=10_000))
    for name, opt in trainer.optimizers.items():
        opt.init({name: model.params[name]})

    opt = trainer.optimizers["means"]
    old_m = mx.arange(model.num_gaussians * 3, dtype=mx.float32).reshape(model.num_gaussians, 3)
    old_v = old_m + 100.0
    opt.state["means"]["m"] = old_m
    opt.state["means"]["v"] = old_v
    opt.state["step"] = mx.array(17, dtype=mx.uint64)

    keep_idx_np = np.array([1, 3, 6], dtype=np.int32)
    keep_idx = mx.array(keep_idx_np)
    model.select(keep_idx_np)
    model.append({
        k: mx.zeros((2, *v.shape[1:]), dtype=v.dtype)
        for k, v in model.params.items()
    })

    trainer._resize_optimizer_states_after_densify({
        "_keep_idx": keep_idx_np,
        "_new_count": 2,
    })

    state = trainer.optimizers["means"].state
    expected_m = mx.concatenate([old_m[keep_idx], mx.zeros((2, 3))], axis=0)
    expected_v = mx.concatenate([old_v[keep_idx], mx.zeros((2, 3))], axis=0)
    assert_close(state["means"]["m"], expected_m, atol=0)
    assert_close(state["means"]["v"], expected_v, atol=0)
    assert int(state["step"]) == 17


def test_trainer_mcmc_relocation_keeps_count_and_resets_moments():
    mx.random.seed(13)
    cam = Camera.look_at(eye=(0, 0, -3.0), at=(0, 0, 0), width=32, height=32)
    target = mx.random.uniform(shape=(32, 32, 3))
    model = GaussianModel.from_points(mx.random.normal((40, 3)) * 0.4, sh_degree=0)
    n0 = model.num_gaussians
    config = TrainerConfig(
        method="mcmc",
        densify_from=1,
        densify_every=2,
        densify_grad_threshold=1e-9,
        mcmc_relocate_frac=0.25,
        mcmc_min_opacity=0.2,
        mcmc_jitter_scale=0.0,
        mcmc_noise_scale=0.0,
    )
    trainer = GaussianTrainer(model, config)
    event = None
    for _ in range(3):
        info = trainer.step(cam, target)
        if info["densify"] is not None:
            event = info["densify"]

    assert model.num_gaussians == n0
    assert event is not None
    assert event["relocated"] > 0
    state = trainer.optimizers["means"].state["means"]
    assert np.isfinite(np.array(state["m"])).all()


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
