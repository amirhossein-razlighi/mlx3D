"""Masks, appearance compensation and geometry priors for Gaussian training.

Covers the acceptance criteria of issue #28 (masked / transparent foreground
training) plus per-view appearance and depth / normal prior losses, on small
generated fixtures.
"""

import os

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

from mlx3d.cameras import Camera
from mlx3d.datasets import MaskCollection, find_mask_file, load_colmap, save_colmap
from mlx3d.losses import masked_l1, masked_psnr, masked_ssim, ssim
from mlx3d.splatting import GaussianModel, GaussianTrainer, TrainerConfig

W, H = 24, 20


def _scene(seed=0):
    mx.random.seed(seed)
    cam = Camera.look_at(eye=(0, 0, -3.0), at=(0, 0, 0), width=W, height=H)
    model = GaussianModel.from_points(mx.random.normal((40, 3)) * 0.4, sh_degree=0)
    target = mx.random.uniform(shape=(H, W, 3))
    return cam, model, target


def _half_mask():
    mask = np.zeros((H, W), np.float32)
    mask[:, : W // 2] = 1.0
    return mx.array(mask)


def _loss_and_grads(trainer, cam, target, **kwargs):
    params = {k: mx.array(v) for k, v in trainer.model.params.items()}
    probe = mx.zeros((trainer.model.num_gaussians, 2))
    bg = mx.zeros((3,))

    def fn(p):
        return trainer._render_loss(p, probe, cam, target, bg, **kwargs)[0]

    loss, grads = mx.value_and_grad(fn)(params)
    return float(loss), grads


# --------------------------------------------------------------- losses


def test_masked_losses_ignore_outside_pixels_and_coverage():
    rng = np.random.default_rng(0)
    pred = mx.array(rng.uniform(size=(H, W, 3)).astype(np.float32))
    target = mx.array(rng.uniform(size=(H, W, 3)).astype(np.float32))
    mask = _half_mask()
    other = mx.where(mask[..., None] > 0, target, 1.0 - target)  # differs only outside
    for fn in (masked_l1, masked_ssim, masked_psnr):
        assert float(fn(pred, target, mask)) == pytest.approx(float(fn(pred, other, mask)))

    # Coverage-normalized: a uniform error gives the same masked L1 for any mask size.
    err = pred + 0.1
    small = mx.zeros((H, W)).at[2:5, 2:5].add(1.0)
    assert float(masked_l1(err, pred, small)) == pytest.approx(0.1, rel=1e-5)
    assert float(masked_l1(err, pred, mask)) == pytest.approx(0.1, rel=1e-5)
    # Full mask reduces to the unmasked metric.
    ones = mx.ones((H, W))
    assert float(masked_ssim(pred, target, ones)) == pytest.approx(float(ssim(pred, target)))


def test_masked_ssim_has_no_gradient_outside_the_mask():
    rng = np.random.default_rng(1)
    pred = mx.array(rng.uniform(size=(H, W, 3)).astype(np.float32))
    target = mx.array(rng.uniform(size=(H, W, 3)).astype(np.float32))
    mask = _half_mask()
    grad = np.array(mx.grad(lambda p: masked_ssim(p, target, mask))(pred))
    assert np.abs(grad[:, W // 2 :]).max() == 0.0
    assert np.abs(grad[:, : W // 2]).max() > 0.0


# --------------------------------------------------------------- trainer


def test_masked_mode_is_independent_of_target_outside_mask():
    cam, model, target = _scene()
    trainer = GaussianTrainer(model, TrainerConfig(densify_from=10_000))
    mask = _half_mask()
    changed = mx.where(mask[..., None] > 0, target, mx.random.uniform(shape=target.shape))
    loss_a, grads_a = _loss_and_grads(trainer, cam, target, mask=mask)
    loss_b, grads_b = _loss_and_grads(trainer, cam, changed, mask=mask)
    assert loss_a == pytest.approx(loss_b, rel=1e-6)
    for k in grads_a:
        np.testing.assert_allclose(np.array(grads_a[k]), np.array(grads_b[k]), atol=1e-7)


def test_transparent_mode_supervises_alpha():
    cam, model, target = _scene()
    trainer = GaussianTrainer(model, TrainerConfig(alpha_mode="transparent", densify_from=10_000))
    full = mx.ones((H, W))
    empty_ish = mx.zeros((H, W)).at[0, 0].add(1.0)
    loss_full, _ = _loss_and_grads(trainer, cam, target, mask=full)
    loss_empty, grads = _loss_and_grads(trainer, cam, target, mask=empty_ish)
    assert loss_full != pytest.approx(loss_empty)  # alpha target matters
    assert float(mx.abs(grads["opacities"]).sum()) > 0

    # Steps run end to end (with a random background) and report the alpha loss.
    cfg = TrainerConfig(alpha_mode="transparent", random_background=True, densify_from=10_000)
    info = GaussianTrainer(model.copy(), cfg).step(cam, target, mask=_half_mask())
    assert info["geometry"]["alpha_l1"] >= 0.0


def test_2dgs_geometry_losses_respect_the_mask():
    cam, model, target = _scene()
    cfg = TrainerConfig(
        method="2dgs",
        densify_from=10_000,
        lambda_2d_depth_variance=0.1,
        lambda_2d_normal_consistency=0.05,
        geometry_min_alpha=0.001,
    )
    trainer = GaussianTrainer(model, cfg)
    params = {k: mx.array(v) for k, v in model.params.items()}
    probe = mx.zeros((model.num_gaussians, 2))
    bg = mx.zeros((3,))

    def metrics(mask=None):
        return trainer._render_loss(params, probe, cam, target, bg, mask=mask)[1][2]

    unmasked = metrics()
    assert float(unmasked["depth_variance"]) > 0.0
    full = metrics(mx.ones((H, W)))
    assert float(full["depth_variance"]) == pytest.approx(float(unmasked["depth_variance"]))
    # A mask covering nothing leaves the geometry losses no pixels at all.
    assert float(metrics(mx.zeros((H, W)))["depth_variance"]) == 0.0


def test_appearance_compensation_learns_per_view_exposure():
    cam, model, target = _scene()
    with pytest.raises(ValueError, match="num_views"):
        GaussianTrainer(model.copy(), TrainerConfig(appearance=True))
    cfg = TrainerConfig(appearance=True, lr_appearance=1e-2, densify_from=10_000)
    trainer = GaussianTrainer(model, cfg, num_views=2)
    with pytest.raises(ValueError, match="view_index"):
        trainer.step(cam, target)
    for _ in range(5):
        trainer.step(cam, target * 0.5, view_index=1)  # a darker exposure for view 1
    appearance = np.array(trainer.appearance)
    assert np.abs(appearance[1]).max() > 0.0
    assert np.abs(appearance[0]).max() == 0.0  # untouched view stays identity


def test_depth_prior_loss_is_scale_and_shift_invariant():
    cam, model, target = _scene()
    cfg = TrainerConfig(
        lambda_depth_prior=1.0, lambda_normal_prior=0.0, prior_min_alpha=0.01, densify_from=10_000
    )
    trainer = GaussianTrainer(model, cfg)
    params = {k: mx.array(v) for k, v in model.params.items()}
    probe = mx.zeros((model.num_gaussians, 2))
    bg = mx.zeros((3,))
    prior = mx.random.uniform(shape=(H, W)) + 1.0

    def metric(depth_prior):
        out = trainer._render_loss(params, probe, cam, target, bg, depth_prior=depth_prior)
        return float(out[1][2]["depth_prior"])

    assert metric(prior) == pytest.approx(metric(5.0 * prior + 2.0), rel=1e-4)
    info = trainer.step(cam, target, depth_prior=prior)
    assert 0.0 <= info["geometry"]["depth_prior"] <= 2.0


def test_normal_prior_vanishes_for_matching_depth():
    cam, model, target = _scene()
    cfg = TrainerConfig(lambda_normal_prior=1.0, prior_min_alpha=0.01, densify_from=10_000)
    trainer = GaussianTrainer(model, cfg)
    params = {k: mx.array(v) for k, v in model.params.items()}
    probe = mx.zeros((model.num_gaussians, 2))
    bg = mx.zeros((3,))
    # Render the model's own depth and use it as the prior: normals agree exactly.
    from mlx3d.splatting import render_gaussian_depth

    own = render_gaussian_depth(
        cam, model.params["means"], model.params["quats"], model.scales_act, model.opacities_act
    )
    out = trainer._render_loss(params, probe, cam, target, bg, depth_prior=own["depth"])
    assert float(out[1][2]["normal_prior"]) == pytest.approx(0.0, abs=1e-3)


# --------------------------------------------------------------- datasets


def _colmap_scene(tmp_path, n=3, rgba=False):
    root = str(tmp_path)
    os.makedirs(os.path.join(root, "images"))
    rng = np.random.default_rng(0)
    cams, names = [], []
    for i in range(n):
        name = f"img_{i}.png"
        pixels = rng.integers(0, 255, (16, 24, 4 if rgba else 3), dtype=np.uint8)
        if rgba:
            pixels[..., 3] = 0
            pixels[4:12, 6:18, 3] = 255
        Image.fromarray(pixels).save(os.path.join(root, "images", name))
        cams.append(
            Camera(
                R=mx.eye(3),
                t=mx.array([float(i), 0.0, 3.0]),
                fx=20.0,
                fy=20.0,
                cx=12.0,
                cy=8.0,
                width=24,
                height=16,
            )
        )
        names.append(name)
    save_colmap(root, cams, names, np.zeros((4, 3), np.float32), np.ones((4, 3), np.float32))
    return root, names


def _write_mask(path, h=16, w=24):
    mask = np.zeros((h, w), np.uint8)
    mask[2:10, 4:20] = 255
    Image.fromarray(mask).save(path)


@pytest.mark.parametrize("cache", ["ram", "uint8", "disk"])
@pytest.mark.parametrize("downscale", [1, 2])
def test_load_colmap_masks_aligned_across_cache_modes(tmp_path, cache, downscale):
    root, names = _colmap_scene(tmp_path)
    os.makedirs(os.path.join(root, "masks"))
    for i, name in enumerate(names):  # both naming conventions
        fname = f"{name}.png" if i % 2 == 0 else f"{os.path.splitext(name)[0]}.png"
        _write_mask(os.path.join(root, "masks", fname))
    ds = load_colmap(root, masks_dir="masks", cache=cache, downscale=downscale)
    assert isinstance(ds.masks, MaskCollection) and len(ds.masks) == len(ds)
    for i in range(len(ds)):
        cam, image = ds[i]  # the tuple API is unchanged
        mask = ds.masks[i]
        assert mask.shape == (cam.height, cam.width) == image.shape[:2]
        assert 0.0 < float(mask.mean()) < 1.0


def test_masks_from_rgba_alpha(tmp_path):
    root, _ = _colmap_scene(tmp_path, rgba=True)
    ds = load_colmap(root, masks_from_alpha=True)
    assert ds.images[0].shape == (16, 24, 3)  # RGB loading is unchanged
    np.testing.assert_allclose(float(ds.masks[0].mean()), 8 * 12 / (16 * 24))


def test_load_colmap_mask_diagnostics(tmp_path):
    root, names = _colmap_scene(tmp_path)
    masks = os.path.join(root, "masks")
    os.makedirs(masks)
    _write_mask(os.path.join(masks, f"{names[0]}.png"))
    with pytest.raises(FileNotFoundError, match="2 of 3 images have no mask"):
        load_colmap(root, masks_dir="masks")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_colmap(root, masks_dir="nope")
    for name in names[1:]:
        _write_mask(os.path.join(masks, f"{name}.png"), h=8, w=8)  # wrong size
    with pytest.raises(ValueError, match="but its image is"):
        load_colmap(root, masks_dir="masks")
    for name in names[1:]:
        Image.fromarray(np.zeros((16, 24), np.uint8)).save(os.path.join(masks, f"{name}.png"))
    with pytest.raises(ValueError, match="empty"):
        load_colmap(root, masks_dir="masks")
    assert find_mask_file(masks, names[0]).endswith(f"{names[0]}.png")
    assert find_mask_file(masks, "missing.jpg") is None


def test_prior_and_geometry_gradients_stay_finite_on_empty_pixels():
    """Background pixels give zero-length tangents; normalizing them must not
    produce NaN gradients (a zero mask weight does not cancel a NaN)."""
    mx.random.seed(0)
    cam = Camera.look_at(eye=(0, 0, -3.0), at=(0, 0, 0), width=48, height=40)
    model = GaussianModel.from_points(mx.random.normal((10, 3)) * 0.2, sh_degree=0)
    configs = [
        TrainerConfig(lambda_depth_prior=0.1, lambda_normal_prior=0.05),
        TrainerConfig(method="2dgs", lambda_2d_normal_consistency=0.05),
    ]
    for cfg in configs:
        trainer = GaussianTrainer(model.copy(), cfg)
        _, grads = _loss_and_grads(
            trainer, cam, mx.zeros((40, 48, 3)), depth_prior=mx.full((40, 48), 2.0)
        )
        for name, g in grads.items():
            assert not bool(mx.isnan(g).any()), name
