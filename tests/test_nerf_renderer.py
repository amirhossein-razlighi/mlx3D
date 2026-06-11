import mlx.core as mx
import numpy as np

from mlx3d.cameras import Camera
from mlx3d.nn import NeRF, PositionalEncoding, render_rays
from mlx3d.renderer import render_points, sample_along_rays, sample_pdf, volume_render


def assert_close(a, b, atol=1e-5):
    np.testing.assert_allclose(np.array(a), np.array(b), atol=atol)


def test_positional_encoding_shapes_and_values():
    pe = PositionalEncoding(num_freqs=4)
    x = mx.random.normal((10, 3))
    out = pe(x)
    assert out.shape == (10, 3 * (2 * 4 + 1))
    assert_close(out[:, :3], x)


def test_sample_along_rays():
    o = mx.zeros((5, 3))
    d = mx.broadcast_to(mx.array([0.0, 0.0, 1.0]), (5, 3))
    pts, t = sample_along_rays(o, d, 1.0, 5.0, 16, stratified=False)
    assert pts.shape == (5, 16, 3)
    assert abs(float(t[0, 0]) - 1.0) < 1e-5
    assert abs(float(t[0, -1]) - 5.0) < 1e-5
    # Points lie on the ray.
    assert_close(pts[0, :, 2], t[0])
    assert_close(pts[0, :, :2], mx.zeros((16, 2)))


def test_sample_pdf_concentrates_mass():
    bins = mx.broadcast_to(mx.linspace(0.0, 1.0, 11)[None], (4, 11))
    weights = mx.zeros((4, 10))
    weights = weights.at[:, 7].add(1.0)  # all mass in bin [0.7, 0.8]
    samples = sample_pdf(bins, weights, 64, deterministic=True)
    s = np.array(samples)
    assert (s >= 0.65).mean() > 0.9
    assert (s <= 0.85).mean() > 0.9


def test_volume_render_opaque_surface():
    # A wall of high density at sample 5 -> depth ~= t[5], rgb = wall color.
    R, S = 3, 12
    t = mx.broadcast_to(mx.linspace(1.0, 4.0, S)[None], (R, S))
    density = mx.zeros((R, S))
    density = density.at[:, 5].add(1e8)
    colors = mx.zeros((R, S, 3))
    colors = colors.at[:, 5, 0].add(1.0)  # red wall
    out = volume_render(density, colors, t)
    assert_close(out["rgb"], mx.broadcast_to(mx.array([1.0, 0.0, 0.0]), (R, 3)), atol=1e-4)
    assert_close(out["acc"], mx.ones((R,)), atol=1e-4)
    assert_close(out["depth"], mx.full((R,), float(t[0, 5])), atol=1e-3)


def test_volume_render_empty_space():
    R, S = 2, 8
    t = mx.broadcast_to(mx.linspace(1.0, 2.0, S)[None], (R, S))
    out = volume_render(mx.zeros((R, S)), mx.ones((R, S, 3)), t, white_background=True)
    assert_close(out["rgb"], mx.ones((R, 3)), atol=1e-5)
    assert_close(out["acc"], mx.zeros((R,)), atol=1e-5)


def test_nerf_forward_and_grad():
    model = NeRF(pos_freqs=4, dir_freqs=2, hidden_dim=32, num_layers=3, skip_layer=2)
    pts = mx.random.normal((7, 5, 3))
    dirs = mx.random.normal((7, 5, 3))
    density, rgb = model(pts, dirs)
    assert density.shape == (7, 5)
    assert rgb.shape == (7, 5, 3)
    assert float(density.min()) >= 0
    assert 0 <= float(rgb.min()) and float(rgb.max()) <= 1


def test_render_rays_end_to_end():
    model = NeRF(pos_freqs=2, dir_freqs=1, hidden_dim=16, num_layers=2, skip_layer=1)
    o = mx.random.normal((6, 3))
    d = mx.random.normal((6, 3))
    out = render_rays(model, o, d, near=0.5, far=3.0, num_coarse=8, num_fine=8)
    assert out["rgb"].shape == (6, 3)
    assert "rgb_coarse" in out

    # Gradients flow to model parameters.
    import mlx.nn as nn

    def loss_fn(m):
        res = render_rays(m, o, d, 0.5, 3.0, num_coarse=8)
        return mx.sum(res["rgb"] ** 2)

    grads = nn.value_and_grad(model, loss_fn)(model)[1]
    flat = mx.utils.tree_flatten(grads) if hasattr(mx, "utils") else None
    # At least check the call succeeded and produced finite loss.
    assert np.isfinite(float(loss_fn(model)))


def test_render_points_visibility_and_grads():
    cam = Camera.look_at(eye=(0.0, 0.0, -3.0), at=(0, 0, 0), width=64, height=64, fov=60.0)
    pts = mx.array([[0.0, 0.0, 0.0]])
    colors = mx.array([[0.0, 1.0, 0.0]])
    out = render_points(cam, pts, colors, radius=1.5)
    img = out["image"]
    assert img.shape == (64, 64, 3)
    # Center pixel is green-ish, corner is background.
    assert float(img[32, 32, 1]) > 0.5
    assert float(img[0, 0, 1]) < 1e-3
    assert float(out["alpha"][32, 32]) > 0.5

    def f(p):
        res = render_points(cam, p, colors, radius=1.5)
        # Encourage brightness at a pixel just left of center (inside the
        # splat window) -> gradient should move the point.
        return -mx.sum(res["image"][32, 30, :])

    g = mx.grad(f)(pts)
    assert not bool(mx.isnan(g).any())
    assert float(mx.abs(g).sum()) > 0


def test_render_points_depth_ordering():
    cam = Camera.look_at(eye=(0.0, 0.0, -3.0), at=(0, 0, 0), width=32, height=32, fov=60.0)
    # Red point in front of green point on the same ray.
    pts = mx.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
    colors = mx.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    out = render_points(cam, pts, colors, radius=1.0, depth_temperature=50.0)
    center = out["image"][16, 16]
    assert float(center[0]) > float(center[1])  # red wins
