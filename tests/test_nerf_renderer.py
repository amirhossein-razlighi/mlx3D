import mlx.core as mx
import numpy as np

from mlx3d.cameras import Camera
from mlx3d.nn import (
    FusedMLP,
    HashGridEncoding,
    HashGridNeRF,
    NeRF,
    OccupancyGrid,
    PositionalEncoding,
    render_rays,
    render_rays_occupancy,
)
from mlx3d.renderer import render_points, sample_along_rays, sample_pdf, volume_render


def test_fused_mlp_kernel_matches_mlx_reference():
    mx.random.seed(0)
    mlp = FusedMLP([24, 64, 64, 8])
    x = mx.random.normal((5000, 24))
    ref = mlp(x)
    fused = mlp.forward_fused(x)
    mx.eval(ref, fused)
    assert ref.shape == (5000, 8)
    np.testing.assert_allclose(np.array(fused), np.array(ref), atol=1e-4)


def test_fused_mlp_is_trainable():
    import mlx.nn as nn

    mlp = FusedMLP([8, 32, 4])
    x = mx.random.normal((64, 8))
    target = mx.zeros((64, 4))

    def loss(m):
        return mx.mean((m(x) - target) ** 2)

    _, grads = nn.value_and_grad(mlp, loss)(mlp)
    mx.eval(grads)
    import mlx.utils as mu

    assert sum(float(mx.abs(v).sum()) for _, v in mu.tree_flatten(grads)) > 0


def assert_close(a, b, atol=1e-5):
    np.testing.assert_allclose(np.array(a), np.array(b), atol=atol)


def test_render_rays_occupancy_matches_dense_when_fully_occupied():
    # With a fully-occupied grid and full budget, occupancy rendering must
    # reproduce the dense render exactly (same samples, no skipping).
    mx.random.seed(0)
    model = HashGridNeRF(bounds=(-1.5, 1.5))
    o = mx.random.normal((128, 3))
    d = mx.random.normal((128, 3))
    d = d / mx.linalg.norm(d, axis=-1, keepdims=True)
    full = OccupancyGrid(resolution=8, bounds=(-100.0, 100.0))
    full.occupancy = mx.ones((8, 8, 8), dtype=mx.bool_)
    dense = render_rays(model, o, d, 2.0, 6.0, num_coarse=64, stratified=False)["rgb"]
    occ = render_rays_occupancy(
        model, o, d, 2.0, 6.0, full, num_samples=64, eval_fraction=1.0, stratified=False
    )["rgb"]
    np.testing.assert_allclose(np.array(occ), np.array(dense), atol=1e-5)


def test_render_rays_occupancy_gradient_flows():
    import mlx.nn as nn
    import mlx.utils as mu

    mx.random.seed(0)
    model = HashGridNeRF(bounds=(-1.5, 1.5))
    o = mx.random.normal((64, 3)) * 0.1
    d = mx.random.normal((64, 3))
    d = d / mx.linalg.norm(d, axis=-1, keepdims=True)
    grid = OccupancyGrid(resolution=32, bounds=(-1.5, 1.5))
    grid.update(lambda p: model(p, mx.zeros_like(p))[0], threshold=0.01)

    def loss(m):
        out = render_rays_occupancy(m, o, d, 1.5, 4.5, grid, num_samples=48, eval_fraction=0.5)
        return mx.mean(out["rgb"] ** 2)

    _, grads = nn.value_and_grad(model, loss)(model)
    total = sum(float(mx.abs(v).sum()) for _, v in mu.tree_flatten(grads))
    assert total > 0.0


def test_occupancy_grid_matches_analytic_sphere():
    def density(p):
        r = mx.linalg.norm(p, axis=-1)
        return mx.where(r < 0.6, 10.0, 0.0)

    grid = OccupancyGrid(resolution=64, bounds=(-1.5, 1.5))
    grid.update(density, threshold=1.0)
    # Occupied fraction ~ sphere volume / box volume.
    expected = (4.0 / 3.0 * np.pi * 0.6**3) / (3.0**3)
    assert abs(grid.occupied_fraction - expected) < 0.005
    # Point queries: inside occupied, outside/empty not.
    q = grid.query(mx.array([[0.0, 0, 0], [1.4, 1.4, 1.4], [0.5, 0, 0], [0.65, 0, 0]]))
    assert [bool(x) for x in q] == [True, False, True, False]
    # Out-of-bounds is empty.
    assert not bool(grid.query(mx.array([[5.0, 5, 5]]))[0])


def test_hashgrid_nerf_forward_and_render_rays():
    model = HashGridNeRF(bounds=(-1.0, 1.0), num_levels=4, log2_hashmap_size=14)
    pts = mx.random.uniform(low=-1, high=1, shape=(32, 8, 3))
    dirs = mx.broadcast_to(mx.array([0.0, 0.0, 1.0]), pts.shape)
    density, rgb = model(pts, dirs)
    assert density.shape == (32, 8)
    assert rgb.shape == (32, 8, 3)
    assert bool((rgb >= 0).all() and (rgb <= 1).all())  # sigmoid output
    # Drops into render_rays unchanged.
    o = mx.random.normal((16, 3))
    d = mx.array([[0.0, 0.0, 1.0]] * 16)
    out = render_rays(model, o, d, 2.0, 6.0, num_coarse=16)
    assert out["rgb"].shape == (16, 3)


def test_hashgrid_nerf_density_zero_outside_bounds():
    # Density must vanish outside the scene AABB (the fix that lets the hash
    # grid localize geometry instead of collapsing to the background).
    model = HashGridNeRF(bounds=(-1.0, 1.0))
    inside = mx.zeros((4, 3))  # origin, inside
    outside = mx.full((4, 3), 5.0)  # far outside the cube
    pts = mx.concatenate([inside, outside])[:, None, :]
    dirs = mx.broadcast_to(mx.array([0.0, 0.0, 1.0]), pts.shape)
    density, _ = model(pts, dirs)
    assert float(mx.abs(density[4:]).max()) == 0.0  # outside -> exactly zero
    assert float(density[:4].max()) > 0.0  # inside -> nonzero (trunc-exp)


def test_positional_encoding_shapes_and_values():
    pe = PositionalEncoding(num_freqs=4)
    x = mx.random.normal((10, 3))
    out = pe(x)
    assert out.shape == (10, 3 * (2 * 4 + 1))
    assert_close(out[:, :3], x)


def test_hash_grid_encoding_shape_and_gradients():
    enc = HashGridEncoding(
        num_levels=3,
        features_per_level=2,
        log2_hashmap_size=5,
        base_resolution=4,
        finest_resolution=16,
    )
    x = mx.random.uniform(shape=(8, 3), low=-1.0, high=1.0)
    out = enc(x)
    assert out.shape == (8, 6)

    def loss_fn(points):
        return mx.sum(enc(points) ** 2)

    grad = mx.grad(loss_fn)(x)
    assert not bool(mx.isnan(grad).any())


def test_hash_grid_table_params_are_trainable():
    """Hash-grid tables must receive gradients via nn.value_and_grad."""
    import mlx.nn as nn

    enc = HashGridEncoding(
        num_levels=4,
        features_per_level=2,
        log2_hashmap_size=6,
        base_resolution=4,
        finest_resolution=32,
    )
    x = mx.random.uniform(shape=(16, 3), low=-1.0, high=1.0)
    target = mx.zeros((16, 8))

    def loss_fn(model):
        return mx.mean((model(x) - target) ** 2)

    # Confirm tables are in parameters()
    params = enc.parameters()
    assert "tables" in params, "HashGridEncoding.tables missing from parameters()"
    assert len(params["tables"]) == 4

    loss, grads = nn.value_and_grad(enc, loss_fn)(enc)
    mx.eval(grads)
    table_grads = grads["tables"]
    assert len(table_grads) == 4
    for i, g in enumerate(table_grads):
        assert g is not None, f"table[{i}] gradient is None"
        assert not bool(mx.isnan(g).any()), f"NaN in table[{i}] gradient"
        assert float(mx.abs(g).max()) > 0, f"table[{i}] gradient is all zeros"


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

    loss, grads = nn.value_and_grad(model, loss_fn)(model)
    mx.eval(grads)
    assert np.isfinite(float(loss))


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
