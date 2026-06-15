"""Behavior tests: gradients flow and optimization reduces loss end to end.

These span multiple modules (cameras + renderers + losses + optimizers) and act
as integration coverage for the differentiable pipeline.
"""

import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from mlx3d.cameras import Camera, refine_camera
from mlx3d.nn import HashGridNeRF, NeRF, render_rays
from mlx3d.ops import marching_cubes
from mlx3d.renderer import (
    interpolate_face_attributes,
    rasterize_meshes,
    render_mesh,
    render_mesh_soft,
    sample_along_rays,
    volume_render,
)
from mlx3d.splatting import GaussianModel, GaussianTrainer, TrainerConfig
from mlx3d.utils import ico_sphere


def test_multiface_mesh_renders_without_nan():
    """Regression: a closed mesh (many faces) must not produce NaNs.

    Faces far from a pixel get out-of-triangle barycentric coords, which can
    push ``z_face`` below ``znear`` and overflow the depth weighting; the
    rasterizer must stay finite and produce real coverage.
    """
    mesh = ico_sphere(level=3, radius=1.0)
    verts = mesh.verts_packed()
    colors = 0.5 * verts / mx.maximum(mx.linalg.norm(verts, axis=-1, keepdims=True), 1e-6) + 0.5
    cam = Camera.look_at(eye=(2.2, 1.6, 2.2), at=(0, 0, 0), fov=45.0, width=64, height=64)

    out = render_mesh_soft(cam, mesh, verts_colors=colors, sigma=3e-3, background=0.0)
    mx.eval(out["image"], out["alpha"], out["depth"])
    for key in ("image", "alpha", "depth"):
        assert not bool(mx.isnan(out[key]).any()), f"NaN in {key}"
    assert float(out["alpha"].mean()) > 0.05  # the sphere is actually visible


def test_mesh_color_optimization_reduces_loss():
    cam = Camera.look_at(eye=(0, 0, -3.0), at=(0, 0, 0), fov=60.0, width=32, height=32)
    verts = mx.array([[-0.7, -0.6, 0.0], [0.7, -0.6, 0.0], [0.0, 0.7, 0.0]])
    faces = mx.array([[0, 1, 2]], dtype=mx.int32)
    # Target is a real render with a known color, so it is exactly reachable.
    target = render_mesh_soft(
        cam, verts, faces, face_colors=mx.array([[0.2, 0.8, 0.3]]), sigma=0.02
    )["image"]

    opt = optim.Adam(learning_rate=0.1)

    def loss_fn(color):
        out = render_mesh_soft(cam, verts, faces, face_colors=color, sigma=0.02)
        return mx.mean((out["image"] - target) ** 2)

    losses = []
    color = mx.array([[0.5, 0.5, 0.5]])
    lg = mx.value_and_grad(loss_fn)
    for _ in range(20):
        loss, grad = lg(color)
        color = opt.apply_gradients({"c": grad}, {"c": color})["c"]
        mx.eval(color)
        losses.append(float(loss))
    assert losses[-1] < 0.2 * losses[0]


def test_volume_render_color_fit_reduces_loss():
    """Optimizing per-sample colors through the volume renderer fits a target."""
    rays, samples = 64, 16
    densities = mx.ones((rays, samples)) * 5.0  # high opacity -> strong color signal
    t_vals = mx.broadcast_to(mx.linspace(0.0, 1.0, samples)[None], (rays, samples))
    target = mx.broadcast_to(mx.array([0.9, 0.1, 0.4])[None], (rays, 3))

    colors = mx.zeros((rays, samples, 3)) + 0.5
    opt = optim.Adam(learning_rate=0.2)

    def loss_fn(colors):
        out = volume_render(densities, colors, t_vals)
        return mx.mean((out["rgb"] - target) ** 2)

    lg = mx.value_and_grad(loss_fn)
    first = last = None
    for i in range(60):
        loss, grad = lg(colors)
        colors = opt.apply_gradients({"c": grad}, {"c": colors})["c"]
        mx.eval(colors)
        if i == 0:
            first = float(loss)
        last = float(loss)
    assert last < 0.2 * first


def test_nerf_render_rays_trains_a_few_steps():
    """A tiny NeRF reduces its loss on synthetic volume-rendered rays."""
    mx.random.seed(0)
    near, far = 1.0, 4.0
    cam = Camera.look_at(eye=(0, 0, 2.8), at=(0, 0, 0), fov=45.0, width=24, height=24)
    o, d = cam.generate_rays()
    o, d = o.reshape(-1, 3), d.reshape(-1, 3)
    pts, t = sample_along_rays(o, d, near, far, 48, stratified=False)
    r = mx.linalg.norm(pts, axis=-1)
    density = 30.0 * mx.maximum(0.6 - mx.abs(r - 0.6), 0.0)
    rgb = 0.5 * pts / mx.maximum(r[..., None], 1e-6) + 0.5
    target = volume_render(density, rgb, t, d)["rgb"]

    model = NeRF(pos_freqs=4, dir_freqs=2, hidden_dim=32, num_layers=3, skip_layer=1)
    opt = optim.Adam(learning_rate=2e-3)

    def loss_fn(model):
        out = render_rays(model, o, d, near, far, num_coarse=32)
        return mx.mean((out["rgb"] - target) ** 2)

    lg = nn.value_and_grad(model, loss_fn)
    first = float(loss_fn(model))
    for _ in range(40):
        loss, grads = lg(model)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)
    assert float(loss) < 0.8 * first
    assert not bool(mx.isnan(loss))


def test_hashgrid_nerf_converges_fast_on_synthetic():
    """The hash-grid NeRF should fit synthetic rays quickly and reach a low loss."""
    mx.random.seed(0)
    near, far = 1.0, 4.0
    cam = Camera.look_at(eye=(0, 0, 2.8), at=(0, 0, 0), fov=45.0, width=24, height=24)
    o, d = cam.generate_rays()
    o, d = o.reshape(-1, 3), d.reshape(-1, 3)
    pts, t = sample_along_rays(o, d, near, far, 48, stratified=False)
    r = mx.linalg.norm(pts, axis=-1)
    density = 30.0 * mx.maximum(0.6 - mx.abs(r - 0.6), 0.0)
    rgb = 0.5 * pts / mx.maximum(r[..., None], 1e-6) + 0.5
    target = volume_render(density, rgb, t, d)["rgb"]

    model = HashGridNeRF(
        bounds=(-1.5, 1.5), num_levels=8, log2_hashmap_size=15, finest_resolution=256
    )
    opt = optim.Adam(learning_rate=5e-3)

    def loss_fn(model):
        out = render_rays(model, o, d, near, far, num_coarse=48)
        return mx.mean((out["rgb"] - target) ** 2)

    lg = nn.value_and_grad(model, loss_fn)
    first = float(loss_fn(model))
    for _ in range(60):
        loss, grads = lg(model)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)
    assert float(loss) < 0.3 * first  # converges quickly
    assert not bool(mx.isnan(loss))


def test_gaussian_trainer_step_reduces_loss():
    mx.random.seed(0)
    cam = Camera.look_at(eye=(0, 0, 2.8), at=(0, 0, 0), fov=45.0, width=32, height=32)
    o, d = cam.generate_rays()
    o, d = o.reshape(-1, 3), d.reshape(-1, 3)
    pts, t = sample_along_rays(o, d, 1.0, 4.0, 48, stratified=False)
    r = mx.linalg.norm(pts, axis=-1)
    density = 30.0 * mx.maximum(0.6 - mx.abs(r - 0.6), 0.0)
    rgb = 0.5 * pts / mx.maximum(r[..., None], 1e-6) + 0.5
    target = volume_render(density, rgb, t, d)["rgb"].reshape(32, 32, 3)

    key = mx.random.normal((500, 3))
    points = key / mx.maximum(mx.linalg.norm(key, axis=-1, keepdims=True), 1e-6) * 0.6
    model = GaussianModel.from_points(points, mx.random.uniform(shape=(500, 3)), sh_degree=0)
    trainer = GaussianTrainer(model, TrainerConfig(densify_from=10_000), scene_extent=1.0)

    first = trainer.step(cam, target)["loss"]
    for _ in range(20):
        info = trainer.step(cam, target)
    assert info["loss"] < first


def test_hard_rasterizer_faster_than_soft():
    """The hard z-buffer rasterizer must be substantially faster than the soft one."""
    mesh = ico_sphere(level=3, radius=1.0)
    v, f = mesh.verts_packed(), mesh.faces_packed()
    vc = 0.5 * v + 0.5
    cam = Camera.look_at(eye=(2.2, 1.6, 2.2), at=(0, 0, 0), fov=45.0, width=160, height=160)

    def hard():
        frag = rasterize_meshes(cam, v, f)
        mx.eval(interpolate_face_attributes(frag, vc))

    def soft():
        mx.eval(render_mesh_soft(cam, mesh, verts_colors=vc, sigma=3e-3)["image"])

    hard()
    soft()  # warmup / compile
    t = time.perf_counter()
    for _ in range(3):
        hard()
    th = time.perf_counter() - t
    t = time.perf_counter()
    for _ in range(3):
        soft()
    ts = time.perf_counter() - t
    assert th < ts, f"hard ({th:.3f}s) should beat soft ({ts:.3f}s)"
    # Comfortable margin so the test is meaningful, not just noise.
    assert ts / th > 3.0


def test_render_mesh_color_optimization_reduces_loss():
    mesh = ico_sphere(level=2, radius=1.0)
    cam = Camera.look_at(eye=(2.2, 1.6, 2.2), at=(0, 0, 0), fov=45.0, width=48, height=48)
    target = render_mesh(
        cam,
        mesh,
        verts_colors=mx.full((mesh.verts_packed().shape[0], 3), mx.array([0.2, 0.7, 0.4])),
        shading="none",
    )["image"]

    opt = optim.Adam(learning_rate=0.1)
    color = mx.full((mesh.verts_packed().shape[0], 3), 0.5)

    def loss_fn(c):
        out = render_mesh(cam, mesh, verts_colors=c, shading="none")
        return mx.mean((out["image"] - target) ** 2)

    lg = mx.value_and_grad(loss_fn)
    first = last = None
    for i in range(25):
        loss, grad = lg(color)
        color = opt.apply_gradients({"c": grad}, {"c": color})["c"]
        mx.eval(color)
        if i == 0:
            first = float(loss)
        last = float(loss)
    assert last < 0.25 * first


def test_pose_optimization_recovers_perturbed_camera():
    """Optimizing an SE(3) twist through refine_camera recovers a known pose."""
    mx.random.seed(0)
    true_cam = Camera.look_at(eye=(0.4, 0.3, -3.0), at=(0, 0, 0), fov=50.0, width=128, height=128)
    pts = mx.random.normal((40, 3)) * 0.6
    target_xy, _ = true_cam.project_points(pts)
    # Start from a wrongly-posed camera.
    perturbed = refine_camera(true_cam, mx.array([0.15, -0.1, 0.2, 0.08, -0.05, 0.06]))

    def loss(xi):
        xy, _ = refine_camera(perturbed, xi).project_points(pts)
        return mx.mean((xy - target_xy) ** 2)

    xi = mx.zeros((6,))
    opt = optim.Adam(learning_rate=0.05)
    first = float(loss(xi))
    for _ in range(200):
        loss_val, g = mx.value_and_grad(loss)(xi)
        xi = opt.apply_gradients({"x": g}, {"x": xi})["x"]
        mx.eval(xi)
    assert float(loss(xi)) < 0.05 * first  # reprojection error collapses


def test_marching_cubes_recovers_sphere_extent():
    n = 32
    lin = mx.linspace(-1.5, 1.5, n)
    z, y, x = mx.meshgrid(lin, lin, lin, indexing="ij")
    sdf = mx.sqrt(x**2 + y**2 + z**2) - 1.0
    spacing = 3.0 / (n - 1)
    mesh = marching_cubes(sdf, level=0.0, spacing=(spacing,) * 3, origin=(-1.5, -1.5, -1.5))
    verts = mesh.verts_packed()
    assert verts.shape[0] > 0
    radii = mx.linalg.norm(verts, axis=-1)
    # All surface vertices lie ~on the unit sphere.
    assert abs(float(radii.mean()) - 1.0) < 0.1
    assert float(radii.max()) < 1.2
