"""Unit tests for the hard mesh rasterizer and shading."""

import mlx.core as mx
import numpy as np

from mlx3d.cameras import Camera
from mlx3d.renderer import (
    AmbientLights,
    DirectionalLights,
    PointLights,
    interpolate_face_attributes,
    rasterize_meshes,
    render_mesh,
)
from mlx3d.utils import ico_sphere, torus


def _tri_cam():
    cam = Camera.look_at(eye=(0, 0, -3.0), at=(0, 0, 0), fov=60.0, width=32, height=32)
    verts = mx.array([[-0.6, -0.5, 0.0], [0.6, -0.5, 0.0], [0.0, 0.6, 0.0]])
    faces = mx.array([[0, 1, 2]], dtype=mx.int32)
    return cam, verts, faces


def test_rasterize_triangle_hit_and_miss():
    cam, verts, faces = _tri_cam()
    frag = rasterize_meshes(cam, verts, faces)
    assert int(frag.pix_to_face[16, 16]) == 0  # center hits the triangle
    assert int(frag.pix_to_face[0, 0]) == -1  # corner is empty
    assert not bool(frag.valid[0, 0])


def test_barycentrics_sum_to_one_on_hits():
    cam, verts, faces = _tri_cam()
    frag = rasterize_meshes(cam, verts, faces)
    s = mx.sum(frag.bary, axis=-1)
    covered = frag.valid
    err = mx.abs(mx.where(covered, s, 1.0) - 1.0)
    assert float(err.max()) < 1e-4


def test_depth_ordering_near_face_wins():
    cam = Camera.look_at(eye=(0, 0, -3.0), at=(0, 0, 0), fov=60.0, width=16, height=16)
    verts = mx.array(
        [
            [-0.6, -0.5, 0.5],
            [0.6, -0.5, 0.5],
            [0.0, 0.6, 0.5],  # far
            [-0.6, -0.5, -0.5],
            [0.6, -0.5, -0.5],
            [0.0, 0.6, -0.5],  # near
        ]
    )
    faces = mx.array([[0, 1, 2], [3, 4, 5]], dtype=mx.int32)
    frag = rasterize_meshes(cam, verts, faces)
    assert int(frag.pix_to_face[8, 8]) == 1  # nearer triangle


def test_interpolate_face_attributes_recovers_vertex_colors():
    cam, verts, faces = _tri_cam()
    frag = rasterize_meshes(cam, verts, faces)
    colors = mx.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    img = interpolate_face_attributes(frag, colors)
    # Interpolated color is a convex combo of the three on covered pixels.
    covered = frag.valid
    csum = mx.sum(img, axis=-1)
    err = mx.abs(mx.where(covered, csum, 1.0) - 1.0)
    assert float(err.max()) < 1e-4
    assert float(img.min()) >= -1e-5


def test_rasterizer_gradient_flows_to_vertices():
    mesh = ico_sphere(level=2, radius=1.0)
    cam = Camera.look_at(eye=(2.2, 1.6, 2.2), at=(0, 0, 0), fov=45.0, width=48, height=48)
    v, f = mesh.verts_packed(), mesh.faces_packed()
    vc = 0.5 * v + 0.5

    def loss(v_):
        frag = rasterize_meshes(cam, v_, f)
        return mx.sum(interpolate_face_attributes(frag, vc))

    g = mx.grad(loss)(v)
    assert bool(mx.isfinite(g).all())
    assert float(mx.abs(g).sum()) > 0.0


def test_render_mesh_outputs_and_no_nan():
    mesh = ico_sphere(level=3, radius=1.0)
    cam = Camera.look_at(eye=(2.4, 1.8, 2.4), at=(0, 0, 0), fov=45.0, width=64, height=64)
    out = render_mesh(
        cam, mesh, lights=[PointLights(location=(3, 3, -2)), AmbientLights(color=(0.2, 0.2, 0.2))]
    )
    for k in ("image", "alpha", "depth", "normals"):
        assert k in out
        assert not bool(mx.isnan(out[k]).any())
    assert out["image"].shape == (64, 64, 3)
    assert float(out["alpha"].mean()) > 0.1
    # Depth is positive on covered pixels.
    covered = out["alpha"] > 0.5
    assert float(mx.where(covered, out["depth"], 1.0).min()) > 0.0


def test_two_sided_shading_lights_inward_wound_mesh():
    # The torus primitive winds normals inward; two-sided shading must still
    # light its camera-facing surface rather than render it black.
    tor = torus(r=0.4, R=1.0, sides=24, rings=48)
    cam = Camera.look_at(eye=(2.4, 1.8, 2.4), at=(0, 0, 0), fov=45.0, width=64, height=64)
    out = render_mesh(
        cam,
        tor,
        lights=[
            DirectionalLights(direction=(-1, -1.2, -0.6)),
            AmbientLights(color=(0.2, 0.2, 0.2)),
        ],
        verts_colors=mx.full((tor.verts_packed().shape[0], 3), 0.6),
    )
    img, alpha = np.array(out["image"]), np.array(out["alpha"])
    lit = img[alpha > 0.5]
    assert float(lit.mean()) > 0.25  # clearly more than pure ambient on black


def test_textured_render_samples_texture():
    # A 2x2 texture mapped onto a full-UV quad: corners read the corner texels.
    tex = mx.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
        ]
    )
    verts = mx.array([[-1.0, -1, 0], [1.0, -1, 0], [1.0, 1, 0], [-1.0, 1, 0]])
    faces = mx.array([[0, 1, 2], [0, 2, 3]], dtype=mx.int32)
    verts_uvs = mx.array([[0.0, 0], [1.0, 0], [1.0, 1], [0.0, 1]])
    cam = Camera.look_at(eye=(0, 0, -3.0), at=(0, 0, 0), fov=50.0, width=64, height=64)
    out = render_mesh(
        cam, verts, faces, texture=tex, verts_uvs=verts_uvs, faces_uvs=faces, shading="none"
    )
    assert not bool(mx.isnan(out["image"]).any())
    img, alpha = np.array(out["image"]), np.array(out["alpha"])
    assert alpha.mean() > 0.3  # the quad is visible
    # The rendered colors come from the texture palette, not mid-grey default.
    assert img[alpha > 0.5].std() > 0.1

    # Missing UVs is an error.
    try:
        render_mesh(cam, verts, faces, texture=tex)
        raise AssertionError("expected ValueError without UVs")
    except ValueError:
        pass


def test_shading_none_returns_albedo():
    mesh = ico_sphere(level=2, radius=1.0)
    cam = Camera.look_at(eye=(0, 0, -3.0), at=(0, 0, 0), fov=60.0, width=32, height=32)
    vc = mx.full((mesh.verts_packed().shape[0], 3), mx.array([0.2, 0.5, 0.9]))
    out = render_mesh(cam, mesh, verts_colors=vc, shading="none", background=0.0)
    img, alpha = np.array(out["image"]), np.array(out["alpha"])
    lit = img[alpha > 0.5]
    # Unlit albedo: covered pixels carry the flat color, no lighting modulation.
    np.testing.assert_allclose(lit.mean(axis=0), [0.2, 0.5, 0.9], atol=0.05)
