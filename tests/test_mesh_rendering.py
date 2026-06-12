import mlx.core as mx
import numpy as np

from mlx3d.cameras import Camera
from mlx3d.renderer import render_mesh_soft, sample_texture


def test_soft_mesh_rasterizer_renders_triangle_and_gradients():
    cam = Camera.look_at(eye=(0.0, 0.0, -3.0), at=(0, 0, 0), width=48, height=48, fov=60.0)
    verts = mx.array([[-0.6, -0.5, 0.0], [0.6, -0.5, 0.0], [0.0, 0.6, 0.0]])
    faces = mx.array([[0, 1, 2]], dtype=mx.int32)
    colors = mx.array([[0.0, 0.8, 1.0]])

    out = render_mesh_soft(cam, verts, faces, face_colors=colors, sigma=0.02)
    assert out["image"].shape == (48, 48, 3)
    assert float(out["alpha"][24, 24]) > 0.4
    assert float(out["image"][24, 24, 2]) > 0.4

    def loss_fn(v):
        rendered = render_mesh_soft(cam, v, faces, face_colors=colors, sigma=0.02)
        return -mx.sum(rendered["alpha"][22:26, 22:26])

    grad = mx.grad(loss_fn)(verts)
    assert not bool(mx.isnan(grad).any())
    assert float(mx.abs(grad).sum()) > 0.0


def test_texture_sampling_and_textured_mesh_render():
    tex = mx.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
        ]
    )
    sampled = sample_texture(tex, mx.array([[0.0, 1.0], [1.0, 0.0]]))
    np.testing.assert_allclose(np.array(sampled[0]), [1.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(np.array(sampled[1]), [1.0, 1.0, 1.0], atol=1e-6)

    cam = Camera.look_at(eye=(0.0, 0.0, -3.0), at=(0, 0, 0), width=32, height=32, fov=60.0)
    verts = mx.array([[-0.7, -0.6, 0.0], [0.7, -0.6, 0.0], [0.0, 0.7, 0.0]])
    faces = mx.array([[0, 1, 2]], dtype=mx.int32)
    uv = mx.array([[0.0, 1.0], [1.0, 1.0], [0.5, 0.0]])
    out = render_mesh_soft(
        cam,
        verts,
        faces,
        texcoords=uv,
        faces_texcoords_idx=faces,
        texture=tex,
        sigma=0.03,
    )
    assert out["image"].shape == (32, 32, 3)
    assert float(out["alpha"].max()) > 0.5
