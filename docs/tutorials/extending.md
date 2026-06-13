# Extending MLX3D

MLX3D is built to be extended without ceremony. There are no renderer base
classes to subclass and no registry to register with — the library leans on a
single convention and ordinary Python callables.

## The renderer convention

Every image-space renderer in MLX3D — [`render_mesh_soft`][mlx3d.renderer.render_mesh_soft],
[`render_points`][mlx3d.renderer.render_points], and
[`GaussianModel.render`][mlx3d.splatting.GaussianModel] — is just a callable:

```python
renderer(camera, scene, **options) -> {"image": ..., "alpha": ..., "depth": ...}
```

That shape is captured by the [`Renderer`][mlx3d.renderer.Renderer] protocol and
the [`RenderOutput`][mlx3d.renderer.RenderOutput] dict. Because it is a
`typing.Protocol`, your own functions satisfy it automatically — you only use it
as a type hint, never as a parent class.

## Writing a custom renderer

Suppose you want a renderer that shades a mesh by its surface normals. Write a
function with the right shape and you are done — it is differentiable and works
with everything else (saving, the viewer, turntable loops):

```python
import mlx.core as mx
from mlx3d.cameras import Camera
from mlx3d.renderer import RenderOutput, render_mesh_soft
from mlx3d.structures import Meshes


def render_normals(camera: Camera, mesh: Meshes, **kwargs) -> RenderOutput:
    normals = mesh.verts_normals_packed()
    colors = 0.5 * normals + 0.5  # map [-1, 1] -> [0, 1]
    return render_mesh_soft(camera, mesh, verts_colors=colors, **kwargs)
```

Anything that accepts a `Renderer` now accepts `render_normals`:

```python
from mlx3d.io import save_image


def turntable(renderer, mesh, *, frames=8, size=256):
    for i in range(frames):
        cam = Camera.look_at(
            eye=(2.4 * mx.cos(mx.array(i / frames * 6.28)).item(), 1.4,
                 2.4 * mx.sin(mx.array(i / frames * 6.28)).item()),
            at=(0, 0, 0), fov=45.0, width=size, height=size,
        )
        out = renderer(cam, mesh, sigma=3e-3)
        save_image(f"frame_{i:02d}.png", out["image"])


turntable(render_normals, my_mesh)   # the same loop drives any renderer
```

A complete, runnable version of this is in
[`examples/extend_renderer.py`](https://github.com/amirhossein-razlighi/mlx3D/blob/main/examples/extend_renderer.py).

## Building your own pipeline from the pieces

If you need lower-level control, compose the public building blocks directly
instead of using a high-level renderer. For example, a custom volume renderer
that uses your own field:

```python
from mlx3d.renderer import sample_along_rays, volume_render

def render_my_field(camera, field, near, far, samples=96) -> dict:
    o, d = camera.generate_rays()
    h, w = o.shape[:2]
    pts, t = sample_along_rays(o.reshape(-1, 3), d.reshape(-1, 3), near, far, samples)
    density, rgb = field(pts)                       # your model / closure
    out = volume_render(density, rgb, t, d.reshape(-1, 3))
    return {"image": out["rgb"].reshape(h, w, 3), "alpha": out["acc"].reshape(h, w)}
```

The same approach works for the Gaussian Splatting stack: `project_gaussians`,
`bin_gaussians`, and `rasterize` are all exported from `mlx3d.splatting`, so you
can assemble a custom splatting pass while keeping the Metal kernels.

Everything stays differentiable end to end, so any pipeline you build this way
can be dropped straight into an MLX optimization loop.
