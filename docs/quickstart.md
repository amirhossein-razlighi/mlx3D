# Quickstart

## Install

```bash
pip install mlx3d
```

For development (with [uv](https://docs.astral.sh/uv/)):

```bash
git clone https://github.com/amirhossein-razlighi/mlx3D
cd mlx3D
uv sync              # creates .venv with all dev dependencies
uv run pytest tests/
```

## Meshes and point clouds

```python
import mlx.core as mx
from mlx3d.structures import Meshes, Pointclouds
from mlx3d.utils import ico_sphere, torus

sphere = ico_sphere(level=3)          # a Meshes batch with one mesh
print(sphere.verts_packed().shape)    # (642, 3)
print(sphere.faces_packed().shape)    # (1280, 3)

normals = sphere.verts_normals_packed()   # area-weighted vertex normals
areas = sphere.faces_areas_packed()

# Batches can mix meshes of different sizes:
batch = Meshes(
    verts=[sphere.verts_packed(), torus().verts_packed()],
    faces=[sphere.faces_packed(), torus().faces_packed()],
)
padded = batch.verts_padded()   # (2, max_V, 3), zero padded
```

Everything that depends on vertex positions is differentiable — you can
rebuild a `Meshes` from an optimized vertex array every iteration and
gradients flow through normals, areas and losses.

## Loading and saving 3D files

```python
from mlx3d.io import load_obj, load_ply, save_obj, save_ply

data = load_obj("bunny.obj")        # verts, faces, normals, texcoords, colors
save_ply("bunny.ply", data.verts, faces=data.faces)
```

## Cameras and rendering points

```python
from mlx3d.cameras import Camera
from mlx3d.renderer import render_points
from mlx3d.ops import sample_points_from_meshes

camera = Camera.look_at(eye=(2, 1, -3), at=(0, 0, 0), fov=60.0,
                        width=512, height=512)

points = sample_points_from_meshes(ico_sphere(3), 20_000)[0]
out = render_points(camera, points, radius=1.5)
image = out["image"]   # (512, 512, 3), differentiable w.r.t. points & colors
```

## Computing losses

```python
from mlx3d.losses import chamfer_distance, mesh_laplacian_smoothing

p1 = mx.random.normal((1, 1000, 3))
p2 = mx.random.normal((1, 1000, 3))
loss, _ = chamfer_distance(p1, p2)

smooth = mesh_laplacian_smoothing(sphere)
```

## Optimization loops in MLX

MLX is functional: compute gradients with `mx.value_and_grad` and apply them
with an optimizer. A minimal template used throughout the tutorials:

```python
import mlx.optimizers as optim

params = {"points": mx.random.normal((1000, 3))}
optimizer = optim.Adam(learning_rate=1e-2)
target = sample_points_from_meshes(torus(), 1000)[0]

def loss_fn(points):
    loss, _ = chamfer_distance(points, target)
    return loss

for step in range(200):
    loss, grads = mx.value_and_grad(loss_fn)(params["points"])
    params = optimizer.apply_gradients({"points": grads}, params)
    mx.eval(params["points"])   # evaluate the lazy graph once per step
```

!!! tip "MLX is lazy"
    Operations build a graph; nothing runs until `mx.eval` (or a value is
    inspected). Evaluate **once per iteration** — more often adds overhead,
    much less often grows the graph too large.

Next: read the [Conventions](conventions.md) page (camera axes, quaternion
order) — it will save you debugging time with real datasets.
