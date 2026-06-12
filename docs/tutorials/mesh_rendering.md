# Mesh Rendering and Extraction

MLX3D includes a differentiable soft mesh rasterizer for optimization loops,
UV texture rendering for OBJ assets, and scalar-field mesh extraction.

## Soft Rasterization

`render_mesh_soft` projects triangles through an MLX3D camera and blends them
with SoftRas-style coverage and depth weighting. The renderer is written in
MLX, so gradients flow to vertices, colors, and texture values.

```python
import mlx.core as mx
from mlx3d.cameras import Camera
from mlx3d.renderer import render_mesh_soft

cam = Camera.look_at(eye=(0, 0, -3), at=(0, 0, 0), width=256, height=256)
verts = mx.array([[-0.7, -0.6, 0.0], [0.7, -0.6, 0.0], [0.0, 0.7, 0.0]])
faces = mx.array([[0, 1, 2]], dtype=mx.int32)
face_colors = mx.array([[0.1, 0.8, 1.0]])

out = render_mesh_soft(cam, verts, faces, face_colors=face_colors, sigma=0.02)
image = out["image"]
alpha = out["alpha"]
```

Use small `sigma` values for sharper silhouettes and larger values for smoother
optimization gradients. This renderer targets differentiable losses and
medium-resolution previews; Gaussian Splatting remains the real-time viewer
path for large splat scenes.

## Textured OBJ Rendering

`load_obj` parses `mtllib` and common diffuse `map_Kd` textures. The returned
`ObjData` contains `texcoords`, `faces_texcoords_idx`, and `texture_image`
when they are present.

```python
from mlx3d.io import load_obj
from mlx3d.renderer import render_mesh_soft

obj = load_obj("asset.obj")
out = render_mesh_soft(
    cam,
    obj.verts,
    obj.faces,
    texcoords=obj.texcoords,
    faces_texcoords_idx=obj.faces_texcoords_idx,
    texture=obj.texture_image,
)
```

MLX3D uses OBJ UV convention for sampling: `v=0` is the bottom of the image.

## Mesh Extraction

`marching_cubes` extracts a mesh from a scalar grid and returns a `Meshes`
object. Internally it uses a marching-tetrahedra cube decomposition, which
avoids ambiguous cube cases while keeping the familiar marching-cubes API.

```python
import numpy as np
import mlx.core as mx
from mlx3d.ops import marching_cubes

xs = np.linspace(-1, 1, 64)
z, y, x = np.meshgrid(xs, xs, xs, indexing="ij")
sdf = x * x + y * y + z * z - 0.5

mesh = marching_cubes(
    mx.array(sdf),
    level=0.0,
    spacing=(2 / 63, 2 / 63, 2 / 63),
    origin=(-1, -1, -1),
)
print(mesh.verts_packed().shape, mesh.faces_packed().shape)
```
