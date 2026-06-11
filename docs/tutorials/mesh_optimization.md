# Mesh Optimization

Deform a sphere into a target shape by optimizing per-vertex offsets — the
classic differentiable-geometry "hello world". Full script:
[`examples/fit_mesh.py`](https://github.com/amirhossein-razlighi/mlx3D/blob/main/examples/fit_mesh.py).

## Setup

We start from an icosphere and fit a torus (or any OBJ you pass in):

```python
import mlx.core as mx
import mlx.optimizers as optim

from mlx3d.structures import Meshes
from mlx3d.utils import ico_sphere, torus
from mlx3d.ops import sample_points_from_meshes
from mlx3d.losses import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)

target = torus(r=0.4, R=1.0, sides=24, rings=48)
src = ico_sphere(level=3)

verts0 = src.verts_packed()       # (V, 3) fixed initial vertices
faces = src.faces_list()          # topology never changes
offsets = mx.zeros(verts0.shape)  # the parameter we optimize
```

## The loss

The data term is a **chamfer distance** between points sampled on the two
surfaces. Three regularizers keep the mesh well-behaved:

- `mesh_edge_loss` — penalizes long edges (uniform triangle sizes),
- `mesh_laplacian_smoothing` — pulls vertices toward their neighbors' centroid,
- `mesh_normal_consistency` — penalizes sharp dihedral angles.

```python
def loss_fn(offsets):
    mesh = Meshes([verts0 + offsets], faces)
    pts_src = sample_points_from_meshes(mesh, 3000)
    pts_tgt = sample_points_from_meshes(target, 3000)
    cham, _ = chamfer_distance(pts_src, pts_tgt)
    return (
        cham
        + 0.1 * mesh_edge_loss(mesh)
        + 0.1 * mesh_laplacian_smoothing(mesh)
        + 0.01 * mesh_normal_consistency(mesh)
    )
```

Note that the `Meshes` object is rebuilt **inside** the loss function:
construction is differentiable, so gradients flow from sampled points back
to `offsets`.

## The loop

```python
optimizer = optim.Adam(learning_rate=1e-2)
state = {"offsets": offsets}

for it in range(500):
    loss, grads = mx.value_and_grad(loss_fn)(state["offsets"])
    state = optimizer.apply_gradients({"offsets": grads}, state)
    mx.eval(state["offsets"])
    if it % 50 == 0:
        print(f"iter {it:4d}  loss {float(loss):.5f}")
```

Save the result:

```python
from mlx3d.io import save_obj

fitted = Meshes([verts0 + state["offsets"]], faces)
save_obj("fitted.obj", fitted.verts_packed(), fitted.faces_packed())
```

On an M-series Mac this converges in a few seconds. Try raising the
icosphere level for more detail, or lowering the Laplacian weight to fit
sharper features (at the cost of noisier surfaces).
