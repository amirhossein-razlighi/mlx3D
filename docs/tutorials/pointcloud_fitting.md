# Point Cloud Fitting

Optimize raw 3D points to match a target distribution — the smallest
end-to-end differentiable 3D pipeline, and a good test that your install
works. Full script:
[`examples/fit_pointcloud.py`](https://github.com/amirhossein-razlighi/mlx3D/blob/main/examples/fit_pointcloud.py).

## Chamfer distance

The bidirectional chamfer distance is the workhorse loss for point sets: for
every point in each cloud, the squared distance to its nearest neighbor in
the other cloud. MLX3D computes it with a tiled brute-force k-NN that runs
entirely on the GPU ([`knn_points`](../api/ops.md)) — no KD-tree needed at
these sizes.

```python
import mlx.core as mx
import mlx.optimizers as optim

from mlx3d.losses import chamfer_distance
from mlx3d.ops import sample_points_from_meshes
from mlx3d.utils import torus

target = sample_points_from_meshes(torus(r=0.4, R=1.0), 5000)[0]  # (5000, 3)
points = mx.random.normal((5000, 3)) * 0.5                        # random init

optimizer = optim.Adam(learning_rate=2e-2)

def loss_fn(points):
    loss, _ = chamfer_distance(points, target)
    return loss

state = {"points": points}
for it in range(300):
    loss, grads = mx.value_and_grad(loss_fn)(state["points"])
    state = optimizer.apply_gradients({"points": grads}, state)
    mx.eval(state["points"])
```

After ~300 iterations the random blob has collapsed onto the torus surface.
Save and inspect it:

```python
from mlx3d.io import save_ply
save_ply("fitted_points.ply", state["points"])
```

## With normals

If both clouds carry normals, `chamfer_distance` can also return a normal
agreement term (`1 - |cos|` between matched normals):

```python
loss, loss_normals = chamfer_distance(
    x, y, x_normals=nx, y_normals=ny
)
total = loss + 0.1 * loss_normals
```

## Where to go next

- Use the [`Pointclouds`](../api/structures.md) structure to handle batches
  of variable-sized clouds (packed/padded views).
- Render your clouds differentiably with
  [`render_points`](../api/renderer.md) and optimize against images instead
  of 3D targets.
- For photorealistic view synthesis from points, jump to
  [Gaussian Splatting](gaussian_splatting.md).
