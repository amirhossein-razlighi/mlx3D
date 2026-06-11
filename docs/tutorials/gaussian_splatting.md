# Gaussian Splatting

Train and render [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
scenes on Apple Silicon. MLX3D ships a Metal translation of the reference
CUDA rasterizer — tile-based forward **and** backward kernels — wrapped in
`mx.custom_function` so the whole pipeline is differentiable end to end.
Full script:
[`examples/train_gaussian_splatting.py`](https://github.com/amirhossein-razlighi/mlx3D/blob/main/examples/train_gaussian_splatting.py).

## How the renderer works

1. **Projection** (pure MLX, autodiff): each Gaussian's quaternion + scale
   build a 3D covariance \( \Sigma = R S S^T R^T \), which the EWA splatting
   approximation maps to a 2D screen-space covariance
   \( \Sigma' = J W \Sigma W^T J^T \).
2. **Tile binning**: Gaussians are assigned to all 16×16 pixel tiles their
   3σ extent touches and sorted by (tile, depth) — fully vectorized MLX.
3. **Rasterization** (Metal kernels): one threadgroup per tile streams its
   depth-sorted Gaussians through threadgroup memory while 256 threads
   alpha-composite their pixels front-to-back with early termination. The
   backward kernel traverses back-to-front and accumulates gradients with
   atomic adds — the same design as the reference implementation.

A pure-MLX reference renderer (`render_gaussians_reference`) verifies the
kernels: forward images match to float32 precision and all gradients match
the autodiff oracle to ~1e-7 relative error (see `tests/test_splatting.py`).

## Rendering

```python
import mlx.core as mx
from mlx3d.cameras import Camera
from mlx3d.splatting import render_gaussians

camera = Camera.look_at(eye=(0, 0, -4), at=(0, 0, 0), width=1280, height=720)

out = render_gaussians(
    camera,
    means,        # (N, 3)
    quats,        # (N, 4) (w, x, y, z)
    scales,       # (N, 3) standard deviations (post-activation)
    opacities,    # (N,) in [0, 1]
    colors=rgb,   # or sh=... for view-dependent color
)
out["image"]   # (720, 1280, 3) — differentiable w.r.t. all inputs
out["alpha"]   # (720, 1280) accumulated opacity — also differentiable
```

On an M-series GPU, 100k Gaussians render at ~30 FPS at 720p, with a full
forward+backward training step around 100 ms.

## Training on a COLMAP scene

Any dataset prepared for the original 3DGS works as-is (a `sparse/0`
reconstruction plus an `images/` folder):

```bash
python examples/train_gaussian_splatting.py --data /path/to/scene --iters 7000
```

What the script does, in code:

```python
from mlx3d.datasets import load_colmap
from mlx3d.splatting import GaussianModel, GaussianTrainer, TrainerConfig

ds = load_colmap("scene/")                       # cameras, images, SfM points
model = GaussianModel.from_points(ds.points, ds.point_colors)
trainer = GaussianTrainer(model, TrainerConfig(), scene_extent=ds.scene_extent)

for it in range(7000):
    cam, img = ds[it % len(ds)]                  # shuffle in practice
    info = trainer.step(cam, img)                # L1 + D-SSIM, ADC, SH growth

model.save_ply("point_cloud.ply")                # standard 3DGS checkpoint
```

`GaussianTrainer` implements the paper's recipe:

- per-parameter Adam learning rates (positions scaled by scene extent),
- loss \( (1-\lambda)\,L_1 + \lambda\,(1 - \text{SSIM}) \) with \( \lambda = 0.2 \),
- **adaptive density control** — screen-space positional gradients are
  accumulated per Gaussian; high-gradient small Gaussians are cloned,
  high-gradient large ones split, transparent/oversized ones pruned,
- periodic opacity resets and progressive SH degree growth.

## Blender-synthetic scenes

No SfM points? Initialize randomly:

```bash
python examples/train_gaussian_splatting.py \
    --data nerf_synthetic/lego --format blender --iters 7000
```

## Inspecting the result

Open any checkpoint in the [interactive viewer](../viewer.md) — orbit, pan
and zoom in the browser, rendered live by the Metal kernels:

```bash
mlx3d-view outputs/gs/point_cloud.ply
```

## Checkpoints

`save_ply` / `GaussianModel.load_ply` use the standard 3DGS PLY layout
(`f_dc_*`, `f_rest_*`, `opacity`, `scale_*`, `rot_*`), so checkpoints open
directly in common viewers (SuperSplat, Polycam viewer, gsplat tools, ...)
and you can fine-tune checkpoints trained elsewhere.

!!! note "Current limitations"
    - Densification rebuilds optimizer state (the reference implementation
      preserves Adam moments through clone/split).
    - One camera per `step` call; no per-step learning-rate schedule yet
      (the paper decays the position LR exponentially).
