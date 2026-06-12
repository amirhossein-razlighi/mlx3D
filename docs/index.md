# MLX3D

**Differentiable 3D computer vision on Apple Silicon, built on [MLX](https://github.com/ml-explore/mlx).**

MLX3D brings the PyTorch3D workflow to Macs: batched 3D data structures, cameras, differentiable rendering, and modern view-synthesis methods (NeRF and 3D Gaussian Splatting) — all running natively on the Apple GPU through MLX, with the Gaussian Splatting rasterizer implemented as custom Metal kernels.

## Highlights

- **Data structures** — batched [`Meshes`](api/structures.md) and [`Pointclouds`](api/structures.md) with list / packed / padded views, differentiable through construction.
- **Cameras & transforms** — OpenCV-convention pinhole [`Camera`](api/cameras.md) (drop-in for COLMAP / NeRF datasets) and a full set of [rotation conversions](api/transforms.md).
- **Geometry ops & losses** — k-NN, chamfer distance, surface sampling, Laplacian smoothing, edge length and normal consistency losses, PSNR/SSIM.
- **NeRF** — positional encoding, the NeRF MLP, stratified + hierarchical sampling and volume rendering, with a training script for the Blender synthetic scenes.
- **Mesh rendering** — differentiable soft triangle rasterization, OBJ/MTL texture loading and UV sampling, plus scalar-field mesh extraction.
- **3D Gaussian Splatting** — a Metal translation of the reference CUDA rasterizer: tile-based forward and backward kernels wrapped in `mx.custom_function`, EWA projection, spherical harmonics, adaptive density control, COLMAP loading, and standard `.ply` checkpoints viewable in any splat viewer.
- **Interactive viewer** — [`mlx3d-view`](viewer.md) opens any 3DGS checkpoint in the browser with orbit/pan/zoom, rendered live by the Metal kernels; NeRFs are supported too.
- **IO** — OBJ and PLY (ascii/binary) loading and saving, including 3DGS checkpoint layouts.

## Installation

```bash
pip install mlx3d
```

Requires an Apple Silicon Mac (M1 or newer) and Python ≥ 3.10.

## A 60-second taste

```python
import mlx.core as mx
from mlx3d.cameras import Camera
from mlx3d.splatting import GaussianModel

# Random colored Gaussians...
model = GaussianModel.from_points(
    points=mx.random.normal((10_000, 3)) * 0.5,
    colors=mx.random.uniform(shape=(10_000, 3)),
)

# ...rendered from a camera, differentiably, on the Apple GPU.
camera = Camera.look_at(eye=(0, 0, -4), at=(0, 0, 0), width=1280, height=720)
out = model.render(camera)
print(out["image"].shape)  # (720, 1280, 3)
```

Continue with the [Quickstart](quickstart.md), then pick a tutorial:

| Tutorial | What you build |
|---|---|
| [Mesh Optimization](tutorials/mesh_optimization.md) | Deform a sphere into a target shape with chamfer + regularizers |
| [Point Cloud Fitting](tutorials/pointcloud_fitting.md) | Optimize raw points with chamfer distance |
| [NeRF](tutorials/nerf.md) | Train a neural radiance field on the Blender synthetic scenes |
| [Gaussian Splatting](tutorials/gaussian_splatting.md) | Train 3DGS on a COLMAP scene with the Metal rasterizer |
