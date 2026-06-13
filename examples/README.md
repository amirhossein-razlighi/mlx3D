# MLX3D examples

Every script here is standalone — run it straight from a clone with
[`uv`](https://docs.astral.sh/uv/):

```bash
uv run python examples/render_mesh.py
```

Outputs (images, meshes, point clouds) are written to `outputs/` by default;
pass `--out` to change the path.

## Self-contained (no downloads, run anywhere)

These generate their own synthetic data in-process and finish in seconds on an
Apple-Silicon Mac — ideal for a first look or for CI.

| Script | Feature | What it does |
| --- | --- | --- |
| [`render_mesh.py`](render_mesh.py) | Rasterization | Soft-rasterize a position-colored icosphere with `render_mesh_soft`. |
| [`raytrace_volume.py`](raytrace_volume.py) | Ray tracing | Cast camera rays through an analytic radiance field and composite with `volume_render`. |
| [`extract_mesh.py`](extract_mesh.py) | Mesh extraction | Recover a mesh from an implicit SDF with `marching_cubes`, save OBJ, render it. |
| [`fit_pointcloud.py`](fit_pointcloud.py) | Point-cloud optimization | Fit a noisy point cloud to a target shape with chamfer distance. |
| [`fit_mesh.py`](fit_mesh.py) | Mesh fitting | Deform an icosphere onto a target with chamfer + mesh regularizers. |
| [`fit_nerf.py`](fit_nerf.py) | NeRF | Train a compact NeRF on synthetic views, render a held-out view. |
| [`fit_gaussians.py`](fit_gaussians.py) | Gaussian splatting | Fit 3D Gaussians to synthetic views with the real `GaussianTrainer`. |
| [`extend_renderer.py`](extend_renderer.py) | Extensibility | Plug a custom renderer into the pipeline via the `Renderer` protocol. |

## Real-scene training (bring your own dataset)

These take a COLMAP or NeRF-synthetic (Blender) scene. They share the data
loaders in `mlx3d.datasets`.

| Script | Feature | Data |
| --- | --- | --- |
| [`train_nerf.py`](train_nerf.py) | NeRF | A NeRF-synthetic scene, e.g. `nerf_synthetic/lego`. |
| [`train_gaussian_splatting.py`](train_gaussian_splatting.py) | Gaussian splatting | A COLMAP scene (`sparse/0` + `images`) or a Blender scene. |
| [`benchmark_gaussian_splatting.py`](benchmark_gaussian_splatting.py) | Profiling | A COLMAP scene; reports startup/train/render timings and memory. |

## Writing your own

Every image renderer in MLX3D is just a callable
`(camera, scene) -> {"image", "alpha", "depth"}` (the
[`Renderer`](../src/mlx3d/renderer/protocols.py) protocol). `extend_renderer.py`
shows how to drop your own shading pass into the same pipeline — saving, the
viewer, turntable loops — without subclassing anything.
