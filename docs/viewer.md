# Interactive Viewer

MLX3D ships a browser-based viewer for Gaussian Splatting scenes and NeRFs.
Frames are rendered server-side on the Apple GPU — splats go through the
Metal rasterization kernels, so typical scenes orbit in real time — and
streamed to a canvas page with orbit / pan / zoom controls. No extra
dependencies, nothing to install.

## Viewing a Gaussian Splatting checkpoint

Any 3DGS-format `.ply` (trained with MLX3D or elsewhere):

```bash
mlx3d-view point_cloud.ply
# or: python -m mlx3d.viewer point_cloud.ply --background 1 1 1 --port 8090
```

This opens `http://127.0.0.1:8090` in your browser.

From Python:

```python
from mlx3d.splatting import GaussianModel
from mlx3d.viewer import view_gaussians

model = GaussianModel.load_ply("point_cloud.ply")
view_gaussians(model, background=(0, 0, 0))   # blocking; Ctrl-C to stop
```

## Controls

| Input | Action |
|---|---|
| drag | orbit |
| `shift` + drag, right-drag | pan |
| scroll / pinch | zoom |
| `R` | reset camera |
| `U` | flip the up axis (handy for COLMAP scenes, which are often "upside down") |
| `D` | toggle Gaussian RGB / expected-depth rendering |
| `M` | toggle Gaussian RGB / mesh-style depth-contour rendering |
| `H` | toggle the help panel |
| `[` / `]` | render resolution down / up |

The page adapts resolution automatically: while you drag it renders at
reduced resolution for responsiveness, then refines to full resolution when
the camera settles. When nothing changes, no frames are requested at all —
the GPU idles.

Gaussian checkpoints expose two display modes in the browser: RGB and depth.
Depth uses a forward-only Metal splatting pass that accumulates
transmittance-weighted expected depth per pixel, then colorizes it on the GPU
before JPEG encoding. It is meant for geometry inspection during training and
does not add work to the differentiable RGB training path.

The mesh-style mode uses the same GPU depth pass and overlays screen-space
depth/alpha contours in MLX. It is a fast inspection view for geometry and
holes, not an exported triangle mesh.

## Viewing a NeRF

```python
from mlx3d.viewer import view_nerf

view_nerf(model, near=2.0, far=6.0)   # model: a trained mlx3d.nn.NeRF
```

NeRF rendering is orders of magnitude heavier than splatting; the viewer
starts at half resolution and leans on adaptive degradation. Expect seconds
per full-resolution frame for a full-size NeRF — fine for inspecting a
training run, not a real-time experience.

## Viewing anything else

`Viewer` works with any callback that maps a camera to an image, so you can
inspect custom renderers too:

```python
import mlx.core as mx
from mlx3d.viewer import Viewer
from mlx3d.renderer import render_points

points = ...   # (P, 3)
mx.eval(points)  # arrays captured by the callback must be evaluated

viewer = Viewer(lambda cam: render_points(cam, points)["image"])
viewer.serve(port=8090)
```

!!! warning "Threading note"
    Frames are rendered on HTTP handler threads, and MLX cannot evaluate
    lazy arrays that were created on a different thread. Call ``mx.eval`` on
    everything your callback captures before ``serve()`` — the
    ``view_gaussians`` / ``view_nerf`` helpers already do this.
