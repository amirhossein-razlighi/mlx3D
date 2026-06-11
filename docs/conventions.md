# Conventions

A page worth reading once: most 3D bugs are convention bugs.

## Camera model

MLX3D uses the **OpenCV / COLMAP convention** everywhere:

- Camera frame: **+x right, +y down, +z forward** (into the screen).
- Extrinsics are world-to-camera: `X_cam = R @ X_world + t`.
- Intrinsics in pixels: `u = fx * x/z + cx`, `v = fy * y/z + cy`.
- Pixel `(0, 0)` is the top-left corner; pixel centers are at `+0.5`.

Why this convention? It is what COLMAP outputs, what most NeRF datasets and
every Gaussian Splatting implementation use — your real-world data loads
without axis flips. The [Blender dataset loader](api/datasets.md) converts
the OpenGL-style (`y` up, `z` backward) c2w matrices for you.

```python
from mlx3d.cameras import Camera

cam = Camera.look_at(eye=(0, 0, -4), at=(0, 0, 0), up=(0, 1, 0),
                     fov=60.0, width=640, height=480)
cam.camera_center            # (3,) position in world space
cam.world_to_camera(points)  # (..., 3) -> camera frame
xy, depth = cam.project_points(points)
origins, dirs = cam.generate_rays()   # one ray per pixel, world space
```

!!! warning "Points behind the camera"
    `project_points` returns z-depth alongside pixel coordinates; it does
    **not** discard points behind the camera. Mask on `depth > 0` yourself.

## Rotations

- Quaternions are **scalar-first** `(w, x, y, z)` — the same as PyTorch3D and
  the 3DGS reference implementation.
- `euler_angles_to_matrix(angles, "XYZ")` applies `R_X @ R_Y @ R_Z`.
- All conversions are batched and differentiable; see
  [Transforms](api/transforms.md).

## Meshes

- Faces are integer triangles `(F, 3)`, counter-clockwise = outward normal.
- Padded faces use `-1` for padding rows.
- The **packed** representation concatenates all meshes of the batch and
  offsets face indices so they index into the packed vertex array directly.

## Images

- Images are `(H, W, 3)` float32 in `[0, 1]`, row 0 at the top.
- Batched image metrics accept `(N, H, W, C)`.

## Gaussian Splatting parameters

The raw (optimizable) parameterization follows the 3DGS paper:

| Parameter | Shape | Activation |
|---|---|---|
| `means` | (N, 3) | — |
| `scales` | (N, 3) | `exp` → standard deviations |
| `quats` | (N, 4) | normalized in the renderer |
| `opacities` | (N,) | `sigmoid` |
| `sh_dc`, `sh_rest` | (N, 1, 3), (N, K-1, 3) | SH evaluation |

Checkpoints saved with `GaussianModel.save_ply` use the standard 3DGS PLY
layout (`f_dc_*`, `f_rest_*`, `opacity`, `scale_*`, `rot_*`) and are loadable
by common splat viewers.
