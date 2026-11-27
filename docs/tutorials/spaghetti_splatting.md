# Spaghetti Splatting

[Spaghetti Splatting](https://spaghetti-splatting.github.io/) (Shramko et al.,
2026) replaces the *point* at the heart of 3D Gaussian Splatting with a
*line segment*: each primitive is a segment convolved with a 3D Gaussian
kernel. Chained segments describe curves, so spokes, cables, hair and grass
take a handful of primitives instead of long strings of blobs — and since a
zero-length segment is exactly a Gaussian, the representation is a strict
superset of 3DGS. `mlx3d.splatting.spaghetti` implements it with custom Metal
kernels.

<p align="center">
  <img src="../../assets/spaghetti_bicycle_compare.jpg" width="100%" alt="Held-out bicycle view: photo, spaghetti splats and 3D Gaussians" />
  <br/>
  <em>A held-out view of Mip-NeRF 360 <em>bicycle</em>: the photo (left), spaghetti splats (middle) and 3D Gaussians (right) trained with the same budget.</em>
</p>

## Results on Mip-NeRF 360 *bicycle*

Both methods start from the scene's 54k SfM points, with the same 300k
primitive budget, 7k iterations and 618×411 training views (every 8th photo
held out), on an M1 Pro:

| Method | PSNR ↑ | SSIM ↑ | Primitives | Training |
|---|---|---|---|---|
| 3D Gaussian Splatting | 24.99 | 0.715 | 308k Gaussians | 58 min |
| Spaghetti Splatting | 24.29 | 0.700 | 300k segments | 54 min |

On this scene the two are close — as in the paper, where spaghetti splats
roughly match 3DGS-MCMC on Mip-NeRF 360 and pull ahead on hair and fur — and
both resolve the spokes. This is an untuned 7k-iteration schedule at reduced
resolution; the paper trains 30k iterations at full resolution.

<p align="center">
  <img src="../../assets/spaghetti_bicycle_spokes.jpg" width="96%" alt="Front wheel of the bicycle: photo, spaghetti splats and 3D Gaussians" />
  <br/>
  <em>The front wheel of the same view at 2.5× — photo, spaghetti, 3DGS.</em>
</p>

<p align="center">
  <video src="../../assets/spaghetti_bicycle_flythrough.mp4" width="80%" autoplay loop muted playsinline controls></video>
  <br/>
  <em>Flying along the capture path through the spaghetti reconstruction.</em>
</p>

Reproduce it on any COLMAP scene:

```bash
python examples/train_spaghetti_splatting.py --data path/to/bicycle \
    --downscale 2 --iters 7000 --budget 300000 --compare-3dgs --out outputs/bicycle
```

## Python API

```python
from mlx3d.datasets import load_colmap
from mlx3d.splatting.spaghetti import SpaghettiModel, SpaghettiTrainer, default_config_for

ds = load_colmap("bicycle", downscale=2)
model = SpaghettiModel.from_points(ds.points, ds.point_colors)   # short segments at SfM points
config = default_config_for(iterations=7000, max_segments=300_000, lambda_connect=0.5)
trainer = SpaghettiTrainer(model, config, scene_extent=ds.scene_extent)
for step in range(7000):
    camera, image = ds[step % len(ds)]
    trainer.step(camera, image)
trainer.finalize_polylines()            # merge nearby ends into shared joints

out = model.render(camera)              # {"image", "alpha", ...}, differentiable
model.save_ply("spaghetti.ply")
curves = model.polylines(tolerance=1e-6)  # list of (K, 3) polylines
```

`render_spaghetti(camera, means, halves, quats, scales, rho, colors)` is the
functional, differentiable renderer (including with respect to the camera
pose), and `SpaghettiModel.from_gaussians(model)` lifts a trained 3DGS model
into zero-length segments.

## How it works

**The primitive.** A segment ``x(u) = a + u v`` blurred by ``N(0, Σ)`` with
density ``ρ`` has a closed-form optical depth along any ray (the paper's
Eq. 19): a Gaussian falloff from the ray to the segment's line times an `erf`
integral along it. mlx3d evaluates it **exactly per pixel** in the kernel's
whitened space (where ``Σ`` becomes the identity), instead of linearizing each
segment in screen space — numerically robust for very thin kernels far from
the camera.

**The rasterizer.** Segments are binned into 16×16 tiles by an opacity-aware
footprint ellipse that also rejects most pixels before the 3D math, sorted by
depth and alpha-composited front to back. The backward pass is hand-derived
down to the 16 numbers each segment contributes per pixel, and MLX autodiff
takes it from there; gradients are summed across each SIMD group before the
atomic adds, which makes the backward pass 10× faster than per-pixel atomics.

**Training.** `SpaghettiTrainer` follows the paper's recipe on top of
3DGS-MCMC: L1 + D-SSIM with the MCMC opacity and scale regularizers,
covariance-shaped exploration noise for near-transparent segments, and every
100 steps relocation of dead segments plus growth toward the budget. Because
optical depth adds along rays, the two densification moves are **exact** for
a segment in isolation: splitting a long one at its midpoint, or cloning a
short one with half the optical depth each, renders identically. In the final
sixth of training an attraction loss pulls nearby endpoints together, and
`finalize_polylines()` merges them into connected curves.

## Validation

- The closed form matches brute-force numerical integration of the paper's
  definition to within 4e-5 for points, short and long segments.
- The Metal rasterizer matches a dense pure-MLX reference to 1e-6, and its
  backward pass matches MLX autodiff of that reference to 1e-5 (relative) for
  every parameter, including camera pose.

## Notes and differences from the paper

- The official code and supplement were not yet released, so the formulas the
  main text leaves to the supplement are re-derived: per-pixel evaluation is
  exact rather than linearized, and the attraction loss is the mean squared
  distance between nearest endpoints of different segments within
  `connect_radius` kernel widths.
- Only the Gaussian kernel is implemented (the paper also uses a Beta kernel),
  and appearance uses spherical harmonics instead of spherical Voronoi.
- The rasterizer needs undistorted pinhole cameras.
