# Changelog

## 0.4.0

### Added

- **Spaghetti Splatting** (`mlx3d.splatting.spaghetti`): radiance fields of
  Gaussian-blurred line segments (Shramko et al., 2026), a strict superset of
  3DGS that represents hair, wires, spokes and grass with far fewer
  primitives. Custom Metal forward/backward rasterizer that evaluates the
  paper's closed-form segment optical depth exactly per pixel (in the
  kernel's whitened space, robust for thin kernels), with opacity-aware
  footprint ellipses for binning and early rejection, and SIMD-group-reduced
  gradient atomics (10x faster backward). `SpaghettiModel` (PLY IO, 3DGS
  lift, polyline extraction), `SpaghettiTrainer` (3DGS-MCMC-style relocation
  and growth with exact split / clone moves, endpoint attraction and
  merging into polylines), and `examples/train_spaghetti_splatting.py` for
  COLMAP scenes with a like-for-like 3DGS comparison. Validated against
  numerical quadrature (4e-5), a dense MLX reference (1e-6) and autodiff
  gradients (1e-5).
- Foreground masks for Gaussian training (#28): `load_colmap(masks_dir=...,
  masks_from_alpha=...)` loads COLMAP-layout masks (`<name>.png` or
  `<stem>.png`; grayscale / RGB / RGBA) or the images' own alpha into a
  `MaskCollection` aligned with every cache mode and downscale, failing early
  on missing, empty or mis-sized masks. `GaussianTrainer.step(..., mask=)`
  supports `alpha_mode="masked"` (coverage-normalized L1 + mask-aware SSIM;
  background pixels get no loss or gradient; 2DGS geometry losses restricted
  to the mask) and `"transparent"` (mask = target opacity: alpha supervision,
  target composited over an optionally random background).
- Mask-aware metrics `masked_l1`, `masked_psnr`, `masked_ssim` and a per-pixel
  `ssim_map`; `mlx3d-eval --masks / --masks-from-alpha / --alpha-mode` reports
  the matching metrics plus coverage, alpha MAE and foreground IoU.
- Per-view appearance compensation (`TrainerConfig.appearance`): a learnable
  affine color transform per image absorbs exposure / white-balance changes.
- Geometry priors: `TrainerConfig.lambda_depth_prior` (scale/shift-invariant
  Pearson depth correlation) and `lambda_normal_prior` (normals of rendered vs
  prior depth) with `step(..., depth_prior=)`.
- `mlx3d-capture --masks / --masks-from-alpha / --alpha-mode / --mask-sfm /
  --appearance / --depth-prior`. The DA3 pose stage now saves per-frame depth
  maps (`<out>/depth/`) for the priors; `--mask-sfm` opts into masked feature
  extraction for COLMAP and the built-in SfM.
- "Training on Real Captures" tutorial.
- `mlx3d.models.DepthAnything3`: an MLX port of the Depth Anything 3
  feed-forward geometry model. From any number of views of a static scene,
  one forward pass predicts per-view depth, confidence, intrinsics and
  world-to-camera poses (`GeometryPrediction`, with `cameras()`,
  `world_points()`, `point_cloud()` and `to_pointclouds()` helpers). Loads the
  official checkpoints (`da3-small` / `da3-base`, Apache-2.0; `da3-large`,
  CC BY-NC) through an exhaustive, strict weight conversion; backbone in
  float32 or half precision, and matches the official PyTorch implementation
  to ~1e-6 relative error on every output. Sequences longer than
  `max_views` run in anchored chunks aligned by a similarity fit on shared anchor cameras.
  Weight downloads need the new optional `[models]` extra (`huggingface_hub`).
- `mlx3d-capture --poses da3 [--pose-model da3-small]`: feed-forward camera
  poses and a dense fused point cloud as the capture pipeline's pose stage
  (`mlx3d.capture.run_feedforward_poses`), with joint pose refinement on by
  default.
- `examples/predict_geometry.py`: photos → depth maps, fused `points.ply`,
  and a fly-through video; plus the "Photos → Depth & Cameras" tutorial.

### Changed

- Python 3.11 or newer is required (3.10 reached end of life in October 2026,
  and the patched Pillow and filelock releases need 3.11). CI now tests 3.11,
  3.12 and 3.13.
- `mlx3d-capture --poses auto` now prefers Depth Anything 3 when the
  `[models]` extra is installed, then COLMAP, then the built-in SfM.
- `mlx3d-capture` initializes splat scales from nearest-neighbor spacing
  against up to 100k reference points (was 10k). On dense clouds the old
  subsample inflated every initial splat several-fold, making early training
  steps up to ~10x slower.

### Fixed

- Normal-based losses (the 2DGS normal-consistency loss and the new normal
  prior) produced NaN gradients wherever a pixel's depth tangents vanish
  (empty background): `linalg.norm` has a NaN gradient at zero, which a zero
  mask weight does not cancel. Normalization is now zero-safe.
- `render_points` rendered blank images when any point was behind the camera:
  its soft z-weights were normalized against the nearest point including
  culled ones, which underflowed every visible weight.
- Resumed captures forgot where their poses came from (reported only as
  "existing"), so built-in SfM / DA3 poses lost joint pose refinement on a
  re-run. The pose source is now recorded in `<out>/poses.json`.
- `--poses auto` falls back to COLMAP / the built-in SfM when Depth Anything 3
  cannot load (e.g. offline before the one-time weight download) instead of
  failing the capture.
- Broken images on the Interactive Viewer docs page (raw HTML paths were not
  relative to the page's served URL); a new test checks every raw-HTML media
  path in the docs.
- CI: macOS jobs moved to `macos-15` (GitHub retired the `macos-14` image, so
  every macOS test and docs-deploy job was being cancelled), and all actions
  moved to their Node 24 majors.

### Security

- Lockfile upgrades for the open Dependabot alerts: Pillow 13.0.0, filelock
  4.3.1, pymdown-extensions >= 12.3, mkdocs-material 9.8.2.

## 0.3.2

### Changed

- Faster Gaussian tile binning (`bin_gaussians`), on the critical path of every
  render and training step. The default path now fuses expansion and
  range-finding into two Metal kernels — tile duplicates are emitted in depth
  order so only a stable 32-bit tile-key sort is needed, and per-tile ranges are
  recovered by boundary detection instead of a scatter-min/max pass — replacing
  the int64 composite-key argsort. Bit-identical output; ~2.5× faster binning
  and ~26% faster forward rendering on a 500k-Gaussian scene at 1080p (so
  viewers, evaluation, capture previews, and every training forward speed up).

### Added

- Type support: mlx3d now ships a `py.typed` marker, so downstream projects
  get its inline type hints under mypy / pyright.
- Finite-difference gradient check for the Metal Gaussian rasterizer's
  hand-written backward pass (`examples/validate_gaussian_gradients.py` plus a
  `unit` test). Confirms the analytic gradients match numerical derivatives
  (geometry to ~1e-2, appearance to ~1e-3), a runnable correctness artifact on
  Apple Silicon.

### Security

- Pillow, certifi and urllib3 bumped for Dependabot alerts (#29).

## 0.3.0

### Added

- Fast forward-only Gaussian rasterization (`FastGaussianRenderer`,
  `render_gaussians_fast`), adapting the geometry-shader-style pipeline of
  dendenxu/fast-gaussian-rasterization to Metal compute: a fused per-Gaussian
  geometry kernel, duplicates emitted in globally depth-sorted order (stable
  32-bit tile sort), boundary-detection tile ranges, fp16 splat colors, and
  zero mid-frame CPU/GPU synchronization via persistent grow-on-demand
  buffers. 1.5–2× faster than the training rasterizer on real scenes (up to
  3.7× at high pixel-to-point ratios) at 45+ dB PSNR parity; not
  differentiable (training keeps `render_gaussians`). Exposed as
  `mlx3d-view --fast` and `mlx3d-render --fast`.
- Dynamic (4D) Gaussian sequence playback: `FastGaussianRenderer.update()`
  swaps per-timestep arrays cheaply;
  `examples/play_dynamic_gaussians.py` streams a Dynamic 3D Gaussians
  `params.npz` at 67 fps (vs 40 fps on the training path) with GIF/frame
  export. New `examples/benchmark_fast_rasterization.py` reports
  latency/FPS/PSNR on synthetic scenes or any 3DGS `.ply`.
- `mlx3d-capture`: a one-command capture pipeline (photos directory or video
  file → camera poses → 3D Gaussian Splatting with a live browser viewer → a
  compacted `splat.ply`). Stages are cached and resumable; `--quality
  fast/balanced/best` presets pick iterations, resolution, frame count and SH
  degree. Python API in `mlx3d.capture` (`run_capture`, `CaptureConfig`).
- Built-in COLMAP-free structure-from-motion (`mlx3d.capture.run_sfm`): SIFT +
  ratio matching, essential-matrix initialization, incremental PnP-RANSAC
  registration, filtered triangulation, and periodic sparse bundle adjustment
  with shared-focal refinement (EXIF 35mm-equivalent prior when available).
  Requires the new optional `[capture]` extra (OpenCV + SciPy). A thin COLMAP
  CLI wrapper (`mlx3d.capture.run_colmap`) is preferred automatically when the
  binary is installed.
- Video ingestion for captures: ffmpeg-based even sampling with automatic
  motion-blur filtering (sharpest frame per time bucket by variance of
  Laplacian).
- `GaussianTrainer.step` accepts an optional per-view SE(3) twist and returns
  its gradient, enabling BARF-style joint pose refinement during training. The
  capture pipeline turns this on automatically for built-in SfM poses and
  exports the refined poses as a COLMAP model.
- `save_colmap`: binary COLMAP sparse-model writer (inverse of `load_colmap`),
  with intrinsics deduplication and PINHOLE/OPENCV/OPENCV_FISHEYE support.

### Packaging

- Tag-driven releases: the version comes from the git tag (hatch-vcs) and
  pushing a `v*` tag builds and publishes to PyPI (#25).

## 0.2.1

### Added

- SDF utilities in `mlx3d.ops`: analytic primitives (`sdf_sphere`, `sdf_box`,
  `sdf_torus`, `sdf_plane`), constructive-solid-geometry operators
  (`sdf_union`, `sdf_intersection`, `sdf_difference` and smooth `sdf_smooth_*`
  variants), and `sample_sdf_grid` / `sdf_to_mesh` to turn an SDF callable into
  a mesh via marching cubes. All pure-MLX and differentiable w.r.t. shape
  parameters. New `examples/sdf_csg.py` shows CSG modeling through to a render.

### Fixed

- NeRF training: the classic `NeRF` model used a ReLU density activation that
  dies at initialization (density 0 everywhere, zero gradients), so the network
  never trained. Switched to softplus; added a regression test.
- Soft mesh rasterizer (`render_mesh_soft`): per-chunk accumulators are now
  evaluated each iteration so intermediates are freed between chunks. Peak
  memory for dense meshes is bounded (a 46k-face mesh at 256² went from OOM on
  16 GB to ~1.2 GB) while staying fully differentiable.
- `marching_cubes` now welds duplicate crossing-point vertices (≈5× fewer verts
  on a 64³ grid) and drops collapsed faces, cutting downstream memory and cost.
- `examples/extract_mesh.py` visualizes dense meshes with the O(H·W) hard
  rasterizer instead of the soft renderer (68 s + OOM → ~0.4 s).

### Packaging

- Added a standard `project.optional-dependencies` `dev` extra so
  `pip install -e ".[dev]"` works with any installer (the PEP 735
  `dependency-groups` entry is kept for `uv sync`). Documented both dev setups,
  including the uv-venv `pip` caveat, in the README.

## 0.2.0

### Added

- Arbitrary-channel Gaussian feature rendering through the existing Metal
  rasterizer, with alpha-composited and normalized expected-feature modes.
- Mip-Splatting-style opacity compensation for anti-aliased Gaussian rendering,
  available from render APIs and Gaussian Splatting training.
- glTF scene loading across default-scene nodes and triangle primitives,
  including node transforms, UVs, material IDs, and PBR base-color summaries.
- glTF base-color texture loading for embedded/data-URI and external image
  assets, wired into `mlx3d-render` for textured mesh previews.
- glTF metallic/roughness factor import, with `mlx3d-render --shading pbr`
  forwarding uniform material factors into the PBR mesh shader.
- glTF export of self-contained GLB assets with embedded base-color PNG
  textures.
- glTF export of UV coordinates and a simple PBR base color material.
- `mlx3d-render` CLI for rendering Gaussian checkpoints and mesh assets to
  RGB, depth, or normal images.
- `mlx3d-eval` CLI for deterministic Gaussian checkpoint evaluation with PSNR,
  SSIM, L1, per-view metrics, and JSON output.
- Gaussian checkpoint compaction by opacity/footprint importance, Gaussian
  count cap, and optional spherical-harmonic degree truncation.
- `mlx3d-compact` CLI for checkpoint pruning and SH-degree truncation.
- Opt-in 2DGS depth-variance and normal-depth consistency regularizers that
  reuse the differentiable Metal-backed feature rasterizer.
- 2DGS surfel extraction helpers on `GaussianModel`, including a Poisson
  reconstruction wrapper for mesh extraction from oriented Gaussian disks.
- 3DGUT-style Unscented Transform Gaussian projection for distortion-aware
  rendering/training with `projection="ut"` and `--projection ut`.
- PBR-style Cook-Torrance/GGX mesh shading with roughness and metallic controls.
- Morton-order face sorting for ray-mesh intersection, improving chunk-level
  AABB culling on arbitrarily ordered meshes while preserving original face IDs.

### Fixed

- Aligned Gaussian depth rasterization with the RGB kernel's early-transmittance
  cutoff semantics.
