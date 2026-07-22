# Changelog

## Unreleased

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

Release branch in progress.

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
