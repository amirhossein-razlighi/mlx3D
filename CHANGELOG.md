# Changelog

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
