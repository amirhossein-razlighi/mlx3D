# Changelog

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
- glTF export of UV coordinates and a simple PBR base color material.
- `mlx3d-render` CLI for rendering Gaussian checkpoints and mesh assets to
  RGB, depth, or normal images.
- `mlx3d-eval` CLI for deterministic Gaussian checkpoint evaluation with PSNR,
  SSIM, L1, per-view metrics, and JSON output.
- Gaussian checkpoint compaction by opacity/footprint importance, Gaussian
  count cap, and optional spherical-harmonic degree truncation.
- `mlx3d-compact` CLI for checkpoint pruning and SH-degree truncation.
- PBR-style Cook-Torrance/GGX mesh shading with roughness and metallic controls.
- Morton-order face sorting for ray-mesh intersection, improving chunk-level
  AABB culling on arbitrarily ordered meshes while preserving original face IDs.

### Fixed

- Aligned Gaussian depth rasterization with the RGB kernel's early-transmittance
  cutoff semantics.
