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
- glTF export of UV coordinates and a simple PBR base color material.
- `mlx3d-render` CLI for rendering Gaussian checkpoints and mesh assets to
  RGB, depth, or normal images.

### Fixed

- Aligned Gaussian depth rasterization with the RGB kernel's early-transmittance
  cutoff semantics.
