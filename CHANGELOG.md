# Changelog

## 0.3.0

### Added

- Multi-asset interactive viewer. The browser viewer now renders **meshes**
  (tile-based hard rasterizer: shaded / normals / depth / wireframe modes,
  with a steerable light) and **point clouds** (splat-based point renderer)
  in addition to Gaussian Splatting and NeRF, via new `view_mesh`,
  `view_pointcloud`, and a `view_file` auto-detector. `mlx3d-view` takes any
  `.ply` / `.obj` / `.gltf` / `.glb` and picks the right viewer.
- Live control panel. Each scene type surfaces runtime controls (background,
  exposure/gamma, splat scale, SH degree, a debug clip plane for Gaussians;
  light direction and wireframe width for meshes; sample counts and near/far
  for NeRF) that take effect per frame without restarting. Controls are
  declared on `Viewer` and rendered as sliders/color pickers/selects in-page.
- New Gaussian display modes (`alpha`, turbo-colormapped `depth`) and a shared
  clip plane that applies across RGB, depth, and alpha passes.
- Viewer UX: orbit axis gizmo, camera presets (front/back/left/right/top/iso),
  number-key display-mode switching, a full-resolution PNG screenshot endpoint
  and button, copy-camera-to-clipboard, and camera/control persistence across
  reloads.
- `examples/view_scene.py` opens the viewer on any scene file or a built-in
  procedural demo mesh.

### Changed

- Viewer render callbacks may now accept a second `params` argument carrying
  live control values; one-argument callbacks keep working unchanged.

### Efficiency

- The GPU lock is held only across render + `mx.eval`, so one frame's
  (CPU-side) encode overlaps the next request's render.
- A small frame cache serves byte-identical repeat requests for a settled view
  without re-rendering.

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
