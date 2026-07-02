# Photos → Splat in Minutes

`mlx3d-capture` turns a folder of photos — or a phone video — into a trained
3D Gaussian Splat with **one command**, entirely on your Mac:

```bash
mlx3d-capture ./my_photos/
# or
mlx3d-capture walkaround.mp4
```

That single command runs the whole pipeline:

1. **Frames** — photos are used as-is; videos are sampled with ffmpeg and
   motion-blurred frames are dropped automatically (the sharpest frame per
   time bucket wins, scored by variance of Laplacian).
2. **Camera poses** — COLMAP if it is installed, otherwise mlx3d's built-in
   COLMAP-free structure-from-motion (see below). Either way the poses land in
   the standard COLMAP binary format under `<out>/sparse/0`.
3. **Training** — 3D Gaussian Splatting on the Apple GPU, with a **live
   browser viewer** that opens automatically so you can watch the splat
   converge in real time.
4. **Export** — a compacted `splat.ply` you can open in any splat viewer
   (including `mlx3d-view`), plus eval renders and a `capture.json` summary.

Everything is resumable: re-running the same command skips stages whose
outputs already exist (pass `--overwrite` to recompute).

## Requirements

```bash
pip install mlx3d

# for videos (frame extraction):
brew install ffmpeg

# pick ONE of the two pose engines:
brew install colmap                  # best quality, handles hard captures
pip install "mlx3d[capture]"         # built-in SfM (OpenCV + SciPy), no COLMAP needed
```

## Capturing good inputs

The usual photogrammetry rules apply:

- Orbit the subject with **lots of overlap** between neighboring shots
  (aim for 60–80% overlap; 50–150 photos or a slow 30–60 s video).
- Prefer texture-rich, well-lit, static scenes. Avoid glass, mirrors, and
  moving objects.
- For video, move slowly — motion blur is the main quality killer (the
  pipeline filters the worst frames, but it cannot invent sharpness).

## Options you will actually use

```bash
mlx3d-capture ./my_photos/ --quality fast        # quick preview (~3k iters)
mlx3d-capture ./my_photos/ --quality best        # 30k iters, full resolution
mlx3d-capture clip.mp4 --frames 200              # keep more video frames
mlx3d-capture ./my_photos/ --poses builtin       # force the built-in SfM
mlx3d-capture ./my_photos/ --no-viewer           # headless (CI, ssh)
mlx3d-capture ./my_photos/ --low-mem             # 8–16 GB machines
mlx3d-capture ./my_photos/ --out captures/lion   # choose the output folder
```

`--quality` presets pick iterations, training resolution, video frame count
and SH degree (`fast` = 3k iters / 960 px, `balanced` = 7k / 1280 px,
`best` = 30k / 1920 px). Every choice can be overridden individually
(`--iters`, `--max-dim`, `--sh-degree`, `--method mcmc`, ...); see
`mlx3d-capture --help`.

## COLMAP-free poses: the built-in SfM

If COLMAP is not installed, mlx3d estimates poses itself with a compact
classical SfM pipeline (`mlx3d.capture.run_sfm`):

- SIFT features + ratio-test matching — exhaustive for small captures,
  sequential-window with log-spaced loop-closure pairs for ordered/video
  captures;
- essential-matrix two-view initialization from the strongest pair with a
  healthy triangulation angle;
- incremental PnP-RANSAC registration and filtered triangulation;
- periodic sparse bundle adjustment (robust soft-L1 loss) that also refines a
  shared focal length, seeded from the EXIF 35 mm-equivalent focal length
  when present (COLMAP's `1.2 × max(width, height)` prior otherwise).

Because SfM-only poses are a bit noisier than COLMAP's, the trainer then
**refines poses jointly with the splats** (BARF-style): each view gets a
learnable SE(3) twist, optimized by gradient descent through the
differentiable projection with a decaying learning rate. This is on by
default for built-in poses (`--refine-poses auto`) and can be forced on for
COLMAP poses too (`--refine-poses on`). The refined poses are exported as a
COLMAP model under `<out>/refined/sparse/0`.

On a synthetic 24-view benchmark the built-in SfM recovers camera centers
within 0.3% of the orbit radius, rotations within ~0.5°, and the focal length
within 1% — and it reconstructs real captures (e.g. the 11-photo Sceaux Castle
dataset) end to end in a few minutes on an M-series laptop.

## Python API

The pipeline is a normal library call:

```python
from mlx3d.capture import CaptureConfig, run_capture

summary = run_capture(
    "my_photos/",
    "captures/scene",
    CaptureConfig(quality="balanced", poses="auto"),
)
print(summary["stages"]["train"]["psnr_mean"], summary["splat"])
```

and the stages are usable on their own:

```python
from mlx3d.capture import extract_video_frames, run_sfm, run_colmap, list_images
from mlx3d.datasets import load_colmap, save_colmap

frames = extract_video_frames("walk.mp4", "scene/images", num_frames=150)
result = run_sfm(list_images("scene/images"), "scene")   # -> scene/sparse/0
ds = load_colmap("scene", images_dir="scene/images")     # ready for training
```

## Output layout

```
captures/scene/
├── images/            extracted video frames (photo inputs stay in place)
├── sparse/0/          COLMAP-format poses + sparse points
├── refined/sparse/0/  poses after joint refinement (when enabled)
├── renders/           eval renders from training views
├── point_cloud.ply    raw training checkpoint
├── splat.ply          compacted splat — share this one
└── capture.json       per-stage timings and stats
```

## When things go wrong

- **"No image pairs with enough matches"** — the images don't overlap enough
  or lack texture. Capture again with smaller steps between shots.
- **Only part of the images registered** — the unregistered ones are skipped
  automatically; training proceeds with the rest.
- **Poor quality with built-in poses** — install COLMAP (`brew install
  colmap`); the pipeline picks it up automatically, and `--overwrite` redoes
  the pose stage.
- **Out of memory** — add `--low-mem`, or lower `--max-dim` / use
  `--quality fast`.
