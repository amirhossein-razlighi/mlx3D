"""Photos -> depth, cameras and a fused point cloud with Depth Anything 3.

One forward pass of the pretrained model predicts every view's depth map,
intrinsics and pose — no feature matching or bundle adjustment. This script
saves what it predicted and renders a fly-through of the fused point cloud
as Gaussian splats with mlx3d's fast rasterizer.

    pip install "mlx3d[models]"
    python examples/predict_geometry.py path/to/photos --out outputs/da3

Outputs in ``--out``:
    depth.png      input frames above their predicted depth maps
    points.ply     the fused, confidence-filtered colored point cloud
    flythrough.mp4 the point cloud rendered along the predicted trajectory
                   (``flythrough.gif`` when ffmpeg is not installed)
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from mlx3d.cameras import Camera
from mlx3d.capture import list_images
from mlx3d.io import save_ply
from mlx3d.models import DepthAnything3
from mlx3d.splatting import FastGaussianRenderer


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    """Inverse-depth (disparity) to an 8-bit "magma"-like ramp, near = bright."""
    disp = 1.0 / np.maximum(depth, 1e-6)
    lo, hi = np.percentile(disp, [2, 98])
    x = np.clip((disp - lo) / max(hi - lo, 1e-8), 0, 1)[..., None]
    stops = np.array([[0, 0, 4], [81, 18, 124], [183, 55, 121], [252, 137, 97], [252, 253, 191]])
    pos = x * (len(stops) - 1)
    i = np.minimum(pos.astype(int), len(stops) - 2)
    return (stops[i[..., 0]] * (1 - (pos - i)) + stops[i[..., 0] + 1] * (pos - i)).astype(np.uint8)


def depth_strip(pred, max_views: int = 4) -> Image.Image:
    idx = np.linspace(0, len(pred) - 1, min(max_views, len(pred))).round().astype(int)
    top = np.concatenate([pred.rgb[i] for i in idx], axis=1)
    bottom = np.concatenate([colorize_depth(np.array(pred.depth[i])) for i in idx], axis=1)
    return Image.fromarray(np.concatenate([top, bottom], axis=0))


def _nearest_rotation(m: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(m)
    return u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt


def flythrough_cameras(pred, n_frames: int, width: int, pull_back: float = 0.15) -> list[Camera]:
    """Cameras gliding along the predicted trajectory (interpolated poses),
    stepped back along the viewing direction so the reconstruction's extent
    shows."""
    R, t = np.array(pred.R, np.float64), np.array(pred.t, np.float64)
    centers = -np.einsum("nji,nj->ni", R, t)
    step = pull_back * np.linalg.norm(centers - centers.mean(0), axis=1).mean()

    K = np.array(pred.K[0])
    pw, ph = pred.image_size
    scale = width / pw
    height = 2 * round(ph * scale / 2)  # even, as H.264 requires
    cams = []
    for s in np.linspace(0, len(centers) - 1, n_frames):
        i = min(int(s), len(centers) - 2)
        w = s - i
        Ri = _nearest_rotation((1 - w) * R[i] + w * R[i + 1])
        center = (1 - w) * centers[i] + w * centers[i + 1] - step * Ri[2]  # Ri[2]: view axis
        cams.append(
            Camera(
                R=mx.array(Ri, dtype=mx.float32),
                t=mx.array(-Ri @ center, dtype=mx.float32),
                fx=float(K[0, 0] * scale),
                fy=float(K[1, 1] * scale),
                cx=width / 2,
                cy=height / 2,
                width=width,
                height=height,
            )
        )
    return cams


def point_splats(pred, points: np.ndarray, colors: np.ndarray) -> FastGaussianRenderer:
    """Opaque isotropic splats about one source pixel wide (distance / focal)."""
    centers = -np.einsum("nji,nj->ni", np.array(pred.R), np.array(pred.t))
    nearest = np.min(np.linalg.norm(points[:, None] - centers[None], axis=-1), axis=1)
    radius = 1.1 * nearest / float(np.array(pred.K)[:, 0, 0].mean())
    n = len(points)
    return FastGaussianRenderer(
        means=mx.array(points),
        quats=mx.broadcast_to(mx.array([1.0, 0.0, 0.0, 0.0]), (n, 4)),
        scales=mx.broadcast_to(mx.array(radius, dtype=mx.float32)[:, None], (n, 3)),
        opacities=mx.full((n,), 0.95),
        colors=mx.array(colors),
    )


def write_video(frames: list[np.ndarray], out_dir: Path, fps: int = 30) -> Path:
    if shutil.which("ffmpeg") is None:
        path = out_dir / "flythrough.gif"
        images = [Image.fromarray(f) for f in frames]
        images[0].save(path, save_all=True, append_images=images[1:], duration=1000 // fps, loop=0)
        return path
    path = out_dir / "flythrough.mp4"
    with tempfile.TemporaryDirectory() as tmp:
        for i, frame in enumerate(frames):
            Image.fromarray(frame).save(f"{tmp}/{i:04d}.png")
        cmd = ["ffmpeg", "-y", "-v", "error", "-framerate", str(fps), "-i", f"{tmp}/%04d.png"]
        cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(path)]
        subprocess.run(cmd, check=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("images", help="directory of photos (one static scene)")
    parser.add_argument("--out", default="outputs/da3")
    parser.add_argument("--model", default="da3-base", help="da3-small | da3-base | da3-large")
    parser.add_argument("--max-images", type=int, default=32, help="evenly subsample to this")
    parser.add_argument("--frames", type=int, default=120, help="fly-through length")
    parser.add_argument("--width", type=int, default=640, help="fly-through width in pixels")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    paths = list_images(args.images)
    if len(paths) > args.max_images:
        paths = [paths[i] for i in np.linspace(0, len(paths) - 1, args.max_images).astype(int)]

    model = DepthAnything3.from_pretrained(args.model)
    start = time.perf_counter()
    pred = model.predict(paths)
    print(
        f"{args.model}: {len(pred)} views at {pred.image_size} in {time.perf_counter() - start:.1f}s"
    )

    depth_strip(pred).save(out / "depth.png")
    points, colors = pred.point_cloud(conf_quantile=0.4, max_points=800_000)
    save_ply(str(out / "points.ply"), mx.array(points), colors=mx.array(colors))
    print(f"fused {len(points)} points -> {out / 'points.ply'}")

    renderer = point_splats(pred, points, colors)
    frames = []
    for cam in flythrough_cameras(pred, args.frames, args.width):
        image = renderer.render(cam, background=mx.ones(3))["image"]
        frames.append((np.array(image).clip(0, 1) * 255).astype(np.uint8))
    print(f"wrote {write_video(frames, out)}")


if __name__ == "__main__":
    main()
