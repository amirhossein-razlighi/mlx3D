"""Play back a dynamic (4D) Gaussian sequence with the fast forward-only renderer.

Loads a `Dynamic 3D Gaussians <https://dynamic3dgaussians.github.io/>`_
``params.npz`` (per-timestep means / rotations / colors with shared scales and
opacities), streams timesteps through
:meth:`~mlx3d.splatting.FastGaussianRenderer.update`, and renders an orbiting
camera. Optionally exports frames, an animated GIF, and a playback benchmark
against the training rasterizer.

Get a scene (~2 GB each; this extracts one member of the release zip):

    pip install remotezip && python - <<'EOF'
    from remotezip import RemoteZip
    import shutil
    url = "https://omnomnom.vision.rwth-aachen.de/data/Dynamic3DGaussians/output.zip"
    with RemoteZip(url) as z:
        with z.open("output/pretrained/juggle/params.npz") as s, \
             open("juggle_params.npz", "wb") as o:
            shutil.copyfileobj(s, o, length=1 << 20)
    EOF

Then:

    python examples/play_dynamic_gaussians.py juggle_params.npz --gif juggle.gif
    python examples/play_dynamic_gaussians.py juggle_params.npz --benchmark
"""

import argparse
import math
import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.exists():
    sys.path.insert(0, str(_SRC))

import mlx.core as mx
import numpy as np

from mlx3d.cameras import Camera
from mlx3d.io import save_image
from mlx3d.splatting import FastGaussianRenderer, render_gaussians


class DynamicGaussianSequence:
    """A Dynamic-3D-Gaussians ``params.npz`` as per-timestep activated arrays."""

    def __init__(self, path: str, foreground_only: bool = False):
        d = np.load(path)
        keep = slice(None)
        if foreground_only:
            keep = d["seg_colors"][:, 0] > 0.5  # the release's foreground mask
        self.means = d["means3D"][:, keep]  # (T, N, 3)
        self.quats = d["unnorm_rotations"][:, keep]  # (T, N, 4)
        self.colors = d["rgb_colors"][:, keep]  # (T, N, 3)
        self.scales = np.exp(d["log_scales"][keep])  # (N, 3) shared across time
        self.opacities = 1.0 / (1.0 + np.exp(-d["logit_opacities"][keep, 0]))  # (N,)
        self.num_timesteps = self.means.shape[0]
        self.num_gaussians = self.means.shape[1]

    def timestep(self, t: int) -> dict[str, mx.array]:
        return {
            "means": mx.array(self.means[t]),
            "quats": mx.array(self.quats[t]),
            "scales": mx.array(self.scales),
            "opacities": mx.array(self.opacities),
            "colors": mx.array(np.clip(self.colors[t], 0.0, 1.0)),
        }


def orbit_camera(i, num_timesteps, width, height, center_dist=2.4, cam_height=1.3):
    """The Dynamic3DGaussians studio convention: y-down world, orbit about +y."""
    ry = 2.0 * math.pi * i / num_timesteps
    R = mx.array(
        [
            [math.cos(ry), 0.0, -math.sin(ry)],
            [0.0, 1.0, 0.0],
            [math.sin(ry), 0.0, math.cos(ry)],
        ],
        dtype=mx.float32,
    )
    t = mx.array([0.0, cam_height, center_dist], dtype=mx.float32)
    f = 0.82 * width
    return Camera(
        R=R,
        t=t,
        fx=f,
        fy=f,
        cx=width / 2,
        cy=height / 2,
        width=width,
        height=height,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", help="Dynamic 3D Gaussians params.npz")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--frames", type=int, default=None, help="number of timesteps to play")
    parser.add_argument("--static-camera", action="store_true", help="don't orbit while playing")
    parser.add_argument(
        "--foreground-only", action="store_true", help="drop background gaussians (seg mask)"
    )
    parser.add_argument("--export-dir", type=str, default=None, help="write PNG frames here")
    parser.add_argument("--gif", type=str, default=None, help="write an animated GIF here")
    parser.add_argument("--benchmark", action="store_true", help="compare against render_gaussians")
    args = parser.parse_args()

    seq = DynamicGaussianSequence(args.npz, foreground_only=args.foreground_only)
    frames_n = min(args.frames or seq.num_timesteps, seq.num_timesteps)
    print(f"{seq.num_gaussians:,} gaussians x {seq.num_timesteps} timesteps; playing {frames_n}")

    first = seq.timestep(0)
    renderer = FastGaussianRenderer(**first)
    bg = mx.zeros((3,))

    gif_frames = []
    t_update = t_render = 0.0
    t0 = time.perf_counter()
    for i in range(frames_n):
        state = seq.timestep(i)
        tu = time.perf_counter()
        renderer.update(**state)
        t_update += time.perf_counter() - tu
        cam_i = 0 if args.static_camera else i
        cam = orbit_camera(cam_i, seq.num_timesteps, args.width, args.height)
        tr = time.perf_counter()
        out = renderer.render(cam, background=bg)
        mx.eval(out["image"])
        t_render += time.perf_counter() - tr
        if args.export_dir:
            Path(args.export_dir).mkdir(parents=True, exist_ok=True)
            save_image(f"{args.export_dir}/frame_{i:04d}.png", out["image"])
        if args.gif:
            arr = (np.clip(np.array(out["image"]), 0, 1) * 255).astype(np.uint8)
            gif_frames.append(arr)
    wall = time.perf_counter() - t0
    print(
        f"fast playback: {wall / frames_n * 1000:.1f} ms/frame ({frames_n / wall:.1f} fps) "
        f"[update {t_update / frames_n * 1000:.1f} ms, render {t_render / frames_n * 1000:.1f} ms]"
    )

    if args.gif and gif_frames:
        from PIL import Image

        imgs = [Image.fromarray(f) for f in gif_frames]
        imgs[0].save(
            args.gif, save_all=True, append_images=imgs[1:], duration=33, loop=0, optimize=True
        )
        print(f"wrote {args.gif} ({Path(args.gif).stat().st_size / 1e6:.1f} MB)")

    if args.benchmark:
        n_bench = min(30, frames_n)
        t0 = time.perf_counter()
        for i in range(n_bench):
            state = seq.timestep(i)
            cam = orbit_camera(i, seq.num_timesteps, args.width, args.height)
            out = render_gaussians(
                cam,
                state["means"],
                state["quats"],
                state["scales"],
                state["opacities"],
                colors=state["colors"],
                background=bg,
            )
            mx.eval(out["image"])
        wall_ref = time.perf_counter() - t0
        print(
            f"reference playback: {wall_ref / n_bench * 1000:.1f} ms/frame "
            f"({n_bench / wall_ref:.1f} fps)"
        )


if __name__ == "__main__":
    main()
