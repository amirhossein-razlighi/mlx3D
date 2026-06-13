"""Fit 3D Gaussians to a synthetic multi-view scene (no data needed).

Renders ground-truth views of an analytic colored sphere, initializes a random
Gaussian point cloud, and optimizes it with the real :class:`GaussianTrainer`
(the same training loop used for COLMAP/Blender scenes), then renders a held-out
view. This is the quickest way to see the splatting pipeline end to end; for
real scenes with densification tuned for quality, use
``train_gaussian_splatting.py``.

Usage:
    python examples/fit_gaussians.py [--iters 400]
"""

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.exists():
    sys.path.insert(0, str(_SRC))

import mlx.core as mx

from mlx3d.cameras import Camera
from mlx3d.io import save_image
from mlx3d.losses import psnr
from mlx3d.renderer import sample_along_rays, volume_render
from mlx3d.splatting import GaussianModel, GaussianTrainer, TrainerConfig

NEAR, FAR = 1.0, 4.0


def _camera(azim: float, elev: float, res: int) -> Camera:
    a, e = mx.radians(mx.array(azim)), mx.radians(mx.array(elev))
    d = 2.8
    eye = (
        d * float(mx.cos(e) * mx.sin(a)),
        d * float(mx.sin(e)),
        d * float(mx.cos(e) * mx.cos(a)),
    )
    return Camera.look_at(eye=eye, at=(0, 0, 0), up=(0, 1, 0), fov=45.0, width=res, height=res)


def _ground_truth(camera: Camera, samples: int = 96) -> mx.array:
    """A soft glowing colored sphere, volume-rendered for multi-view consistency."""
    o, d = camera.generate_rays()
    h, w = o.shape[:2]
    points, t_vals = sample_along_rays(
        o.reshape(-1, 3), d.reshape(-1, 3), NEAR, FAR, samples, False
    )
    r = mx.linalg.norm(points, axis=-1)
    density = 30.0 * mx.maximum(0.6 - mx.abs(r - 0.6), 0.0)
    rgb = 0.5 * (points / mx.maximum(r[..., None], 1e-6)) + 0.5
    return volume_render(density, rgb, t_vals, d.reshape(-1, 3))["rgb"].reshape(h, w, 3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=400)
    parser.add_argument("--res", type=int, default=64)
    parser.add_argument("--num-gaussians", type=int, default=4000)
    parser.add_argument("--out", type=str, default="outputs/fit_gaussians.png")
    args = parser.parse_args()

    # Training views: a ring of cameras at two elevations.
    views = []
    for elev in (-20.0, 20.0):
        for azim in range(0, 360, 45):
            cam = _camera(float(azim), elev, args.res)
            views.append((cam, _ground_truth(cam)))
    print(f"{len(views)} synthetic training views at {args.res}x{args.res}")

    # Random initial Gaussians inside a unit ball with random colors.
    key = mx.random.normal((args.num_gaussians, 3))
    points = key / mx.maximum(mx.linalg.norm(key, axis=-1, keepdims=True), 1e-6)
    points = points * mx.random.uniform(shape=(args.num_gaussians, 1)) ** (1 / 3) * 0.9
    colors = mx.random.uniform(shape=(args.num_gaussians, 3))
    model = GaussianModel.from_points(points, colors, sh_degree=2)

    config = TrainerConfig(
        densify_from=100,
        densify_until=args.iters,
        densify_every=100,
        opacity_reset_every=10_000,
        sh_increase_every=200,
        white_background=False,
    )
    trainer = GaussianTrainer(model, config, scene_extent=1.0)

    for it in range(args.iters):
        cam, target = views[it % len(views)]
        info = trainer.step(cam, target)
        if it % 100 == 0 or it == args.iters - 1:
            print(f"iter {it:4d}  loss {info['loss']:.5f}  gaussians {info['num_gaussians']}")

    # Held-out view between training azimuths.
    cam = _camera(22.5, 0.0, args.res * 2)
    out = model.render(cam)
    mx.eval(out["image"])
    save_image(args.out, out["image"])
    ref = _ground_truth(cam)
    print(f"Held-out PSNR: {float(psnr(out['image'], ref)):.2f} dB  ->  saved {args.out}")


if __name__ == "__main__":
    main()
