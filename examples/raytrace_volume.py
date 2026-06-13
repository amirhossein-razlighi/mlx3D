"""Ray-trace an analytic volume with the NeRF compositing rule (no data, no training).

Casts one ray per pixel from a camera, samples an analytic radiance field (a
soft glowing colored sphere), and composites it with
:func:`mlx3d.renderer.volume_render` — the exact quadrature a NeRF uses, here
driven by a closed-form field instead of an MLP. A good sanity check that the
camera ray generation and volume renderer agree.

Usage:
    python examples/raytrace_volume.py [--out outputs/volume.png] [--size 256]
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
from mlx3d.renderer import sample_along_rays, volume_render


def radiance_field(points: mx.array) -> tuple[mx.array, mx.array]:
    """Analytic density + RGB for a soft colored sphere of radius ~0.6.

    Returns ``(density, rgb)`` with shapes ``(..., )`` and ``(..., 3)``.
    """
    r = mx.linalg.norm(points, axis=-1)
    # Dense near the surface, fading smoothly outward.
    density = 30.0 * mx.maximum(0.6 - mx.abs(r - 0.6), 0.0)
    # Color by direction on the sphere -> a smooth RGB field.
    rgb = 0.5 * (points / mx.maximum(r[..., None], 1e-6)) + 0.5
    return density, rgb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="outputs/volume_render.png")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--samples", type=int, default=96)
    args = parser.parse_args()

    camera = Camera.look_at(
        eye=(1.8, 1.2, 1.8),
        at=(0, 0, 0),
        up=(0, 1, 0),
        fov=45.0,
        width=args.size,
        height=args.size,
    )
    origins, directions = camera.generate_rays()  # (H, W, 3) each
    h, w = args.size, args.size
    origins = origins.reshape(-1, 3)
    directions = directions.reshape(-1, 3)

    points, t_vals = sample_along_rays(
        origins, directions, near=1.0, far=4.0, num_samples=args.samples, stratified=False
    )
    density, rgb = radiance_field(points)
    out = volume_render(density, rgb, t_vals, directions, white_background=True)

    image = out["rgb"].reshape(h, w, 3)
    mx.eval(image)
    save_image(args.out, image)
    print(f"Saved {args.out}  (mean opacity: {float(out['acc'].mean()):.3f})")


if __name__ == "__main__":
    main()
