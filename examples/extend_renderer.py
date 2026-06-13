"""Plug your own renderer into the MLX3D pipeline (no data needed).

MLX3D's image renderers are just callables ``(camera, scene) -> RenderOutput``
(see :class:`mlx3d.renderer.Renderer`). That means you can drop in a custom
shading pass without subclassing anything and reuse the rest of the pipeline —
here, the same :func:`mlx3d.io.save_image` sink and a generic turntable loop
that accepts *any* renderer.

This defines a normal-shaded renderer and runs it side by side with the
built-in soft rasterizer.

Usage:
    python examples/extend_renderer.py [--size 256]
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
from mlx3d.renderer import Renderer, RenderOutput, render_mesh_soft
from mlx3d.structures import Meshes
from mlx3d.utils import ico_sphere


def render_normals(camera: Camera, mesh: Meshes, **kwargs) -> RenderOutput:
    """A custom renderer: shade each surface point by its world-space normal.

    Satisfies the :class:`~mlx3d.renderer.Renderer` protocol simply by taking a
    camera and returning ``image``/``alpha``/``depth``. Internally it reuses the
    library's soft rasterizer with per-vertex colors set to the normals, so it
    is differentiable and tile-free like everything else in the pipeline.
    """
    normals = mesh.verts_normals_packed()
    verts_colors = 0.5 * normals + 0.5  # map [-1, 1] -> [0, 1]
    return render_mesh_soft(camera, mesh, verts_colors=verts_colors, **kwargs)


def turntable(renderer: Renderer, mesh: Meshes, *, size: int, azim: float) -> RenderOutput:
    """Generic driver that works with ANY renderer satisfying the protocol."""
    camera = Camera.look_at(
        eye=(
            2.4 * mx.cos(mx.radians(mx.array(azim))).item(),
            1.4,
            2.4 * mx.sin(mx.radians(mx.array(azim))).item(),
        ),
        at=(0, 0, 0),
        up=(0, 1, 0),
        fov=45.0,
        width=size,
        height=size,
    )
    return renderer(camera, mesh, sigma=3e-3, background=(0.05, 0.05, 0.08))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--outdir", type=str, default="outputs")
    args = parser.parse_args()

    mesh = ico_sphere(level=3, radius=1.0)

    # The exact same turntable loop drives both renderers interchangeably.
    renderers: dict[str, Renderer] = {
        "builtin_soft": lambda cam, m, **kw: render_mesh_soft(
            cam, m, verts_colors=0.5 * m.verts_packed() + 0.5, **kw
        ),
        "custom_normals": render_normals,
    }

    for name, renderer in renderers.items():
        out = turntable(renderer, mesh, size=args.size, azim=35.0)
        mx.eval(out["image"])
        path = f"{args.outdir}/extend_{name}.png"
        save_image(path, out["image"])
        print(f"{name:>16}: saved {path}")


if __name__ == "__main__":
    main()
