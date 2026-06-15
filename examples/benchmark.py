"""Micro-benchmarks for the MLX3D render paths (self-contained, no data).

Reports forward throughput on the local Apple-Silicon GPU for the mesh
rasterizers, Gaussian Splatting, and NeRF ray rendering. Use it to track
performance and to sanity-check that a change didn't regress speed.

    python examples/benchmark.py [--markdown]
"""

import argparse
import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.exists():
    sys.path.insert(0, str(_SRC))

import mlx.core as mx

from mlx3d.cameras import Camera
from mlx3d.nn import HashGridNeRF, NeRF, render_rays
from mlx3d.renderer import (
    interpolate_face_attributes,
    rasterize_meshes,
    render_mesh_soft,
)
from mlx3d.splatting import GaussianModel
from mlx3d.utils import ico_sphere


def _time(fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t) / iters


def bench_rasterizers(rows):
    mesh = ico_sphere(level=5, radius=1.0)  # 20480 faces
    v, f = mesh.verts_packed(), mesh.faces_packed()
    vc = 0.5 * v + 0.5
    nf = int(f.shape[0])
    for size in (256, 512, 1024):
        cam = Camera.look_at(eye=(2.2, 1.6, 2.2), at=(0, 0, 0), fov=45.0, width=size, height=size)

        def hard():
            frag = rasterize_meshes(cam, v, f)
            mx.eval(interpolate_face_attributes(frag, vc))

        dt = _time(hard)
        rows.append(
            (f"hard rasterizer  {nf} faces @ {size}px", f"{dt * 1e3:.1f} ms", f"{1 / dt:.0f} fps")
        )

    # Soft rasterizer is O(F*H*W); benchmark only at a modest size.
    cam = Camera.look_at(eye=(2.2, 1.6, 2.2), at=(0, 0, 0), fov=45.0, width=256, height=256)

    def soft():
        mx.eval(render_mesh_soft(cam, mesh, verts_colors=vc, sigma=3e-3)["image"])

    dt = _time(soft, iters=5)
    rows.append((f"soft rasterizer  {nf} faces @ 256px", f"{dt * 1e3:.1f} ms", f"{1 / dt:.1f} fps"))


def bench_splatting(rows):
    mx.random.seed(0)
    for n, size in ((10_000, 512), (100_000, 720)):
        pts = mx.random.normal((n, 3)) * 0.6
        model = GaussianModel.from_points(pts, mx.random.uniform(shape=(n, 3)), sh_degree=0)
        cam = Camera.look_at(eye=(0, 0, -3.0), at=(0, 0, 0), fov=60.0, width=size, height=size)

        def render():
            mx.eval(model.render(cam)["image"])

        dt = _time(render)
        rows.append(
            (
                f"gaussian splatting  {n:,} splats @ {size}px",
                f"{dt * 1e3:.1f} ms",
                f"{1 / dt:.0f} fps",
            )
        )


def bench_nerf(rows):
    mx.random.seed(0)
    o = mx.random.normal((4096, 3))
    d = mx.random.normal((4096, 3))
    d = d / mx.linalg.norm(d, axis=-1, keepdims=True)
    for name, model in (("vanilla NeRF", NeRF()), ("hash-grid NeRF", HashGridNeRF())):

        def render(m=model):
            mx.eval(render_rays(m, o, d, 2.0, 6.0, num_coarse=64)["rgb"])

        dt = _time(render, iters=5)
        rays_s = 4096 / dt
        rows.append(
            (f"{name}  4096 rays x 64 samples", f"{dt * 1e3:.1f} ms", f"{rays_s / 1e3:.0f}k rays/s")
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--markdown", action="store_true", help="emit a Markdown table")
    args = parser.parse_args()

    rows: list[tuple[str, str, str]] = []
    bench_rasterizers(rows)
    bench_splatting(rows)
    bench_nerf(rows)

    if args.markdown:
        print("| Benchmark | Latency | Throughput |")
        print("| --- | --- | --- |")
        for name, lat, thr in rows:
            print(f"| {name} | {lat} | {thr} |")
    else:
        width = max(len(r[0]) for r in rows)
        for name, lat, thr in rows:
            print(f"{name:<{width}}  {lat:>10}  {thr:>14}")


if __name__ == "__main__":
    main()
