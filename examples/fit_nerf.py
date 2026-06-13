"""Train a small NeRF on a synthetic scene generated in-process (no data needed).

Builds ground-truth views by volume-rendering an analytic glowing sphere from
six cameras, then fits a compact NeRF to them by optimizing random ray batches
with the volume-rendering loss, and finally renders a held-out view. This
exercises the full NeRF path — the ``NeRF`` MLP, ``render_rays``, and the
volume renderer — without any dataset download. To train on real
NeRF-synthetic scenes, see ``train_nerf.py``.

Usage:
    python examples/fit_nerf.py [--iters 600] [--res 48]
"""

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.exists():
    sys.path.insert(0, str(_SRC))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from mlx3d.cameras import Camera
from mlx3d.io import save_image
from mlx3d.nn import NeRF, render_rays
from mlx3d.renderer import sample_along_rays, volume_render

NEAR, FAR = 1.0, 4.0


def _camera(azim: float, res: int) -> Camera:
    a = mx.radians(mx.array(azim))
    return Camera.look_at(
        eye=(2.8 * float(mx.sin(a)), 1.0, 2.8 * float(mx.cos(a))),
        at=(0, 0, 0),
        up=(0, 1, 0),
        fov=45.0,
        width=res,
        height=res,
    )


def _analytic_field(points: mx.array) -> tuple[mx.array, mx.array]:
    """Ground-truth radiance field: a soft glowing colored sphere."""
    r = mx.linalg.norm(points, axis=-1)
    density = 30.0 * mx.maximum(0.6 - mx.abs(r - 0.6), 0.0)
    rgb = 0.5 * (points / mx.maximum(r[..., None], 1e-6)) + 0.5
    return density, rgb


def _ground_truth(camera: Camera, samples: int = 96) -> mx.array:
    o, d = camera.generate_rays()
    h, w = o.shape[:2]
    points, t_vals = sample_along_rays(
        o.reshape(-1, 3), d.reshape(-1, 3), NEAR, FAR, samples, stratified=False
    )
    density, rgb = _analytic_field(points)
    out = volume_render(density, rgb, t_vals, d.reshape(-1, 3))
    return out["rgb"].reshape(h, w, 3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=600)
    parser.add_argument("--res", type=int, default=48)
    parser.add_argument("--batch-rays", type=int, default=1024)
    parser.add_argument("--out", type=str, default="outputs/fit_nerf.png")
    args = parser.parse_args()

    # Flatten all training rays + their ground-truth colors.
    all_o, all_d, all_c = [], [], []
    for azim in (0.0, 60.0, 120.0, 180.0, 240.0, 300.0):
        cam = _camera(azim, args.res)
        o, d = cam.generate_rays()
        all_o.append(o.reshape(-1, 3))
        all_d.append(d.reshape(-1, 3))
        all_c.append(_ground_truth(cam).reshape(-1, 3))
    origins = mx.concatenate(all_o)
    directions = mx.concatenate(all_d)
    targets = mx.concatenate(all_c)
    n_rays = origins.shape[0]

    model = NeRF(pos_freqs=6, dir_freqs=4, hidden_dim=64, num_layers=4, skip_layer=2)
    optimizer = optim.Adam(learning_rate=1e-3)

    def loss_fn(model, o, d, c):
        out = render_rays(model, o, d, NEAR, FAR, num_coarse=64)
        return mx.mean((out["rgb"] - c) ** 2)

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    for it in range(args.iters):
        idx = mx.random.randint(0, n_rays, shape=(args.batch_rays,))
        loss, grads = loss_and_grad(model, origins[idx], directions[idx], targets[idx])
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        if it % 150 == 0 or it == args.iters - 1:
            print(f"iter {it:4d}  loss {float(loss):.5f}")

    # Render a held-out view (azimuth between training cameras), in ray chunks.
    cam = _camera(30.0, args.res * 2)
    o, d = cam.generate_rays()
    h, w = o.shape[:2]
    o, d = o.reshape(-1, 3), d.reshape(-1, 3)
    chunks = [
        render_rays(
            model, o[s : s + 4096], d[s : s + 4096], NEAR, FAR, num_coarse=64, stratified=False
        )["rgb"]
        for s in range(0, o.shape[0], 4096)
    ]
    image = mx.concatenate(chunks).reshape(h, w, 3)
    mx.eval(image)
    save_image(args.out, image)
    print(f"Saved held-out render to {args.out}")


if __name__ == "__main__":
    main()
