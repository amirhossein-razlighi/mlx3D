"""Train an Instant-NGP-style hash-grid NeRF on a Blender-synthetic scene.

Much faster than the vanilla NeRF (``train_nerf.py``): the multi-resolution
hash grid reaches a recognizable reconstruction in a couple of minutes on an
Apple-Silicon GPU instead of hours.

    python examples/train_ngp_nerf.py --data /path/to/nerf_synthetic/lego \\
        --iters 3000 --downscale 4

Renders a training view to ``outputs/ngp_nerf.png`` periodically.
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
import numpy as np

from mlx3d.datasets import load_blender
from mlx3d.io import save_image
from mlx3d.nn import HashGridNeRF, render_rays


def render_view(model, cam, near, far, path, chunk=8192):
    o, d = cam.generate_rays()
    h, w = o.shape[:2]
    o, d = o.reshape(-1, 3), d.reshape(-1, 3)
    chunks = []
    for s in range(0, o.shape[0], chunk):
        out = render_rays(
            model,
            o[s : s + chunk],
            d[s : s + chunk],
            near,
            far,
            num_coarse=160,
            stratified=False,
            white_background=True,
        )
        chunks.append(out["rgb"])
        mx.eval(chunks[-1])
    save_image(path, mx.concatenate(chunks).reshape(h, w, 3))
    print(f"Saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--iters", type=int, default=3000)
    parser.add_argument("--downscale", type=int, default=4)
    parser.add_argument("--batch-rays", type=int, default=4096)
    parser.add_argument("--num-samples", type=int, default=96)
    parser.add_argument("--near", type=float, default=2.0)
    parser.add_argument("--far", type=float, default=6.0)
    parser.add_argument("--bound", type=float, default=1.5)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--out", type=str, default="outputs/ngp_nerf.png")
    args = parser.parse_args()

    train = load_blender(args.data, "train", downscale=args.downscale, white_background=True)
    all_o, all_d, all_c = [], [], []
    for cam, img in zip(train.cameras, train.images):
        o, d = cam.generate_rays()
        all_o.append(o.reshape(-1, 3))
        all_d.append(d.reshape(-1, 3))
        all_c.append(img.reshape(-1, 3))
    origins, dirs, colors = mx.concatenate(all_o), mx.concatenate(all_d), mx.concatenate(all_c)
    n_rays = origins.shape[0]
    print(f"{len(train)} views, {n_rays} rays")

    model = HashGridNeRF(bounds=(-args.bound, args.bound))
    optimizer = optim.Adam(learning_rate=args.lr)

    def loss_fn(model, o, d, c):
        out = render_rays(
            model, o, d, args.near, args.far, num_coarse=args.num_samples, white_background=True
        )
        return mx.mean((out["rgb"] - c) ** 2)

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    perm, cursor = np.random.permutation(n_rays), 0
    for it in range(1, args.iters + 1):
        if cursor + args.batch_rays > n_rays:
            perm, cursor = np.random.permutation(n_rays), 0
        idx = mx.array(perm[cursor : cursor + args.batch_rays])
        cursor += args.batch_rays
        loss, grads = loss_and_grad(model, origins[idx], dirs[idx], colors[idx])
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        if it % 200 == 0:
            print(
                f"iter {it:5d}  loss {float(loss):.5f}  psnr {-10 * np.log10(float(loss) + 1e-9):.2f}"
            )
        if it % 1000 == 0 or it == args.iters:
            render_view(model, train.cameras[0], args.near, args.far, args.out)


if __name__ == "__main__":
    main()
