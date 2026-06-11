"""Train a NeRF on a Blender-synthetic scene (e.g. the NeRF lego dataset).

Download a scene from the official NeRF synthetic dataset, then:

    python examples/train_nerf.py --data /path/to/nerf_synthetic/lego \\
        --iters 20000 --downscale 4

Renders a test view to outputs/nerf_view.png every 1000 iterations.
"""

import argparse
import os

import numpy as np

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from mlx3d.datasets import load_blender
from mlx3d.losses import psnr
from mlx3d.nn import NeRF, render_rays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--iters", type=int, default=20000)
    parser.add_argument("--batch-rays", type=int, default=1024)
    parser.add_argument("--downscale", type=int, default=4)
    parser.add_argument("--num-coarse", type=int, default=64)
    parser.add_argument("--num-fine", type=int, default=64)
    parser.add_argument("--near", type=float, default=2.0)
    parser.add_argument("--far", type=float, default=6.0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--out", type=str, default="outputs")
    args = parser.parse_args()

    print("Loading dataset...")
    train = load_blender(args.data, "train", downscale=args.downscale)
    print(f"{len(train)} training views, {train.images[0].shape[1]}x{train.images[0].shape[0]}")

    # Pre-generate all rays (origins, directions, colors) and shuffle.
    all_o, all_d, all_c = [], [], []
    for cam, img in zip(train.cameras, train.images):
        o, d = cam.generate_rays()
        all_o.append(o.reshape(-1, 3))
        all_d.append(d.reshape(-1, 3))
        all_c.append(img.reshape(-1, 3))
    origins = mx.concatenate(all_o)
    dirs = mx.concatenate(all_d)
    colors = mx.concatenate(all_c)
    n_rays = origins.shape[0]
    print(f"{n_rays} rays")

    model = NeRF()
    optimizer = optim.Adam(learning_rate=args.lr)

    def loss_fn(model, o, d, c):
        out = render_rays(
            model, o, d, args.near, args.far,
            num_coarse=args.num_coarse, num_fine=args.num_fine,
            white_background=True,
        )
        loss = ((out["rgb"] - c) ** 2).mean()
        if "rgb_coarse" in out:
            loss = loss + ((out["rgb_coarse"] - c) ** 2).mean()
        return loss

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    os.makedirs(args.out, exist_ok=True)

    perm = np.random.permutation(n_rays)
    cursor = 0
    for it in range(1, args.iters + 1):
        if cursor + args.batch_rays > n_rays:
            perm = np.random.permutation(n_rays)
            cursor = 0
        idx = mx.array(perm[cursor : cursor + args.batch_rays])
        cursor += args.batch_rays

        loss, grads = loss_and_grad(model, origins[idx], dirs[idx], colors[idx])
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        if it % 100 == 0:
            print(f"iter {it:6d}  loss {float(loss):.5f}  psnr {-10*np.log10(float(loss)/2):.2f}")

        if it % 1000 == 0:
            render_view(model, train.cameras[0], args, os.path.join(args.out, "nerf_view.png"))


def render_view(model, cam, args, path, chunk=4096):
    from PIL import Image

    o, d = cam.generate_rays()
    o, d = o.reshape(-1, 3), d.reshape(-1, 3)
    pixels = []
    for s in range(0, o.shape[0], chunk):
        out = render_rays(
            model, o[s : s + chunk], d[s : s + chunk], args.near, args.far,
            num_coarse=args.num_coarse, num_fine=args.num_fine,
            stratified=False, white_background=True,
        )
        pixels.append(out["rgb"])
        mx.eval(pixels[-1])
    img = mx.concatenate(pixels).reshape(cam.height, cam.width, 3)
    arr = (np.clip(np.array(img), 0, 1) * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
