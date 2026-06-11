"""Train 3D Gaussian Splatting on a COLMAP or Blender-synthetic scene.

COLMAP scene (the standard 3DGS input — a directory with sparse/0 and images):

    python examples/train_gaussian_splatting.py --data /path/to/scene --iters 7000

Blender-synthetic scene:

    python examples/train_gaussian_splatting.py --data /path/to/nerf_synthetic/lego \\
        --format blender --iters 7000

Saves a standard 3DGS PLY checkpoint (viewable in any splat viewer) and
periodic renders.
"""

import argparse
import os

import mlx.core as mx
import numpy as np

from mlx3d.losses import psnr
from mlx3d.splatting import GaussianModel, GaussianTrainer, TrainerConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--format", choices=["colmap", "blender"], default="colmap")
    parser.add_argument("--iters", type=int, default=7000)
    parser.add_argument("--downscale", type=int, default=1)
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--init-points", type=int, default=30000,
                        help="random init size for blender scenes (no SfM points)")
    parser.add_argument("--out", type=str, default="outputs/gs")
    args = parser.parse_args()

    print("Loading dataset...")
    if args.format == "colmap":
        from mlx3d.datasets import load_colmap

        ds = load_colmap(args.data, downscale=args.downscale)
        init_points, init_colors = ds.points, ds.point_colors
        scene_extent = ds.scene_extent
        white_bg = False
    else:
        from mlx3d.datasets import load_blender

        ds = load_blender(args.data, "train", downscale=args.downscale)
        # Random init inside the synthetic scenes' bounding box.
        init_points = mx.random.uniform(low=-1.5, high=1.5, shape=(args.init_points, 3))
        init_colors = mx.random.uniform(shape=(args.init_points, 3))
        scene_extent = 3.0
        white_bg = True

    print(f"{len(ds)} views, init with {init_points.shape[0]} points, "
          f"extent {scene_extent:.2f}")

    model = GaussianModel.from_points(init_points, init_colors, sh_degree=args.sh_degree)
    config = TrainerConfig(white_background=white_bg, densify_until=args.iters // 2)
    trainer = GaussianTrainer(model, config, scene_extent=scene_extent)

    os.makedirs(args.out, exist_ok=True)
    order = np.random.permutation(len(ds))
    cursor = 0
    for it in range(1, args.iters + 1):
        if cursor >= len(ds):
            order = np.random.permutation(len(ds))
            cursor = 0
        cam, img = ds[int(order[cursor])]
        cursor += 1

        info = trainer.step(cam, img)
        if it % 100 == 0:
            print(f"iter {it:6d}  loss {info['loss']:.5f}  gaussians {info['num_gaussians']}")
        if it % 1000 == 0 or it == args.iters:
            save_view(model, ds, white_bg, os.path.join(args.out, f"render_{it:06d}.png"))
            model.save_ply(os.path.join(args.out, "point_cloud.ply"))

    print(f"Done. Checkpoint: {args.out}/point_cloud.ply")


def save_view(model, ds, white_bg, path):
    from PIL import Image

    cam, img = ds[0]
    bg = mx.ones((3,)) if white_bg else mx.zeros((3,))
    out = model.render(cam, background=bg)
    print(f"  eval view PSNR: {float(psnr(out['image'], img)):.2f} dB")
    arr = (np.clip(np.array(out["image"]), 0, 1) * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


if __name__ == "__main__":
    main()
