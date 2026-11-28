# Training on Real-World Captures

Phone captures break the assumptions of a clean benchmark: the object sits on
a turntable or in a changing room, auto-exposure shifts between shots, and
thin or textureless surfaces get no SfM points. Three opt-in training extras
address this, from the CLI or the Python API:

| Problem | Fix | Flag |
|---|---|---|
| Background you don't want (turntable, hands, room) | foreground masks | `--masks DIR`, `--alpha-mode` |
| Exposure / white balance changing between photos | per-image appearance | `--appearance` |
| Floaters, flat or wobbly surfaces | depth + normal priors | `--depth-prior` |

All three are off by default; without them training is unchanged.

## Foreground masks

Masks follow [COLMAP's layout](https://colmap.github.io/faq.html#mask-image-regions),
so one mask set serves training, evaluation and SfM: for `images/frame_001.jpg`
the mask is `masks/frame_001.jpg.png` (`masks/frame_001.png` also works).
White keeps a pixel, black ignores it, grays are soft weights. Grayscale,
RGB and RGBA mask files work, or use the photos' own alpha channel with
`--masks-from-alpha`. Missing, empty or wrongly sized masks fail before
training starts, with the offending file names.

There are two ways to use them:

- **`--alpha-mode masked`** (default) — background pixels are simply ignored:
  they get no loss and no gradient, and losses are normalized by mask area so
  small and large objects train alike. Use it when you want the object
  without caring what the splat does elsewhere.
- **`--alpha-mode transparent`** — the mask is the *target opacity*. The
  rendered alpha is supervised to match it and the photo is composited over
  a random background each step, so the result is a clean cut-out with an
  empty background and no color fringe.

```bash
mlx3d-capture photos/ --masks masks/                          # ignore the background
mlx3d-capture photos/ --masks masks/ --alpha-mode transparent # train a cut-out
```

<p align="center">
  <img src="../../assets/robust_cutout.jpg" width="90%" alt="Photo, mask and transparent cut-out splat" />
  <br/>
  <em>A garden photo, its mask, and the <code>--alpha-mode transparent</code> splat over a checkerboard from two views: an empty background and 0.92–0.96 IoU against the masks. (The masks here were made from Depth Anything 3 depth: points near the orbit's center, above the ground.)</em>
</p>

SSIM needs care with masks because it looks at 11×11 windows: pixels outside
the mask are replaced by the same constant in both images before filtering,
so windows on the silhouette never compare background content. The same
semantics apply when evaluating:

```bash
mlx3d-eval splat.ply --data captures/scene --masks masks/                          # masked metrics
mlx3d-eval splat.ply --data captures/scene --masks masks/ --alpha-mode transparent # + alpha MAE, IoU
```

Masks are not used for camera poses unless you ask: `--mask-sfm` restricts
COLMAP / built-in SfM features to the foreground, which helps when the
background moves but can hurt small objects that rely on background texture.

## Exposure and white-balance compensation

`--appearance` gives every photo a learnable 3×4 color transform (gain,
color mixing and offset), applied to the rendered foreground before the loss.
Lighting changes between photos are absorbed there instead of being "explained"
by floaters and view-dependent color. The saved splat is the uncorrected,
consistent scene.

<p align="center">
  <img src="../../assets/robust_appearance.jpg" width="90%" alt="Training with and without appearance compensation" />
  <br/>
  <em>Every one of 24 photos got a random gain (0.6–1.4×) and white-balance shift (left: one input). Trained without (2nd) and with <code>--appearance</code> (3rd), compared with the original photo (right): the color cast is gone and PSNR against the originals rises from 18.9 to 20.2 dB.</em>
</p>

## Depth and normal priors

When poses come from [Depth Anything 3](feedforward_geometry.md), every frame
also has a dense depth map. `--depth-prior` adds two losses against it:

- a **depth correlation** loss, `1 − Pearson(rendered depth, prior)`, which
  ignores the prior's unknown scale and shift;
- a **normal** loss between surface normals of the rendered and the prior
  depth, which flattens wobbly surfaces.

Low-confidence prior pixels are marked unknown and skipped, as are pixels the
splat does not cover yet.

```bash
mlx3d-capture photos/ --depth-prior     # needs DA3 poses (the default with mlx3d[models])
```

<p align="center">
  <img src="../../assets/robust_depth_prior.jpg" width="90%" alt="Rendered depth with and without the depth prior" />
  <br/>
  <em>Rendered depth after the same short training run, without (middle) and with <code>--depth-prior</code> (right): the floaters and streaks disappear, and PSNR improves from 20.7 to 21.2 dB.</em>
</p>

## Python API

Everything above is a `TrainerConfig` switch plus per-step inputs, so custom
training loops get the same behavior:

```python
from mlx3d.datasets import load_colmap
from mlx3d.splatting import GaussianTrainer, TrainerConfig

ds = load_colmap("scene", masks_dir="masks")            # ds.masks[i]: (H, W) in [0, 1]
config = TrainerConfig(
    alpha_mode="transparent", random_background=True,   # masks as target opacity
    appearance=True,                                     # per-view color transforms
    lambda_depth_prior=0.1, lambda_normal_prior=0.05,   # geometry priors
)
trainer = GaussianTrainer(model, config, scene_extent=ds.scene_extent, num_views=len(ds))
for i in order:
    cam, image = ds[i]
    trainer.step(cam, image, mask=ds.masks[i], depth_prior=priors[i], view_index=i)
```

The masked metrics are available as `mlx3d.losses.masked_l1`, `masked_psnr`
and `masked_ssim`.
