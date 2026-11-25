"""DualDPT dense head: per-pixel depth (+ confidence) and, optionally, ray maps.

A DPT decoder turns the four tapped token grids into an image-resolution map:
each stage is projected and resized into a feature pyramid (4x, 2x, 1x, 1/2x
the patch grid) and fused top-down by RefineNet-style blocks. DA3 runs two
independent fusion chains over the same pyramid — the *main* chain predicts
depth, the *aux* chain predicts a per-pixel camera ray (origin + direction) —
so the ray branch is only built when asked for.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .._resample import resize_bilinear
from .backbone import LayerOutput

__all__ = ["DualDPT"]


def _conv3x3(c_in: int, c_out: int, bias: bool = True) -> nn.Conv2d:
    return nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=bias)


def _conv1x1(c_in: int, c_out: int) -> nn.Conv2d:
    return nn.Conv2d(c_in, c_out, kernel_size=1)


@lru_cache(maxsize=32)
def _uv_embedding(height: int, width: int, channels: int, aspect: float) -> mx.array:
    """Sinusoidal embedding ``(H, W, C)`` of an aspect-correct UV grid.

    The grid spans ``[-x_span, x_span] x [-y_span, y_span]`` with spans
    normalized by the image diagonal; ``C / 2`` channels encode u, the rest v.
    """
    diag = (aspect**2 + 1.0) ** 0.5
    span_x, span_y = aspect / diag, 1.0 / diag
    u = np.linspace(-span_x * (width - 1) / width, span_x * (width - 1) / width, width)
    v = np.linspace(-span_y * (height - 1) / height, span_y * (height - 1) / height, height)
    uu, vv = np.meshgrid(u, v, indexing="xy")  # (H, W)

    half = channels // 2
    omega = 1.0 / 100.0 ** (np.arange(half // 2, dtype=np.float64) / (half / 2.0))

    def sincos(coord: np.ndarray) -> np.ndarray:
        angles = coord[..., None] * omega
        return np.concatenate([np.sin(angles), np.cos(angles)], axis=-1)

    return mx.array(np.concatenate([sincos(uu), sincos(vv)], axis=-1), dtype=mx.float32)


def _add_uv_embedding(x: mx.array, image_hw: tuple[int, int], ratio: float = 0.1) -> mx.array:
    H, W = image_hw
    pe = _uv_embedding(x.shape[1], x.shape[2], x.shape[3], W / H)
    return x + (ratio * pe).astype(x.dtype)


class ResidualConvUnit(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.conv1 = _conv3x3(features, features)
        self.conv2 = _conv3x3(features, features)

    def __call__(self, x: mx.array) -> mx.array:
        out = self.conv1(nn.relu(x))
        return self.conv2(nn.relu(out)) + x


class FusionBlock(nn.Module):
    """Merge a lateral feature into the top-down path, refine, upsample."""

    def __init__(self, features: int, has_residual: bool = True):
        super().__init__()
        self.has_residual = has_residual
        if has_residual:
            self.res_unit1 = ResidualConvUnit(features)
        self.res_unit2 = ResidualConvUnit(features)
        self.out_conv = _conv1x1(features, features)

    def __call__(
        self, x: mx.array, lateral: mx.array | None = None, size: tuple[int, int] | None = None
    ) -> mx.array:
        if lateral is not None and self.has_residual:
            x = x + self.res_unit1(lateral)
        x = self.res_unit2(x)
        size = size or (2 * x.shape[1], 2 * x.shape[2])
        return self.out_conv(resize_bilinear(x, size))


class FusionChain(nn.Module):
    """Four fusion blocks walking the pyramid from coarsest to finest."""

    def __init__(self, features: int):
        super().__init__()
        self.refine = [FusionBlock(features) for _ in range(3)] + [
            FusionBlock(features, has_residual=False)
        ]

    def __call__(self, pyramid: list[mx.array]) -> mx.array:
        l1, l2, l3, l4 = pyramid
        x = self.refine[3](l4, size=l3.shape[1:3])
        x = self.refine[2](x, l3, size=l2.shape[1:3])
        x = self.refine[1](x, l2, size=l1.shape[1:3])
        return self.refine[0](x, l1)


class DualDPT(nn.Module):
    """Depth + ray DPT head over ``(B*S, P, 2C)`` patch tokens.

    Args:
        dim_in: token width (``2 * embed_dim``: local and global halves).
        features: fusion width.
        out_channels: per-stage projection widths.
        patch_size: backbone patch size.
    """

    def __init__(
        self,
        dim_in: int,
        features: int,
        out_channels: tuple[int, int, int, int],
        patch_size: int = 14,
    ):
        super().__init__()
        self.patch_size = patch_size
        c1, c2, c3, c4 = out_channels
        neck, head = features // 2, 32

        self.norm = nn.LayerNorm(dim_in, eps=1e-5)
        self.projects = [_conv1x1(dim_in, c) for c in out_channels]
        # Stage resizers to (4x, 2x, 1x, 1/2x) of the patch grid; stage 3 is identity.
        self.upsample4 = nn.ConvTranspose2d(c1, c1, kernel_size=4, stride=4)
        self.upsample2 = nn.ConvTranspose2d(c2, c2, kernel_size=2, stride=2)
        self.downsample2 = nn.Conv2d(c4, c4, kernel_size=3, stride=2, padding=1)
        self.layer_rn = [_conv3x3(c, features, bias=False) for c in out_channels]

        # Depth branch.
        self.fusion = FusionChain(features)
        self.depth_neck = _conv3x3(features, neck)
        self.depth_conv = _conv3x3(neck, head)
        self.depth_out = _conv1x1(head, 2)  # log-depth, log-confidence

        # Ray branch (only the finest level of DA3's aux pyramid is ever used).
        self.ray_fusion = FusionChain(features)
        self.ray_neck = [
            _conv3x3(features, neck),
            _conv3x3(neck, features),
            _conv3x3(features, neck),
            _conv3x3(neck, features),
            _conv3x3(features, neck),
        ]
        self.ray_conv = _conv3x3(neck, head)
        self.ray_norm = nn.LayerNorm(head, eps=1e-5)
        self.ray_out = _conv1x1(head, 7)  # origin (3), direction (3), log-confidence

    def _pyramid(self, layers: list[mx.array], grid_hw: tuple[int, int], image_hw) -> list:
        resizers = (self.upsample4, self.upsample2, None, self.downsample2)
        pyramid = []
        for tokens, project, resize, rn in zip(layers, self.projects, resizers, self.layer_rn):
            x = self.norm(tokens).reshape(tokens.shape[0], *grid_hw, tokens.shape[-1])
            x = _add_uv_embedding(project(x), image_hw)
            if resize is not None:
                x = resize(x)
            pyramid.append(rn(x))
        return pyramid

    def __call__(
        self, layers: list[mx.array], image_hw: tuple[int, int], with_rays: bool = False
    ) -> dict[str, mx.array]:
        """Decode ``(B', P, 2C)`` token maps for ``B'`` frames.

        Returns ``depth`` and ``depth_conf`` ``(B', H, W)``; with ``with_rays``
        also ``ray`` ``(B', h, w, 6)`` and ``ray_conf`` ``(B', h, w)`` at 8x the
        patch grid.
        """
        H, W = image_hw
        grid = (H // self.patch_size, W // self.patch_size)
        pyramid = self._pyramid(layers, grid, image_hw)

        x = self.depth_neck(self.fusion(pyramid))
        x = _add_uv_embedding(resize_bilinear(x, (H, W)), image_hw)
        logits = self.depth_out(nn.relu(self.depth_conv(x)))
        out = {"depth": mx.exp(logits[..., 0]), "depth_conf": mx.exp(logits[..., 1]) + 1.0}

        if with_rays:
            r = self.ray_fusion(pyramid)
            for conv in self.ray_neck:
                r = conv(r)
            r = _add_uv_embedding(r, image_hw)
            r = self.ray_out(nn.relu(self.ray_norm(self.ray_conv(r))))
            out["ray"] = r[..., :6]
            out["ray_conf"] = mx.exp(r[..., 6]) + 1.0
        return out

    def decode(
        self, layers: list[LayerOutput], image_hw: tuple[int, int], with_rays: bool = False
    ) -> dict[str, mx.array]:
        """Decode backbone taps ``(B, S, P, 2C)`` to ``(B, S, ...)`` maps."""
        B, S = layers[0].patch_tokens.shape[:2]
        flat = [t.patch_tokens.reshape(B * S, *t.patch_tokens.shape[2:]) for t in layers]
        out = self(flat, image_hw, with_rays)
        return {k: v.reshape(B, S, *v.shape[1:]) for k, v in out.items()}
