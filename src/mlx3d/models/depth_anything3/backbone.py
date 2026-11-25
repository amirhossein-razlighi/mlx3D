"""DINOv2 ViT with DA3's alternating frame/cross-view attention.

The first ``alt_start`` blocks are a plain DINOv2 encoder run independently on
every view. From ``alt_start`` on, the class token is swapped for a learned
camera token (one for the reference view, one shared by the others) and odd
blocks attend jointly over the tokens of *all* views, which is what lets the
network reason about relative pose. Tokens are laid out ``(B, S, N, C)``:
batch, views, tokens per view (class + patches), channels.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .._resample import resize_bicubic
from .config import DA3Config

__all__ = ["DinoBackbone", "LayerOutput", "RefViewStrategy"]

RefViewStrategy = Literal["first", "middle", "saddle_balanced", "saddle_sim_range"]

# The reference view only changes anything once there are this many views.
_MIN_VIEWS_FOR_REF_SELECTION = 3


@dataclass
class LayerOutput:
    """Tokens of one tapped block, as consumed by the DPT head and camera decoder.

    Attributes:
        patch_tokens: ``(B, S, P, 2C)`` frame-local tokens concatenated with the
            (final-normed) cross-view tokens, for the ``P`` patch positions.
        camera_token: ``(B, S, 2C)`` the per-view camera token, un-normalized.
    """

    patch_tokens: mx.array
    camera_token: mx.array

    def arrays(self) -> tuple[mx.array, mx.array]:
        """Both arrays, e.g. for ``mx.eval(*tap.arrays())`` (``mx.eval`` does not
        traverse dataclasses)."""
        return self.patch_tokens, self.camera_token


# --------------------------------------------------------------------------- #
# Rotary position embedding
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _Rope:
    """Precomputed 2D RoPE rotation for a fixed token layout (broadcastable)."""

    cos: mx.array
    sin: mx.array


@lru_cache(maxsize=32)
def _rope_tables(positions: tuple[tuple[int, int], ...], head_dim: int, base: float) -> _Rope:
    """Cos/sin tables for 2D RoPE: half the channels rotate with y, half with x."""
    pos = np.asarray(positions, dtype=np.float64)  # (N, 2) as (y, x)
    axis_dim = head_dim // 2
    inv_freq = 1.0 / base ** (np.arange(0, axis_dim, 2, dtype=np.float64) / axis_dim)
    tables = []
    for axis in (0, 1):
        angles = pos[:, axis : axis + 1] * inv_freq[None]  # (N, axis_dim / 2)
        tables.append(np.concatenate([angles, angles], axis=-1))
    angles = np.concatenate(tables, axis=-1)  # (N, head_dim)
    return _Rope(cos=mx.array(np.cos(angles), mx.float32), sin=mx.array(np.sin(angles), mx.float32))


@mx.compile
def _apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Rotate ``(..., N, D)`` by RoPE tables; each D/2 axis half is rotated in pairs
    of its own halves (``rotate_half`` convention)."""
    *lead, d = x.shape
    q = x.reshape(*lead, 2, 2, d // 4)  # (axis, half, d/4)
    rotated = mx.concatenate([-q[..., 1:2, :], q[..., 0:1, :]], axis=-2).reshape(x.shape)
    return x * cos.astype(x.dtype) + rotated * sin.astype(x.dtype)


def _local_positions(grid_h: int, grid_w: int) -> tuple[tuple[int, int], ...]:
    """Per-view positions: special token at (0, 0), patches at (y + 1, x + 1)."""
    patches = [(y + 1, x + 1) for y in range(grid_h) for x in range(grid_w)]
    return ((0, 0), *patches)


def _global_positions(n_tokens: int, n_views: int) -> tuple[tuple[int, int], ...]:
    """Cross-view positions: every patch shares (1, 1), so attention across views
    is translation-free while still separating the camera token."""
    per_view = ((0, 0),) + ((1, 1),) * (n_tokens - 1)
    return per_view * n_views


# --------------------------------------------------------------------------- #
# Transformer block
# --------------------------------------------------------------------------- #


class LayerScale(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        return x * self.gamma


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int, out: int | None = None):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, out or dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qk_norm: bool):
        super().__init__()
        self.num_heads = num_heads
        self.qk_norm = qk_norm
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        if qk_norm:
            head_dim = dim // num_heads
            self.q_norm = nn.LayerNorm(head_dim, eps=1e-5)
            self.k_norm = nn.LayerNorm(head_dim, eps=1e-5)

    def __call__(self, x: mx.array, rope: _Rope | None = None) -> mx.array:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.transpose(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        if rope is not None:
            q = _apply_rope(q, rope.cos, rope.sin)
            k = _apply_rope(k, rope.cos, rope.sin)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=q.shape[-1] ** -0.5)
        return self.proj(out.transpose(0, 2, 1, 3).reshape(B, N, C))


class Block(nn.Module):
    """Pre-norm ViT block with LayerScale (DINOv2 layout)."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: int, qk_norm: bool = False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads, qk_norm)
        self.ls1 = LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, dim * mlp_ratio)
        self.ls2 = LayerScale(dim)

    def __call__(self, x: mx.array, rope: _Rope | None = None) -> mx.array:
        x = x + self.ls1(self.attn(self.norm1(x), rope))
        return x + self.ls2(self.mlp(self.norm2(x)))


# --------------------------------------------------------------------------- #
# Reference-view selection
# --------------------------------------------------------------------------- #


def select_reference_view(tokens: mx.array, strategy: RefViewStrategy) -> mx.array:
    """Pick the view that anchors the world frame, from ``(B, S, N, C)`` tokens.

    ``saddle_balanced`` (DA3's default) picks the view whose class token is
    closest to the median across similarity-to-others, norm, and variance — a
    "typical" view rather than an outlier. Returns ``(B,)`` int32 indices.
    """
    B, S = tokens.shape[:2]
    if strategy == "first" or S <= 1:
        return mx.zeros((B,), dtype=mx.int32)
    if strategy == "middle":
        return mx.full((B,), S // 2, dtype=mx.int32)

    cls = tokens[:, :, 0].astype(mx.float32)
    norm = mx.linalg.norm(cls, axis=-1, keepdims=True)
    unit = cls / norm
    sim = (unit @ unit.transpose(0, 2, 1)) - mx.eye(S)[None]
    if strategy == "saddle_sim_range":
        return mx.argmax(sim.max(axis=-1) - sim.min(axis=-1), axis=1).astype(mx.int32)
    if strategy != "saddle_balanced":
        raise ValueError(f"Unknown reference view strategy {strategy!r}.")

    def to_unit_range(m: mx.array) -> mx.array:
        lo, hi = m.min(axis=1, keepdims=True), m.max(axis=1, keepdims=True)
        return (m - lo) / (hi - lo + 1e-8)

    metrics = (
        sim.sum(axis=-1) / (S - 1),
        norm[..., 0],
        mx.var(unit, axis=-1, ddof=1),
    )
    score = sum(mx.abs(to_unit_range(m) - 0.5) for m in metrics)
    return mx.argmin(score, axis=1).astype(mx.int32)


def _reference_first_order(ref: mx.array, n_views: int) -> mx.array:
    """``(B, S)`` permutation moving view ``ref`` to the front, others in order."""
    pos = mx.arange(n_views)[None]
    ref = ref[:, None]
    order = mx.where((pos > 0) & (pos <= ref), pos - 1, pos)
    return mx.where(pos == 0, ref, order)


def _permute_views(x: mx.array, order: mx.array) -> mx.array:
    index = order.reshape(*order.shape, *([1] * (x.ndim - 2)))
    return mx.take_along_axis(x, index, axis=1)


# --------------------------------------------------------------------------- #
# Backbone
# --------------------------------------------------------------------------- #


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int, dim: int):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.proj(x)  # (B, h, w, C)
        return x.reshape(x.shape[0], -1, x.shape[-1])


class DinoBackbone(nn.Module):
    """DA3's multi-view DINOv2 encoder.

    Call with normalized NHWC images ``(B, S, H, W, 3)`` (H, W multiples of the
    patch size); returns one :class:`LayerOutput` per ``config.out_layers``.
    """

    def __init__(self, config: DA3Config):
        super().__init__()
        c = config
        self.config = c
        self.patch_embed = PatchEmbed(c.patch_size, c.embed_dim)
        self.cls_token = mx.zeros((1, 1, c.embed_dim))
        self.camera_token = mx.zeros((1, 2, c.embed_dim))
        self.pos_embed = mx.zeros((1, 1 + c.pretrain_grid**2, c.embed_dim))
        self.blocks = [
            Block(c.embed_dim, c.num_heads, c.mlp_ratio, qk_norm=i >= c.qknorm_start)
            for i in range(c.depth)
        ]
        self.norm = nn.LayerNorm(c.embed_dim, eps=1e-5)

    def _positional_embedding(self, grid_h: int, grid_w: int) -> mx.array:
        """Pretrained (1, 1 + 37*37, C) embedding resampled to the input grid."""
        g = self.config.pretrain_grid
        if (grid_h, grid_w) == (g, g):
            return self.pos_embed
        cls_pos, patch_pos = self.pos_embed[:, :1], self.pos_embed[:, 1:]
        patch_pos = patch_pos.reshape(1, g, g, -1)
        # DINOv2's historical +0.1 offset keeps floor(g * scale) == grid size.
        scale = ((grid_h + 0.1) / g, (grid_w + 0.1) / g)
        patch_pos = resize_bicubic(patch_pos, scale).reshape(1, grid_h * grid_w, -1)
        return mx.concatenate([cls_pos, patch_pos], axis=1)

    def _embed(self, images: mx.array) -> mx.array:
        B, S, H, W, _ = images.shape
        p = self.config.patch_size
        x = self.patch_embed(images.reshape(B * S, H, W, 3))
        cls = mx.broadcast_to(self.cls_token, (B * S, 1, x.shape[-1])).astype(x.dtype)
        x = mx.concatenate([cls, x], axis=1)
        x = x + self._positional_embedding(H // p, W // p).astype(x.dtype)
        return x.reshape(B, S, *x.shape[1:])

    def _camera_tokens(self, batch: int, n_views: int, dtype: mx.Dtype) -> mx.array:
        ref, src = self.camera_token[:, :1], self.camera_token[:, 1:]
        tokens = mx.concatenate([ref, mx.repeat(src, n_views - 1, axis=1)], axis=1)
        return mx.broadcast_to(tokens, (batch, n_views, tokens.shape[-1])).astype(dtype)

    def __call__(
        self, images: mx.array, ref_view_strategy: RefViewStrategy = "saddle_balanced"
    ) -> list[LayerOutput]:
        c = self.config
        B, S, H, W, _ = images.shape
        if H % c.patch_size or W % c.patch_size:
            raise ValueError(f"Image size {H}x{W} must be a multiple of {c.patch_size}.")

        x = self._embed(images)
        N = x.shape[2]
        head_dim = c.head_dim
        local_rope = _rope_tables(
            _local_positions(H // c.patch_size, W // c.patch_size), head_dim, c.rope_frequency
        )
        global_rope = _rope_tables(_global_positions(N, S), head_dim, c.rope_frequency)

        order = None  # view permutation applied for reference-view selection
        local_x = x
        outputs: list[LayerOutput] = []
        for i, block in enumerate(self.blocks):
            if i == c.alt_start - 1 and S >= _MIN_VIEWS_FOR_REF_SELECTION:
                order = _reference_first_order(select_reference_view(x, ref_view_strategy), S)
                x, local_x = _permute_views(x, order), _permute_views(local_x, order)
            if i == c.alt_start:
                cam = self._camera_tokens(B, S, x.dtype)
                x = mx.concatenate([cam[:, :, None], x[:, :, 1:]], axis=2)

            use_rope = i >= c.rope_start
            if i >= c.alt_start and i % 2 == 1:
                y = block(x.reshape(B, S * N, -1), global_rope if use_rope else None)
                x = y.reshape(B, S, N, -1)
            else:
                y = block(x.reshape(B * S, N, -1), local_rope if use_rope else None)
                x = local_x = y.reshape(B, S, N, -1)

            if i in c.out_layers:
                outputs.append(self._tap(local_x, x, order))
        return outputs

    def _tap(self, local_x: mx.array, x: mx.array, order: mx.array | None) -> LayerOutput:
        tokens = mx.concatenate([local_x, x], axis=-1)
        if order is not None:  # undo the reference-first permutation
            tokens = _permute_views(tokens, mx.argsort(order, axis=1))
        C = local_x.shape[-1]
        normed = mx.concatenate([tokens[..., :C], self.norm(tokens[..., C:])], axis=-1)
        return LayerOutput(patch_tokens=normed[:, :, 1:], camera_token=tokens[:, :, 0])
