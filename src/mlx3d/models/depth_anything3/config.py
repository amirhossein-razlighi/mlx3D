"""Architecture presets for Depth Anything 3 (DA3) multi-view geometry models."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["DA3Config", "PRESETS"]


@dataclass(frozen=True)
class DA3Config:
    """Hyperparameters of a DA3 network (DINOv2 backbone + DualDPT head).

    Attributes:
        repo_id: Hugging Face repository holding the pretrained weights.
        license: license of those weights (the MLX code itself is MIT).
        embed_dim, depth, num_heads: ViT width, number of blocks, attention heads.
        out_layers: blocks whose tokens feed the dense head (and camera decoder).
        alt_start: first block that alternates frame-local and cross-view
            (global) attention; odd blocks from here on are global.
        qknorm_start, rope_start: first block using QK-LayerNorm / 2D RoPE.
        head_features: DPT fusion width.
        head_out_channels: per-stage DPT projection widths.
    """

    repo_id: str
    license: str
    embed_dim: int
    depth: int
    num_heads: int
    out_layers: tuple[int, ...]
    alt_start: int
    qknorm_start: int
    rope_start: int
    head_features: int
    head_out_channels: tuple[int, int, int, int]
    patch_size: int = 14
    pretrain_grid: int = 37
    """Side length of the patch grid the positional embedding was trained at
    (518 px / 14)."""
    rope_frequency: float = 100.0
    mlp_ratio: int = 4

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads


PRESETS: dict[str, DA3Config] = {
    "da3-small": DA3Config(
        repo_id="depth-anything/DA3-SMALL",
        license="apache-2.0",
        embed_dim=384,
        depth=12,
        num_heads=6,
        out_layers=(5, 7, 9, 11),
        alt_start=4,
        qknorm_start=4,
        rope_start=4,
        head_features=64,
        head_out_channels=(48, 96, 192, 384),
    ),
    "da3-base": DA3Config(
        repo_id="depth-anything/DA3-BASE",
        license="apache-2.0",
        embed_dim=768,
        depth=12,
        num_heads=12,
        out_layers=(5, 7, 9, 11),
        alt_start=4,
        qknorm_start=4,
        rope_start=4,
        head_features=128,
        head_out_channels=(96, 192, 384, 768),
    ),
    "da3-large": DA3Config(
        repo_id="depth-anything/DA3-LARGE",
        license="cc-by-nc-4.0",
        embed_dim=1024,
        depth=24,
        num_heads=16,
        out_layers=(11, 15, 19, 23),
        alt_start=8,
        qknorm_start=8,
        rope_start=8,
        head_features=256,
        head_out_channels=(256, 512, 1024, 1024),
    ),
}
