"""Map official (PyTorch) DA3 checkpoints onto the MLX module tree.

The rules are explicit and exhaustive: every checkpoint tensor is either
renamed onto exactly one MLX parameter or listed as intentionally unused, so a
new checkpoint layout fails loudly instead of loading half the weights.
"""

from __future__ import annotations

import re

import mlx.core as mx

__all__ = ["convert_checkpoint"]

# (pattern, replacement) applied to keys after stripping the ``model.`` prefix.
_RENAMES: list[tuple[str, str]] = [
    (r"^backbone\.pretrained\.", "backbone."),
    (r"^head\.resize_layers\.0\.", "head.upsample4."),
    (r"^head\.resize_layers\.1\.", "head.upsample2."),
    (r"^head\.resize_layers\.3\.", "head.downsample2."),
    (r"^head\.scratch\.layer(\d)_rn\.", lambda m: f"head.layer_rn.{int(m[1]) - 1}."),
    (
        r"^head\.scratch\.refinenet(\d)(_aux)?\.",
        lambda m: f"head.{'ray_fusion' if m[2] else 'fusion'}.refine.{int(m[1]) - 1}.",
    ),
    (r"\.resConfUnit(\d)\.", r".res_unit\1."),
    (r"^head\.scratch\.output_conv1\.", "head.depth_neck."),
    (r"^head\.scratch\.output_conv2\.0\.", "head.depth_conv."),
    (r"^head\.scratch\.output_conv2\.2\.", "head.depth_out."),
    # Only the finest aux level is used. Its LayerNorm is the one module shared
    # by all levels, which safetensors stores once, under level 0.
    (r"^head\.scratch\.output_conv1_aux\.3\.", "head.ray_neck."),
    (r"^head\.scratch\.output_conv2_aux\.3\.0\.", "head.ray_conv."),
    (r"^head\.scratch\.output_conv2_aux\.0\.2\.", "head.ray_norm."),
    (r"^head\.scratch\.output_conv2_aux\.3\.5\.", "head.ray_out."),
    (r"^cam_dec\.backbone\.0\.", "cam_dec.fc1."),
    (r"^cam_dec\.backbone\.2\.", "cam_dec.fc2."),
    (r"^cam_dec\.fc_fov\.0\.", "cam_dec.fc_fov."),
]

# Tensors with no MLX counterpart (yet): the pose-conditioning encoder and the
# coarser aux-pyramid levels, whose outputs DA3 computes but never returns.
_UNUSED = re.compile(
    r"^(cam_enc\."
    r"|head\.scratch\.output_conv1_aux\.[012]\."
    r"|head\.scratch\.output_conv2_aux\.[012]\.[05]\.)"
)

_CONV_TRANSPOSE = re.compile(r"^head\.upsample\d\.weight$")


def _rename(key: str) -> str:
    for pattern, repl in _RENAMES:
        key = re.sub(pattern, repl, key)
    return key


def convert_checkpoint(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """Convert a DA3 ``model.safetensors`` dict to MLX parameter names/layouts.

    Linear weights keep PyTorch's ``(out, in)`` layout; convolution kernels move
    from ``(out, in, kh, kw)`` to MLX's ``(out, kh, kw, in)`` and transposed
    convolutions from ``(in, out, kh, kw)`` to ``(out, kh, kw, in)``.
    """
    converted: dict[str, mx.array] = {}
    for raw_key, value in weights.items():
        key = raw_key.removeprefix("model.")
        if _UNUSED.match(key):
            continue
        key = _rename(key)
        if value.ndim == 4:
            axes = (1, 2, 3, 0) if _CONV_TRANSPOSE.match(key) else (0, 2, 3, 1)
            value = value.transpose(axes)
        if key in converted:
            raise ValueError(f"Two checkpoint tensors map onto {key!r}.")
        converted[key] = value
    return converted
