"""Fused-MLP Metal kernel for small networks (Instant-NGP / tiny-cuda-nn style).

A small MLP (a few <=64-wide layers) is the inner loop of hash-grid NeRF and
splatting colour fields. Evaluated layer-by-layer with separate matmuls, each
intermediate activation is written to and read back from global memory; this
kernel instead evaluates the **whole MLP per sample in a single launch**,
keeping activations in registers.

Performance note: on Apple GPUs, MLX's native matmuls (MPS-backed GEMM) are
currently *faster* than this hand-written per-thread kernel — a naive fused
matvec doesn't exploit the GPU's matrix hardware the way a tiled GEMM does. So
:meth:`FusedMLP.__call__` (the differentiable MLX path) is the recommended path
for both training and inference. The fused kernel is provided as a correct,
self-contained reference (it matches ``__call__`` exactly) and a starting point
for a properly tiled fused kernel; it is not a drop-in speedup today.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

__all__ = ["FusedMLP"]

_MAX_WIDTH = 64

# One thread per input row. Reads layer dims from `dims` =
# [n_layers, d0, d1, ..., d_n]; weights are per-layer (d_in, d_out) row-major,
# concatenated in `weights`; biases concatenated in `biases`. ReLU on every
# layer except the last. meta = [num_rows].
_FUSED_SRC = """
    constexpr int MAXW = 64;
    const uint s = thread_position_in_grid.x;
    const int N = meta[0];
    if ((int)s >= N) return;

    const int n_layers = dims[0];
    int din = dims[1];
    const int out_dim = dims[1 + n_layers];

    float cur[MAXW];
    float nxt[MAXW];
    for (int i = 0; i < din; ++i) cur[i] = x[s * din + i];

    int woff = 0;
    int boff = 0;
    for (int L = 0; L < n_layers; ++L) {
        const int dout = dims[2 + L];
        for (int j = 0; j < dout; ++j) {
            float acc = biases[boff + j];
            for (int i = 0; i < din; ++i) {
                acc += weights[woff + i * dout + j] * cur[i];
            }
            nxt[j] = (L < n_layers - 1) ? metal::fmax(acc, 0.0f) : acc;
        }
        woff += din * dout;
        boff += dout;
        for (int j = 0; j < dout; ++j) cur[j] = nxt[j];
        din = dout;
    }

    for (int j = 0; j < out_dim; ++j) out[s * out_dim + j] = cur[j];
"""

_fused_kernel = mx.fast.metal_kernel(
    name="fused_mlp_forward",
    input_names=["x", "weights", "biases", "dims", "meta"],
    output_names=["out"],
    source=_FUSED_SRC,
)


class FusedMLP(nn.Module):
    """A small ReLU MLP with a fused Metal forward path.

    Args:
        layer_dims: sizes ``[in, h1, ..., out]``; every hidden/in/out dimension
            must be ``<= 64``. ReLU is applied after every layer except the last.
    """

    def __init__(self, layer_dims: list[int]):
        super().__init__()
        if any(d > _MAX_WIDTH for d in layer_dims):
            raise ValueError(f"FusedMLP supports dimensions <= {_MAX_WIDTH}; got {layer_dims}.")
        self.layer_dims = list(layer_dims)
        # Weights stored (in, out) so x @ W matches the kernel's row-major layout.
        self.weights = []
        self.biases = []
        for din, dout in zip(layer_dims[:-1], layer_dims[1:]):
            scale = (2.0 / din) ** 0.5
            self.weights.append(mx.random.normal((din, dout)) * scale)
            self.biases.append(mx.zeros((dout,)))

    def __call__(self, x: mx.array) -> mx.array:
        """Differentiable MLX forward (use for training)."""
        h = x
        n = len(self.weights)
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = h @ w + b
            if i < n - 1:
                h = nn.relu(h)
        return h

    def forward_fused(self, x: mx.array) -> mx.array:
        """Fused single-kernel forward; matches :meth:`__call__` exactly.

        A correct reference for the fused-MLP idea. See the module note: MLX's
        native matmuls are currently faster on Apple GPUs, so prefer
        :meth:`__call__` in practice.
        """
        rows = int(x.shape[0])
        dims = mx.array([len(self.weights), *self.layer_dims], dtype=mx.int32)
        meta = mx.array([rows], dtype=mx.int32)
        flat_w = mx.concatenate([w.reshape(-1) for w in self.weights])
        flat_b = mx.concatenate([b.reshape(-1) for b in self.biases])
        tg = 256
        grid = ((rows + tg - 1) // tg * tg, 1, 1)
        (out,) = _fused_kernel(
            inputs=[
                mx.contiguous(x.astype(mx.float32)),
                mx.contiguous(flat_w.astype(mx.float32)),
                mx.contiguous(flat_b.astype(mx.float32)),
                dims,
                meta,
            ],
            output_shapes=[(rows * self.layer_dims[-1],)],
            output_dtypes=[mx.float32],
            grid=grid,
            threadgroup=(tg, 1, 1),
        )
        return out.reshape(rows, self.layer_dims[-1])
