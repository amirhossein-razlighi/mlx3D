"""K-nearest-neighbor search between point sets."""

import mlx.core as mx

__all__ = ["knn_points", "knn_gather"]

_METAL_TOPK_MAX_K = 8
_METAL_BLOCK = 256

_KNN3D_SMALLK_SRC = """
    constexpr int BLOCK = 256;
    constexpr int MAX_K = 8;
    constexpr float INF = 3.402823466e+38F;

    const int p1_count = params[0];
    const int p2_count = params[1];
    const int K = params[2];

    const uint q = thread_position_in_grid.x;
    const uint lid = thread_index_in_threadgroup;
    const bool active = q < uint(p1_count);

    threadgroup float3 sm_ref[BLOCK];

    float3 query = float3(0.0f);
    if (active) {
        query = float3(p1[3 * q + 0], p1[3 * q + 1], p1[3 * q + 2]);
    }

    float best_d[MAX_K];
    int best_i[MAX_K];
    for (int k = 0; k < MAX_K; k++) {
        best_d[k] = INF;
        best_i[k] = 0;
    }

    for (int base = 0; base < p2_count; base += BLOCK) {
        const int r = base + int(lid);
        if (r < p2_count) {
            sm_ref[lid] = float3(p2[3 * r + 0], p2[3 * r + 1], p2[3 * r + 2]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active) {
            const int batch_size = min(BLOCK, p2_count - base);
            for (int j = 0; j < batch_size; j++) {
                const float3 diff = query - sm_ref[j];
                const float dist = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
                const int ref_i = base + j;

                if (dist < best_d[K - 1] || (dist == best_d[K - 1] && ref_i < best_i[K - 1])) {
                    int pos = K - 1;
                    while (
                        pos > 0
                        && (dist < best_d[pos - 1]
                            || (dist == best_d[pos - 1] && ref_i < best_i[pos - 1]))
                    ) {
                        best_d[pos] = best_d[pos - 1];
                        best_i[pos] = best_i[pos - 1];
                        pos--;
                    }
                    best_d[pos] = dist;
                    best_i[pos] = ref_i;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (active) {
        for (int k = 0; k < K; k++) {
            dists[q * K + k] = max(best_d[k], 0.0f);
            idx[q * K + k] = best_i[k];
        }
    }
"""

_knn3d_smallk_kernel = mx.fast.metal_kernel(
    name="knn3d_smallk",
    input_names=["p1", "p2", "params"],
    output_names=["dists", "idx"],
    source=_KNN3D_SMALLK_SRC,
)


def _pairwise_sqdist(a: mx.array, b: mx.array) -> mx.array:
    """Squared euclidean distances between ``a`` (..., P1, D) and ``b`` (..., P2, D)."""
    a2 = mx.sum(a * a, axis=-1, keepdims=True)  # (..., P1, 1)
    b2 = mx.sum(b * b, axis=-1, keepdims=True)  # (..., P2, 1)
    cross = a @ b.swapaxes(-1, -2)  # (..., P1, P2)
    d = a2 - 2.0 * cross + b2.swapaxes(-1, -2)
    return mx.maximum(d, 0.0)


def _can_use_metal_knn(p1: mx.array, p2: mx.array, K: int) -> bool:
    return (
        p1.ndim == 2
        and p2.ndim == 2
        and p1.shape[-1] == 3
        and p2.shape[-1] == 3
        and 1 <= K <= min(_METAL_TOPK_MAX_K, p2.shape[0])
    )


def _knn_points_3d_smallk(p1: mx.array, p2: mx.array, K: int) -> tuple[mx.array, mx.array]:
    p1_count, p2_count = p1.shape[0], p2.shape[0]
    params = mx.array([p1_count, p2_count, K], dtype=mx.int32)
    groups = (p1_count + _METAL_BLOCK - 1) // _METAL_BLOCK
    dists, idx = _knn3d_smallk_kernel(
        inputs=[p1.astype(mx.float32), p2.astype(mx.float32), params],
        output_shapes=[(p1_count, K), (p1_count, K)],
        output_dtypes=[mx.float32, mx.int32],
        grid=(groups * _METAL_BLOCK, 1, 1),
        threadgroup=(_METAL_BLOCK, 1, 1),
        init_value=0,
    )
    return dists, idx


def knn_points(
    p1: mx.array,
    p2: mx.array,
    K: int = 1,
    chunk_size: int = 4096,
) -> tuple[mx.array, mx.array]:
    """For each point in ``p1`` find its ``K`` nearest neighbors in ``p2``.

    Args:
        p1: (N, P1, D) or (P1, D) query points.
        p2: (N, P2, D) or (P2, D) reference points.
        K: number of neighbors.
        chunk_size: queries processed per tile to bound the (P1, P2) distance
            matrix memory.

    Returns:
        ``(dists, idx)`` of shapes (..., P1, K): squared distances and indices
        into ``p2``, sorted by increasing distance.
    """
    if K < 1:
        raise ValueError("K must be at least 1.")
    if p2.shape[-2] == 0:
        raise ValueError("p2 must contain at least one point.")
    K = min(K, p2.shape[-2])
    if _can_use_metal_knn(p1, p2, K):
        return _knn_points_3d_smallk(p1, p2, K)

    squeeze = p1.ndim == 2
    if squeeze:
        p1, p2 = p1[None], p2[None]
    P1, P2 = p1.shape[1], p2.shape[1]

    dists_out, idx_out = [], []
    for start in range(0, P1, chunk_size):
        d = _pairwise_sqdist(p1[:, start : start + chunk_size], p2)  # (N, c, P2)
        if K == 1:
            idx = mx.argmin(d, axis=-1, keepdims=True)
            dist = mx.take_along_axis(d, idx, axis=-1)
        elif K < P2:
            # argpartition + small sort instead of a full argsort of the
            # (chunk, P2) matrix, which would allocate K-independent memory.
            part = mx.argpartition(d, kth=K - 1, axis=-1)[..., :K]
            dist_part = mx.take_along_axis(d, part, axis=-1)
            order = mx.argsort(dist_part, axis=-1)
            idx = mx.take_along_axis(part, order, axis=-1)
            dist = mx.take_along_axis(dist_part, order, axis=-1)
        else:
            idx = mx.argsort(d, axis=-1)
            dist = mx.take_along_axis(d, idx, axis=-1)
        dists_out.append(dist)
        idx_out.append(idx)

    dists = mx.concatenate(dists_out, axis=1)
    idx = mx.concatenate(idx_out, axis=1)
    if squeeze:
        dists, idx = dists[0], idx[0]
    return dists, idx


def knn_gather(x: mx.array, idx: mx.array) -> mx.array:
    """Gather neighbor features: ``x`` (N, P2, C), ``idx`` (N, P1, K) -> (N, P1, K, C)."""
    N, P1, K = idx.shape
    C = x.shape[-1]
    flat = idx.reshape(N, P1 * K)
    gathered = mx.take_along_axis(x, flat[..., None].astype(mx.int32), axis=1)
    return gathered.reshape(N, P1, K, C)
