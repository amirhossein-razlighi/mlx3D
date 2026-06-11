"""Tile-based Gaussian Splatting rasterization with custom Metal kernels.

This is a Metal translation of the CUDA rasterizer from "3D Gaussian
Splatting for Real-Time Radiance Field Rendering" (Kerbl et al. 2023), in
the tile-batched style of gsplat:

- **Forward**: one threadgroup per 16x16 tile. The depth-sorted Gaussians of
  the tile are streamed through threadgroup memory in batches of 256 while
  every thread alpha-composites its own pixel front-to-back, terminating
  early once transmittance is exhausted.
- **Backward**: same tiling, traversed back-to-front, accumulating gradients
  with respect to 2D means, conics, colors and opacities via atomic adds.

The pair is wrapped in :func:`mx.custom_function`, so MLX autodiff chains
the kernel gradients into the (pure-MLX) projection math upstream.
"""

import mlx.core as mx

from .tiles import TILE_SIZE

__all__ = ["rasterize"]

_BLOCK = TILE_SIZE * TILE_SIZE  # threads per tile / batch size

_FORWARD_SRC = """
    constexpr int TILE = 16;
    constexpr int BLOCK = 256;

    const int width = params[0];
    const int height = params[1];
    const int tiles_x = params[2];

    uint2 tile = threadgroup_position_in_grid.xy;
    uint2 lid = thread_position_in_threadgroup.xy;
    uint tidx = thread_index_in_threadgroup;

    const int tile_id = tile.y * tiles_x + tile.x;
    const int px = tile.x * TILE + lid.x;
    const int py = tile.y * TILE + lid.y;
    const bool inside = (px < width) && (py < height);
    const float2 pixf = float2(px + 0.5f, py + 0.5f);

    const int range_start = tile_ranges[2 * tile_id];
    const int range_end = tile_ranges[2 * tile_id + 1];
    const int num = range_end - range_start;
    const int n_batches = (num + BLOCK - 1) / BLOCK;

    threadgroup float2 sm_xy[BLOCK];
    threadgroup float4 sm_co[BLOCK];   // conic.a, conic.b, conic.c, opacity
    threadgroup float3 sm_rgb[BLOCK];

    float T = 1.0f;
    float3 acc = float3(0.0f);
    int last_contrib = 0;
    bool done = !inside;

    for (int b = 0; b < n_batches; b++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int load = range_start + b * BLOCK + tidx;
        if (load < range_end) {
            const int g = sorted_ids[load];
            sm_xy[tidx] = float2(means2d[2 * g], means2d[2 * g + 1]);
            sm_co[tidx] = float4(conics[3 * g], conics[3 * g + 1], conics[3 * g + 2],
                                 opacities[g]);
            sm_rgb[tidx] = float3(colors[3 * g], colors[3 * g + 1], colors[3 * g + 2]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (done) continue;

        const int batch_size = min(BLOCK, num - b * BLOCK);
        for (int j = 0; j < batch_size; j++) {
            const float2 d = sm_xy[j] - pixf;
            const float4 co = sm_co[j];
            const float power = -0.5f * (co.x * d.x * d.x + co.z * d.y * d.y)
                                - co.y * d.x * d.y;
            if (power > 0.0f) continue;
            const float alpha = min(0.99f, co.w * metal::exp(power));
            if (alpha < 1.0f / 255.0f) continue;
            const float next_T = T * (1.0f - alpha);
            if (next_T < 1e-4f) { done = true; break; }
            acc += sm_rgb[j] * (alpha * T);
            T = next_T;
            last_contrib = b * BLOCK + j + 1;
        }
    }

    if (inside) {
        const int pid = py * width + px;
        image[3 * pid + 0] = acc.x + T * background[0];
        image[3 * pid + 1] = acc.y + T * background[1];
        image[3 * pid + 2] = acc.z + T * background[2];
        final_T[pid] = T;
        n_contrib[pid] = last_contrib;
    }
"""

_BACKWARD_SRC = """
    constexpr int TILE = 16;
    constexpr int BLOCK = 256;

    const int width = params[0];
    const int height = params[1];
    const int tiles_x = params[2];

    uint2 tile = threadgroup_position_in_grid.xy;
    uint2 lid = thread_position_in_threadgroup.xy;
    uint tidx = thread_index_in_threadgroup;

    const int tile_id = tile.y * tiles_x + tile.x;
    const int px = tile.x * TILE + lid.x;
    const int py = tile.y * TILE + lid.y;
    const bool inside = (px < width) && (py < height);
    const float2 pixf = float2(px + 0.5f, py + 0.5f);
    const int pid = py * width + px;

    const int range_start = tile_ranges[2 * tile_id];
    const int range_end = tile_ranges[2 * tile_id + 1];
    const int num = range_end - range_start;
    const int n_batches = (num + BLOCK - 1) / BLOCK;

    threadgroup float2 sm_xy[BLOCK];
    threadgroup float4 sm_co[BLOCK];
    threadgroup float3 sm_rgb[BLOCK];
    threadgroup int sm_id[BLOCK];

    // Pixel state for the back-to-front sweep.
    float T_final = inside ? final_T[pid] : 1.0f;
    float T = T_final;
    int last = inside ? n_contrib[pid] : 0;
    float3 dL_dpix = float3(0.0f);
    float bg_dot = 0.0f;
    if (inside) {
        dL_dpix = float3(grad_image[3 * pid], grad_image[3 * pid + 1],
                         grad_image[3 * pid + 2]);
        // Gradients w.r.t. final transmittance (from the alpha output) enter
        // the same way as the background term: T_final = prod(1 - alpha_i).
        bg_dot = background[0] * dL_dpix.x + background[1] * dL_dpix.y
               + background[2] * dL_dpix.z + grad_final_T[pid];
    }
    float3 accum = float3(0.0f);
    float last_alpha = 0.0f;
    float3 last_color = float3(0.0f);

    for (int b = n_batches - 1; b >= 0; b--) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int load = range_start + b * BLOCK + tidx;
        if (load < range_end) {
            const int g = sorted_ids[load];
            sm_id[tidx] = g;
            sm_xy[tidx] = float2(means2d[2 * g], means2d[2 * g + 1]);
            sm_co[tidx] = float4(conics[3 * g], conics[3 * g + 1], conics[3 * g + 2],
                                 opacities[g]);
            sm_rgb[tidx] = float3(colors[3 * g], colors[3 * g + 1], colors[3 * g + 2]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!inside) continue;

        const int batch_size = min(BLOCK, num - b * BLOCK);
        for (int j = batch_size - 1; j >= 0; j--) {
            const int global_j = b * BLOCK + j;
            if (global_j >= last) continue;  // did not contribute in forward

            const float2 d = sm_xy[j] - pixf;
            const float4 co = sm_co[j];
            const float power = -0.5f * (co.x * d.x * d.x + co.z * d.y * d.y)
                                - co.y * d.x * d.y;
            if (power > 0.0f) continue;
            const float G = metal::exp(power);
            const float alpha = min(0.99f, co.w * G);
            if (alpha < 1.0f / 255.0f) continue;

            // Transmittance in front of this Gaussian.
            T = T / (1.0f - alpha);
            const float3 c = sm_rgb[j];

            // dL/dcolor = alpha * T * dL/dpix
            const float aT = alpha * T;

            // Color accumulated behind this Gaussian (for dL/dalpha).
            accum = last_alpha * last_color + (1.0f - last_alpha) * accum;
            last_alpha = alpha;
            last_color = c;

            float dL_dalpha = ((c.x - accum.x) * dL_dpix.x
                             + (c.y - accum.y) * dL_dpix.y
                             + (c.z - accum.z) * dL_dpix.z) * T;
            // Background contribution through T_final.
            dL_dalpha += (-T_final / (1.0f - alpha)) * bg_dot;

            const float dL_dG = co.w * dL_dalpha;
            const float gdx = G * d.x;
            const float gdy = G * d.y;
            // d(power)/d(mean) accounting for d = xy - pix.
            const float dL_dx = -(gdx * co.x + gdy * co.y) * dL_dG;
            const float dL_dy = -(gdy * co.z + gdx * co.y) * dL_dG;

            const int g = sm_id[j];
            atomic_fetch_add_explicit(&grad_means2d[2 * g + 0], dL_dx,
                                      memory_order_relaxed);
            atomic_fetch_add_explicit(&grad_means2d[2 * g + 1], dL_dy,
                                      memory_order_relaxed);
            atomic_fetch_add_explicit(&grad_conics[3 * g + 0],
                                      -0.5f * gdx * d.x * dL_dG, memory_order_relaxed);
            atomic_fetch_add_explicit(&grad_conics[3 * g + 1],
                                      -gdx * d.y * dL_dG, memory_order_relaxed);
            atomic_fetch_add_explicit(&grad_conics[3 * g + 2],
                                      -0.5f * gdy * d.y * dL_dG, memory_order_relaxed);
            atomic_fetch_add_explicit(&grad_colors[3 * g + 0], aT * dL_dpix.x,
                                      memory_order_relaxed);
            atomic_fetch_add_explicit(&grad_colors[3 * g + 1], aT * dL_dpix.y,
                                      memory_order_relaxed);
            atomic_fetch_add_explicit(&grad_colors[3 * g + 2], aT * dL_dpix.z,
                                      memory_order_relaxed);
            atomic_fetch_add_explicit(&grad_opacities[g], G * dL_dalpha,
                                      memory_order_relaxed);
        }
    }
"""

_forward_kernel = mx.fast.metal_kernel(
    name="gs_rasterize_forward",
    input_names=[
        "means2d", "conics", "colors", "opacities",
        "sorted_ids", "tile_ranges", "background", "params",
    ],
    output_names=["image", "final_T", "n_contrib"],
    source=_FORWARD_SRC,
)

_backward_kernel = mx.fast.metal_kernel(
    name="gs_rasterize_backward",
    input_names=[
        "means2d", "conics", "colors", "opacities",
        "sorted_ids", "tile_ranges", "background", "params",
        "final_T", "n_contrib", "grad_image", "grad_final_T",
    ],
    output_names=["grad_means2d", "grad_conics", "grad_colors", "grad_opacities"],
    source=_BACKWARD_SRC,
    atomic_outputs=True,
)


@mx.custom_function
def _rasterize_core(means2d, conics, colors, opacities, sorted_ids, tile_ranges,
                    background, params):
    width = int(params[0].item())
    height = int(params[1].item())
    tiles_x = int(params[2].item())
    tiles_y = int(params[3].item())
    image, final_T, n_contrib = _forward_kernel(
        inputs=[means2d, conics, colors, opacities, sorted_ids, tile_ranges,
                background, params],
        output_shapes=[(height, width, 3), (height, width), (height, width)],
        output_dtypes=[mx.float32, mx.float32, mx.int32],
        grid=(tiles_x * TILE_SIZE, tiles_y * TILE_SIZE, 1),
        threadgroup=(TILE_SIZE, TILE_SIZE, 1),
        init_value=0,
    )
    return image, final_T, n_contrib


@_rasterize_core.vjp
def _rasterize_core_vjp(primals, cotangents, outputs):
    (means2d, conics, colors, opacities, sorted_ids, tile_ranges,
     background, params) = primals
    image, final_T, n_contrib = outputs
    grad_image = cotangents[0]
    grad_final_T = cotangents[1]
    tiles_x = int(params[2].item())
    tiles_y = int(params[3].item())
    N = means2d.shape[0]
    g_means2d, g_conics, g_colors, g_opacities = _backward_kernel(
        inputs=[means2d, conics, colors, opacities, sorted_ids, tile_ranges,
                background, params, final_T, n_contrib, grad_image, grad_final_T],
        output_shapes=[(N, 2), (N, 3), (N, 3), (N,)],
        output_dtypes=[mx.float32, mx.float32, mx.float32, mx.float32],
        grid=(tiles_x * TILE_SIZE, tiles_y * TILE_SIZE, 1),
        threadgroup=(TILE_SIZE, TILE_SIZE, 1),
        init_value=0,
    )
    return (
        g_means2d,
        g_conics,
        g_colors,
        g_opacities,
        mx.zeros_like(sorted_ids),
        mx.zeros_like(tile_ranges),
        mx.zeros_like(background),
        mx.zeros_like(params),
    )


def rasterize(
    means2d: mx.array,
    conics: mx.array,
    colors: mx.array,
    opacities: mx.array,
    sorted_ids: mx.array,
    tile_ranges: mx.array,
    width: int,
    height: int,
    tiles_x: int,
    tiles_y: int,
    background: mx.array | None = None,
) -> dict[str, mx.array]:
    """Alpha-composite binned 2D Gaussians into an image.

    All per-Gaussian inputs are differentiable; gradients are produced by the
    backward Metal kernel.

    Returns:
        dict with ``image`` (H, W, 3), ``alpha`` (H, W), and the saved
        auxiliary buffers ``final_T`` / ``n_contrib``.
    """
    if background is None:
        background = mx.zeros((3,))
    params = mx.array([width, height, tiles_x, tiles_y], dtype=mx.int32)
    image, final_T, n_contrib = _rasterize_core(
        means2d.astype(mx.float32),
        conics.astype(mx.float32),
        colors.astype(mx.float32),
        opacities.astype(mx.float32),
        sorted_ids,
        tile_ranges,
        background.astype(mx.float32),
        params,
    )
    return {
        "image": image,
        "alpha": 1.0 - final_T,
        "final_T": final_T,
        "n_contrib": n_contrib,
    }
