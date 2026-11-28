"""Tile-based rasterization of spaghetti segments with custom Metal kernels.

Same structure as the 3DGS rasterizer (:mod:`mlx3d.splatting.rasterize`):
one 16x16-pixel threadgroup per tile, depth-sorted segments streamed through
threadgroup memory in batches of 256, front-to-back compositing with early
termination; the backward pass walks back-to-front and accumulates gradients
with atomics. The difference is the per-pixel primitive: instead of a 2D
Gaussian footprint, each pixel evaluates the exact closed-form optical depth
of a Gaussian-blurred segment along its own ray (see
:mod:`~mlx3d.splatting.spaghetti.projection`), so there is no per-segment
screen-space linearization.

Per segment the kernels read 16 floats — the whitening map ``W`` (row-major
3x3), whitened start ``a`` and direction ``v``, and density ``rho`` — plus an
RGB color. Gradients are produced for exactly those inputs; MLX autodiff
carries them back to positions, kernel shape and camera pose.
"""

from __future__ import annotations

import mlx.core as mx

from ..tiles import TILE_SIZE

__all__ = ["SEGMENT_FLOATS", "pack_segments", "rasterize_segments"]

SEGMENT_FLOATS = 16

_HEADER = """
// erf (Abramowitz & Stegun 7.1.26, |error| < 1.5e-7); Metal has no erf().
inline float spaghetti_erf(float x) {
    const float t = 1.0f / (1.0f + 0.3275911f * metal::abs(x));
    const float poly = t * (0.254829592f + t * (-0.284496736f + t * (1.421413741f
                     + t * (-1.453152027f + t * 1.061405429f))));
    const float r = 1.0f - poly * metal::exp(-x * x);
    return x >= 0.0f ? r : -r;
}

struct SegmentTerms {
    float3 e;      // whitened ray direction W d
    float n;       // |e|^2
    float3 y;      // -P a
    float3 z;      // P v
    float k;       // |z|
    float ub;      // closest line parameter
    float E;       // squared whitened ray-to-line distance
    float I;       // integral of the kernel profile along the segment
    float S;       // sqrt(pi / n)
    bool is_line;
    float tau;
};

constant float SQRT_PI = 1.7724538509055159f;
constant float POINT_EPS2 = 1e-8f;
constant float MIN_ALPHA = 1.0f / 255.0f;

inline float3 seg_row(threadgroup const float* s, int r) {
    return float3(s[3 * r], s[3 * r + 1], s[3 * r + 2]);
}

// Optical depth of one segment along unit camera ray d. Returns false (and
// leaves tau unset) when the segment cannot reach MIN_ALPHA at this pixel.
inline bool segment_tau(threadgroup const float* s, float3 d, thread SegmentTerms& t) {
    t.e = float3(dot(seg_row(s, 0), d), dot(seg_row(s, 1), d), dot(seg_row(s, 2), d));
    const float3 a = float3(s[9], s[10], s[11]);
    const float3 v = float3(s[12], s[13], s[14]);
    const float rho = s[15];
    t.n = dot(t.e, t.e);
    const float inv_n = 1.0f / t.n;
    t.y = -(a - t.e * (dot(t.e, a) * inv_n));
    t.z = v - t.e * (dot(t.e, v) * inv_n);
    const float k2 = dot(t.z, t.z);
    t.is_line = k2 > POINT_EPS2;
    if (t.is_line) {
        t.k = metal::sqrt(k2);
        t.ub = dot(t.z, t.y) / k2;
        const float3 r = t.y - t.ub * t.z;
        t.E = dot(r, r);
    } else {
        t.k = 0.0f;
        t.ub = 0.0f;
        t.E = dot(t.y, t.y);
    }
    t.S = metal::sqrt(M_PI_F * inv_n);
    const float upper = rho * t.S * metal::exp(-t.E);  // tau <= upper since I <= 1
    if (upper < MIN_ALPHA) return false;
    t.I = t.is_line
        ? SQRT_PI / (2.0f * t.k) * (spaghetti_erf(t.k * (1.0f - t.ub)) + spaghetti_erf(t.k * t.ub))
        : 1.0f;
    t.tau = upper * t.I;
    return true;
}

// Cheap 2D rejection before the 3D ray math (same test in both passes).
inline bool outside_footprint(float2 pix, float2 c, float3 co) {
    const float2 x = pix - c;
    return co.x * x.x * x.x + 2.0f * co.y * x.x * x.y + co.z * x.y * x.y > 1.0f;
}

inline float3 pixel_ray(int px, int py, float fx, float fy, float cx, float cy) {
    return normalize(float3((px + 0.5f - cx) / fx, (py + 0.5f - cy) / fy, 1.0f));
}
"""

_PREAMBLE = """
    constexpr int TILE = 16;
    constexpr int BLOCK = 256;
    constexpr int SF = 16;  // floats per segment

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
    const float3 d = pixel_ray(px, py, intr[0], intr[1], intr[2], intr[3]);
    const int pid = py * width + px;

    const int range_start = tile_ranges[2 * tile_id];
    const int range_end = tile_ranges[2 * tile_id + 1];
    const int num = range_end - range_start;
    const int n_batches = (num + BLOCK - 1) / BLOCK;

    threadgroup float sm_seg[BLOCK * SF];
    threadgroup float3 sm_rgb[BLOCK];
    threadgroup float2 sm_xy[BLOCK];    // footprint ellipse center (pixels)
    threadgroup float3 sm_conic[BLOCK]; // footprint ellipse x^T C x <= 1
    const float2 pixf = float2(px + 0.5f, py + 0.5f);
"""

_LOAD = """
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int load = range_start + b * BLOCK + tidx;
        if (load < range_end) {
            const int g = sorted_ids[load];
            for (int q = 0; q < SF; q++) sm_seg[tidx * SF + q] = segments[g * SF + q];
            sm_rgb[tidx] = float3(colors[3 * g], colors[3 * g + 1], colors[3 * g + 2]);
            sm_xy[tidx] = float2(ellipses[5 * g], ellipses[5 * g + 1]);
            sm_conic[tidx] = float3(ellipses[5 * g + 2], ellipses[5 * g + 3], ellipses[5 * g + 4]);
            LOAD_EXTRA
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
"""

_FORWARD_SRC = (
    _PREAMBLE
    + """
    float T = 1.0f;
    float3 acc = float3(0.0f);
    int last_contrib = 0;
    bool done = !inside;

    for (int b = 0; b < n_batches; b++) {
"""
    + _LOAD.replace("LOAD_EXTRA", "")
    + """
        if (done) continue;
        const int batch_size = min(BLOCK, num - b * BLOCK);
        for (int j = 0; j < batch_size; j++) {
            if (outside_footprint(pixf, sm_xy[j], sm_conic[j])) continue;
            SegmentTerms t;
            if (!segment_tau(&sm_seg[j * SF], d, t)) continue;
            const float alpha = min(0.99f, 1.0f - metal::exp(-t.tau));
            if (alpha < MIN_ALPHA) continue;
            const float next_T = T * (1.0f - alpha);
            if (next_T < 1e-4f) { done = true; break; }
            acc += sm_rgb[j] * (alpha * T);
            T = next_T;
            last_contrib = b * BLOCK + j + 1;
        }
    }

    if (inside) {
        image[3 * pid + 0] = acc.x + T * background[0];
        image[3 * pid + 1] = acc.y + T * background[1];
        image[3 * pid + 2] = acc.z + T * background[2];
        final_T[pid] = T;
        n_contrib[pid] = last_contrib;
    }
"""
)

_BACKWARD_SRC = (
    _PREAMBLE
    + """
    threadgroup int sm_id[BLOCK];
    constexpr int GRAD_FLOATS = SF + 3;  // segment floats + RGB

    const float T_final = inside ? final_T[pid] : 1.0f;
    float T = T_final;
    const int last = inside ? n_contrib[pid] : 0;
    float3 dL_dpix = float3(0.0f);
    float bg_dot = 0.0f;
    if (inside) {
        dL_dpix = float3(grad_image[3 * pid], grad_image[3 * pid + 1], grad_image[3 * pid + 2]);
        bg_dot = background[0] * dL_dpix.x + background[1] * dL_dpix.y
               + background[2] * dL_dpix.z + grad_final_T[pid];
    }
    float3 accum = float3(0.0f);
    float last_alpha = 0.0f;
    float3 last_color = float3(0.0f);

    for (int b = n_batches - 1; b >= 0; b--) {
"""
    + _LOAD.replace("LOAD_EXTRA", "sm_id[tidx] = g;")
    + """
        const int batch_size = min(BLOCK, num - b * BLOCK);
        for (int j = batch_size - 1; j >= 0; j--) {
            // Every lane visits every segment (uniform control flow) so the
            // per-segment gradients can be summed across the SIMD group and
            // written with one atomic per group instead of one per pixel.
            // Inactive lanes contribute zeros.
            float grad[GRAD_FLOATS];
            for (int q = 0; q < GRAD_FLOATS; q++) grad[q] = 0.0f;
            threadgroup const float* s = &sm_seg[j * SF];
            SegmentTerms t;
            bool active = inside && (b * BLOCK + j < last)
                       && !outside_footprint(pixf, sm_xy[j], sm_conic[j])
                       && segment_tau(s, d, t);
            float raw_alpha = 0.0f, alpha = 0.0f;
            if (active) {
                raw_alpha = 1.0f - metal::exp(-t.tau);
                alpha = min(0.99f, raw_alpha);
                active = alpha >= MIN_ALPHA;
            }
            if (active) {
                T = T / (1.0f - alpha);
                const float3 c = sm_rgb[j];
                const float aT = alpha * T;
                accum = last_alpha * last_color + (1.0f - last_alpha) * accum;
                last_alpha = alpha;
                last_color = c;

                float dL_dalpha = dot(c - accum, dL_dpix) * T;
                dL_dalpha += (-T_final / (1.0f - alpha)) * bg_dot;
                // alpha = 1 - exp(-tau); the 0.99 clamp has zero gradient.
                const float gt = raw_alpha < 0.99f ? dL_dalpha * (1.0f - raw_alpha) : 0.0f;

                // ---- tau = rho * S * exp(-E) * I
                const float rho = s[15];
                const float g_rho = gt * t.tau / rho;
                const float g_E = -gt * t.tau;
                const float g_I = gt * rho * t.S * metal::exp(-t.E);
                const float inv_n = 1.0f / t.n;
                const float g_n = -0.5f * t.tau * inv_n * gt;

                float3 g_y, g_z;
                if (t.is_line) {
                    const float k = t.k, ub = t.ub;
                    const float s1 = k * (1.0f - ub), s2 = k * ub;
                    const float e1 = metal::exp(-s1 * s1), e2 = metal::exp(-s2 * s2);
                    const float g_k = g_I * (-t.I / k + (e1 * (1.0f - ub) + e2 * ub) / k);
                    float g_ub = g_I * (e2 - e1);
                    const float3 r = t.y - ub * t.z;
                    const float3 g_r = 2.0f * g_E * r;
                    g_y = g_r;
                    g_ub -= dot(g_r, t.z);
                    g_z = -ub * g_r;
                    const float k2 = k * k;
                    g_z += g_ub * t.y / k2;
                    g_y += g_ub * t.z / k2;
                    const float g_k2 = -g_ub * ub / k2 + g_k / (2.0f * k);
                    g_z += 2.0f * g_k2 * t.z;
                } else {
                    g_y = 2.0f * g_E * t.y;
                    g_z = float3(0.0f);
                }

                // ---- y = -(a - e (e.a)/n),  z = v - e (e.v)/n
                const float3 a = float3(s[9], s[10], s[11]);
                const float3 v = float3(s[12], s[13], s[14]);
                const float ea = dot(t.e, a), ev = dot(t.e, v);
                const float gy_e = dot(g_y, t.e), gz_e = dot(g_z, t.e);
                const float3 g_a = -(g_y - t.e * (gy_e * inv_n));
                const float3 g_v = g_z - t.e * (gz_e * inv_n);
                float3 g_e = 2.0f * g_n * t.e;
                g_e += a * (gy_e * inv_n) + g_y * (ea * inv_n)
                     - t.e * (2.0f * ea * gy_e * inv_n * inv_n);
                g_e -= v * (gz_e * inv_n) + g_z * (ev * inv_n)
                     - t.e * (2.0f * ev * gz_e * inv_n * inv_n);

                // ---- e = W d (row-major W), then a, v, rho, color
                for (int row = 0; row < 3; row++) {
                    grad[3 * row + 0] = g_e[row] * d.x;
                    grad[3 * row + 1] = g_e[row] * d.y;
                    grad[3 * row + 2] = g_e[row] * d.z;
                }
                for (int q = 0; q < 3; q++) {
                    grad[9 + q] = g_a[q];
                    grad[12 + q] = g_v[q];
                    grad[16 + q] = aT * dL_dpix[q];
                }
                grad[15] = g_rho;
            }
            if (simd_any(active)) {
                const int g = sm_id[j];
                for (int q = 0; q < GRAD_FLOATS; q++) {
                    const float total = simd_sum(grad[q]);
                    if (simd_is_first()) {
                        device atomic_float* dst = q < SF ? &grad_segments[g * SF + q]
                                                          : &grad_colors[3 * g + (q - SF)];
                        atomic_fetch_add_explicit(dst, total, memory_order_relaxed);
                    }
                }
            }
        }
    }
"""
)

_INPUTS = [
    "segments",
    "colors",
    "ellipses",
    "sorted_ids",
    "tile_ranges",
    "background",
    "params",
    "intr",
]

_forward_kernel = mx.fast.metal_kernel(
    name="spaghetti_rasterize_forward",
    input_names=_INPUTS,
    output_names=["image", "final_T", "n_contrib"],
    header=_HEADER,
    source=_FORWARD_SRC,
)

_backward_kernel = mx.fast.metal_kernel(
    name="spaghetti_rasterize_backward",
    input_names=_INPUTS + ["final_T", "n_contrib", "grad_image", "grad_final_T"],
    output_names=["grad_segments", "grad_colors"],
    header=_HEADER,
    source=_BACKWARD_SRC,
    atomic_outputs=True,
)


def pack_segments(W: mx.array, a: mx.array, v: mx.array, rho: mx.array) -> mx.array:
    """Pack per-segment kernel inputs into the ``(N, 16)`` layout the kernels read."""
    n = W.shape[0]
    return mx.concatenate([W.reshape(n, 9), a, v, rho[:, None]], axis=1).astype(mx.float32)


@mx.custom_function
def _rasterize_core(segments, colors, ellipses, sorted_ids, tile_ranges, background, params, intr):
    width, height, tiles_x, tiles_y = (int(params[i].item()) for i in range(4))
    return _forward_kernel(
        inputs=[segments, colors, ellipses, sorted_ids, tile_ranges, background, params, intr],
        output_shapes=[(height, width, 3), (height, width), (height, width)],
        output_dtypes=[mx.float32, mx.float32, mx.int32],
        grid=(tiles_x * TILE_SIZE, tiles_y * TILE_SIZE, 1),
        threadgroup=(TILE_SIZE, TILE_SIZE, 1),
        init_value=0,
    )


@_rasterize_core.vjp
def _rasterize_core_vjp(primals, cotangents, outputs):
    segments, colors, ellipses, sorted_ids, tile_ranges, background, params, intr = primals
    _, final_T, n_contrib = outputs
    tiles_x, tiles_y = int(params[2].item()), int(params[3].item())
    n = segments.shape[0]
    grad_segments, grad_colors = _backward_kernel(
        inputs=[
            segments,
            colors,
            ellipses,
            sorted_ids,
            tile_ranges,
            background,
            params,
            intr,
            final_T,
            n_contrib,
            cotangents[0],
            cotangents[1],
        ],
        output_shapes=[(n, SEGMENT_FLOATS), (n, 3)],
        output_dtypes=[mx.float32, mx.float32],
        grid=(tiles_x * TILE_SIZE, tiles_y * TILE_SIZE, 1),
        threadgroup=(TILE_SIZE, TILE_SIZE, 1),
        init_value=0,
    )
    zeros = [
        mx.zeros_like(p) for p in (ellipses, sorted_ids, tile_ranges, background, params, intr)
    ]
    return (grad_segments, grad_colors, *zeros)


def rasterize_segments(
    segments: mx.array,
    colors: mx.array,
    means2d: mx.array,
    conics: mx.array,
    sorted_ids: mx.array,
    tile_ranges: mx.array,
    intrinsics: tuple[float, float, float, float],
    width: int,
    height: int,
    tiles_x: int,
    tiles_y: int,
    background: mx.array | None = None,
) -> dict[str, mx.array]:
    """Composite binned segments into an image (differentiable).

    Args:
        segments: (N, 16) packed segments (:func:`pack_segments`).
        colors: (N, 3) RGB.
        means2d: (N, 2) footprint-ellipse centers in pixels.
        conics: (N, 3) footprint ellipses ``x^T C x <= 1`` around ``means2d``;
            pixels outside are skipped.
        intrinsics: pinhole ``(fx, fy, cx, cy)`` for per-pixel ray directions.

    Returns:
        dict with ``image`` (H, W, 3), ``alpha`` (H, W), ``final_T`` and
        ``n_contrib``.
    """
    if background is None:
        background = mx.zeros((3,))
    params = mx.array([width, height, tiles_x, tiles_y], dtype=mx.int32)
    intr = mx.array(intrinsics, dtype=mx.float32)
    ellipses = mx.stop_gradient(mx.concatenate([means2d, conics], axis=1)).astype(mx.float32)
    image, final_T, n_contrib = _rasterize_core(
        segments.astype(mx.float32),
        colors.astype(mx.float32),
        ellipses,
        sorted_ids,
        tile_ranges,
        background.astype(mx.float32),
        params,
        intr,
    )
    return {"image": image, "alpha": 1.0 - final_T, "final_T": final_T, "n_contrib": n_contrib}
