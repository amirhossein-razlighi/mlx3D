"""Fast forward-only Gaussian Splatting rasterization.

A viewing-oriented rendering path in the spirit of
`fast-gaussian-rasterization <https://github.com/dendenxu/fast-gaussian-rasterization>`_,
which replaces the training rasterizer's generic pipeline with a
hardware-style "geometry shader" flow: compute everything per Gaussian once,
sort back-to-front once, then splat. This Metal/MLX adaptation keeps the
tile-based compositing (Metal compute has no hardware blend stage) but applies
the same ideas end to end:

- **One fused geometry kernel** replaces ~60 elementwise MLX ops: camera
  transform, EWA projection from a *cached* 3D covariance, conic, culling,
  and tile-bounding-box math run in a single pass per Gaussian.
- **Cross-frame caching** (their ``OptimizedGaussians`` TODO): activations
  (``exp`` / ``sigmoid``), the 3D covariance, and SH-evaluated colors are
  computed once and reused; colors refresh only when the camera has moved
  enough to matter.
- **Cheaper sorting**: Gaussians are globally depth-sorted once per frame
  (N keys), so tile duplicates are emitted already depth-ordered and only a
  *stable 32-bit* tile-key sort remains — instead of a 64-bit
  ``(tile, depth)`` composite sort over every duplicate.
- **Zero mid-frame CPU/GPU synchronization** (their non-blocking transfer
  point): duplicate buffers use a persistent, grow-on-demand capacity with a
  sentinel tile, so the whole frame is submitted as one lazy graph. The
  training path must sync mid-frame to size its buffers.
- **Forward-only compositing kernel**: no backward bookkeeping
  (``n_contrib``), a half-precision splat payload (conic / opacity / color
  quantization error < 0.1%), and an adjustable early-termination
  transmittance for extra speed.

No backward pass — use :func:`~mlx3d.splatting.render_gaussians` for
training. The natural consumers are viewers, flythrough export, benchmark
loops, and dynamic (4D) sequence playback via :meth:`FastGaussianRenderer.update`.

Example:
    >>> from mlx3d.splatting import GaussianModel, FastGaussianRenderer
    >>> model = GaussianModel.load_ply("point_cloud.ply")
    >>> renderer = FastGaussianRenderer(model)
    >>> out = renderer.render(camera)          # {"image", "alpha"}
"""

from __future__ import annotations

import mlx.core as mx

from ..cameras import Camera
from .sh import eval_sh
from .tiles import TILE_SIZE

__all__ = ["FastGaussianRenderer", "render_gaussians_fast"]

_BLOCK = TILE_SIZE * TILE_SIZE
_SENTINEL = 0xFFFFFFFF  # padding tile key; sorts after every real tile

# --------------------------------------------------------------------------
# Kernel 1 — "geometry shader": everything per Gaussian, fused in one pass.
# --------------------------------------------------------------------------
_GEOMETRY_SRC = """
    const uint g = thread_position_in_grid.x;
    const uint N = (uint)iparams[0];
    if (g >= N) return;

    const int width = iparams[1];
    const int height = iparams[2];
    const int tiles_x = iparams[3];
    const int tiles_y = iparams[4];
    const bool antialias = iparams[5] != 0;

    // fparams: R(9) t(3) fx fy cx cy tanfx tanfy blur znear
    const float fx = fparams[12], fy = fparams[13];
    const float cx = fparams[14], cy = fparams[15];
    const float tanfx = fparams[16], tanfy = fparams[17];
    const float blur = fparams[18], znear = fparams[19];

    const float px = means[3 * g + 0];
    const float py = means[3 * g + 1];
    const float pz = means[3 * g + 2];

    // world -> camera
    const float x = fparams[0] * px + fparams[1] * py + fparams[2] * pz + fparams[9];
    const float y = fparams[3] * px + fparams[4] * py + fparams[5] * pz + fparams[10];
    const float z = fparams[6] * px + fparams[7] * py + fparams[8] * pz + fparams[11];
    const float z_safe = metal::max(z, 1e-6f);
    const float inv_z = 1.0f / z_safe;

    const float u = fx * x * inv_z + cx;
    const float v = fy * y * inv_z + cy;

    // EWA Jacobian with padded-frustum clamping (matches the training path).
    const float txc = metal::clamp(x * inv_z, -1.3f * tanfx, 1.3f * tanfx) * z_safe;
    const float tyc = metal::clamp(y * inv_z, -1.3f * tanfy, 1.3f * tanfy) * z_safe;
    const float inv_z2 = inv_z * inv_z;
    const float j00 = fx * inv_z;
    const float j02 = -fx * txc * inv_z2;
    const float j11 = fy * inv_z;
    const float j12 = -fy * tyc * inv_z2;

    // T = J @ R (two rows of a 2x3).
    const float t00 = j00 * fparams[0] + j02 * fparams[6];
    const float t01 = j00 * fparams[1] + j02 * fparams[7];
    const float t02 = j00 * fparams[2] + j02 * fparams[8];
    const float t10 = j11 * fparams[3] + j12 * fparams[6];
    const float t11 = j11 * fparams[4] + j12 * fparams[7];
    const float t12 = j11 * fparams[5] + j12 * fparams[8];

    // cov2d = T @ Sigma @ T^T from the cached upper-triangular 3D covariance.
    const float s00 = cov3d[6 * g + 0], s01 = cov3d[6 * g + 1], s02 = cov3d[6 * g + 2];
    const float s11 = cov3d[6 * g + 3], s12 = cov3d[6 * g + 4], s22 = cov3d[6 * g + 5];
    const float w0x = t00 * s00 + t01 * s01 + t02 * s02;
    const float w0y = t00 * s01 + t01 * s11 + t02 * s12;
    const float w0z = t00 * s02 + t01 * s12 + t02 * s22;
    const float w1x = t10 * s00 + t11 * s01 + t12 * s02;
    const float w1y = t10 * s01 + t11 * s11 + t12 * s12;
    const float w1z = t10 * s02 + t11 * s12 + t12 * s22;
    const float a0 = w0x * t00 + w0y * t01 + w0z * t02;
    const float b  = w0x * t10 + w0y * t11 + w0z * t12;
    const float c0 = w1x * t10 + w1y * t11 + w1z * t12;

    const float a = a0 + blur;
    const float c = c0 + blur;
    const float det = a * c - b * b;
    const float det_safe = metal::max(det, 1e-12f);

    float comp = 1.0f;
    if (antialias) {
        comp = metal::sqrt(metal::max(a0 * c0 - b * b, 0.0f) / det_safe);
    }

    // 3-sigma radius from the larger eigenvalue.
    const float mid = 0.5f * (a + c);
    const float lam1 = mid + metal::sqrt(metal::max(mid * mid - det, 0.01f));
    float radius = metal::ceil(3.0f * metal::sqrt(metal::max(lam1, 0.0f)));

    const bool valid = (z > znear) && (det > 0.0f);
    if (!valid) radius = 0.0f;

    // Tile bounding box (inclusive), clipped to the screen.
    const float TILEF = 16.0f;
    int xmin = (int)metal::clamp(metal::floor((u - radius) / TILEF), 0.0f, (float)(tiles_x - 1));
    int xmax = (int)metal::clamp(metal::floor((u + radius) / TILEF), 0.0f, (float)(tiles_x - 1));
    int ymin = (int)metal::clamp(metal::floor((v - radius) / TILEF), 0.0f, (float)(tiles_y - 1));
    int ymax = (int)metal::clamp(metal::floor((v + radius) / TILEF), 0.0f, (float)(tiles_y - 1));

    const bool on_screen = (radius > 0.0f)
        && (u + radius >= 0.0f) && (u - radius < (float)width)
        && (v + radius >= 0.0f) && (v - radius < (float)height);
    const int count = on_screen ? (xmax - xmin + 1) * (ymax - ymin + 1) : 0;

    xy[2 * g + 0] = u;
    xy[2 * g + 1] = v;
    depths[g] = z;
    payload[8 * g + 0] = (half)(c / det_safe);
    payload[8 * g + 1] = (half)(-b / det_safe);
    payload[8 * g + 2] = (half)(a / det_safe);
    payload[8 * g + 3] = (half)(opacities[g] * comp);
    payload[8 * g + 4] = (half)colors[3 * g + 0];
    payload[8 * g + 5] = (half)colors[3 * g + 1];
    payload[8 * g + 6] = (half)colors[3 * g + 2];
    payload[8 * g + 7] = (half)0.0f;
    bbox[4 * g + 0] = xmin;
    bbox[4 * g + 1] = xmax;
    bbox[4 * g + 2] = ymin;
    bbox[4 * g + 3] = ymax;
    counts[g] = count;
"""

# --------------------------------------------------------------------------
# Kernel 2 — duplicate expansion in depth order into a persistent capacity.
# --------------------------------------------------------------------------
_EXPAND_SRC = """
    const uint i = thread_position_in_grid.x;   // depth rank
    const uint N = (uint)iparams[0];
    if (i >= N) return;

    const int tiles_x = iparams[1];
    const int capacity = iparams[2];

    const int g = order[i];
    const int count = counts[g];
    if (count == 0) return;
    int slot = offsets[i];

    const int xmin = bbox[4 * g + 0];
    const int xmax = bbox[4 * g + 1];
    const int ymin = bbox[4 * g + 2];
    const int ymax = bbox[4 * g + 3];

    for (int ty = ymin; ty <= ymax; ty++) {
        for (int tx = xmin; tx <= xmax; tx++) {
            if (slot >= capacity) return;   // overflow: dropped this frame, grown next
            keys[slot] = (uint)(ty * tiles_x + tx);
            dup_ids[slot] = g;
            slot++;
        }
    }
"""

# --------------------------------------------------------------------------
# Kernel 3 — tile ranges from sorted keys by boundary detection (no atomics).
# --------------------------------------------------------------------------
_RANGES_SRC = """
    const uint i = thread_position_in_grid.x;
    const uint capacity = (uint)iparams[0];
    const uint num_tiles = (uint)iparams[1];
    if (i >= capacity) return;

    const uint k = keys[sort_idx[i]];
    const uint kp = (i == 0) ? 0xFFFFFFFFu : keys[sort_idx[i - 1]];

    if (i == 0 || k != kp) {
        if (k < num_tiles) tile_ranges[2 * k] = (int)i;         // start
        if (i > 0 && kp < num_tiles) tile_ranges[2 * kp + 1] = (int)i;  // end
    }
    if (i == capacity - 1 && k < num_tiles) tile_ranges[2 * k + 1] = (int)(i + 1);
"""

# --------------------------------------------------------------------------
# Kernel 4 — forward-only tile compositing with a packed half payload.
# --------------------------------------------------------------------------
_RASTER_FAST_SRC = """
    constexpr int TILE = 16;
    constexpr int BLOCK = 256;

    const int width = iparams[0];
    const int height = iparams[1];
    const int tiles_x = iparams[2];
    const float t_min = fparams[0];

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
    bool done = !inside;

    for (int b = 0; b < n_batches; b++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int load = range_start + b * BLOCK + tidx;
        if (load < range_end) {
            const int g = dup_ids[sort_idx[load]];
            // Vectorized loads: one float2 + two half4 per splat.
            sm_xy[tidx] = ((device const float2*)xy)[g];
            const float4 co = float4(((device const half4*)payload)[2 * g]);
            const float4 rgba = float4(((device const half4*)payload)[2 * g + 1]);
            sm_co[tidx] = co;
            sm_rgb[tidx] = rgba.xyz;
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
            if (next_T < t_min) { done = true; T = next_T; break; }
            acc += sm_rgb[j] * (alpha * T);
            T = next_T;
        }
    }

    if (inside) {
        const int pid = py * width + px;
        image[3 * pid + 0] = acc.x + T * background[0];
        image[3 * pid + 1] = acc.y + T * background[1];
        image[3 * pid + 2] = acc.z + T * background[2];
        final_T[pid] = T;
    }
"""

_geometry_kernel = mx.fast.metal_kernel(
    name="gs_fast_geometry",
    input_names=["means", "cov3d", "opacities", "colors", "fparams", "iparams"],
    output_names=["xy", "depths", "payload", "bbox", "counts"],
    source=_GEOMETRY_SRC,
)

_expand_kernel = mx.fast.metal_kernel(
    name="gs_fast_expand",
    input_names=["order", "offsets", "counts", "bbox", "iparams"],
    output_names=["keys", "dup_ids"],
    source=_EXPAND_SRC,
)

_ranges_kernel = mx.fast.metal_kernel(
    name="gs_fast_ranges",
    input_names=["keys", "sort_idx", "iparams"],
    output_names=["tile_ranges"],
    source=_RANGES_SRC,
)

_raster_fast_kernel = mx.fast.metal_kernel(
    name="gs_fast_raster",
    input_names=[
        "xy", "payload", "dup_ids", "sort_idx", "tile_ranges", "background", "fparams", "iparams"
    ],
    output_names=["image", "final_T"],
    source=_RASTER_FAST_SRC,
)


def _cov3d_upper(quats: mx.array, scales: mx.array) -> mx.array:
    """Upper-triangular world covariance (N, 6): s00 s01 s02 s11 s12 s22."""
    q = quats / mx.maximum(mx.linalg.norm(quats, axis=-1, keepdims=True), 1e-12)
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
    r01 = 2.0 * (qx * qy - qw * qz)
    r02 = 2.0 * (qx * qz + qw * qy)
    r10 = 2.0 * (qx * qy + qw * qz)
    r11 = 1.0 - 2.0 * (qx * qx + qz * qz)
    r12 = 2.0 * (qy * qz - qw * qx)
    r20 = 2.0 * (qx * qz - qw * qy)
    r21 = 2.0 * (qy * qz + qw * qx)
    r22 = 1.0 - 2.0 * (qx * qx + qy * qy)
    sx, sy, sz = scales[:, 0], scales[:, 1], scales[:, 2]
    m00, m01, m02 = r00 * sx, r01 * sy, r02 * sz
    m10, m11, m12 = r10 * sx, r11 * sy, r12 * sz
    m20, m21, m22 = r20 * sx, r21 * sy, r22 * sz
    s00 = m00 * m00 + m01 * m01 + m02 * m02
    s01 = m00 * m10 + m01 * m11 + m02 * m12
    s02 = m00 * m20 + m01 * m21 + m02 * m22
    s11 = m10 * m10 + m11 * m11 + m12 * m12
    s12 = m10 * m20 + m11 * m21 + m12 * m22
    s22 = m20 * m20 + m21 * m21 + m22 * m22
    return mx.stack([s00, s01, s02, s11, s12, s22], axis=-1)


class FastGaussianRenderer:
    """Forward-only, cache-heavy Gaussian renderer for viewing and playback.

    Precomputes activations, 3D covariances, and SH colors once; per frame it
    runs three fused Metal kernels with no mid-frame CPU synchronization.
    Roughly matching the reference renderer at the pixel level (identical
    compositing math; the splat payload is quantized to fp16 and early
    termination is slightly more aggressive).

    Args:
        model: a :class:`~mlx3d.splatting.GaussianModel` (or anything exposing
            ``params``, ``scales_act``, ``opacities_act``, ``sh``,
            ``active_sh_degree``).
        sh_degree: cap the evaluated SH degree (``None`` = the model's active
            degree; ``0`` renders view-independent color, fastest).
        t_min: transmittance early-termination threshold. The training kernel
            uses ``1e-4``; the default here (``1/255``) stops as soon as any
            further splat could no longer change an 8-bit pixel.
        color_refresh: fraction of the scene radius the camera must move
            before view-dependent SH colors are re-evaluated (``0`` =
            every frame).
        antialias: Mip-Splatting opacity compensation (as in the training
            path).

    Not differentiable. For training use ``render_gaussians``.
    """

    def __init__(
        self,
        model=None,
        *,
        means: mx.array | None = None,
        quats: mx.array | None = None,
        scales: mx.array | None = None,
        opacities: mx.array | None = None,
        colors: mx.array | None = None,
        sh: mx.array | None = None,
        sh_degree: int | None = None,
        t_min: float = 1.0 / 255.0,
        color_refresh: float = 0.005,
        antialias: bool = False,
    ):
        self.t_min = float(t_min)
        self.color_refresh = float(color_refresh)
        self.antialias = bool(antialias)
        self._sh_degree_cap = sh_degree

        self._capacity = 0
        self._cached_colors: mx.array | None = None
        self._cached_color_center: mx.array | None = None

        if model is not None:
            self.update(
                means=model.params["means"],
                quats=model.params["quats"],
                scales=model.scales_act,
                opacities=model.opacities_act,
                sh=model.sh,
                sh_degree=(
                    model.active_sh_degree if sh_degree is None else min(sh_degree, model.sh_degree)
                ),
            )
        else:
            if means is None or opacities is None or (colors is None and sh is None):
                raise ValueError(
                    "Provide either `model` or (means, quats, scales, opacities, colors|sh)."
                )
            self.update(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                sh=sh,
                sh_degree=3 if sh_degree is None else sh_degree,
            )

    # ------------------------------------------------------------- scene state
    def update(
        self,
        means: mx.array,
        quats: mx.array | None = None,
        scales: mx.array | None = None,
        opacities: mx.array | None = None,
        colors: mx.array | None = None,
        sh: mx.array | None = None,
        sh_degree: int | None = None,
        cov3d: mx.array | None = None,
    ) -> None:
        """Replace scene arrays (e.g. per timestep of a 4D sequence).

        ``scales`` / ``opacities`` are the *activated* values. Pass ``cov3d``
        (N, 6 upper-triangular) directly to skip the quaternion math when a
        sequence precomputes covariances. Caches are rebuilt lazily.
        """
        self.means = means.astype(mx.float32)
        self.n = int(means.shape[0])
        if cov3d is not None:
            self.cov3d = cov3d.astype(mx.float32)
        else:
            if quats is None or scales is None:
                raise ValueError("Provide quats+scales or cov3d.")
            self.cov3d = _cov3d_upper(
                quats.astype(mx.float32), scales.astype(mx.float32)
            )
        if opacities is not None:
            self.opacities = opacities.astype(mx.float32)
        if (colors is None) == (sh is None) and colors is not None:
            raise ValueError("Provide only one of `colors` or `sh`.")
        if colors is not None:
            self._colors_static = colors.astype(mx.float32)
            self._sh = None
        elif sh is not None:
            self._sh = sh.astype(mx.float32)
            self._colors_static = None
        if sh_degree is not None:
            self.sh_degree = int(sh_degree)

        # A cheap scene radius for the color-refresh heuristic.
        if self.n > 0:
            lo = mx.min(self.means, axis=0)
            hi = mx.max(self.means, axis=0)
            self._scene_radius = float(mx.linalg.norm(hi - lo).item()) * 0.5 + 1e-6
        else:
            self._scene_radius = 1.0

        self._cached_colors = None
        self._cached_color_center = None
        # Evaluate everything the render graph will capture so frames start hot.
        mx.eval(self.means, self.cov3d, self.opacities)

    # ----------------------------------------------------------------- colors
    def _view_colors(self, camera: Camera) -> mx.array:
        if self._colors_static is not None:
            return self._colors_static
        # The cache key lives in plain Python floats: viewers render on HTTP
        # handler threads, and MLX cannot evaluate lazy arrays created on a
        # different thread, so no cross-frame MLX graph may survive here.
        center = camera.camera_center
        center_f = tuple(float(c) for c in center)
        if self._cached_colors is not None and self.color_refresh > 0:
            moved = sum((a - b) ** 2 for a, b in zip(center_f, self._cached_color_center)) ** 0.5
            if moved < self.color_refresh * self._scene_radius:
                return self._cached_colors
        deg = self.sh_degree if self._sh_degree_cap is None else self._sh_degree_cap
        deg = min(deg, self.sh_degree)
        dirs = self.means - center
        dirs = dirs / mx.maximum(mx.linalg.norm(dirs, axis=-1, keepdims=True), 1e-8)
        colors = mx.maximum(eval_sh(deg, self._sh, dirs), 0.0)
        mx.eval(colors)
        self._cached_colors = colors
        self._cached_color_center = center_f
        return colors

    # ----------------------------------------------------------------- render
    def render(self, camera: Camera, background: mx.array | None = None) -> dict[str, mx.array]:
        """Render one frame. Returns ``{"image", "alpha"}`` (not differentiable)."""
        w, h = camera.width, camera.height
        tiles_x = (w + TILE_SIZE - 1) // TILE_SIZE
        tiles_y = (h + TILE_SIZE - 1) // TILE_SIZE
        num_tiles = tiles_x * tiles_y
        if background is None:
            background = mx.zeros((3,))
        background = background.astype(mx.float32)
        if self.n == 0:
            image = mx.broadcast_to(background[None, None, :], (h, w, 3))
            return {"image": image, "alpha": mx.zeros((h, w))}
        colors = self._view_colors(camera)

        R, t = camera.R, camera.t
        fparams = mx.concatenate(
            [
                R.reshape(-1).astype(mx.float32),
                t.reshape(-1).astype(mx.float32),
                mx.array(
                    [
                        camera.fx,
                        camera.fy,
                        camera.cx,
                        camera.cy,
                        0.5 * w / camera.fx,
                        0.5 * h / camera.fy,
                        0.3,
                        camera.znear,
                    ],
                    dtype=mx.float32,
                ),
            ]
        )
        iparams_geo = mx.array(
            [self.n, w, h, tiles_x, tiles_y, int(self.antialias)], dtype=mx.int32
        )

        xy, depths, payload, bbox, counts = _geometry_kernel(
            inputs=[self.means, self.cov3d, self.opacities, colors, fparams, iparams_geo],
            output_shapes=[(self.n, 2), (self.n,), (self.n, 8), (self.n, 4), (self.n,)],
            output_dtypes=[mx.float32, mx.float32, mx.float16, mx.int32, mx.int32],
            grid=(self.n, 1, 1),
            threadgroup=(min(256, self.n), 1, 1),
        )

        # Global back-to-front order (front-to-back for the compositor):
        # duplicates emitted in this order are depth-sorted within every tile.
        order = mx.argsort(depths).astype(mx.int32)
        counts_sorted = counts[order]
        offsets = (mx.cumsum(counts_sorted) - counts_sorted).astype(mx.int32)
        total = mx.sum(counts)  # stays on GPU; checked only after the frame

        if self._capacity == 0:
            # First frame: one-time sync to size the duplicate buffers.
            self._capacity = max(int(total.item()) + 1024, 1) * 5 // 4

        for _attempt in range(2):
            image, final_T = self._composite(
                xy, payload, order, offsets, counts, bbox, background, total,
                w, h, tiles_x, tiles_y, num_tiles,
            )
            # The capacity check rides on the frame's own eval: no extra sync.
            mx.eval(image, final_T, total)
            needed = int(total.item())
            if needed <= self._capacity:
                break
            # Rare (e.g. zooming in): grow and re-render this frame correctly.
            self._capacity = needed * 5 // 4 + 1024
        return {"image": image, "alpha": 1.0 - final_T}

    def _composite(
        self, xy, payload, order, offsets, counts, bbox, background, total,
        w, h, tiles_x, tiles_y, num_tiles,
    ):
        capacity = self._capacity
        iparams_exp = mx.array([self.n, tiles_x, capacity], dtype=mx.int32)
        keys, dup_ids = _expand_kernel(
            inputs=[order, offsets, counts, bbox, iparams_exp],
            output_shapes=[(capacity,), (capacity,)],
            output_dtypes=[mx.uint32, mx.int32],
            grid=(self.n, 1, 1),
            threadgroup=(min(256, self.n), 1, 1),
            init_value=0,
        )
        # Offsets are a perfect prefix sum, so slots [0, total) are exactly the
        # written ones. Mask the tail with the sentinel *on the GPU* — `total`
        # is a 0-d array, so this costs no synchronization.
        positions = mx.arange(capacity, dtype=mx.int32)
        keys = mx.where(positions < total, keys, mx.array(_SENTINEL, dtype=mx.uint32))

        # Stable 32-bit sort: ties keep depth order from the expansion pass.
        # Downstream kernels read through `sort_idx`, so no gathered copies of
        # the sorted arrays are ever materialized.
        sort_idx = mx.argsort(keys).astype(mx.int32)

        iparams_rg = mx.array([capacity, num_tiles], dtype=mx.int32)
        (tile_ranges,) = _ranges_kernel(
            inputs=[keys, sort_idx, iparams_rg],
            output_shapes=[(num_tiles, 2)],
            output_dtypes=[mx.int32],
            grid=(capacity, 1, 1),
            threadgroup=(256, 1, 1),
            init_value=0,  # empty tiles keep start == end == 0
        )

        iparams_r = mx.array([w, h, tiles_x], dtype=mx.int32)
        fparams_r = mx.array([self.t_min], dtype=mx.float32)
        image, final_T = _raster_fast_kernel(
            inputs=[xy, payload, dup_ids, sort_idx, tile_ranges, background, fparams_r, iparams_r],
            output_shapes=[(h, w, 3), (h, w)],
            output_dtypes=[mx.float32, mx.float32],
            grid=(tiles_x * TILE_SIZE, tiles_y * TILE_SIZE, 1),
            threadgroup=(TILE_SIZE, TILE_SIZE, 1),
            init_value=0,
        )
        return image, final_T


def render_gaussians_fast(
    camera: Camera,
    means: mx.array,
    quats: mx.array,
    scales: mx.array,
    opacities: mx.array,
    colors: mx.array | None = None,
    sh: mx.array | None = None,
    sh_degree: int = 3,
    background: mx.array | None = None,
    t_min: float = 1.0 / 255.0,
    antialias: bool = False,
) -> dict[str, mx.array]:
    """One-shot fast forward-only render (signature mirrors ``render_gaussians``).

    For loops (viewers, flythroughs, 4D playback) construct a
    :class:`FastGaussianRenderer` once instead — it caches activations,
    covariances, colors, and buffer capacity across frames.

    Not differentiable; ``scales`` / ``opacities`` are activated values.
    """
    renderer = FastGaussianRenderer(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        sh=sh,
        sh_degree=sh_degree,
        t_min=t_min,
        antialias=antialias,
    )
    return renderer.render(camera, background=background)
