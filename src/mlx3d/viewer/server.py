"""An interactive browser-based viewer for MLX3D scenes.

The viewer runs a tiny local HTTP server. The browser page sends orbit-camera
state (azimuth / elevation / radius / target) plus live control values, and
receives JPEG frames rendered on the Apple GPU — for Gaussian Splatting that
means the Metal rasterization kernels, so a 100k-Gaussian scene orbits in real
time. Meshes go through the tile-based hard rasterizer and point clouds through
the splat-based point renderer, so one viewer covers every MLX3D scene type.

No dependencies beyond the package itself (Pillow encodes the frames).

Example:
    >>> from mlx3d.viewer import view_file
    >>> view_file("point_cloud.ply")   # gaussians, mesh, or points — auto-detected
    >>> view_file("bunny.obj")         # opens http://127.0.0.1:8090
"""

import inspect
import io
import json
import math
import threading
import time
import webbrowser
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Callable
from urllib.parse import parse_qs, urlparse

import mlx.core as mx
import numpy as np

from ..cameras import Camera

__all__ = [
    "LiveGaussianViewer",
    "Viewer",
    "view_file",
    "view_gaussians",
    "view_live_gaussians",
    "view_mesh",
    "view_nerf",
    "view_pointcloud",
]

# A render callback maps a camera (and optional live control values) to an
# (H, W, 3) image in [0, 1]. Both 1-arg and 2-arg callbacks are accepted.
RenderFn = Callable[..., mx.array]

# Query keys consumed by the camera / transport; everything else is a control.
_RESERVED_KEYS = frozenset(
    {"theta", "phi", "radius", "tx", "ty", "tz", "fov", "w", "h", "up", "q", "mode"}
)


# --------------------------------------------------------------------- helpers
def _wrap_render_fn(fn: RenderFn) -> Callable[[Camera, dict], mx.array]:
    """Normalize a render callback to the ``(camera, params)`` calling convention.

    Legacy callbacks take only a camera; new ones take ``(camera, params)`` so
    they can react to live control values. We inspect the signature once and
    adapt, keeping every existing ``view_*`` helper working unchanged.
    """
    try:
        positional = [
            p
            for p in inspect.signature(fn).parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        wants_params = len(positional) >= 2 or any(
            p.kind == p.VAR_POSITIONAL for p in inspect.signature(fn).parameters.values()
        )
    except (TypeError, ValueError):
        wants_params = False
    if wants_params:
        return fn
    return lambda camera, params: fn(camera)


def hex_to_rgb(value: str, default: tuple[float, float, float]) -> tuple[float, float, float]:
    """Parse ``#rrggbb`` (with or without ``#``) into a 0..1 RGB tuple."""
    if not isinstance(value, str):
        return default
    s = value.lstrip("#")
    if len(s) != 6:
        return default
    try:
        return tuple(int(s[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]
    except ValueError:
        return default


def tone_map(image: mx.array, exposure: float = 1.0, gamma: float = 1.0) -> mx.array:
    """Apply an exposure multiplier and gamma to a [0, 1] image (no-op at 1, 1)."""
    out = image
    if exposure != 1.0:
        out = out * float(exposure)
    out = mx.clip(out, 0.0, 1.0)
    if gamma != 1.0 and gamma > 0.0:
        out = out ** (1.0 / float(gamma))
    return out


def _slider(name, label, lo, hi, step, value):
    return {"name": name, "label": label, "kind": "slider", "min": lo, "max": hi, "step": step, "value": value}


def _color(name, label, value):
    return {"name": name, "label": label, "kind": "color", "value": value}


def _select(name, label, options, value):
    return {"name": name, "label": label, "kind": "select", "options": list(options), "value": value}


class Viewer:
    """Serve an interactive orbit-camera viewer for any camera-to-image renderer.

    Args:
        render_fn: callback mapping a :class:`~mlx3d.cameras.Camera` (and, if it
            accepts a second argument, a ``params`` dict of live control values)
            to an (H, W, 3) image in [0, 1]. Rendering happens on HTTP handler
            threads, and MLX cannot evaluate lazy arrays created on another
            thread — call ``mx.eval`` on any arrays the callback captures
            (model parameters etc.) before serving. The ``view_*`` helpers do
            this for you.
        info: metadata shown in the page HUD (e.g. ``{"gaussians": 100000}``).
        render_modes: named alternate render callbacks selectable in the page
            (e.g. ``{"depth": ..., "normals": ...}``).
        controls: a list of control specs (built with :func:`_slider` /
            :func:`_color` / :func:`_select`) surfaced as live sliders in the
            page; their values arrive in the ``params`` dict.
        initial_radius / initial_target / fov: starting orbit pose.
    """

    def __init__(
        self,
        render_fn: RenderFn,
        info: dict | None = None,
        render_modes: dict[str, RenderFn] | None = None,
        controls: list[dict] | None = None,
        initial_radius: float = 4.0,
        initial_target: tuple[float, float, float] = (0.0, 0.0, 0.0),
        fov: float = 60.0,
    ):
        self.render_fn = _wrap_render_fn(render_fn)
        self.render_modes = {k: _wrap_render_fn(v) for k, v in (render_modes or {}).items()}
        self.controls = controls or []
        self.info = {
            "radius": initial_radius,
            "target": list(initial_target),
            "fov": fov,
            "display_modes": ["rgb", *self.render_modes.keys()],
            "controls": self.controls,
            **(info or {}),
        }
        self._info_lock = threading.Lock()
        self._lock = threading.Lock()  # one render at a time on the GPU
        self._cache: "OrderedDict[str, bytes]" = OrderedDict()
        self._cache_lock = threading.Lock()
        self._cache_max = 6

    def get_info(self) -> dict:
        with self._info_lock:
            return dict(self.info)

    def update_info(self, **info) -> None:
        with self._info_lock:
            self.info.update(info)

    # ------------------------------------------------------------------ camera
    @staticmethod
    def camera_from_query(q: dict[str, list[str]]) -> Camera:
        """Build a Camera from the orbit parameters sent by the page."""

        def f(name: str, default: float) -> float:
            return float(q.get(name, [default])[0])

        theta = f("theta", 0.0)  # azimuth, radians
        phi = f("phi", 0.0)  # elevation, radians
        radius = max(f("radius", 4.0), 1e-3)
        target = mx.array([f("tx", 0.0), f("ty", 0.0), f("tz", 0.0)])
        fov = f("fov", 60.0)
        width = max(int(f("w", 640)), 16)
        height = max(int(f("h", 480)), 16)
        up_sign = 1.0 if f("up", 1.0) >= 0 else -1.0

        eye = target + radius * mx.array(
            [
                math.cos(phi) * math.sin(theta),
                up_sign * math.sin(phi),
                math.cos(phi) * math.cos(theta),
            ]
        )
        return Camera.look_at(
            eye=eye,
            at=target,
            up=(0.0, up_sign, 0.0),
            fov=fov,
            width=width,
            height=height,
        )

    @staticmethod
    def params_from_query(q: dict[str, list[str]]) -> dict:
        """Collect live control values (everything that isn't a camera key).

        Numeric strings become floats; ``#rrggbb`` colors and other tokens stay
        as strings, so render callbacks can read them with ``params.get(...)``.
        """
        params: dict[str, object] = {}
        for key, values in q.items():
            if key in _RESERVED_KEYS or not values:
                continue
            raw = values[0]
            try:
                params[key] = float(raw)
            except (TypeError, ValueError):
                params[key] = raw
        return params

    # ------------------------------------------------------------------ render
    def render_image(self, camera: Camera, mode: str, params: dict) -> np.ndarray:
        """Render one frame to a uint8 ``(H, W, 3)`` numpy array.

        The GPU lock is held only across render + ``mx.eval`` so the (CPU-side)
        encode of one frame overlaps the render of the next request.
        """
        fn = self.render_modes.get(mode, self.render_fn)
        with self._lock:
            img = fn(camera, params)
            mx.eval(img)
            host = np.array(img)
        return (np.clip(host, 0.0, 1.0) * 255).astype(np.uint8)

    def render_jpeg(
        self, camera: Camera, quality: int = 85, mode: str = "rgb", params: dict | None = None
    ) -> bytes:
        from PIL import Image

        arr = self.render_image(camera, mode, params or {})
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
        return buf.getvalue()

    def render_png(self, camera: Camera, mode: str = "rgb", params: dict | None = None) -> bytes:
        from PIL import Image

        arr = self.render_image(camera, mode, params or {})
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        return buf.getvalue()

    def _cached_jpeg(self, key: str, build: Callable[[], bytes]) -> bytes:
        with self._cache_lock:
            hit = self._cache.get(key)
            if hit is not None:
                self._cache.move_to_end(key)
                return hit
        body = build()
        with self._cache_lock:
            self._cache[key] = body
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_max:
                self._cache.popitem(last=False)
        return body

    def _warmup(self) -> None:
        """Compile every render mode's Metal kernels on the main thread.

        ``mx.fast.metal_kernel`` JIT-compiles on first dispatch, and that first
        compilation must happen on the thread that owns the default GPU stream
        (the main thread). The HTTP server hands each request to a fresh worker
        thread, so without this the very first ``/render`` — or the first use of
        a mode whose kernel hasn't been compiled yet — fails with
        "There is no Stream(gpu, 0) in current thread". Rendering one tiny frame
        per mode here primes the global kernel cache so handler threads only
        ever re-dispatch already-compiled kernels.
        """
        info = self.get_info()
        target = info.get("target", [0.0, 0.0, 0.0])
        q = {
            "radius": [str(info.get("radius", 4.0))],
            "fov": [str(info.get("fov", 60.0))],
            "tx": [str(target[0])],
            "ty": [str(target[1])],
            "tz": [str(target[2])],
            "w": ["64"],
            "h": ["64"],
        }
        camera = self.camera_from_query(q)
        for mode in ("rgb", *self.render_modes.keys()):
            try:
                self.render_image(camera, mode, {})
            except Exception as e:  # don't block serving if one mode can't warm up
                print(f"  (warmup skipped for mode '{mode}': {e})")

    # ------------------------------------------------------------------ server
    def serve(self, host: str = "127.0.0.1", port: int = 8090, open_browser: bool = True):
        """Start the viewer (blocking). Press Ctrl-C to stop."""
        viewer = self
        page = resources.files("mlx3d.viewer").joinpath("viewer.html").read_text()
        self._warmup()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # silence per-request logging
                pass

            def _send(self, code: int, content_type: str, body: bytes, extra: dict | None = None):
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                for k, v in (extra or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                url = urlparse(self.path)
                try:
                    if url.path == "/":
                        self._send(200, "text/html; charset=utf-8", page.encode())
                    elif url.path == "/info":
                        self._send(200, "application/json", json.dumps(viewer.get_info()).encode())
                    elif url.path == "/render":
                        q = parse_qs(url.query)
                        quality = int(float(q.get("q", [85])[0]))
                        mode = q.get("mode", ["rgb"])[0]
                        cam = viewer.camera_from_query(q)
                        params = viewer.params_from_query(q)
                        body = viewer._cached_jpeg(
                            url.query, lambda: viewer.render_jpeg(cam, quality, mode, params)
                        )
                        self._send(200, "image/jpeg", body)
                    elif url.path == "/screenshot":
                        q = parse_qs(url.query)
                        mode = q.get("mode", ["rgb"])[0]
                        cam = viewer.camera_from_query(q)
                        params = viewer.params_from_query(q)
                        body = viewer.render_png(cam, mode, params)
                        self._send(
                            200,
                            "image/png",
                            body,
                            {"Content-Disposition": 'attachment; filename="mlx3d.png"'},
                        )
                    else:
                        self._send(404, "text/plain", b"not found")
                except BrokenPipeError:
                    pass  # client cancelled the request (e.g. fast interaction)
                except Exception as e:  # surface render errors to the page
                    self._send(500, "text/plain", str(e).encode())

        server = ThreadingHTTPServer((host, port), Handler)
        addr = f"http://{host}:{port}"
        print(f"MLX3D viewer running at {addr}  (Ctrl-C to stop)")
        if open_browser:
            threading.Timer(0.3, lambda: webbrowser.open(addr)).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nViewer stopped.")
        finally:
            server.server_close()


# --------------------------------------------------------------- shared shading
def _depth_shade(depth: mx.array, alpha: mx.array) -> tuple[mx.array, mx.array]:
    valid = alpha > 1e-3
    near = mx.min(mx.where(valid, depth, mx.full(depth.shape, 1e9, dtype=depth.dtype)))
    far = mx.max(mx.where(valid, depth, mx.zeros_like(depth)))
    denom = mx.maximum(far - near, 1e-6)
    shade = 1.0 - (depth - near) / denom
    shade = mx.where(valid, mx.clip(shade, 0.0, 1.0), mx.zeros_like(shade))
    return shade, valid


# Compact "turbo"-like ramp (5 control colors) for legible depth/heatmaps.
_TURBO = np.array(
    [
        [0.19, 0.07, 0.23],
        [0.13, 0.57, 0.55],
        [0.47, 0.82, 0.27],
        [0.97, 0.79, 0.18],
        [0.90, 0.16, 0.11],
    ],
    dtype=np.float32,
)


def _colormap(shade: mx.array, valid: mx.array) -> mx.array:
    """Map a [0, 1] scalar field through the turbo ramp; black where invalid."""
    ramp = mx.array(_TURBO)
    x = mx.clip(shade, 0.0, 1.0) * (ramp.shape[0] - 1)
    lo = mx.floor(x)
    t = (x - lo)[..., None]
    lo_i = lo.astype(mx.int32)
    hi_i = mx.minimum(lo_i + 1, ramp.shape[0] - 1)
    c = ramp[lo_i] * (1.0 - t) + ramp[hi_i] * t
    return mx.where(valid[..., None], c, mx.zeros_like(c))


def _depth_to_rgb(depth: mx.array, alpha: mx.array, colormap: bool = False) -> mx.array:
    shade, valid = _depth_shade(depth, alpha)
    if colormap:
        return _colormap(shade, valid)
    return mx.stack([shade, shade, shade], axis=-1)


def _depth_to_mesh_rgb(depth: mx.array, alpha: mx.array) -> mx.array:
    shade, valid = _depth_shade(depth, alpha)
    zero_col = mx.zeros((depth.shape[0], 1), dtype=depth.dtype)
    zero_row = mx.zeros((1, depth.shape[1]), dtype=depth.dtype)
    dzx = mx.concatenate([mx.abs(shade[:, 1:] - shade[:, :-1]), zero_col], axis=1)
    dzy = mx.concatenate([mx.abs(shade[1:, :] - shade[:-1, :]), zero_row], axis=0)
    dax = mx.concatenate([mx.abs(alpha[:, 1:] - alpha[:, :-1]), zero_col], axis=1)
    day = mx.concatenate([mx.abs(alpha[1:, :] - alpha[:-1, :]), zero_row], axis=0)
    contour = mx.maximum(mx.maximum(dzx, dzy) * 18.0, mx.maximum(dax, day) * 4.0)
    contour = mx.where(valid, mx.clip(contour, 0.0, 1.0), mx.zeros_like(contour))
    surface = mx.where(valid, 0.18 + 0.70 * shade, mx.zeros_like(shade))
    value = surface * (1.0 - contour) + 0.02 * contour
    return mx.stack([value, value, value], axis=-1)


def _alpha_to_rgb(alpha: mx.array) -> mx.array:
    a = mx.clip(alpha, 0.0, 1.0)
    return mx.stack([a, a, a], axis=-1)


# ------------------------------------------------------------------ gaussians
class LiveGaussianViewer:
    """Live viewer adapter for a Gaussian model that changes during training.

    The training thread calls :meth:`publish` at a controlled cadence. Each
    publish stores references to already-evaluated MLX arrays, so render
    requests never read a half-mutated training model and no CPU copies are
    made just to refresh the preview.
    """

    def __init__(
        self,
        model,
        background: tuple[float, float, float] = (0.0, 0.0, 0.0),
        max_scale: float = 0.5,
        poll_ms: int = 750,
        initial_radius: float | None = None,
        initial_target: tuple[float, float, float] | None = None,
    ):
        from ..splatting import GaussianModel

        self._model_cls = GaussianModel
        self._lock = threading.Lock()
        self._background = mx.array(background)
        self._params: dict[str, mx.array] = {}
        self._sh_degree = model.sh_degree
        self._active_sh_degree = model.active_sh_degree
        self._revision = -1
        self.publish(model, step=0, loss=None)

        if initial_radius is None or initial_target is None:
            means = np.array(self._params["means"])
            center = means.mean(axis=0)
            radius = float(np.percentile(np.linalg.norm(means - center, axis=1), 90)) * 2.5 + 1e-3
        else:
            center = np.array(initial_target, dtype=np.float32)
            radius = float(initial_radius)
        self.viewer = Viewer(
            lambda camera, params: self._render(camera, "rgb", params),
            info={
                "mode": "live gaussian splatting",
                "live": True,
                "max_scale": float(max_scale),
                "poll_info_ms": int(poll_ms),
                "status": "training",
            },
            render_modes={
                "depth": lambda camera, params: self._render(camera, "depth", params),
                "mesh": lambda camera, params: self._render(camera, "mesh", params),
            },
            controls=_gaussian_controls(model.active_sh_degree, background),
            initial_radius=radius,
            initial_target=tuple(float(c) for c in center),
        )
        self.viewer.update_info(
            revision=self._revision,
            gaussians=model.num_gaussians,
            sh_degree=model.active_sh_degree,
        )

    def publish(
        self,
        model,
        step: int | None = None,
        loss: float | None = None,
        lr_means: float | None = None,
    ) -> None:
        params = dict(model.params)
        mx.eval(params, self._background)
        with self._lock:
            self._params = params
            self._sh_degree = model.sh_degree
            self._active_sh_degree = model.active_sh_degree
            self._revision += 1
            revision = self._revision
        info = {
            "revision": revision,
            "gaussians": model.num_gaussians,
            "sh_degree": model.active_sh_degree,
            "updated_at": time.time(),
        }
        if step is not None:
            info["step"] = int(step)
        if loss is not None:
            info["loss"] = float(loss)
        if lr_means is not None:
            info["lr_means"] = float(lr_means)
        if hasattr(self, "viewer"):
            self.viewer.update_info(**info)

    def mark_done(self) -> None:
        self.viewer.update_info(status="done")

    def _render(self, camera: Camera, mode: str, params: dict | None = None) -> mx.array:
        with self._lock:
            model_params = dict(self._params)
            sh_degree = self._sh_degree
            active_sh_degree = self._active_sh_degree
        model = self._model_cls(model_params, sh_degree=sh_degree)
        model.active_sh_degree = active_sh_degree
        return _render_gaussian(model, camera, mode, params or {}, self._background)

    def serve(self, host: str = "127.0.0.1", port: int = 8090, open_browser: bool = True):
        self.viewer.serve(host=host, port=port, open_browser=open_browser)


def _gaussian_controls(active_sh_degree: int, background) -> list[dict]:
    bg = "#%02x%02x%02x" % tuple(int(round(c * 255)) for c in background)
    return [
        _color("bg", "Background", bg),
        _slider("exposure", "Exposure", 0.1, 4.0, 0.05, 1.0),
        _slider("gamma", "Gamma", 0.4, 2.4, 0.05, 1.0),
        _slider("splat_scale", "Splat scale", 0.1, 2.5, 0.05, 1.0),
        _slider("sh_degree", "SH degree", 0, 3, 1, active_sh_degree),
        _select("clip_axis", "Clip axis", ["off", "x", "y", "z", "-x", "-y", "-z"], "off"),
        _slider("clip", "Clip position", 0.0, 1.0, 0.01, 1.0),
    ]


def _gaussian_bounds(means_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo = np.percentile(means_np, 1, axis=0)
    hi = np.percentile(means_np, 99, axis=0)
    return lo.astype(np.float32), hi.astype(np.float32)


def _clip_opacities(
    opacities: mx.array, means: mx.array, params: dict, bounds: tuple[np.ndarray, np.ndarray]
) -> mx.array:
    """Hide Gaussians on the far side of a debug clip plane by zeroing opacity."""
    axis_name = str(params.get("clip_axis", "off"))
    if axis_name == "off":
        return opacities
    negative = axis_name.startswith("-")
    axis = {"x": 0, "y": 1, "z": 2}[axis_name.lstrip("-")]
    lo, hi = bounds
    t = float(params.get("clip", 1.0))
    threshold = float(lo[axis]) + t * float(hi[axis] - lo[axis])
    coord = means[:, axis]
    keep = coord >= threshold if negative else coord <= threshold
    return mx.where(keep, opacities, mx.zeros_like(opacities))


def _render_gaussian(
    model,
    camera: Camera,
    mode: str,
    params: dict,
    default_bg: mx.array,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> mx.array:
    from ..splatting import render_gaussian_depth, render_gaussians

    bg_rgb = hex_to_rgb(
        params.get("bg"), tuple(float(c) for c in np.array(default_bg).reshape(-1)[:3])
    )
    # Share the clip plane and splat-scale knobs across every display mode.
    means = model.params["means"]
    quats = model.params["quats"]
    scales = model.scales_act * float(params.get("splat_scale", 1.0))
    opacities = model.opacities_act
    if bounds is not None:
        opacities = _clip_opacities(opacities, means, params, bounds)

    if mode in ("depth", "mesh", "alpha"):
        out = render_gaussian_depth(camera, means, quats, scales, opacities)
        if mode == "depth":
            return _depth_to_rgb(out["depth"], out["alpha"], colormap=True)
        if mode == "mesh":
            return _depth_to_mesh_rgb(out["depth"], out["alpha"])
        return _alpha_to_rgb(out["alpha"])

    sh_degree = int(params.get("sh_degree", model.active_sh_degree))
    sh_degree = max(0, min(sh_degree, model.sh_degree))
    out = render_gaussians(
        camera,
        means,
        quats,
        scales,
        opacities,
        sh=model.sh,
        sh_degree=sh_degree,
        background=mx.array(bg_rgb),
    )
    return tone_map(out["image"], params.get("exposure", 1.0), params.get("gamma", 1.0))


def view_live_gaussians(
    model,
    background: tuple[float, float, float] = (0.0, 0.0, 0.0),
    host: str = "127.0.0.1",
    port: int = 8090,
    open_browser: bool = True,
    serve: bool = True,
    max_scale: float = 0.5,
    poll_ms: int = 750,
    initial_radius: float | None = None,
    initial_target: tuple[float, float, float] | None = None,
) -> LiveGaussianViewer:
    """Open a live viewer for a Gaussian model being updated by training.

    Returns a :class:`LiveGaussianViewer`; call ``publish(model, step, loss)``
    whenever a fresh preview should be visible in the browser.
    """
    live = LiveGaussianViewer(
        model,
        background=background,
        max_scale=max_scale,
        poll_ms=poll_ms,
        initial_radius=initial_radius,
        initial_target=initial_target,
    )
    if serve:
        live.serve(host=host, port=port, open_browser=open_browser)
    return live


def view_gaussians(
    model,
    background: tuple[float, float, float] = (0.0, 0.0, 0.0),
    host: str = "127.0.0.1",
    port: int = 8090,
    open_browser: bool = True,
    serve: bool = True,
) -> Viewer:
    """Open an interactive viewer for a :class:`~mlx3d.splatting.GaussianModel`.

    Renders with the Metal rasterization kernels — real-time orbiting for
    typical scene sizes. Live controls cover background, exposure/gamma, splat
    scale, SH degree, and a debug clip plane; extra display modes show depth,
    alpha, and a contour "mesh" preview. The orbit radius is initialized from
    the scene's bounding sphere.

    Args:
        model (GaussianModel): the scene (e.g. ``GaussianModel.load_ply(...)``).
        background: RGB background color.
        serve: start the (blocking) server; pass ``False`` to get the
            configured :class:`Viewer` back instead (used in tests).
    """
    bg = mx.array(background)
    # Handler threads cannot evaluate lazy arrays created here on the main
    # thread, so materialize everything the render closure captures.
    mx.eval(model.params, bg)

    means = np.array(model.params["means"])
    center = means.mean(axis=0)
    radius = float(np.percentile(np.linalg.norm(means - center, axis=1), 90)) * 2.5 + 1e-3
    bounds = _gaussian_bounds(means)

    def render(camera: Camera, params: dict) -> mx.array:
        return _render_gaussian(model, camera, "rgb", params, bg, bounds)

    viewer = Viewer(
        render,
        info={"mode": "gaussian splatting", "gaussians": model.num_gaussians},
        render_modes={
            "depth": lambda camera, params: _render_gaussian(model, camera, "depth", params, bg, bounds),
            "alpha": lambda camera, params: _render_gaussian(model, camera, "alpha", params, bg, bounds),
            "mesh": lambda camera, params: _render_gaussian(model, camera, "mesh", params, bg, bounds),
        },
        controls=_gaussian_controls(model.active_sh_degree, background),
        initial_radius=radius,
        initial_target=tuple(float(c) for c in center),
    )
    if serve:
        viewer.serve(host=host, port=port, open_browser=open_browser)
    return viewer


# ----------------------------------------------------------------------- meshes
def _orbit_pose(centers: np.ndarray) -> tuple[float, tuple[float, float, float]]:
    center = centers.mean(axis=0)
    spread = np.linalg.norm(centers - center, axis=1)
    radius = float(np.percentile(spread, 95)) * 2.6 + 1e-3
    return radius, tuple(float(c) for c in center)


def _light_dir(params: dict) -> mx.array:
    az = math.radians(float(params.get("light_az", 35.0)))
    el = math.radians(float(params.get("light_el", 35.0)))
    return mx.array(
        [math.cos(el) * math.sin(az), math.sin(el), math.cos(el) * math.cos(az)],
        dtype=mx.float32,
    )


def _mesh_modes(mesh, verts_colors, default_bg) -> tuple[Callable, dict[str, Callable], list[dict]]:
    from ..renderer import (
        DirectionalLights,
        interpolate_face_attributes,
        rasterize_meshes,
        render_mesh,
    )

    normals = mesh.verts_normals_packed()

    def _bg(params):
        return hex_to_rgb(params.get("bg"), default_bg)

    def shaded(camera: Camera, params: dict) -> mx.array:
        lights = [DirectionalLights(direction=tuple(np.array(_light_dir(params)).tolist()))]
        out = render_mesh(
            camera,
            mesh,
            verts_colors=verts_colors,
            lights=lights,
            background=_bg(params),
            shading="phong",
        )
        return tone_map(out["image"], params.get("exposure", 1.0), params.get("gamma", 1.0))

    def normals_mode(camera: Camera, params: dict) -> mx.array:
        frag = rasterize_meshes(camera, mesh)
        n = interpolate_face_attributes(frag, normals)
        n = n / mx.maximum(mx.linalg.norm(n, axis=-1, keepdims=True), 1e-8)
        rgb = 0.5 * n + 0.5
        bg = mx.array(_bg(params))
        return mx.where(frag.valid[..., None], rgb, bg)

    def depth_mode(camera: Camera, params: dict) -> mx.array:
        frag = rasterize_meshes(camera, mesh)
        alpha = frag.valid.astype(mx.float32)
        return _depth_to_rgb(frag.zbuf, alpha, colormap=True)

    def wireframe(camera: Camera, params: dict) -> mx.array:
        frag = rasterize_meshes(camera, mesh)
        edge = mx.min(frag.bary, axis=-1)
        width = float(params.get("wire", 0.02))
        line = mx.clip(1.0 - edge / max(width, 1e-4), 0.0, 1.0)
        line = mx.where(frag.valid, line, mx.zeros_like(line))
        bg = mx.array(_bg(params))
        ink = mx.array([0.85, 0.88, 1.0])
        rgb = bg[None, None, :] * (1.0 - line[..., None]) + ink[None, None, :] * line[..., None]
        return rgb

    controls = [
        _color("bg", "Background", "#%02x%02x%02x" % tuple(int(round(c * 255)) for c in default_bg)),
        _slider("exposure", "Exposure", 0.1, 4.0, 0.05, 1.0),
        _slider("gamma", "Gamma", 0.4, 2.4, 0.05, 1.0),
        _slider("light_az", "Light azimuth", -180, 180, 5, 35),
        _slider("light_el", "Light elevation", -89, 89, 5, 35),
        _slider("wire", "Wire width", 0.005, 0.08, 0.005, 0.02),
    ]
    modes = {"normals": normals_mode, "depth": depth_mode, "wireframe": wireframe}
    return shaded, modes, controls


def view_mesh(
    mesh,
    verts_colors: mx.array | None = None,
    background: tuple[float, float, float] = (0.04, 0.04, 0.06),
    host: str = "127.0.0.1",
    port: int = 8090,
    open_browser: bool = True,
    serve: bool = True,
) -> Viewer:
    """Open an interactive viewer for a :class:`~mlx3d.structures.Meshes`.

    Renders with the tile-based hard rasterizer and the MLX3D shader stack.
    Display modes: shaded (Phong, with a steerable light), surface normals,
    depth (turbo colormap), and wireframe. ``mesh`` may also be a path to an
    OBJ / PLY / glTF asset.

    Args:
        mesh: a :class:`Meshes`, or a path string to load.
        verts_colors: optional ``(V, 3)`` per-vertex albedo in [0, 1].
        serve: start the (blocking) server; pass ``False`` to get the
            :class:`Viewer` back instead (used in tests).
    """
    from ..structures import Meshes

    if isinstance(mesh, str):
        mesh, loaded_colors = _load_mesh(mesh)
        verts_colors = verts_colors if verts_colors is not None else loaded_colors
    if not isinstance(mesh, Meshes):
        raise TypeError("view_mesh expects a Meshes or a path to a mesh asset.")

    verts = mesh.verts_packed()
    faces = mesh.faces_packed()
    if verts_colors is None:
        verts_colors = mx.full(verts.shape, 0.75, dtype=mx.float32)
    mx.eval(verts, faces, verts_colors)

    radius, target = _orbit_pose(np.array(verts))
    shaded, modes, controls = _mesh_modes(mesh, verts_colors, background)

    viewer = Viewer(
        shaded,
        info={
            "mode": "mesh",
            "verts": int(verts.shape[0]),
            "faces": int(faces.shape[0]),
        },
        render_modes=modes,
        controls=controls,
        initial_radius=radius,
        initial_target=target,
    )
    if serve:
        viewer.serve(host=host, port=port, open_browser=open_browser)
    return viewer


# ------------------------------------------------------------------ pointclouds
def view_pointcloud(
    points,
    colors: mx.array | None = None,
    background: tuple[float, float, float] = (0.02, 0.02, 0.03),
    point_radius: float = 2.0,
    host: str = "127.0.0.1",
    port: int = 8090,
    open_browser: bool = True,
    serve: bool = True,
) -> Viewer:
    """Open an interactive viewer for a point cloud.

    Uses the differentiable splat-based point renderer. ``points`` may be a
    :class:`~mlx3d.structures.Pointclouds`, an ``(N, 3)`` array, or a path to a
    PLY point cloud. The live ``point size`` control adjusts the splat radius.
    """
    from ..renderer import render_points
    from ..structures import Pointclouds

    if isinstance(points, str):
        points, colors = _load_points(points, colors)
    if isinstance(points, Pointclouds):
        colors = colors if colors is not None else (
            points.features_packed() if points.features_packed() is not None else None
        )
        points = points.points_packed()
    pts = mx.array(points) if not isinstance(points, mx.array) else points
    if colors is None:
        colors = mx.full(pts.shape, 0.8, dtype=mx.float32)
    mx.eval(pts, colors)

    radius, target = _orbit_pose(np.array(pts))

    def render(camera: Camera, params: dict) -> mx.array:
        bg = hex_to_rgb(params.get("bg"), background)
        r = float(params.get("point_size", point_radius))
        out = render_points(camera, pts, colors, radius=r, background=mx.array(bg))
        img = out["image"] if isinstance(out, dict) else out
        return tone_map(img, params.get("exposure", 1.0), params.get("gamma", 1.0))

    controls = [
        _color("bg", "Background", "#%02x%02x%02x" % tuple(int(round(c * 255)) for c in background)),
        _slider("point_size", "Point size", 0.5, 8.0, 0.25, point_radius),
        _slider("exposure", "Exposure", 0.1, 4.0, 0.05, 1.0),
        _slider("gamma", "Gamma", 0.4, 2.4, 0.05, 1.0),
    ]
    viewer = Viewer(
        render,
        info={"mode": "point cloud", "points": int(pts.shape[0])},
        controls=controls,
        initial_radius=radius,
        initial_target=target,
    )
    if serve:
        viewer.serve(host=host, port=port, open_browser=open_browser)
    return viewer


# --------------------------------------------------------------------- loaders
def _is_gaussian_ply(ply) -> bool:
    keys = set(ply.extra.keys())
    return "opacity" in keys and any(k.startswith("scale_") for k in keys)


def _load_mesh(path: str):
    """Load a mesh (with optional per-vertex colors) from OBJ / PLY / glTF."""
    from ..io import load_gltf, load_obj, load_ply
    from ..structures import Meshes

    lower = path.lower()
    if lower.endswith(".obj"):
        obj = load_obj(path)
        return Meshes([obj.verts], [obj.faces]), obj.verts_colors
    if lower.endswith((".gltf", ".glb")):
        asset = load_gltf(path)
        return Meshes([asset.verts], [asset.faces]), None
    if lower.endswith(".ply"):
        ply = load_ply(path)
        if ply.faces is None:
            raise ValueError(f"{path} is a point cloud, not a mesh (no faces).")
        return Meshes([ply.verts], [ply.faces]), ply.colors
    raise ValueError(f"Unsupported mesh format: {path}")


def _load_points(path: str, colors):
    from ..io import load_ply

    ply = load_ply(path)
    return ply.verts, colors if colors is not None else ply.colors


def view_file(
    path: str,
    host: str = "127.0.0.1",
    port: int = 8090,
    open_browser: bool = True,
    serve: bool = True,
    kind: str | None = None,
    background: tuple[float, float, float] | None = None,
) -> Viewer:
    """Open the right viewer for a scene file, auto-detecting its type.

    - ``.ply`` with Gaussian attributes  → :func:`view_gaussians`
    - ``.ply`` with faces                → :func:`view_mesh`
    - ``.ply`` points only               → :func:`view_pointcloud`
    - ``.obj`` / ``.gltf`` / ``.glb``    → :func:`view_mesh`

    Pass ``kind`` (``"gaussians"`` / ``"mesh"`` / ``"points"``) to override the
    auto-detection.
    """
    from ..io import load_ply
    from ..splatting import GaussianModel

    lower = path.lower()
    if kind is None:
        if lower.endswith((".obj", ".gltf", ".glb")):
            kind = "mesh"
        elif lower.endswith(".ply"):
            ply = load_ply(path)
            if _is_gaussian_ply(ply):
                kind = "gaussians"
            elif ply.faces is not None:
                kind = "mesh"
            else:
                kind = "points"
        else:
            raise ValueError(f"Cannot infer scene type for {path}; pass kind=...")

    common = dict(host=host, port=port, open_browser=open_browser, serve=serve)
    if kind == "gaussians":
        model = GaussianModel.load_ply(path)
        bg = background if background is not None else (0.0, 0.0, 0.0)
        return view_gaussians(model, background=bg, **common)
    if kind == "mesh":
        mesh, colors = _load_mesh(path)
        bg = background if background is not None else (0.04, 0.04, 0.06)
        return view_mesh(mesh, verts_colors=colors, background=bg, **common)
    if kind == "points":
        pts, colors = _load_points(path, None)
        bg = background if background is not None else (0.02, 0.02, 0.03)
        return view_pointcloud(pts, colors=colors, background=bg, **common)
    raise ValueError(f"Unknown kind: {kind!r}")


# ------------------------------------------------------------------------- nerf
def view_nerf(
    model,
    near: float = 2.0,
    far: float = 6.0,
    num_coarse: int = 64,
    num_fine: int = 64,
    white_background: bool = True,
    chunk: int = 16384,
    host: str = "127.0.0.1",
    port: int = 8090,
    open_browser: bool = True,
    serve: bool = True,
) -> Viewer:
    """Open an interactive viewer for a trained :class:`~mlx3d.nn.NeRF`.

    NeRF rendering is far heavier than splatting; the page's adaptive
    resolution keeps interaction usable (it drops resolution while you drag
    and refines when you stop). Live controls trade quality for speed: sample
    count, near/far, and exposure/gamma.
    """
    from ..nn import render_rays

    mx.eval(model.parameters())  # see Viewer docstring: threads need evaluated arrays

    def render(camera: Camera, params: dict) -> mx.array:
        n_coarse = int(params.get("num_coarse", num_coarse))
        n_fine = int(params.get("num_fine", num_fine))
        near_p = float(params.get("near", near))
        far_p = float(params.get("far", far))
        origins, dirs = camera.generate_rays()
        o = origins.reshape(-1, 3)
        d = dirs.reshape(-1, 3)
        out = []
        for s in range(0, o.shape[0], chunk):
            res = render_rays(
                model,
                o[s : s + chunk],
                d[s : s + chunk],
                near_p,
                far_p,
                num_coarse=n_coarse,
                num_fine=n_fine,
                stratified=False,
                white_background=white_background,
            )
            out.append(res["rgb"])
            mx.eval(out[-1])
        img = mx.concatenate(out).reshape(camera.height, camera.width, 3)
        return tone_map(img, params.get("exposure", 1.0), params.get("gamma", 1.0))

    controls = [
        _slider("num_coarse", "Coarse samples", 16, 128, 8, num_coarse),
        _slider("num_fine", "Fine samples", 0, 128, 8, num_fine),
        _slider("near", "Near", 0.1, far, 0.1, near),
        _slider("far", "Far", near, far * 2, 0.1, far),
        _slider("exposure", "Exposure", 0.1, 4.0, 0.05, 1.0),
        _slider("gamma", "Gamma", 0.4, 2.4, 0.05, 1.0),
    ]
    viewer = Viewer(
        render,
        info={"mode": "nerf", "heavy": True},
        controls=controls,
        initial_radius=0.5 * (near + far),
    )
    if serve:
        viewer.serve(host=host, port=port, open_browser=open_browser)
    return viewer
