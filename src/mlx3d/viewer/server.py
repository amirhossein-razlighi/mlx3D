"""An interactive browser-based viewer for MLX3D scenes.

The viewer runs a tiny local HTTP server. The browser page sends orbit-camera
state (azimuth / elevation / radius / target) and receives JPEG frames
rendered on the Apple GPU — for Gaussian Splatting that means the Metal
rasterization kernels, so a 100k-Gaussian scene orbits in real time.

No dependencies beyond the package itself (Pillow encodes the JPEGs).

Example:
    >>> from mlx3d.splatting import GaussianModel
    >>> from mlx3d.viewer import view_gaussians
    >>> model = GaussianModel.load_ply("point_cloud.ply")
    >>> view_gaussians(model)  # opens http://127.0.0.1:8090
"""

import io
import json
import math
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Callable
from urllib.parse import parse_qs, urlparse

import mlx.core as mx
import numpy as np

from ..cameras import Camera

__all__ = ["LiveGaussianViewer", "Viewer", "view_gaussians", "view_live_gaussians", "view_nerf"]

RenderFn = Callable[[Camera], mx.array]


class Viewer:
    """Serve an interactive orbit-camera viewer for any camera-to-image renderer.

    Args:
        render_fn: callback mapping a :class:`~mlx3d.cameras.Camera` to an
            (H, W, 3) image in [0, 1]. Rendering happens on HTTP handler
            threads, and MLX cannot evaluate lazy arrays created on another
            thread — call ``mx.eval`` on any arrays the callback captures
            (model parameters etc.) before serving. The ``view_*`` helpers
            do this for you.
        info: metadata shown in the page HUD (e.g. ``{"gaussians": 100000}``).
        initial_radius: starting orbit distance.
        initial_target: starting orbit target.
        fov: vertical field of view in degrees.
    """

    def __init__(
        self,
        render_fn: RenderFn,
        info: dict | None = None,
        initial_radius: float = 4.0,
        initial_target: tuple[float, float, float] = (0.0, 0.0, 0.0),
        fov: float = 60.0,
    ):
        self.render_fn = render_fn
        self.info = {
            "radius": initial_radius,
            "target": list(initial_target),
            "fov": fov,
            **(info or {}),
        }
        self._info_lock = threading.Lock()
        self._lock = threading.Lock()  # one render at a time on the GPU

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

        theta = f("theta", 0.0)        # azimuth, radians
        phi = f("phi", 0.0)            # elevation, radians
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
            eye=eye, at=target, up=(0.0, up_sign, 0.0),
            fov=fov, width=width, height=height,
        )

    def render_jpeg(self, camera: Camera, quality: int = 85) -> bytes:
        from PIL import Image

        with self._lock:
            img = self.render_fn(camera)
            mx.eval(img)
        arr = (np.clip(np.array(img), 0.0, 1.0) * 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
        return buf.getvalue()

    # ------------------------------------------------------------------ server
    def serve(self, host: str = "127.0.0.1", port: int = 8090, open_browser: bool = True):
        """Start the viewer (blocking). Press Ctrl-C to stop."""
        viewer = self
        page = resources.files("mlx3d.viewer").joinpath("viewer.html").read_text()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # silence per-request logging
                pass

            def _send(self, code: int, content_type: str, body: bytes):
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
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
                        cam = viewer.camera_from_query(q)
                        self._send(200, "image/jpeg", viewer.render_jpeg(cam, quality))
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
            radius = (
                float(np.percentile(np.linalg.norm(means - center, axis=1), 90)) * 2.5 + 1e-3
            )
        else:
            center = np.array(initial_target, dtype=np.float32)
            radius = float(initial_radius)
        self.viewer = Viewer(
            self._render,
            info={
                "mode": "live gaussian splatting",
                "live": True,
                "max_scale": float(max_scale),
                "poll_info_ms": int(poll_ms),
                "status": "training",
            },
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

    def _render(self, camera: Camera) -> mx.array:
        with self._lock:
            params = dict(self._params)
            sh_degree = self._sh_degree
            active_sh_degree = self._active_sh_degree
        model = self._model_cls(params, sh_degree=sh_degree)
        model.active_sh_degree = active_sh_degree
        return model.render(camera, background=self._background)["image"]

    def serve(self, host: str = "127.0.0.1", port: int = 8090, open_browser: bool = True):
        self.viewer.serve(host=host, port=port, open_browser=open_browser)


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
    typical scene sizes. The orbit radius is initialized from the scene's
    bounding sphere.

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

    def render(camera: Camera) -> mx.array:
        return model.render(camera, background=bg)["image"]

    viewer = Viewer(
        render,
        info={"mode": "gaussian splatting", "gaussians": model.num_gaussians},
        initial_radius=radius,
        initial_target=tuple(float(c) for c in center),
    )
    if serve:
        viewer.serve(host=host, port=port, open_browser=open_browser)
    return viewer


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
    and refines when you stop).
    """
    from ..nn import render_rays

    mx.eval(model.parameters())  # see Viewer docstring: threads need evaluated arrays

    def render(camera: Camera) -> mx.array:
        origins, dirs = camera.generate_rays()
        o = origins.reshape(-1, 3)
        d = dirs.reshape(-1, 3)
        out = []
        for s in range(0, o.shape[0], chunk):
            res = render_rays(
                model, o[s : s + chunk], d[s : s + chunk], near, far,
                num_coarse=num_coarse, num_fine=num_fine,
                stratified=False, white_background=white_background,
            )
            out.append(res["rgb"])
            mx.eval(out[-1])
        return mx.concatenate(out).reshape(camera.height, camera.width, 3)

    viewer = Viewer(
        render,
        info={"mode": "nerf", "heavy": True},
        initial_radius=0.5 * (near + far),
    )
    if serve:
        viewer.serve(host=host, port=port, open_browser=open_browser)
    return viewer
