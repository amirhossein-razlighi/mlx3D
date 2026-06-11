import json
import socket
import threading
import time
import urllib.error
import urllib.request

import mlx.core as mx
import numpy as np

from mlx3d.cameras import Camera
from mlx3d.splatting import GaussianModel
from mlx3d.viewer import Viewer, view_gaussians, view_live_gaussians


def test_camera_from_query_orbit():
    q = {"theta": ["0.0"], "phi": ["0.0"], "radius": ["3.0"],
         "tx": ["0"], "ty": ["0"], "tz": ["0"], "fov": ["60"],
         "w": ["64"], "h": ["48"]}
    cam = Viewer.camera_from_query(q)
    assert cam.width == 64 and cam.height == 48
    # theta=phi=0 puts the eye at target + (0, 0, radius), looking at target.
    np.testing.assert_allclose(np.array(cam.camera_center), [0, 0, 3.0], atol=1e-5)
    xy, z = cam.project_points(mx.zeros((1, 3)))
    np.testing.assert_allclose(np.array(xy[0]), [32.0, 24.0], atol=1e-3)
    assert float(z[0]) > 0


def test_camera_from_query_flipped_up():
    q = {"phi": ["0.5"], "radius": ["2.0"], "up": ["-1"], "w": ["32"], "h": ["32"]}
    cam = Viewer.camera_from_query(q)
    # With up flipped, positive elevation moves the eye to negative y.
    assert float(cam.camera_center[1]) < 0


def test_render_jpeg_bytes():
    model = GaussianModel.from_points(mx.random.normal((50, 3)) * 0.3, sh_degree=0)
    viewer = view_gaussians(model, serve=False)
    cam = Camera.look_at(eye=(0, 0, -3.0), at=(0, 0, 0), width=64, height=48)
    data = viewer.render_jpeg(cam)
    assert data[:2] == b"\xff\xd8"  # JPEG magic
    assert len(data) > 500


def test_live_gaussian_viewer_publish():
    model = GaussianModel.from_points(mx.random.normal((50, 3)) * 0.3, sh_degree=0)
    live = view_live_gaussians(
        model,
        serve=False,
        max_scale=0.25,
        poll_ms=250,
        initial_radius=2.5,
        initial_target=(1.0, 2.0, 3.0),
    )
    info0 = live.viewer.get_info()

    model.params["means"] = model.params["means"] + 0.01
    mx.eval(model.params)
    live.publish(model, step=7, loss=0.1234, lr_means=1.2e-4)
    info = live.viewer.get_info()

    assert info["live"] is True
    assert info["max_scale"] == 0.25
    assert info["poll_info_ms"] == 250
    assert info["radius"] == 2.5
    assert info["target"] == [1.0, 2.0, 3.0]
    assert info["revision"] > info0["revision"]
    assert info["step"] == 7
    assert abs(info["loss"] - 0.1234) < 1e-6
    assert abs(info["lr_means"] - 1.2e-4) < 1e-8

    cam = Camera.look_at(eye=(0, 0, -3.0), at=(0, 0, 0), width=64, height=48)
    data = live.viewer.render_jpeg(cam)
    assert data[:2] == b"\xff\xd8"


def _serve_in_background(viewer) -> str:
    """Start viewer.serve on a free port; return the base URL once it's up."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    threading.Thread(
        target=viewer.serve,
        kwargs={"host": "127.0.0.1", "port": port, "open_browser": False},
        daemon=True,
    ).start()
    base = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            urllib.request.urlopen(base + "/info", timeout=1)
            return base
        except Exception:
            time.sleep(0.05)
    raise RuntimeError("viewer server did not start")


def test_http_endpoints():
    model = GaussianModel.from_points(mx.random.normal((50, 3)) * 0.3, sh_degree=0)
    viewer = view_gaussians(model, serve=False)
    base = _serve_in_background(viewer)

    page = urllib.request.urlopen(base + "/", timeout=5).read()
    assert b"MLX3D Viewer" in page

    info = json.loads(urllib.request.urlopen(base + "/info", timeout=5).read())
    assert info["gaussians"] == 50
    assert info["mode"] == "gaussian splatting"
    assert info["radius"] > 0

    frame = urllib.request.urlopen(
        base + "/render?theta=0.4&phi=0.2&radius=3&w=64&h=48", timeout=10
    ).read()
    assert frame[:2] == b"\xff\xd8"

    try:
        urllib.request.urlopen(base + "/nope", timeout=5)
        raised = False
    except urllib.error.HTTPError as e:
        raised = e.code == 404
    assert raised


def test_live_http_info_updates():
    model = GaussianModel.from_points(mx.random.normal((50, 3)) * 0.3, sh_degree=0)
    live = view_live_gaussians(model, serve=False)
    base = _serve_in_background(live.viewer)

    info0 = json.loads(urllib.request.urlopen(base + "/info", timeout=5).read())
    live.publish(model, step=11, loss=0.25)
    info = json.loads(urllib.request.urlopen(base + "/info", timeout=5).read())

    assert info["live"] is True
    assert info["revision"] > info0["revision"]
    assert info["step"] == 11
    assert info["loss"] == 0.25
