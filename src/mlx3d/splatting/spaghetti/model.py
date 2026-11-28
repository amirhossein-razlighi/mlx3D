"""A learnable set of spaghetti segments: geometry, kernels, density, appearance."""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

from ...cameras import Camera
from ..sh import num_sh_bases, rgb_to_sh
from .render import render_spaghetti, segment_colors

__all__ = ["SpaghettiModel"]

_SQRT_2PI = math.sqrt(2.0 * math.pi)


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


class SpaghettiModel:
    """Gaussian-blurred line segments with spherical-harmonic colors.

    Parameters (``params``, all ``(N, ...)``):

    - ``means``, ``halves``: segment midpoint and half-vector (segment = ``m ± h``);
    - ``quats``, ``scales``: kernel rotation and log standard deviations;
    - ``opacities``: logit of the *peak point opacity* ``o``. The density is
      ``rho = -ln(1 - o) / (sqrt(2 pi) sigma)`` with ``sigma`` the kernel's
      geometric-mean scale, so a zero-length segment seen through its center
      has opacity ~``o``, exactly like a 3DGS splat; and ``rho`` (density per
      unit length) is what splitting preserves.
    - ``sh_dc`` (N, 1, 3), ``sh_rest`` (N, K-1, 3).
    """

    def __init__(self, params: dict[str, mx.array], sh_degree: int = 3):
        self.params = params
        self.sh_degree = sh_degree
        self.active_sh_degree = 0

    # ------------------------------------------------------------- construction
    @classmethod
    def from_points(
        cls,
        points: mx.array,
        colors: mx.array | None = None,
        sh_degree: int = 3,
        initial_opacity: float = 0.1,
        initial_length: float = 2.0,
        max_scale: float | None = None,
        seed: int = 0,
    ) -> SpaghettiModel:
        """Short segments at ``points`` in random directions.

        Kernel scales come from nearest-neighbor spacing; each segment is
        ``initial_length`` kernel widths long (a nonzero length is needed: at
        exactly zero the gradient with respect to it vanishes by symmetry).
        """
        from ...ops import knn_points

        points = mx.array(points, dtype=mx.float32)
        n = points.shape[0]
        if n > 20_000:  # a reference subsample keeps the k-NN cheap for big clouds
            pick = np.random.default_rng(seed).choice(n, 20_000, replace=False)
            ref = points[mx.array(pick)]
        else:
            ref = points
        dists, _ = knn_points(points, ref, K=4)
        spacing = mx.sqrt(mx.maximum(mx.mean(dists[:, 1:], axis=-1), 1e-14))
        if max_scale is not None:
            spacing = mx.minimum(spacing, max_scale)
        rng = np.random.default_rng(seed)
        dirs = rng.normal(size=(n, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        quats = rng.normal(size=(n, 4))
        colors = mx.full((n, 3), 0.5) if colors is None else mx.array(colors, dtype=mx.float32)
        k = num_sh_bases(sh_degree)
        params = {
            "means": points,
            "halves": mx.array(dirs, dtype=mx.float32) * (0.5 * initial_length * spacing[:, None]),
            "quats": mx.array(
                quats / np.linalg.norm(quats, axis=1, keepdims=True), dtype=mx.float32
            ),
            "scales": mx.broadcast_to(mx.log(spacing)[:, None], (n, 3)),
            "opacities": mx.full((n,), _logit(initial_opacity)),
            "sh_dc": rgb_to_sh(colors)[:, None, :],
            "sh_rest": mx.zeros((n, k - 1, 3)),
        }
        return cls(params, sh_degree)

    @classmethod
    def from_gaussians(cls, gaussians, length_frac: float = 1e-3) -> SpaghettiModel:
        """Lift a :class:`~mlx3d.splatting.GaussianModel` to (near-)zero-length
        segments along each Gaussian's major axis — the representation is a
        strict superset of 3DGS, so this is a valid starting point."""
        from ...transforms import quaternion_to_matrix

        p = gaussians.params
        R = quaternion_to_matrix(p["quats"] / mx.linalg.norm(p["quats"], axis=-1, keepdims=True))
        major = mx.argmax(p["scales"], axis=-1)
        axis = mx.take_along_axis(R, major[:, None, None].astype(mx.int32), axis=2)[..., 0]
        halves = axis * (length_frac * mx.exp(mx.max(p["scales"], axis=-1)))[:, None]
        params = {
            "means": p["means"],
            "halves": halves,
            "quats": p["quats"],
            "scales": p["scales"],
            "opacities": p["opacities"],
            "sh_dc": p["sh_dc"],
            "sh_rest": p["sh_rest"],
        }
        model = cls({k: mx.array(v) for k, v in params.items()}, gaussians.sh_degree)
        model.active_sh_degree = gaussians.active_sh_degree
        return model

    # --------------------------------------------------------------- accessors
    @property
    def num_segments(self) -> int:
        return int(self.params["means"].shape[0])

    def __len__(self) -> int:
        return self.num_segments

    @staticmethod
    def density(params: dict[str, mx.array]) -> mx.array:
        """Density ``rho`` (per unit length) from the opacity / scale parameters."""
        o = mx.sigmoid(params["opacities"])
        sigma = mx.exp(mx.mean(params["scales"], axis=-1))
        return -mx.log(mx.maximum(1.0 - o, 1e-6)) / (_SQRT_2PI * sigma)

    @property
    def opacities_act(self) -> mx.array:
        return mx.sigmoid(self.params["opacities"])

    @property
    def scales_act(self) -> mx.array:
        return mx.exp(self.params["scales"])

    @property
    def lengths(self) -> mx.array:
        return 2.0 * mx.linalg.norm(self.params["halves"], axis=-1)

    @property
    def endpoints(self) -> tuple[mx.array, mx.array]:
        m, h = self.params["means"], self.params["halves"]
        return m - h, m + h

    def one_up_sh_degree(self) -> None:
        self.active_sh_degree = min(self.active_sh_degree + 1, self.sh_degree)

    # --------------------------------------------------------------- rendering
    def render_params(self, params: dict[str, mx.array], camera: Camera, background=None):
        """Differentiable render of an explicit parameter dict."""
        sh = mx.concatenate([params["sh_dc"], params["sh_rest"]], axis=1)
        colors = segment_colors(camera, params["means"], sh, self.active_sh_degree)
        return render_spaghetti(
            camera,
            params["means"],
            params["halves"],
            params["quats"],
            mx.exp(params["scales"]),
            self.density(params),
            colors,
            background=background,
        )

    def render(self, camera: Camera, background: mx.array | None = None) -> dict[str, mx.array]:
        return self.render_params(self.params, camera, background)

    # ----------------------------------------------------- density control ops
    def append(self, new: dict[str, mx.array]) -> None:
        self.params = {k: mx.concatenate([v, new[k]], axis=0) for k, v in self.params.items()}

    def select(self, keep: mx.array) -> None:
        self.params = {k: v[keep] for k, v in self.params.items()}

    def split_or_clone(self, idx: mx.array, split_min_length: mx.array) -> dict[str, mx.array]:
        """Duplicate rows ``idx`` without changing the rendered image.

        Rows longer than ``split_min_length`` are split at their midpoint (the
        row keeps one half, the returned copy is the other); shorter rows are
        cloned with half the optical depth each. Optical depth is additive
        along rays, so both are exact. Updates rows ``idx`` in place and
        returns the new rows to place elsewhere.
        """
        p = self.params
        rows = {k: v[idx] for k, v in p.items()}
        split = (2.0 * mx.linalg.norm(rows["halves"], axis=-1) > split_min_length)[:, None]
        h = rows["halves"]
        first = mx.where(split, rows["means"] - 0.5 * h, rows["means"])
        second = mx.where(split, rows["means"] + 0.5 * h, rows["means"])
        halves = mx.where(split, 0.5 * h, h)
        # Clone: tau -> tau / 2, i.e. 1 - o -> sqrt(1 - o).
        o = mx.sigmoid(rows["opacities"])
        cloned = mx.log(mx.maximum(1.0 - mx.sqrt(mx.maximum(1.0 - o, 1e-12)), 1e-12))
        cloned = cloned - mx.log(mx.maximum(mx.sqrt(mx.maximum(1.0 - o, 1e-12)), 1e-12))
        opac = mx.where(split[:, 0], rows["opacities"], cloned)

        updated = {**rows, "means": first, "halves": halves, "opacities": opac}
        new = {**rows, "means": second, "halves": halves, "opacities": opac}
        for k in p:
            p[k] = p[k].at[idx].add(updated[k] - p[k][idx])
        return new

    def copy(self) -> SpaghettiModel:
        model = SpaghettiModel({k: mx.array(v) for k, v in self.params.items()}, self.sh_degree)
        model.active_sh_degree = self.active_sh_degree
        return model

    # ---------------------------------------------------------------- polylines
    def snap_endpoints(self, radius: float | mx.array) -> int:
        """Merge nearby endpoints of different segments into shared joints.

        Every endpoint whose nearest other-segment endpoint lies within
        ``radius`` (scalar, or per-segment) is moved, together with that
        neighbor, to their midpoint. Returns the number of joints created.
        """
        a, b = self.endpoints
        pts = np.concatenate([np.array(a), np.array(b)])
        n = self.num_segments
        owner = np.concatenate([np.arange(n), np.arange(n)])
        r = np.broadcast_to(np.asarray(radius, dtype=np.float64), (n,))
        r = np.concatenate([r, r])
        nbr = nearest_other_within(pts, owner, r)
        k = np.arange(len(pts))
        safe = np.maximum(nbr, 0)
        mutual = (nbr >= 0) & (nbr[safe] == k) & (k < nbr)
        i, j = np.nonzero(mutual)[0], nbr[mutual]
        mid = 0.5 * (pts[i] + pts[j])
        pts[i], pts[j] = mid, mid
        a, b = pts[:n], pts[n:]
        self.params["means"] = mx.array(0.5 * (a + b), dtype=mx.float32)
        self.params["halves"] = mx.array(0.5 * (b - a), dtype=mx.float32)
        return int(len(i))

    def polylines(self, tolerance: float) -> list[np.ndarray]:
        """Chain segments that share endpoints (within ``tolerance``) into
        polylines; returns a list of ``(K, 3)`` vertex arrays."""
        a, b = (np.array(x) for x in self.endpoints)
        n = self.num_segments
        pts = np.concatenate([a, b])
        q = np.round(pts / max(tolerance, 1e-12)).astype(np.int64)
        keys = {}
        vertex = np.empty(2 * n, dtype=np.int64)
        for k, key in enumerate(map(tuple, q)):
            vertex[k] = keys.setdefault(key, len(keys))
        va, vb = vertex[:n], vertex[n:]
        positions = np.zeros((len(keys), 3))
        np.add.at(positions, vertex, pts)
        positions /= np.bincount(vertex, minlength=len(keys))[:, None]
        adj: dict[int, list[int]] = {}
        for s, (u, w) in enumerate(zip(va, vb)):
            adj.setdefault(u, []).append(s)
            adj.setdefault(w, []).append(s)
        used = np.zeros(n, dtype=bool)
        lines = []
        # Start walks at chain ends (degree != 2) first, then close remaining loops.
        starts = [v for v, segs in adj.items() if len(segs) != 2] + list(adj)
        for v0 in starts:
            for s0 in adj[v0]:
                if used[s0]:
                    continue
                chain, v, s = [v0], v0, s0
                while s is not None and not used[s]:
                    used[s] = True
                    v = vb[s] if va[s] == v else va[s]
                    chain.append(v)
                    nxt = [t for t in adj[v] if not used[t]]
                    s = nxt[0] if len(adj[v]) == 2 and nxt else None
                lines.append(positions[chain])
        return lines

    # ----------------------------------------------------------------- PLY IO
    def save_ply(self, path: str) -> None:
        """Save as a PLY vertex list with spaghetti-specific properties."""
        from ...io import save_ply

        p = self.params
        n = self.num_segments
        extra = {}
        for name, key, width in (
            ("h", "halves", 3),
            ("scale", "scales", 3),
            ("rot", "quats", 4),
        ):
            for c in range(width):
                extra[f"{name}_{c}"] = p[key][:, c]
        extra["opacity"] = p["opacities"]
        for c in range(3):
            extra[f"f_dc_{c}"] = p["sh_dc"][:, 0, c]
        rest = p["sh_rest"].transpose(0, 2, 1).reshape(n, -1)
        for c in range(rest.shape[1]):
            extra[f"f_rest_{c}"] = rest[:, c]
        save_ply(path, p["means"], extra=extra)

    @classmethod
    def load_ply(cls, path: str) -> SpaghettiModel:
        from ...io import load_ply

        ply = load_ply(path)
        props = ply.extra
        n = ply.verts.shape[0]

        def stack(prefix, width):
            return mx.stack([props[f"{prefix}_{c}"] for c in range(width)], axis=-1)

        n_rest = sum(1 for k in props if k.startswith("f_rest_"))
        k = n_rest // 3 + 1
        sh_degree = int(round(math.sqrt(k))) - 1
        if n_rest:
            rest = stack("f_rest", n_rest).reshape(n, 3, k - 1).transpose(0, 2, 1)
        else:
            rest = mx.zeros((n, 0, 3))
        params = {
            "means": ply.verts,
            "halves": stack("h", 3),
            "quats": stack("rot", 4),
            "scales": stack("scale", 3),
            "opacities": props["opacity"],
            "sh_dc": stack("f_dc", 3)[:, None, :],
            "sh_rest": rest,
        }
        model = cls({k: mx.array(v, dtype=mx.float32) for k, v in params.items()}, sh_degree)
        model.active_sh_degree = sh_degree
        return model


def nearest_other_within(
    points: np.ndarray, owner: np.ndarray, radius: np.ndarray, max_per_cell: int = 32
) -> np.ndarray:
    """Nearest point of a *different* owner within each point's ``radius``.

    Returns ``(P,)`` indices, ``-1`` where none is close enough. A hash grid
    with cells of (almost) the largest radius makes this linear-time on the
    CPU — the search radii are a few kernel widths, so a brute-force k-NN over
    hundreds of thousands of endpoints would be both wasteful and long enough
    to trip macOS's GPU watchdog. Cells holding more than ``max_per_cell``
    points only search their first ``max_per_cell`` (approximate in extreme
    density, which is harmless for endpoint merging).
    """
    n = len(points)
    radius = np.broadcast_to(np.asarray(radius, dtype=np.float64), (n,))
    cell = max(float(np.percentile(radius, 99)), 1e-12)
    grid = np.floor(points / cell).astype(np.int64)

    def hash_cells(c: np.ndarray) -> np.ndarray:
        return (c[:, 0] * 73856093) ^ (c[:, 1] * 19349663) ^ (c[:, 2] * 83492791)

    order = np.argsort(hash_cells(grid), kind="stable")
    keys, starts, counts = np.unique(hash_cells(grid)[order], return_index=True, return_counts=True)
    best_d = np.full(n, np.inf)
    best = np.full(n, -1, dtype=np.int64)
    offsets = np.stack(np.meshgrid(*[np.arange(-1, 2)] * 3, indexing="ij"), -1).reshape(-1, 3)
    for off in offsets:
        nk = hash_cells(grid + off)
        pos = np.minimum(np.searchsorted(keys, nk), len(keys) - 1)
        count = np.where(keys[pos] == nk, counts[pos], 0)
        for slot in range(min(int(count.max(initial=0)), max_per_cell)):
            q = np.nonzero(count > slot)[0]
            cand = order[starts[pos[q]] + slot]
            d = np.linalg.norm(points[q] - points[cand], axis=1)
            better = (owner[cand] != owner[q]) & (d < best_d[q])
            best_d[q[better]], best[q[better]] = d[better], cand[better]
    return np.where(best_d < radius, best, -1)
