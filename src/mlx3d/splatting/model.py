"""The trainable Gaussian Splatting scene representation."""

import mlx.core as mx
import numpy as np

from ..cameras import Camera
from ..io import load_ply, save_ply
from ..structures import Meshes
from ..transforms import quaternion_to_matrix
from .render import render_gaussian_depth, render_gaussian_features, render_gaussians
from .sh import num_sh_bases, rgb_to_sh

__all__ = ["GaussianModel"]


class GaussianModel:
    """A scene of 3D Gaussians with the standard 3DGS parameterization.

    Raw (unconstrained) parameters live in ``self.params`` so they can be fed
    to MLX optimizers directly:

    - ``means`` (N, 3)
    - ``scales`` (N, 3): log of the per-axis standard deviations.
    - ``quats`` (N, 4): unnormalized rotation quaternions (w, x, y, z).
    - ``opacities`` (N,): logits; sigmoid gives opacity.
    - ``sh_dc`` (N, 1, 3) and ``sh_rest`` (N, K-1, 3): SH color coefficients.
    """

    def __init__(self, params: dict[str, mx.array], sh_degree: int = 3):
        self.params = params
        self.sh_degree = sh_degree
        # Active degree grows during training (as in the reference implementation).
        self.active_sh_degree = 0

    # ----------------------------------------------------------- construction
    @classmethod
    def from_points(
        cls,
        points: mx.array,
        colors: mx.array | None = None,
        sh_degree: int = 3,
        initial_opacity: float = 0.1,
        scale_init_max_ref: int = 10_000,
        scale_init_chunk_size: int = 1024,
        scale_init_max_scale: float | None = None,
    ) -> "GaussianModel":
        """Initialize from a point cloud (e.g. SfM points).

        Scales are set from the mean distance to the 3 nearest neighbors, as
        in the reference implementation.

        Args:
            scale_init_max_ref: Maximum reference points used for the initial
                nearest-neighbor scale estimate. Large COLMAP clouds otherwise
                spend seconds to minutes materializing huge distance tiles.
            scale_init_chunk_size: Query chunk size for the scale-estimation
                KNN. Lower values reduce peak memory during initialization.
            scale_init_max_scale: Optional cap for initial per-axis Gaussian
                scale. This is useful for COLMAP point clouds with sparse
                outliers whose nearest-neighbor distances would otherwise
                create full-screen splats.
        """
        from ..ops import knn_points

        N = points.shape[0]
        if colors is None:
            colors = mx.full((N, 3), 0.5)

        # Mean distance to the 3 nearest neighbors. For huge clouds, a random
        # reference subset keeps the O(N * R) search memory bounded without
        # meaningfully changing the scale estimate.
        ref_count = min(N, max(1, int(scale_init_max_ref)))
        if N > ref_count:
            ref = points[mx.random.permutation(N)[:ref_count]]
        else:
            ref = points

        if ref_count <= 1:
            mean_sq = mx.full((N,), 1e-8)
        else:
            k = min(4, ref_count)
            d, _ = knn_points(points, ref, K=k, chunk_size=scale_init_chunk_size)
            # If the reference set contains the query point, the first
            # neighbor is the point itself. Skip that zero-distance hit;
            # otherwise use the nearest available neighbors directly.
            self_hit = d[:, 0] <= 1e-12
            nearest = mx.where(self_hit[:, None], d[:, 1:k], d[:, : k - 1])
            mean_sq = mx.maximum(nearest.mean(axis=-1), 1e-8)
        scales = mx.log(mx.sqrt(mean_sq))[:, None] * mx.ones((1, 3))
        if scale_init_max_scale is not None:
            max_log_scale = float(np.log(max(scale_init_max_scale, 1e-8)))
            scales = mx.minimum(scales, mx.full(scales.shape, max_log_scale))

        quats = mx.zeros((N, 4))
        quats = quats.at[:, 0].add(mx.ones((N,)))

        K = num_sh_bases(sh_degree)
        params = {
            "means": mx.array(points),
            "scales": scales,
            "quats": quats,
            "opacities": mx.full((N,), float(np.log(initial_opacity / (1 - initial_opacity)))),
            "sh_dc": rgb_to_sh(mx.array(colors))[:, None, :],
            "sh_rest": mx.zeros((N, K - 1, 3)),
        }
        return cls(params, sh_degree=sh_degree)

    # ------------------------------------------------------------- activations
    @property
    def num_gaussians(self) -> int:
        return self.params["means"].shape[0]

    def __len__(self) -> int:
        return self.num_gaussians

    @property
    def scales_act(self) -> mx.array:
        return mx.exp(self.params["scales"])

    @property
    def opacities_act(self) -> mx.array:
        return mx.sigmoid(self.params["opacities"])

    @property
    def sh(self) -> mx.array:
        return mx.concatenate([self.params["sh_dc"], self.params["sh_rest"]], axis=1)

    def apply_2dgs_constraints(self, max_thickness: float) -> None:
        """Constrain Gaussians to thin surfels for 2DGS-style training.

        The covariance projection and rasterizer already support anisotropic
        oriented Gaussians. Keeping the third local scale small turns each
        Gaussian into an oriented disk while preserving standard 3DGS PLY
        compatibility.
        """
        thickness = max(float(max_thickness), 1e-8)
        max_log_thickness = float(np.log(thickness))
        constrained = self.params["scales"]
        z = mx.minimum(
            constrained[:, 2:3],
            mx.full((self.num_gaussians, 1), max_log_thickness, dtype=constrained.dtype),
        )
        self.params["scales"] = mx.concatenate([constrained[:, :2], z], axis=1)

    # ------------------------------------------------------------------ render
    def render(
        self,
        camera: Camera,
        background: mx.array | None = None,
        antialias: bool = False,
        projection: str = "ewa",
    ) -> dict:
        return render_gaussians(
            camera,
            self.params["means"],
            self.params["quats"],
            self.scales_act,
            self.opacities_act,
            sh=self.sh,
            sh_degree=self.active_sh_degree,
            background=background,
            antialias=antialias,
            projection=projection,
        )

    def render_depth(
        self,
        camera: Camera,
        antialias: bool = False,
        projection: str = "ewa",
    ) -> dict:
        return render_gaussian_depth(
            camera,
            self.params["means"],
            self.params["quats"],
            self.scales_act,
            self.opacities_act,
            antialias=antialias,
            projection=projection,
        )

    def render_features(
        self,
        camera: Camera,
        features: mx.array,
        background: mx.array | None = None,
        normalize: bool = False,
        antialias: bool = False,
        projection: str = "ewa",
    ) -> dict:
        """Render arbitrary per-Gaussian feature channels from this model."""
        return render_gaussian_features(
            camera,
            self.params["means"],
            self.params["quats"],
            self.scales_act,
            self.opacities_act,
            features,
            background=background,
            normalize=normalize,
            antialias=antialias,
            projection=projection,
        )

    def one_up_sh_degree(self) -> None:
        if self.active_sh_degree < self.sh_degree:
            self.active_sh_degree += 1

    # --------------------------------------------------------- surface export
    def surfel_points(
        self,
        min_opacity: float = 0.01,
        max_points: int | None = None,
        orient_towards: tuple[float, float, float] | mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Return oriented surfel samples from the Gaussian centers.

        This is intended for 2DGS-style checkpoints, where the local ``z`` axis
        is the surfel normal and the first two local scales span the disk. Rows
        are filtered by activated opacity and, when capped, ranked by
        ``opacity * scale_x * scale_y`` so large opaque surfels survive first.
        """
        if max_points is not None and max_points <= 0:
            raise ValueError("max_points must be positive when provided.")
        n = self.num_gaussians
        if n == 0:
            empty = mx.zeros((0, 3), dtype=mx.float32)
            return empty, empty

        opacity = np.array(self.opacities_act)
        scales = np.array(self.scales_act)
        importance = opacity * scales[:, 0] * scales[:, 1]
        keep = opacity >= float(min_opacity)
        if not keep.any():
            keep[int(np.argmax(importance))] = True
        keep_idx = np.where(keep)[0]
        if max_points is not None and keep_idx.size > max_points:
            local = np.argpartition(-importance[keep_idx], max_points - 1)[:max_points]
            keep_idx = keep_idx[local]
        keep_idx = np.sort(keep_idx.astype(np.int32))
        idx = mx.array(keep_idx)

        points = self.params["means"][idx]
        normals = quaternion_to_matrix(self.params["quats"][idx])[:, :, 2]
        normals = normals / mx.maximum(mx.linalg.norm(normals, axis=-1, keepdims=True), 1e-8)
        if orient_towards is not None:
            target = mx.array(orient_towards, dtype=points.dtype)
            flip = mx.sum((target - points) * normals, axis=-1, keepdims=True) < 0
            normals = mx.where(flip, -normals, normals)
        return points, normals

    def extract_surface_mesh(
        self,
        resolution: int = 64,
        padding: float = 0.1,
        min_opacity: float = 0.01,
        max_points: int | None = 200_000,
        orient_towards: tuple[float, float, float] | mx.array | None = None,
    ) -> Meshes:
        """Reconstruct a mesh from oriented Gaussian surfels via Poisson reconstruction."""
        if resolution <= 1:
            raise ValueError("resolution must be greater than 1.")
        points, normals = self.surfel_points(
            min_opacity=min_opacity,
            max_points=max_points,
            orient_towards=orient_towards,
        )
        from ..ops import poisson_reconstruction

        return poisson_reconstruction(points, normals, resolution=resolution, padding=padding)

    # -------------------------------------------------------------- compaction
    def copy(self) -> "GaussianModel":
        """Return a detached copy of the Gaussian table."""
        out = GaussianModel({k: mx.array(v) for k, v in self.params.items()}, self.sh_degree)
        out.active_sh_degree = self.active_sh_degree
        return out

    def compact(
        self,
        min_opacity: float = 0.0,
        max_gaussians: int | None = None,
        target_sh_degree: int | None = None,
    ) -> "GaussianModel":
        """Return a smaller checkpoint by pruning low-importance Gaussians.

        Importance is a conservative, view-independent proxy:
        ``sigmoid(opacity) * max(scale)^2``. This keeps opaque large-footprint
        splats before transparent/subpixel ones, preserves the original order
        of retained rows for checkpoint diffability, and does not mutate this
        model.

        Args:
            min_opacity: prune Gaussians with activated opacity below this
                threshold. If the threshold removes every row, the most
                important Gaussian is kept so the checkpoint remains renderable.
            max_gaussians: optional hard cap; the highest-importance rows are
                retained.
            target_sh_degree: optional lower SH degree for color coefficient
                truncation. This reduces checkpoint size and render cost for
                view-dependent color at the expense of angular detail.
        """
        if max_gaussians is not None and max_gaussians <= 0:
            raise ValueError("max_gaussians must be positive when provided.")
        if target_sh_degree is not None and not (0 <= target_sh_degree <= self.sh_degree):
            raise ValueError("target_sh_degree must be between 0 and the model sh_degree.")

        n = self.num_gaussians
        if n == 0:
            return self.copy()

        opacity = np.array(self.opacities_act)
        max_scale = np.array(self.scales_act).max(axis=1)
        importance = opacity * max_scale * max_scale
        keep = opacity >= float(min_opacity)
        if not keep.any():
            keep[int(np.argmax(importance))] = True

        keep_idx = np.where(keep)[0]
        if max_gaussians is not None and keep_idx.size > max_gaussians:
            local = np.argpartition(-importance[keep_idx], max_gaussians - 1)[:max_gaussians]
            keep_idx = keep_idx[local]
        keep_idx = np.sort(keep_idx.astype(np.int32))
        idx = mx.array(keep_idx)
        params = {k: v[idx] for k, v in self.params.items()}

        sh_degree = self.sh_degree
        if target_sh_degree is not None:
            sh_degree = int(target_sh_degree)
            k = num_sh_bases(sh_degree)
            params["sh_rest"] = params["sh_rest"][:, : max(0, k - 1), :]

        out = GaussianModel(params, sh_degree=sh_degree)
        out.active_sh_degree = min(self.active_sh_degree, sh_degree)
        return out

    # ------------------------------------------------------------- checkpoints
    def save_ply(self, path: str) -> None:
        """Save in the standard 3DGS PLY layout (compatible with most viewers)."""
        p = self.params
        N = self.num_gaussians
        extra: dict[str, mx.array] = {}
        sh_dc = p["sh_dc"]
        for c in range(3):
            extra[f"f_dc_{c}"] = sh_dc[:, 0, c]
        rest = np.array(p["sh_rest"])  # (N, K-1, 3)
        rest_t = rest.transpose(0, 2, 1).reshape(N, -1)  # channel-major as in 3DGS
        for i in range(rest_t.shape[1]):
            extra[f"f_rest_{i}"] = mx.array(rest_t[:, i])
        extra["opacity"] = p["opacities"]
        for i in range(3):
            extra[f"scale_{i}"] = p["scales"][:, i]
        for i in range(4):
            extra[f"rot_{i}"] = p["quats"][:, i]
        save_ply(
            path,
            p["means"],
            normals=mx.zeros((N, 3)),
            extra=extra,
            binary=True,
        )

    @classmethod
    def load_ply(cls, path: str, sh_degree: int = 3) -> "GaussianModel":
        """Load a 3DGS-format PLY checkpoint."""
        data = load_ply(path)
        e = data.extra
        N = data.verts.shape[0]
        n_rest = len([k for k in e if k.startswith("f_rest_")])
        K = num_sh_bases(sh_degree)
        if n_rest != 3 * (K - 1):
            # Infer degree from the file.
            K = n_rest // 3 + 1
            sh_degree = int(np.sqrt(K)) - 1
        sh_dc = mx.stack([e["f_dc_0"], e["f_dc_1"], e["f_dc_2"]], axis=-1)[:, None, :]
        if n_rest > 0:
            rest = np.stack([np.array(e[f"f_rest_{i}"]) for i in range(n_rest)], axis=1)
            rest = rest.reshape(N, 3, K - 1).transpose(0, 2, 1)
            sh_rest = mx.array(rest.astype(np.float32))
        else:
            sh_rest = mx.zeros((N, K - 1, 3))
        params = {
            "means": data.verts,
            "scales": mx.stack([e[f"scale_{i}"] for i in range(3)], axis=-1),
            "quats": mx.stack([e[f"rot_{i}"] for i in range(4)], axis=-1),
            "opacities": e["opacity"],
            "sh_dc": sh_dc,
            "sh_rest": sh_rest,
        }
        model = cls(params, sh_degree=sh_degree)
        model.active_sh_degree = sh_degree
        return model

    # ----------------------------------------------------------- densification
    def select(self, keep_idx: np.ndarray) -> None:
        """Keep only the Gaussians at ``keep_idx`` (in-place)."""
        idx = mx.array(keep_idx.astype(np.int32))
        self.params = {k: v[idx] for k, v in self.params.items()}

    def append(self, new_params: dict[str, mx.array]) -> None:
        """Concatenate new Gaussians (in-place)."""
        self.params = {
            k: mx.concatenate([v, new_params[k]], axis=0) for k, v in self.params.items()
        }

    def densify_and_prune(
        self,
        grad_accum: mx.array,
        grad_count: mx.array,
        grad_threshold: float = 0.0002,
        scene_extent: float = 1.0,
        percent_dense: float = 0.01,
        min_opacity: float = 0.005,
        return_optimizer_state: bool = False,
    ) -> dict[str, object]:
        """Adaptive density control from the 3DGS paper.

        Args:
            grad_accum: (N,) accumulated NDC-space positional gradient norms.
            grad_count: (N,) number of accumulation steps each Gaussian was visible.

        Under-reconstructed regions (high positional gradient, small scale)
        are cloned; over-reconstructed ones (high gradient, large scale) are
        split in two. Nearly transparent or oversized Gaussians are pruned.
        Returns counts of cloned/split/pruned Gaussians.
        """
        avg_grad = np.array(grad_accum) / np.maximum(np.array(grad_count), 1.0)
        scales = np.array(self.scales_act)
        max_scale = scales.max(axis=1)
        high_grad = avg_grad > grad_threshold

        clone_mask = high_grad & (max_scale <= percent_dense * scene_extent)
        split_mask = high_grad & (max_scale > percent_dense * scene_extent)

        p_np = {k: np.array(v) for k, v in self.params.items()}

        # Clone: duplicate as-is.
        clone_idx = np.where(clone_mask)[0]
        clones = {k: v[clone_idx] for k, v in p_np.items()}

        # Split: two samples from each Gaussian, scales shrunk by 1.6.
        split_idx = np.where(split_mask)[0]
        splits: dict[str, np.ndarray] = {}
        if split_idx.size > 0:
            from ..transforms import quaternion_to_matrix

            q = mx.array(p_np["quats"][split_idx])
            R = np.array(
                quaternion_to_matrix(
                    q / np.linalg.norm(p_np["quats"][split_idx], axis=1, keepdims=True)
                )
            )
            s = scales[split_idx]
            n2 = split_idx.size * 2
            samples = np.random.normal(size=(n2, 3)) * np.repeat(s, 2, axis=0)
            offsets = np.einsum("nij,nj->ni", np.repeat(R, 2, axis=0), samples)
            splits = {k: np.repeat(v[split_idx], 2, axis=0) for k, v in p_np.items()}
            splits["means"] = splits["means"] + offsets.astype(np.float32)
            splits["scales"] = splits["scales"] - float(np.log(1.6))

        # Prune originals: split sources, low opacity, oversized.
        opac = 1.0 / (1.0 + np.exp(-np.clip(p_np["opacities"], -50.0, 50.0)))
        prune_mask = (opac < min_opacity) | split_mask | (max_scale > 0.5 * scene_extent)
        keep_idx = np.where(~prune_mask)[0]

        self.select(keep_idx)
        if clone_idx.size > 0:
            self.append({k: mx.array(v) for k, v in clones.items()})
        if split_idx.size > 0:
            self.append({k: mx.array(v) for k, v in splits.items()})

        stats: dict[str, object] = {
            "cloned": int(clone_idx.size),
            "split": int(split_idx.size),
            "pruned": int(prune_mask.sum()),
        }
        if return_optimizer_state:
            stats["_keep_idx"] = keep_idx.astype(np.int32)
            stats["_new_count"] = int(clone_idx.size + 2 * split_idx.size)
        return stats

    def relocate_mcmc(
        self,
        grad_accum: mx.array,
        grad_count: mx.array,
        relocate_frac: float = 0.02,
        min_opacity: float = 0.01,
        jitter_scale: float = 0.25,
    ) -> dict[str, object]:
        """Fixed-budget MCMC-style relocation of underused Gaussians.

        This keeps ``N`` constant: low-opacity or never-visible rows are
        replaced by jittered copies of high-gradient rows. It is inspired by
        MCMC 3DGS relocation, and is intended as an alternative to vanilla
        clone/split/prune density control.
        """
        n = self.num_gaussians
        max_relocate = int(max(0, min(relocate_frac, 1.0)) * n)
        if max_relocate <= 0 or n <= 1:
            return {"relocated": 0}

        avg_grad = grad_accum / mx.maximum(grad_count, 1.0)
        opac = mx.sigmoid(self.params["opacities"])
        counts = grad_count
        underused = (opac < min_opacity) | (counts <= 0)
        underused_count = int(mx.sum(underused.astype(mx.int32)))
        k = max_relocate if underused_count == 0 else min(max_relocate, underused_count)
        k = min(k, n - 1)
        if k == 0:
            return {"relocated": 0}

        # Prefer underused rows. If none exist, fall back to the lowest-opacity
        target_score = mx.where(underused, 2.0 - opac, -opac)
        dst = mx.argpartition(-target_score, kth=k - 1)[:k]
        source_score = avg_grad.at[dst].add(-1e30 - avg_grad[dst])
        src = mx.argpartition(-source_score, kth=k - 1)[:k]
        source_scales = mx.exp(self.params["scales"][src])
        reset_opacity = float(np.log(0.05 / 0.95))
        split_shrink = float(np.log(1.6))
        for name, arr in self.params.items():
            values = arr[src]
            if name == "means" and jitter_scale > 0:
                values = values + mx.random.normal(values.shape) * source_scales * float(
                    jitter_scale
                )
            elif name == "opacities":
                values = mx.full(values.shape, reset_opacity, dtype=values.dtype)
            elif name == "scales":
                values = values - split_shrink
            self.params[name] = arr.at[dst].add(values - arr[dst])
        return {"relocated": int(k), "_moved_idx": dst}

    def reset_opacities(self, max_opacity: float = 0.01) -> None:
        """Clamp opacities down (periodic reset from the 3DGS paper)."""
        logit = float(np.log(max_opacity / (1 - max_opacity)))
        self.params["opacities"] = mx.minimum(
            self.params["opacities"], mx.full(self.params["opacities"].shape, logit)
        )
