"""The trainable Gaussian Splatting scene representation."""

import mlx.core as mx
import numpy as np

from ..cameras import Camera
from ..io import load_ply, save_ply
from .render import render_gaussians
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
    ) -> "GaussianModel":
        """Initialize from a point cloud (e.g. SfM points).

        Scales are set from the mean distance to the 3 nearest neighbors, as
        in the reference implementation.
        """
        from ..ops import knn_points

        N = points.shape[0]
        if colors is None:
            colors = mx.full((N, 3), 0.5)

        # Mean distance to the 3 nearest neighbors. For huge clouds, a random
        # reference subset keeps the O(N * R) search memory bounded without
        # meaningfully changing the scale estimate.
        max_ref = 50_000
        if N > max_ref:
            ref = points[mx.random.permutation(N)[:max_ref]]
        else:
            ref = points
        d, _ = knn_points(points, ref, K=4)  # K=4: nearest may be self (dist 0)
        mean_sq = mx.maximum(d[:, 1:].mean(axis=-1), 1e-8)
        scales = mx.log(mx.sqrt(mean_sq))[:, None] * mx.ones((1, 3))

        quats = mx.zeros((N, 4))
        quats = quats.at[:, 0].add(mx.ones((N,)))

        K = num_sh_bases(sh_degree)
        params = {
            "means": mx.array(points),
            "scales": scales,
            "quats": quats,
            "opacities": mx.full(
                (N,), float(np.log(initial_opacity / (1 - initial_opacity)))
            ),
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

    # ------------------------------------------------------------------ render
    def render(self, camera: Camera, background: mx.array | None = None) -> dict:
        return render_gaussians(
            camera,
            self.params["means"],
            self.params["quats"],
            self.scales_act,
            self.opacities_act,
            sh=self.sh,
            sh_degree=self.active_sh_degree,
            background=background,
        )

    def one_up_sh_degree(self) -> None:
        if self.active_sh_degree < self.sh_degree:
            self.active_sh_degree += 1

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
            k: mx.concatenate([v, new_params[k]], axis=0)
            for k, v in self.params.items()
        }

    def densify_and_prune(
        self,
        grad_accum: mx.array,
        grad_count: mx.array,
        grad_threshold: float = 0.0002,
        scene_extent: float = 1.0,
        percent_dense: float = 0.01,
        min_opacity: float = 0.005,
    ) -> dict[str, int]:
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
            R = np.array(quaternion_to_matrix(q / np.linalg.norm(p_np["quats"][split_idx], axis=1, keepdims=True)))
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

        return {
            "cloned": int(clone_idx.size),
            "split": int(split_idx.size),
            "pruned": int(prune_mask.sum()),
        }

    def reset_opacities(self, max_opacity: float = 0.01) -> None:
        """Clamp opacities down (periodic reset from the 3DGS paper)."""
        logit = float(np.log(max_opacity / (1 - max_opacity)))
        self.params["opacities"] = mx.minimum(
            self.params["opacities"], mx.full(self.params["opacities"].shape, logit)
        )
