"""NeRF building blocks: positional encoding, the MLP, and a renderer that
ties them to the ray utilities."""

import mlx.core as mx
import mlx.nn as nn

from ..renderer.rays import sample_along_rays, sample_pdf, volume_render

__all__ = ["PositionalEncoding", "NeRF", "render_rays"]


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding from the NeRF paper.

    Maps x to ``[x, sin(2^0 x), cos(2^0 x), ..., sin(2^{L-1} x), cos(2^{L-1} x)]``.
    """

    def __init__(self, num_freqs: int, include_input: bool = True):
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        self._freqs = 2.0 ** mx.arange(num_freqs)

    @property
    def output_dim_multiplier(self) -> int:
        return 2 * self.num_freqs + (1 if self.include_input else 0)

    def __call__(self, x: mx.array) -> mx.array:
        xb = x[..., None] * self._freqs  # (..., D, L)
        enc = mx.concatenate([mx.sin(xb), mx.cos(xb)], axis=-1)
        enc = enc.reshape(*x.shape[:-1], -1)
        if self.include_input:
            enc = mx.concatenate([x, enc], axis=-1)
        return enc


class NeRF(nn.Module):
    """The original NeRF MLP: density from position, color from position + view."""

    def __init__(
        self,
        pos_freqs: int = 10,
        dir_freqs: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 8,
        skip_layer: int = 4,
    ):
        super().__init__()
        self.pos_enc = PositionalEncoding(pos_freqs)
        self.dir_enc = PositionalEncoding(dir_freqs)
        pos_dim = 3 * self.pos_enc.output_dim_multiplier
        dir_dim = 3 * self.dir_enc.output_dim_multiplier
        self.skip_layer = skip_layer

        layers = []
        in_dim = pos_dim
        for i in range(num_layers):
            if i == skip_layer:
                in_dim += pos_dim
            layers.append(nn.Linear(in_dim, hidden_dim))
            in_dim = hidden_dim
        self.layers = layers

        self.density_head = nn.Linear(hidden_dim, 1)
        self.feature = nn.Linear(hidden_dim, hidden_dim)
        self.color_hidden = nn.Linear(hidden_dim + dir_dim, hidden_dim // 2)
        self.color_head = nn.Linear(hidden_dim // 2, 3)

    def __call__(self, points: mx.array, directions: mx.array) -> tuple[mx.array, mx.array]:
        """Evaluate density and color.

        Args:
            points: (..., 3) sample positions.
            directions: (..., 3) normalized view directions (broadcastable).

        Returns:
            ``(density, rgb)`` with shapes (...,) and (..., 3).
        """
        x = self.pos_enc(points)
        h = x
        for i, layer in enumerate(self.layers):
            if i == self.skip_layer:
                h = mx.concatenate([h, x], axis=-1)
            h = nn.relu(layer(h))

        # Softplus (not ReLU) for the density activation: ReLU's zero region
        # has zero gradient, so with the usual Linear init the density head
        # starts negative everywhere, outputs 0, and the network never receives
        # a gradient (dead NeRF). Softplus is positive and differentiable
        # everywhere, so training always has signal.
        density = nn.softplus(self.density_head(h)[..., 0])
        feat = self.feature(h)
        d = self.dir_enc(directions)
        ch = nn.relu(self.color_hidden(mx.concatenate([feat, d], axis=-1)))
        rgb = mx.sigmoid(self.color_head(ch))
        return density, rgb


def render_rays(
    model: NeRF,
    origins: mx.array,
    directions: mx.array,
    near: float,
    far: float,
    num_coarse: int = 64,
    num_fine: int = 0,
    fine_model: NeRF | None = None,
    stratified: bool = True,
    white_background: bool = False,
) -> dict[str, mx.array]:
    """Render a batch of rays with optional hierarchical sampling.

    Returns a dict with ``rgb``, ``depth``, ``acc`` (and ``rgb_coarse`` when
    fine sampling is enabled).
    """
    dirs_n = directions / mx.linalg.norm(directions, axis=-1, keepdims=True)

    points, t_vals = sample_along_rays(
        origins, directions, near, far, num_coarse, stratified=stratified
    )
    view = mx.broadcast_to(dirs_n[:, None, :], points.shape)
    density, rgb = model(points, view)
    out = volume_render(density, rgb, t_vals, directions, white_background)

    if num_fine <= 0:
        return out

    mids = 0.5 * (t_vals[:, 1:] + t_vals[:, :-1])
    t_fine = sample_pdf(mids, out["weights"][:, 1:-1], num_fine, deterministic=not stratified)
    t_all = mx.sort(mx.concatenate([t_vals, t_fine], axis=-1), axis=-1)
    points = origins[:, None, :] + t_all[..., None] * directions[:, None, :]
    view = mx.broadcast_to(dirs_n[:, None, :], points.shape)
    fmodel = fine_model if fine_model is not None else model
    density, rgb = fmodel(points, view)
    fine_out = volume_render(density, rgb, t_all, directions, white_background)
    fine_out["rgb_coarse"] = out["rgb"]
    return fine_out
