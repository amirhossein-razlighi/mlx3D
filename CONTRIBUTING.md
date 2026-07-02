# Contributing to MLX3D

Thanks for your interest in improving MLX3D! Bug reports, feature requests,
docs fixes and pull requests are all welcome.

## Getting set up

You need an Apple Silicon Mac (MLX runs on the Metal GPU) and Python ≥ 3.10.
Development uses [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/amirhossein-razlighi/mlx3D
cd mlx3D
uv sync --extra capture   # .venv with dev tools + the optional SfM deps
uv run pytest tests/
uv run mkdocs serve       # docs at http://127.0.0.1:8000
```

Plain pip works too: `pip install -e ".[dev,capture]"`.

## Before you open a PR

1. **Run the tests.** `uv run pytest tests/` must pass. Tests are grouped by
   marker (`smoke`, `unit`, `data`, `behavior`) — CI runs them all on Apple
   Silicon runners.
2. **Lint and format.** CI enforces both:
   ```bash
   uvx ruff check .
   uvx ruff format .
   ```
3. **Add tests** for new behavior. Dataset/IO code is tested against small
   synthetic fixtures generated in-process (see `tests/test_capture.py` for
   examples) — no downloads in tests.
4. **Update the docs** when you change user-facing behavior: the relevant
   tutorial under `docs/`, the API reference stub if you add a module, and a
   `CHANGELOG.md` entry under `Unreleased`.

## Guidelines

- **Conventions matter.** MLX3D uses the OpenCV/COLMAP camera convention
  everywhere (`X_cam = R @ X_world + t`, `+z` forward); see
  [docs/conventions.md](docs/conventions.md) before touching cameras,
  projection, or dataset loaders.
- **Keep the public API small and composable.** Renderers are plain callables
  `(camera, scene) -> {"image", ...}`; prefer functions and dataclass configs
  over deep class hierarchies.
- **Heavy dependencies stay optional.** Core `mlx3d` depends only on MLX,
  NumPy, Pillow and tqdm; anything heavier (OpenCV, SciPy, ...) belongs in an
  optional extra with a helpful `ImportError` message.
- **Performance-sensitive code** (Metal kernels, training loops) should come
  with a before/after measurement in the PR description.

## Reporting bugs

Use the issue templates. The most useful reports include: macOS + chip model,
Python and `mlx`/`mlx3d` versions, and a minimal script that reproduces the
problem.

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). Be kind.
