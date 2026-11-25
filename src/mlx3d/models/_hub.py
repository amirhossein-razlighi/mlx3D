"""Locate pretrained weights on disk or fetch them from the Hugging Face Hub."""

from __future__ import annotations

from pathlib import Path

__all__ = ["resolve_weights"]

_WEIGHTS_FILE = "model.safetensors"


def resolve_weights(source: str, repo_id: str | None = None) -> Path:
    """Return the path of a ``model.safetensors`` file.

    Args:
        source: a ``.safetensors`` file, a directory containing one, or a Hub
            repository id.
        repo_id: Hub repository to download when ``source`` is not a local path
            (defaults to ``source``).

    Downloads go to the standard Hugging Face cache, so they happen once.
    """
    path = Path(source).expanduser()
    if path.is_file():
        return path
    if path.is_dir():
        candidate = path / _WEIGHTS_FILE
        if not candidate.is_file():
            raise FileNotFoundError(f"No {_WEIGHTS_FILE} in {path}.")
        return candidate

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "Downloading pretrained weights needs huggingface_hub. Install the extra:\n"
            '    pip install "mlx3d[models]"\n'
            "or pass a local path to the weights."
        ) from e
    return Path(hf_hub_download(repo_id or source, _WEIGHTS_FILE))
