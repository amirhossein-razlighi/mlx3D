"""Smoke tests: the package imports and its public API is wired together."""

import importlib
import importlib.util
from pathlib import Path

import pytest

import mlx3d

SUBPACKAGES = [
    "cameras",
    "datasets",
    "io",
    "losses",
    "nn",
    "ops",
    "renderer",
    "splatting",
    "structures",
    "transforms",
    "utils",
]


def test_version_present():
    assert isinstance(mlx3d.__version__, str)
    assert mlx3d.__version__.count(".") >= 1


def test_top_level_exports_importable():
    for name in mlx3d.__all__:
        assert hasattr(mlx3d, name), f"mlx3d.{name} missing"


@pytest.mark.parametrize("pkg_name", SUBPACKAGES)
def test_subpackage_all_symbols_resolve(pkg_name):
    pkg = importlib.import_module(f"mlx3d.{pkg_name}")
    assert hasattr(pkg, "__all__"), f"mlx3d.{pkg_name} has no __all__"
    for name in pkg.__all__:
        assert hasattr(pkg, name), f"mlx3d.{pkg_name}.{name} declared but missing"


_EXAMPLES = sorted(p for p in (Path(__file__).resolve().parents[1] / "examples").glob("*.py"))


@pytest.mark.parametrize("example_path", _EXAMPLES, ids=lambda p: p.name)
def test_examples_import_clean(example_path):
    """Each example module imports without error (``main()`` is not invoked)."""
    spec = importlib.util.spec_from_file_location(example_path.stem, example_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "main"), f"{example_path.name} should expose main()"
