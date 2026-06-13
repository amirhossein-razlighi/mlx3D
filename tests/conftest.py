"""Shared pytest configuration.

Tests are sliced into the marker groups declared in ``pyproject.toml`` so CI can
run them selectively (``pytest -m smoke``, ``-m "unit or data"``, ...). Rather
than sprinkle ``pytestmark`` across every file, categories are assigned here by
module name; anything not listed is a focused unit test.
"""

import pytest

_CATEGORY_BY_MODULE = {
    "test_import": "smoke",
    "test_smoke": "smoke",
    "test_behavior": "behavior",
    "test_datasets": "data",
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        category = _CATEGORY_BY_MODULE.get(item.path.stem, "unit")
        item.add_marker(getattr(pytest.mark, category))
