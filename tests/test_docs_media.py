"""Raw-HTML media paths in the docs must resolve from each page's served URL.

MkDocs rewrites Markdown links but not raw ``<img>`` / ``<video>`` tags, and
``mkdocs build --strict`` does not check them. With directory URLs,
``docs/viewer.md`` is served at ``/viewer/``, so its images need
``../assets/...``; ``docs/tutorials/x.md`` needs ``../../assets/...``.
"""

import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[1] / "docs"
_SRC = re.compile(r'<(?:img|video|source)\b[^>]*\ssrc="([^"]+)"')


def _served_dir(page: Path) -> Path:
    """Directory a page is served from under ``use_directory_urls``."""
    return page.parent if page.name == "index.md" else page.parent / page.stem


def _local_sources(page: Path) -> list[str]:
    return [s for s in _SRC.findall(page.read_text()) if not re.match(r"^([a-z]+:|/|data:)", s)]


PAGES = sorted(DOCS.rglob("*.md"))


@pytest.mark.parametrize("page", PAGES, ids=[str(p.relative_to(DOCS)) for p in PAGES])
def test_raw_html_media_resolves(page):
    broken = [
        src for src in _local_sources(page) if not (_served_dir(page) / src).resolve().is_file()
    ]
    assert not broken, f"{page.relative_to(DOCS)} references missing media: {broken}"
