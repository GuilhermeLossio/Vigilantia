"""The documentation catalog lists every readable document and is linked from the README."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CATALOG = (ROOT / "CATALOG.md").read_text(encoding="utf-8")
LINK = re.compile(r"\]\(([^)#\s]+)(?:#[^)]*)?\)")


def tracked(pattern: str) -> set[str]:
    """Files Git would publish: tracked plus untracked-but-not-ignored."""
    try:
        listed = subprocess.run(
            ["git", "ls-files", "-co", "--exclude-standard", pattern],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("Git is unavailable; the catalog check needs the file list.")
    return set(listed)


def catalog_links() -> set[str]:
    return set(LINK.findall(CATALOG))


def test_every_markdown_file_is_in_the_catalog() -> None:
    # Agent instruction files are local tooling context, not reader-facing docs.
    documents = {
        path for path in tracked("*.md") if Path(path).name not in {"AGENTS.md", "CLAUDE.md"}
    }
    missing = documents - catalog_links() - {"CATALOG.md"}
    assert not missing, f"Add to CATALOG.md: {sorted(missing)}"


def test_every_notebook_is_in_the_catalog() -> None:
    missing = tracked("notebooks/*.ipynb") - catalog_links()
    assert not missing, f"Add to CATALOG.md: {sorted(missing)}"


def test_catalog_links_point_to_existing_files() -> None:
    assert not [link for link in catalog_links() if not (ROOT / link).exists()]


def test_every_document_entry_has_an_audience_and_reading_time() -> None:
    rows = [
        line for line in CATALOG.splitlines() if re.match(r"^\| \[.+\]\(.+\.(md|ipynb)\)", line)
    ]
    assert rows
    assert all(re.search(r"\| ~\d+ min \|$", row) for row in rows)


def test_readme_and_docs_index_link_the_catalog() -> None:
    assert "](CATALOG.md)" in (ROOT / "README.md").read_text(encoding="utf-8")
    assert "](../CATALOG.md)" in (ROOT / "docs" / "README.md").read_text(encoding="utf-8")
