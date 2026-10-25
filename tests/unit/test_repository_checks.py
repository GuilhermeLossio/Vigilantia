"""Unit tests for the repository checks; application tests will accompany code."""

from pathlib import Path

import pytest

from scripts.check_repository import (
    check_environment_example,
    check_local_links,
    check_repository,
)

pytestmark = pytest.mark.unit


def test_environment_example_accepts_comments_and_blanks(tmp_path: Path) -> None:
    example = tmp_path / ".env.example"
    example.write_text("# Instructions\n\n  # TOKEN=replace-locally\n", encoding="utf-8")
    assert check_environment_example(example) == []


@pytest.mark.parametrize("line", ["TOKEN=not-a-real-token", "export TOKEN=placeholder"])
def test_environment_example_rejects_active_values(tmp_path: Path, line: str) -> None:
    example = tmp_path / ".env.example"
    example.write_text(f"# Header\n{line}\n", encoding="utf-8")
    errors = check_environment_example(example)
    assert len(errors) == 1
    assert ":2:" in errors[0]


@pytest.mark.parametrize("target", ["missing.md", "images/missing.svg#diagram"])
def test_missing_local_targets_fail(tmp_path: Path, target: str) -> None:
    document = tmp_path / "README.md"
    document.write_text(f"[Link]({target})", encoding="utf-8")
    assert len(check_local_links(document)) == 1


@pytest.mark.parametrize("target", ["Guide%20One.md#setup", "<Guide One.md>"])
def test_existing_local_targets_with_spaces_pass(tmp_path: Path, target: str) -> None:
    (tmp_path / "Guide One.md").write_text("# Setup", encoding="utf-8")
    document = tmp_path / "README.md"
    document.write_text(f"[Guide]({target})", encoding="utf-8")
    assert check_local_links(document) == []


def test_relative_targets_resolve_from_document_directory(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "README.md").touch()
    document = tmp_path / "docs" / "guide.md"
    document.write_text("[Home](../README.md)", encoding="utf-8")
    assert check_local_links(document) == []


def test_remote_links_and_code_examples_are_not_fetched(tmp_path: Path) -> None:
    document = tmp_path / "README.md"
    document.write_text(
        "[Web](https://example.invalid)\n[Section](#section)\n"
        "```md\n[Example](missing.md)\n```\n"
        "~~~md\n[Example](also-missing.md)\n~~~\n",
        encoding="utf-8",
    )
    assert check_local_links(document) == []


def test_repository_reports_missing_configuration_template(tmp_path: Path) -> None:
    assert "Missing .env.example" in check_repository(tmp_path)


def test_repository_collects_nested_document_errors(tmp_path: Path) -> None:
    (tmp_path / ".env.example").write_text("# Empty example\n", encoding="utf-8")
    nested = tmp_path / "docs" / "nested"
    nested.mkdir(parents=True)
    (nested / "guide.md").write_text("[Broken](missing.md)", encoding="utf-8")
    assert len(check_repository(tmp_path)) == 1
