"""Offline checks for local documentation links and the configuration template."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote, urlsplit


def check_environment_example(path: Path) -> list[str]:
    """The design-stage template must not contain active configuration values."""
    return [
        f"{path}:{number}: example settings must be commented out"
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if line.strip() and not line.lstrip().startswith("#")
    ]


def check_local_links(path: Path) -> list[str]:
    """Check inline relative link targets outside fenced code (not remote URLs)."""
    content = path.read_text(encoding="utf-8")
    content = re.sub(r"```.*?```|~~~.*?~~~", "", content, flags=re.DOTALL)
    errors = []
    for target in re.findall(r"!?\[[^\]\n]*\]\(([^)\n]+)\)", content):
        # Support angle-bracket destinations; optional Markdown titles are excluded.
        target = target[1 : target.index(">")] if target.startswith("<") else target.split()[0]
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        destination = path.parent / unquote(parsed.path)
        if not destination.exists():
            errors.append(f"{path}: missing local link target {target}")
    return errors


def check_repository(root: Path) -> list[str]:
    """Check project documents without traversing environments or downloaded data."""
    documents = sorted(root.glob("*.md")) + sorted((root / "docs").rglob("*.md"))
    data_readme = root / "data" / "README.md"
    if data_readme.exists():
        documents.append(data_readme)
    errors = [error for path in documents for error in check_local_links(path)]
    example = root / ".env.example"
    if example.exists():
        errors.extend(check_environment_example(example))
    else:
        errors.append("Missing .env.example")
    return errors


def main() -> int:
    errors = check_repository(Path(__file__).resolve().parents[1])
    if errors:
        print("\n".join(errors))
        return 1
    print("Repository checks passed (local file links and commented environment example).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
