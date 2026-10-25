"""Requirements files: every package is pinned and the files chain as documented."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
PIN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*==[0-9][0-9A-Za-z.+-]*$")
FILES = (
    "requirements.txt",
    "requirements-app.txt",
    "requirements-model.txt",
    "requirements-dev.txt",
)


def entries(name: str) -> list[str]:
    lines = (ROOT / name).read_text(encoding="utf-8").splitlines()
    return [line.split("#", 1)[0].strip() for line in lines if line.split("#", 1)[0].strip()]


def includes(name: str) -> list[str]:
    return [line.removeprefix("-r ").strip() for line in entries(name) if line.startswith("-r ")]


def packages(name: str) -> dict[str, str]:
    """Resolve -r includes and return {normalized package: version}."""
    resolved: dict[str, str] = {}
    for included in includes(name):
        resolved.update(packages(included))
    for line in entries(name):
        if not line.startswith("-r "):
            package, version = line.split("==")
            resolved[package.lower().replace("_", "-")] = version
    return resolved


@pytest.mark.parametrize("name", FILES)
def test_every_package_is_pinned_exactly(name: str) -> None:
    unpinned = [
        line for line in entries(name) if not line.startswith("-r ") and not PIN.match(line)
    ]
    assert not unpinned


def test_files_chain_as_documented() -> None:
    assert includes("requirements.txt") == ["requirements-app.txt", "requirements-model.txt"]
    assert includes("requirements-dev.txt") == ["requirements.txt"]
    assert includes("requirements-app.txt") == includes("requirements-model.txt") == []


def test_no_package_is_pinned_twice_with_different_versions() -> None:
    seen: dict[str, tuple[str, str]] = {}
    for name in FILES:
        for line in entries(name):
            if line.startswith("-r "):
                continue
            package, version = line.split("==")
            key = package.lower().replace("_", "-")
            assert seen.setdefault(key, (version, name))[0] == version, (key, seen[key], name)


def test_the_full_environment_covers_service_model_and_notebooks() -> None:
    full = packages("requirements.txt")
    for required in ("flask", "scikit-learn", "qiskit", "qiskit-aer", "jupyterlab", "nbconvert"):
        assert required in full
    dev = packages("requirements-dev.txt")
    assert {"pytest", "ruff"} <= set(dev) and set(full) <= set(dev)


def test_the_container_installs_only_the_service_dependencies() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "pip install -r requirements-app.txt" in dockerfile
    assert "qiskit" not in packages("requirements-app.txt")
