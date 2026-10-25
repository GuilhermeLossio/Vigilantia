"""Notebook hygiene: valid structure, no IBM API use, and no private paths in outputs."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = sorted((ROOT / "notebooks").glob("*.ipynb"))
EXPECTED = {
    "01-data-and-split",
    "02-features-scaling-and-kernel",
    "03-training-and-simulated-evaluation",
    "04-ibm-hardware-results-from-recorded-counts",
    "05-live-inference-runtime-and-service",
}
# Notebooks reproduce hardware results from recorded counts; they never reach IBM.
FORBIDDEN_CODE = (
    "QiskitRuntimeService",
    "dotenv",
    "ibm.connect",
    "submit_run",
    "retrieve_run",
    "check_ibm_access",
)
PRIVATE_PATH = re.compile(r"[A-Za-z]:[\/]+Users|/Users/|/home/|AppData", re.IGNORECASE)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def text(value: str | list[str]) -> str:
    return value if isinstance(value, str) else "".join(value)


def test_the_documented_notebooks_exist() -> None:
    assert {path.stem for path in NOTEBOOKS} == EXPECTED


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.stem)
def test_notebook_structure(path: Path) -> None:
    notebook = load(path)
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["kernelspec"]["name"] == "python3"
    assert notebook["cells"][0]["cell_type"] == "markdown"
    assert text(notebook["cells"][0]["source"]).startswith(f"# {path.stem[:2]} ")


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.stem)
def test_notebook_code_never_reaches_ibm(path: Path) -> None:
    code = "\n".join(
        text(cell["source"]) for cell in load(path)["cells"] if cell["cell_type"] == "code"
    )
    assert not [name for name in FORBIDDEN_CODE if name in code]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.stem)
def test_notebook_outputs_have_no_errors_or_private_paths(path: Path) -> None:
    for cell in load(path)["cells"]:
        for output in cell.get("outputs", []):
            assert output["output_type"] != "error", text(output.get("traceback", ""))
            rendered = json.dumps(output)
            assert not PRIVATE_PATH.search(rendered), rendered[:200]


def test_reproduction_guide_lists_every_notebook_in_order() -> None:
    guide = (ROOT / "docs" / "reproducing.md").read_text(encoding="utf-8")
    positions = [guide.index(f"../notebooks/{name}.ipynb") for name in sorted(EXPECTED)]
    assert positions == sorted(positions)
    assert "pip install -r requirements.txt" in guide


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.stem)
def test_committed_notebooks_were_executed_in_order(path: Path) -> None:
    counts = [
        cell["execution_count"] for cell in load(path)["cells"] if cell["cell_type"] == "code"
    ]
    assert counts == list(range(1, len(counts) + 1))
