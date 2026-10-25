"""Results SVG rendering from synthetic inputs; no network."""

from __future__ import annotations

import importlib.util
import json
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
SPEC = importlib.util.spec_from_file_location(
    "render_results_svg", Path(__file__).resolve().parents[2] / "scripts" / "render_results_svg.py"
)
render = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(render)

METRICS = dict.fromkeys(
    ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "average_precision"), 0.5
)
RESULTS = {
    "run_id": "sim-test",
    "split": {"train": 24, "test": 12, "seed": 42},
    "models": [
        {
            "name": "RBF SVM",
            "kernel_source": "classical",
            "metrics": METRICS,
            "confusion_matrix": [[3, 3], [3, 3]],
        },
        {
            "name": "Quantum-kernel SVM",
            "kernel_source": "simulation",
            "metrics": METRICS,
            "confusion_matrix": [[3, 3], [3, 3]],
        },
    ],
}


def point_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(render, "RESULTS", tmp_path / "results.json")
    monkeypatch.setattr(render, "REHEARSAL", tmp_path / "missing-rehearsal.json")
    monkeypatch.setattr(render, "RUNS", tmp_path / "runs")


def test_render_with_only_published_results_marks_hardware_pending(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    point_paths(monkeypatch, tmp_path)
    (tmp_path / "results.json").write_text(json.dumps(RESULTS), encoding="utf-8")
    svg = render.render()
    root = ElementTree.fromstring(svg)
    assert root.find("{http://www.w3.org/2000/svg}title").text == "Vigilantia benchmark results"
    assert "pending: full QPU run not yet evaluated" in svg
    assert "6/12 correct" in svg
    assert "prefers-color-scheme: dark" in svg


def test_render_includes_pilot_measurements_and_projections(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    point_paths(monkeypatch, tmp_path)
    (tmp_path / "results.json").write_text(json.dumps(RESULTS), encoding="utf-8")
    pilot_dir = tmp_path / "runs" / "pilot-ibm_fez-1"
    pilot_dir.mkdir(parents=True)
    pilot = {
        "backend": "ibm_fez",
        "circuits": 12,
        "mean_abs_kernel_error_vs_ideal": 0.034,
        "usage": {"charged_seconds": 5.0, "resource_units": 0.08, "execution_seconds": 3.2},
        "projected_full_run": {
            "charged_seconds": 140.0,
            "resource_units": 2.3,
            "execution_seconds": 89.0,
        },
    }
    (pilot_dir / "evaluation.json").write_text(json.dumps(pilot), encoding="utf-8")
    svg = render.render()
    ElementTree.fromstring(svg)
    assert "ibm_fez pilot, uncorrected" in svg
    assert "about 89 s" in svg and "about 140 s" in svg
    assert 'class="projected"' in svg


def test_missing_results_file_fails_clearly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    point_paths(monkeypatch, tmp_path)
    with pytest.raises(SystemExit, match="missing"):
        render.render()


def test_latest_runs_ignore_external_subset_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    point_paths(monkeypatch, tmp_path)
    for name, circuits in (
        ("full-ibm_fez-20260101T000000Z", 336),
        ("pilot-ibm_fez-20260101T000000Z", 12),
        ("external-ibm_fez-20260102T000000Z", 1008),
        ("rehearsal-fake_fez-20260103T000000Z", 1008),
    ):
        (tmp_path / "runs" / name).mkdir(parents=True)
        (tmp_path / "runs" / name / "evaluation.json").write_text(
            json.dumps({"circuits": circuits}), encoding="utf-8"
        )
    assert render.latest_run("full") == {"circuits": 336}
    assert render.latest_run("pilot") == {"circuits": 12}
