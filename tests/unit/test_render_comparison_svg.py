"""Model comparison SVG rendering from synthetic and published reports; no network."""

from __future__ import annotations

import copy
import json
import re
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import pytest

from scripts import render_comparison_svg as render

pytestmark = pytest.mark.unit
SVG = "{http://www.w3.org/2000/svg}"
METRIC_NAMES = ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "average_precision")


def model(name: str, source: str, accuracy: float, interval: bool, kernel: float | None) -> dict:
    entry = {
        "name": name,
        "kernel_source": source,
        "metrics": dict.fromkeys(METRIC_NAMES, 0.5) | {"accuracy": accuracy},
        "confusion_matrix": [[3, 3], [3, 3]],
        "runtime_seconds": {"fit": 0.001, "test_kernel": kernel, "predict": 0.0001},
    }
    if interval:
        entry["confidence_intervals"] = {"accuracy": [accuracy - 0.1, accuracy + 0.1]}
    return entry


def models(interval: bool, hardware: bool, kernel: float) -> list[dict]:
    rows = [
        model("Majority baseline", "classical", 0.5, interval, None),
        model("Logistic regression", "classical", 0.6, interval, None),
        model("RBF SVM", "classical", 0.65, interval, None),
        model("Quantum-kernel SVM", "simulation", 0.55, interval, 0.01),
    ]
    if hardware:
        rows += [
            model("Quantum-kernel SVM", "qpu", 0.45, interval, kernel),
            model("Quantum-kernel SVM", "qpu_calibrated", 0.475, interval, kernel),
        ]
    return rows


BENCHMARK = {
    "split": {"train": 24, "test": 12, "seed": 42},
    "hardware": {"backend": "ibm_test"},
    "models": models(interval=False, hardware=True, kernel=50.0),
    "runtime": {"qpu": {"execution_seconds": 50.0, "charged_seconds": 55.0}},
}
EXTERNAL = {
    "preparation": {"primary_rows": 1234},
    "models": [
        {k: v for k, v in m.items() if k != "runtime_seconds"}
        for m in models(interval=True, hardware=False, kernel=0.0)
    ],
}
SUBSET = {
    "subset": {"size": 40},
    "hardware": {"backend": "ibm_test"},
    "models": models(interval=True, hardware=True, kernel=150.0),
    "runtime": {"qpu": {"execution_seconds": 150.0, "charged_seconds": 160.0}},
}


def point_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    benchmark: dict | None = BENCHMARK,
    external: dict | None = EXTERNAL,
    subset: dict | None = SUBSET,
) -> None:
    for attribute, data in (("BENCHMARK", benchmark), ("EXTERNAL", external), ("SUBSET", subset)):
        path = tmp_path / f"{attribute.lower()}.json"
        if data is not None:
            path.write_text(json.dumps(data), encoding="utf-8")
        monkeypatch.setattr(render, attribute, path)


def test_render_is_deterministic_valid_and_accessible(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    point_paths(monkeypatch, tmp_path)
    svg = render.render()
    assert svg == render.render()
    root = ElementTree.fromstring(svg)
    assert root.get("role") == "img"
    assert root.get("aria-labelledby") == "title desc"
    assert root.find(f"{SVG}title").text.startswith("Vigilantia model comparison")
    description = root.find(f"{SVG}desc").text
    assert "n=1,234" in description and "not run" in description
    assert "prefers-color-scheme: dark" in svg
    for token in ("--series-1: #3987e5", "--series-2: #d95926", "--neutral: #7d7c76"):
        assert token in svg
    assert 'width="960"' in svg
    assert str(render.ROOT) not in svg


def test_text_never_uses_series_colors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    point_paths(monkeypatch, tmp_path)
    root = ElementTree.fromstring(render.render())
    classes = {element.get("class") for element in root.iter(f"{SVG}text")}
    assert classes <= {"ink", "ink2", "muted"}


def test_values_come_from_the_reports(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    point_paths(monkeypatch, tmp_path)
    before = render.render()
    subset = copy.deepcopy(SUBSET)
    subset["models"][4]["metrics"]["accuracy"] = 0.775
    subset["models"][4]["runtime_seconds"]["test_kernel"] = 123.4
    subset["runtime"]["qpu"]["charged_seconds"] = 321.0
    point_paths(monkeypatch, tmp_path, subset=subset)
    after = render.render()
    assert after != before
    assert ">0.775<" in after and ">0.775<" not in before
    assert "123.4 s on ibm_test" in after
    assert "ibm_test: 150.0 s execution, 321 s billed" in after
    assert "n=12 · no interval published" in after
    assert "12 ms" not in after  # simulation kernel 0.01 s renders as 10 ms
    assert "10 ms local simulation" in after


def test_missing_combinations_are_marked_not_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    point_paths(monkeypatch, tmp_path)
    svg = render.render()
    assert svg.count(">not run on this set<") == 2
    assert "ibm_test, calibrated" in svg


def test_missing_report_is_drawn_as_not_published(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    point_paths(monkeypatch, tmp_path, subset=None)
    svg = render.render()
    ElementTree.fromstring(svg)
    # Six accuracy rows and six runtime rows of the subset columns.
    assert svg.count(">report not published<") == 12 + 2
    assert "ibm_test" in svg  # backend falls back to the benchmark report


def test_all_reports_missing_fails_clearly(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    point_paths(monkeypatch, tmp_path, benchmark=None, external=None, subset=None)
    with pytest.raises(SystemExit, match="No published reports"):
        render.render()


@pytest.mark.parametrize(
    ("seconds", "label"),
    [(267.584, "267.6 s"), (2.5, "2.50 s"), (0.0117, "12 ms"), (0.0013, "1.3 ms"), (7e-6, "7 µs")],
)
def test_format_seconds(seconds: float, label: str) -> None:
    assert render.format_seconds(seconds) == label


def test_published_chart_matches_published_reports() -> None:
    paths = (render.BENCHMARK, render.EXTERNAL, render.SUBSET, render.OUTPUT)
    if not all(path.is_file() for path in paths):
        pytest.skip("published reports or chart not present")
    svg = render.render()
    assert render.OUTPUT.read_text(encoding="utf-8") == svg + "\n"
    assert not re.search(r"(?<![A-Za-z])[A-Za-z]:[\\/]|/Users/|/home/", svg)


def test_interval_below_domain_stays_inside_its_column(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    subset = copy.deepcopy(SUBSET)
    subset["models"][0]["confidence_intervals"]["accuracy"] = [0.05, 0.6]
    subset["models"][1]["confidence_intervals"]["accuracy"] = [0.4, 1.2]
    point_paths(monkeypatch, tmp_path, subset=subset)
    root = ElementTree.fromstring(render.render())
    gap = 28
    width = (render.PLOT_RIGHT - render.PLOT_LEFT - gap * 2) / 3
    x0 = render.PLOT_LEFT + 2 * (width + gap)
    lines = [e for e in root.iter(f"{SVG}line") if (e.get("class") or "").startswith("ci ")]
    subset_lines = [e for e in lines if float(e.get("x1")) >= x0 - 0.05]
    assert subset_lines
    lefts = [render.PLOT_LEFT + c * (width + gap) for c in range(3)]
    for element in lines:
        x1, x2 = float(element.get("x1")), float(element.get("x2"))
        assert any(left - 0.05 <= x1 <= x2 <= left + width + 0.05 for left in lefts)
    # Each clipped end gets a vertical open cap at the column edge.
    caps = [e for e in subset_lines if e.get("x1") == e.get("x2")]
    assert {round(float(e.get("x1")), 1) for e in caps} == {
        round(x0, 1),
        round(x0 + width, 1),
    }


def test_calibrated_runtime_is_marked_as_the_same_jobs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    point_paths(monkeypatch, tmp_path)
    svg = render.render()
    # One hardware total per set, not one per hardware row.
    assert svg.count(">150.0 s on ibm_test<") == 1
    assert svg.count(">50.0 s on ibm_test<") == 1
    assert svg.count(f">{render.CALIBRATED_NOTE}<") == 2
    description = ElementTree.fromstring(svg).find(f"{SVG}desc").text
    assert "calibrated test kernel: computed from the same ibm_test jobs" in description
    assert "External subset, IBM hardware" not in svg
    assert render.SUBSET_TITLE in svg


def test_runtime_key_uses_tighter_spacing() -> None:
    label = "fit (training; quantum fit includes simulating its training kernel)"
    assert render.text_width(label, 12, render.KEY_EM) < render.text_width(label, 12)
