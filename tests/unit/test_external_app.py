"""External-validation route and page: the published report, synthetic variants, no network."""

from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import pytest

from app import create_app
from app import external as external_module
from app.results import ResultsInvalid
from src.evaluation import external as evaluation

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
PUBLISHED = ROOT / "reports" / "external-validation.json"
REPORT = json.loads(PUBLISHED.read_text(encoding="utf-8"))


def write(tmp_path: Path, data: object) -> Path:
    path = tmp_path / "external-validation.json"
    path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")
    return path


def client_for(tmp_path: Path, external: Path):
    return create_app(
        results_path=tmp_path / "no-results.json",
        replay_path=tmp_path / "no-replay.json",
        chart_path=tmp_path / "no-chart.svg",
        external_path=external,
    ).test_client()


def mutated(change) -> dict:
    report = copy.deepcopy(REPORT)
    change(report)
    return report


def test_published_report_is_served_unchanged() -> None:
    response = create_app().test_client().get("/v1/external-validation")
    assert response.status_code == 200
    assert response.get_json() == REPORT


def test_page_shows_dataset_funnel_metrics_and_simulation_label() -> None:
    page = create_app().test_client().get("/").get_data(as_text=True)
    section = page.split(">External validation</h2>", 1)[1].split(">Replay:", 1)[0]
    assert REPORT["dataset"]["doi"] in section
    assert REPORT["dataset"]["license"] in section
    assert str(REPORT["preparation"]["primary_rows"]) in section
    for entry in REPORT["preparation"]["exclusions"]:
        assert external_module.STEPS[entry["step"]] in section
    assert "Quantum kernel: exact simulation" in section
    assert "not IBM Quantum hardware" in section
    quantum = next(m for m in REPORT["models"] if m["model"] == "quantum_kernel_svm")
    low, high = quantum["confidence_intervals"]["accuracy"]
    assert f"{low:.3f}&ndash;{high:.3f}" in section
    assert "n/a" in section  # majority baseline has no average precision
    assert "No claim of quantum advantage is made." in section


def test_section_follows_the_recorded_benchmark(tmp_path: Path) -> None:
    page = client_for(tmp_path, PUBLISHED).get("/").get_data(as_text=True)
    assert (
        page.index(">Recorded benchmark</h2>")
        < page.index(">External validation</h2>")
        < page.index(">Replay: recorded")
    )


def test_missing_report_is_pending_and_does_not_change_the_page_status(tmp_path: Path) -> None:
    client = client_for(tmp_path, tmp_path / "missing.json")
    api = client.get("/v1/external-validation")
    assert api.status_code == 503
    assert api.get_json()["error"] == "results_unavailable"
    page = client.get("/")
    assert page.status_code == 503  # from the missing benchmark results, not the report
    assert "External validation pending." in page.get_data(as_text=True)


def test_invalid_report_is_a_500_and_an_unavailable_section(tmp_path: Path) -> None:
    client = client_for(tmp_path, write(tmp_path, "{not json"))
    api = client.get("/v1/external-validation")
    assert api.status_code == 500
    assert api.get_json()["error"] == "results_invalid"
    assert "not valid JSON" in api.get_json()["detail"]
    assert "External validation unavailable." in client.get("/").get_data(as_text=True)


def test_path_can_come_from_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VIGILANTIA_EXTERNAL_PATH", str(tmp_path / "missing.json"))
    assert create_app().test_client().get("/v1/external-validation").status_code == 503


def test_page_escapes_report_text(tmp_path: Path) -> None:
    report = mutated(lambda r: r["limitations"].append("Synthetic <b>fixture</b>."))
    page = client_for(tmp_path, write(tmp_path, report)).get("/").get_data(as_text=True)
    assert "&lt;b&gt;fixture&lt;/b&gt;" in page
    assert "<b>fixture</b>" not in page


def test_page_keeps_the_strict_csp(tmp_path: Path) -> None:
    response = client_for(tmp_path, PUBLISHED).get("/")
    assert "default-src 'none'" in response.headers["Content-Security-Policy"]
    assert "<script" not in response.get_data(as_text=True)


def _set(path: tuple, value: object):
    def change(report: dict) -> None:
        target = report
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    return change


INVALID = [
    (_set(("schema_version",), 2), "schema_version"),
    (_set(("dataset", "doi"), " "), "dataset must hold"),
    (_set(("protocol", "sha256"), "abc"), "protocol must hold"),
    (_set(("preparation", "primary_rows"), 1), "input_rows must equal"),
    (_set(("preparation", "exclusions"), []), "preparation.exclusions"),
    (_set(("preparation", "class_counts", "phishing"), 0), "class_counts"),
    (_set(("clipped_values", "rows_with_any_clipped"), 10**6), "clipped_values"),
    (_set(("bootstrap", "resamples_used"), 10**6), "bootstrap"),
    (_set(("models", 0, "kernel_source"), "qpu"), "kernel_source"),
    (_set(("models", 3, "kernel_source"), "qpu"), "kernel_source"),
    (_set(("models", 1, "metrics", "accuracy"), 1.5), "metrics.accuracy"),
    (_set(("models", 1, "confidence_intervals", "f1"), [0.7, 0.6]), "confidence_intervals.f1"),
    (_set(("models", 1, "confidence_intervals", "f1"), None), "confidence_intervals.f1"),
    (_set(("models", 1, "confusion_matrix"), [[1, 0], [0, 1]]), "confusion_matrix"),
    (lambda r: r["models"].pop(), "exactly once"),
    (_set(("limitations",), []), "limitations"),
    (lambda r: r["limitations"].append("see https://example.com"), "URLs or local paths"),
    (lambda r: r["limitations"].append("C:/Users/someone/data"), "URLs or local paths"),
]


@pytest.mark.parametrize(("change", "message"), INVALID)
def test_invalid_reports_are_rejected(tmp_path: Path, change, message: str) -> None:
    response = client_for(tmp_path, write(tmp_path, mutated(change))).get("/v1/external-validation")
    assert response.status_code == 500
    assert message in response.get_json()["detail"]


@pytest.mark.parametrize(("change", "message"), INVALID)
def test_service_and_evaluation_validators_agree(change, message: str) -> None:
    """The service copy of the contract must not drift from the one that writes the report."""
    report = mutated(change)
    with pytest.raises(ResultsInvalid):
        external_module.validate(report)
    with pytest.raises(evaluation.ExternalError):
        evaluation.validate(report)


def test_validators_share_constants_and_accept_the_published_report() -> None:
    assert external_module.METRICS == evaluation.METRICS
    assert tuple(external_module.STEPS) == evaluation.STEPS
    assert external_module.MODELS == evaluation.MODELS
    assert external_module._FORBIDDEN_TEXT.pattern == evaluation._FORBIDDEN_TEXT.pattern
    assert external_module.validate(copy.deepcopy(REPORT)) == evaluation.validate(REPORT)


def test_service_module_does_not_import_training_code() -> None:
    """The image ships only app/, src/features/, and src/inference/."""
    tree = ast.parse(Path(external_module.__file__).read_text(encoding="utf-8"))
    imported = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert all(not name.startswith(("src", "pandas", "numpy", "sklearn")) for name in imported)


def test_docker_context_keeps_the_report_and_drops_qpu_evidence() -> None:
    lines = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    assert "!reports/" in lines
    assert "reports/qpu-evidence/" in lines
    assert not any(line.strip().startswith("reports/external") for line in lines)
