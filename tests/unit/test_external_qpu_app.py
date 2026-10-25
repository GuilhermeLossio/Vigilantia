"""External hardware subset route, chart, and page: the published report, synthetic variants."""

from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import pytest

from app import create_app
from app import external_qpu as service
from app.results import ResultsInvalid
from src.evaluation import external_qpu as evaluation

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
PUBLISHED = ROOT / "reports" / "external-qpu-subset.json"
EXTERNAL = ROOT / "reports" / "external-validation.json"
REPORT = json.loads(PUBLISHED.read_text(encoding="utf-8"))
EXTERNAL_REPORT = json.loads(EXTERNAL.read_text(encoding="utf-8"))
HEADING = ">External subset on IBM Quantum hardware</h2>"
SVG = '<svg xmlns="http://www.w3.org/2000/svg" width="4" height="4"></svg>'


def write(tmp_path: Path, data: object, name: str = "external-qpu-subset.json") -> Path:
    path = tmp_path / name
    path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")
    return path


def client_for(tmp_path: Path, subset: Path, external: Path = EXTERNAL, comparison=None):
    return create_app(
        results_path=tmp_path / "no-results.json",
        replay_path=tmp_path / "no-replay.json",
        chart_path=tmp_path / "no-chart.svg",
        external_path=external,
        external_qpu_path=subset,
        comparison_chart_path=comparison or tmp_path / "no-comparison.svg",
    ).test_client()


def section(page: str) -> str:
    return page.split(HEADING, 1)[1].split(">Replay:", 1)[0]


def mutated(change) -> dict:
    report = copy.deepcopy(REPORT)
    change(report)
    return report


def row(report: dict, source: str) -> dict:
    return next(r for r in report["models"] if r["kernel_source"] == source)


def correct(entry: dict) -> int:
    return entry["confusion_matrix"][0][0] + entry["confusion_matrix"][1][1]


def test_published_report_is_served_unchanged() -> None:
    response = create_app().test_client().get("/v1/external-validation/qpu")
    assert response.status_code == 200
    assert response.get_json() == REPORT


def test_page_shows_run_rows_intervals_and_confusion_counts(tmp_path: Path) -> None:
    text = section(client_for(tmp_path, PUBLISHED).get("/").get_data(as_text=True))
    run = REPORT["hardware"]
    assert run["run_id"] in text
    assert run["backend"] in text
    assert f"n={REPORT['subset']['size']}" in text
    assert f"{REPORT['subset']['primary_rows']:,}" in text
    for entry in REPORT["models"]:
        low, high = entry["confidence_intervals"]["accuracy"]
        assert f"{entry['metrics']['accuracy']:.3f}<br>" in text
        assert f"{low:.3f}&ndash;{high:.3f}" in text
        (tn, fp), (fn, tp) = entry["confusion_matrix"]
        assert f"{tn} / {fp} / {fn} / {tp}" in text
    assert "Quantum kernel: IBM Quantum hardware, calibrated" in text
    assert "n/a" in text  # majority baseline has no average precision
    assert "No claim of quantum advantage is made." in text


def test_page_shows_kernel_checks_calibration_and_ibm_usage(tmp_path: Path) -> None:
    text = section(client_for(tmp_path, PUBLISHED).get("/").get_data(as_text=True))
    kernel = REPORT["kernel"]
    for value in kernel["test_kernel_error_vs_ideal"].values():
        assert f"{value:.3f}" in text
    calibration = kernel["calibration"]
    assert f"{calibration['a']:.3f} / {calibration['c']:.4f} / {calibration['rmse']:.3f}" in text
    assert f"<td>{calibration['clipped_test_values']}</td>" in text
    assert f"{kernel['diagonal_controls_mean']:.3f}" in text
    usage = REPORT["runtime"]["qpu"]
    assert f"{usage['execution_seconds']:.1f} s execution" in text
    assert f"{usage['charged_seconds']:.1f} s billed" in text
    assert f"{usage['resource_units']:.2f} resource units" in text


def test_reading_is_derived_from_the_reports(tmp_path: Path) -> None:
    text = section(client_for(tmp_path, PUBLISHED).get("/").get_data(as_text=True))
    simulation = correct(row(REPORT, "simulation"))
    for source, label in service.HARDWARE_LABELS.items():
        delta = correct(row(REPORT, source)) - simulation
        assert f"{delta:+d} for the {label} kernel" in text
    assert "not evidence that hardware improves the model" in text
    assert f"{100 / REPORT['subset']['size']:.1f} percentage points" in text
    full = {r["model"]: r["metrics"]["accuracy"] for r in EXTERNAL_REPORT["models"]}
    assert f"{full['logistic_regression']:.3f} and {full['rbf_svm']:.3f}" in text
    assert f"{EXTERNAL_REPORT['preparation']['primary_rows']:,} rows" in text
    assert "consistent with sampling variation" in text and "not selection" in text
    assert "harder" not in text


def test_full_set_gap_is_called_sampling_variation_only_inside_the_intervals() -> None:
    facts = service.interpretation(copy.deepcopy(REPORT), copy.deepcopy(EXTERNAL_REPORT))
    assert facts["full_set"]["within_subset_intervals"] is True
    names = {r["model"] for r in facts["full_set"]["classical"]}
    assert names == {r["model"] for r in facts["classical"]}

    def narrow(report: dict) -> None:
        for entry in report["models"]:
            if entry["model"] == "logistic_regression":
                accuracy = entry["metrics"]["accuracy"]
                entry["confidence_intervals"]["accuracy"] = [accuracy, accuracy]

    facts = service.interpretation(mutated(narrow), copy.deepcopy(EXTERNAL_REPORT))
    assert facts["full_set"]["within_subset_intervals"] is False


def test_reading_outside_the_intervals_does_not_claim_sampling_variation(
    tmp_path: Path,
) -> None:
    def narrow(report: dict) -> None:
        for entry in report["models"]:
            if entry["model"] in service.CLASSICAL_MODELS:
                accuracy = entry["metrics"]["accuracy"]
                entry["confidence_intervals"]["accuracy"] = [accuracy, accuracy]

    path = write(tmp_path, mutated(narrow), "subset.json")
    text = section(client_for(tmp_path, path).get("/").get_data(as_text=True))
    assert "Compared with the full external set" in text
    assert "sampling variation" not in text
    assert "not selection" in text


def test_interpretation_flags_differences_outside_the_intervals() -> None:
    def widen_gap(report: dict) -> None:
        hardware = row(report, "qpu")
        hardware["metrics"]["accuracy"] = 0.9
        hardware["confidence_intervals"]["accuracy"] = [0.8, 0.95]

    facts = service.interpretation(mutated(widen_gap))
    assert facts["within_intervals"] is False
    assert facts["full_set"] is None
    assert service.interpretation(copy.deepcopy(REPORT))["within_intervals"] is True


def test_reading_without_the_external_report_omits_the_full_set(tmp_path: Path) -> None:
    client = client_for(tmp_path, PUBLISHED, external=tmp_path / "missing.json")
    text = section(client.get("/").get_data(as_text=True))
    assert "Compared with the full external set" not in text
    assert "not evidence that hardware improves the model" in text


def test_section_follows_external_validation(tmp_path: Path) -> None:
    page = client_for(tmp_path, PUBLISHED).get("/").get_data(as_text=True)
    assert page.index(">External validation</h2>") < page.index(HEADING) < page.index("Replay:")
    assert "<code>/v1/external-validation/qpu</code>" in page


def test_invalid_calibration_report_shows_five_rows(tmp_path: Path) -> None:
    def drop_calibration(report: dict) -> None:
        report["models"].pop()
        report["kernel"]["calibration"] = {"valid": False}
        del report["kernel"]["test_kernel_error_vs_ideal"]["qpu_calibrated"]
        report["limitations"].append(service.CALIBRATION_DEVIATION)

    report = mutated(drop_calibration)
    evaluation.validate(copy.deepcopy(report))
    text = section(client_for(tmp_path, write(tmp_path, report)).get("/").get_data(as_text=True))
    assert "invalid; no calibrated row" in text
    assert "Quantum kernel: IBM Quantum hardware, calibrated" not in text
    assert "for the uncorrected hardware kernel" in text
    assert "Protocol deviation" in text


def test_noisy_simulation_report_is_labelled(tmp_path: Path) -> None:
    def rehearse(report: dict) -> None:
        report["source"] = "noisy_simulation"
        report["label"] = service.SOURCES["noisy_simulation"]

    report = mutated(rehearse)
    text = section(client_for(tmp_path, write(tmp_path, report)).get("/").get_data(as_text=True))
    assert "Not a hardware result." in text


def test_missing_report_is_pending(tmp_path: Path) -> None:
    client = client_for(tmp_path, tmp_path / "missing.json")
    api = client.get("/v1/external-validation/qpu")
    assert api.status_code == 503
    assert api.get_json()["error"] == "results_unavailable"
    page = client.get("/")
    assert page.status_code == 503  # from the missing benchmark results, not the subset
    assert "Hardware subset pending." in section(page.get_data(as_text=True))


def test_invalid_report_is_a_500_and_an_unavailable_section(tmp_path: Path) -> None:
    client = client_for(tmp_path, write(tmp_path, "{not json"))
    api = client.get("/v1/external-validation/qpu")
    assert api.status_code == 500
    assert api.get_json()["error"] == "results_invalid"
    assert "not valid JSON" in api.get_json()["detail"]
    assert "Hardware subset unavailable." in client.get("/").get_data(as_text=True)


def test_paths_can_come_from_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    chart = write(tmp_path, SVG, "chart.svg")
    monkeypatch.setenv("VIGILANTIA_EXTERNAL_QPU_PATH", str(tmp_path / "missing.json"))
    monkeypatch.setenv("VIGILANTIA_COMPARISON_CHART_PATH", str(chart))
    client = create_app().test_client()
    assert client.get("/v1/external-validation/qpu").status_code == 503
    assert client.get("/comparison.svg").get_data(as_text=True) == SVG


def test_comparison_chart_is_served_and_embedded(tmp_path: Path) -> None:
    chart = write(tmp_path, SVG, "comparison.svg")
    client = client_for(tmp_path, PUBLISHED, comparison=chart)
    response = client.get("/comparison.svg")
    assert response.status_code == 200
    assert response.mimetype == "image/svg+xml"
    assert response.get_data(as_text=True) == SVG
    assert "default-src 'none'" in response.headers["Content-Security-Policy"]
    page = section(client.get("/").get_data(as_text=True))
    assert '<img class="chart" src="/comparison.svg"' in page
    alt = page.split('src="/comparison.svg" alt="', 1)[1].split('"', 1)[0]
    assert "benchmark (no intervals)" in alt
    assert f"{REPORT['subset']['size']}-row hardware subset" in alt
    assert "runtime for the benchmark and the subset" in alt
    assert "log scale" in alt


def test_missing_comparison_chart_is_a_404_and_not_embedded(tmp_path: Path) -> None:
    client = client_for(tmp_path, PUBLISHED)
    assert client.get("/comparison.svg").status_code == 404
    assert "/comparison.svg" not in client.get("/").get_data(as_text=True)


def test_page_escapes_report_text_and_keeps_the_strict_csp(tmp_path: Path) -> None:
    report = mutated(lambda r: r["limitations"].append("Synthetic <b>fixture</b>."))
    response = client_for(tmp_path, write(tmp_path, report)).get("/")
    page = response.get_data(as_text=True)
    assert "&lt;b&gt;fixture&lt;/b&gt;" in page
    assert "<b>fixture</b>" not in page
    assert "default-src 'none'" in response.headers["Content-Security-Policy"]
    assert "<script" not in page


def _set(path: tuple, value: object):
    def change(report: dict) -> None:
        target = report
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    return change


INVALID = [
    (_set(("schema_version",), 2), "schema_version"),
    (_set(("label",), "IBM hardware"), "source must be one of"),
    (_set(("protocol", "sha256"), "abc"), "protocol must hold"),
    (_set(("parent_protocol", "path"), " "), "parent_protocol must hold"),
    (_set(("benchmark", "training_examples"), 0), "benchmark must hold"),
    (_set(("subset", "size"), 41), "subset.class_counts"),
    (_set(("subset", "primary_rows"), 10), "subset.primary_rows"),
    (_set(("subset", "row_index"), list(range(40, 0, -1))), "subset.row_index"),
    (_set(("subset", "group_hash"), ["0" * 16] * 40), "subset.group_hash"),
    (_set(("hardware", "backend"), ""), "hardware must hold"),
    (_set(("hardware", "plan", "test"), 1), "hardware.plan"),
    (lambda r: r["hardware"]["job_ids"].pop(), "frozen batch size"),
    (_set(("kernel", "calibration", "valid"), "yes"), "kernel.calibration"),
    (_set(("kernel", "calibration", "a"), None), "A valid calibration"),
    (_set(("kernel", "test_kernel_error_vs_ideal", "qpu"), 2.0), "test_kernel_error_vs_ideal"),
    (_set(("kernel", "diagonal_controls_mean"), -0.1), "diagonal_controls_mean"),
    (_set(("runtime", "repeats"), 3), "frozen 25 repeats"),
    (_set(("runtime", "qpu", "charged_seconds"), -1), "runtime must hold"),
    (_set(("bootstrap", "resamples_used"), 10**6), "bootstrap"),
    (_set(("models", 4, "kernel_source"), "classical"), "compared rows"),
    (_set(("models", 0, "name"), "Majority"), "models[0].name"),
    (_set(("models", 1, "metrics", "accuracy"), 1.5), "metrics.accuracy"),
    (_set(("models", 1, "confidence_intervals", "f1"), [0.7, 0.6]), "confidence_intervals.f1"),
    (_set(("models", 1, "confusion_matrix"), [[1, 0], [0, 1]]), "confusion_matrix"),
    (_set(("models", 1, "runtime_seconds", "fit"), -1), "runtime_seconds"),
    (lambda r: r["models"].reverse(), "compared rows in order"),
    (lambda r: r["models"].pop(), "compared rows in order"),
    (_set(("kernel", "calibration"), {"valid": False}), "test_kernel_error_vs_ideal"),
    (_set(("limitations",), []), "limitations"),
    (lambda r: r["limitations"].append("see https://example.com"), "URLs or local paths"),
    (lambda r: r["limitations"].append("C:/Users/someone/data"), "URLs or local paths"),
]


@pytest.mark.parametrize(("change", "message"), INVALID)
def test_invalid_reports_are_rejected(tmp_path: Path, change, message: str) -> None:
    client = client_for(tmp_path, write(tmp_path, mutated(change)))
    response = client.get("/v1/external-validation/qpu")
    assert response.status_code == 500
    assert message in response.get_json()["detail"]


@pytest.mark.parametrize(("change", "message"), INVALID)
def test_service_and_evaluation_validators_agree(change, message: str) -> None:
    """The service copy of the contract must not drift from the one that writes the report."""
    report = mutated(change)
    with pytest.raises(ResultsInvalid):
        service.validate(report)
    with pytest.raises(evaluation.ExternalQpuError):
        evaluation.validate(report)


def test_validators_share_constants_and_accept_the_published_report() -> None:
    assert service.COMPARED == evaluation.COMPARED
    assert service.NAMES == evaluation.NAMES
    assert service.CLASSES == evaluation.CLASSES
    assert service.HARDWARE_SOURCES == evaluation.HARDWARE_SOURCES
    assert service.RUNTIME_KEYS == evaluation.RUNTIME_KEYS
    assert service.USAGE_KEYS == evaluation.USAGE_KEYS
    assert service.SOURCES == evaluation.SOURCES
    assert service.CALIBRATION_DEVIATION == evaluation.CALIBRATION_DEVIATION
    assert service.BATCH_SIZE == evaluation.BATCH_SIZE
    assert service.FROZEN_REPEATS == evaluation.FROZEN_REPEATS
    assert service.validate(copy.deepcopy(REPORT)) == evaluation.validate(REPORT)


def test_service_module_does_not_import_training_code() -> None:
    """The image ships only app/, src/features/, and src/inference/."""
    tree = ast.parse(Path(service.__file__).read_text(encoding="utf-8"))
    imported = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert all(not name.startswith(("src", "pandas", "numpy", "sklearn")) for name in imported)


def test_docker_image_ships_the_subset_and_chart_but_not_the_evidence() -> None:
    lines = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8").splitlines()
    assert "!reports/" in lines
    assert "!docs/model-comparison.svg" in lines
    assert "reports/qpu-evidence-external/" in lines
    assert not any(line.strip().startswith("reports/external") for line in lines)
    assert "COPY reports/ reports/" in dockerfile
    assert "COPY docs/model-comparison.svg docs/model-comparison.svg" in dockerfile
