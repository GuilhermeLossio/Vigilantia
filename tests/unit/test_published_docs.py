"""The documentation quotes the published hardware subset, charts, and routes as they are."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app import create_app
from src.quantum import qpu_run

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
SUBSET = json.loads((ROOT / "reports" / "external-qpu-subset.json").read_text(encoding="utf-8"))
EXTERNAL = json.loads((ROOT / "reports" / "external-validation.json").read_text(encoding="utf-8"))
RUN_ID = SUBSET["hardware"]["run_id"]
OPEN_PLAN_SECONDS = 600
LABELS = {
    "classical": "classical",
    "simulation": "exact simulation",
    "qpu": "`ibm_fez`, uncorrected",
    "qpu_calibrated": "`ibm_fez`, calibrated",
}


def read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def interval(row: dict, metric: str) -> str:
    low, high = row["confidence_intervals"][metric]
    return f"{row['metrics'][metric]:.3f} ({low:.3f}–{high:.3f})"


def cells(row: dict) -> str:
    """TN / FP / FN / TP, as the README tables write a confusion matrix."""
    (tn, fp), (fn, tp) = row["confusion_matrix"]
    return f"{tn} / {fp} / {fn} / {tp}"


def row_id(row: dict) -> str:
    return f"{row['model']}-{row['kernel_source']}"


def evidence_usage() -> list[dict]:
    """IBM usage of every recorded run, from the versioned evidence."""
    records = sorted(ROOT.glob("reports/qpu-evidence*/*/run.json"))
    assert records
    return [qpu_run.usage_totals(json.loads(path.read_text(encoding="utf-8"))) for path in records]


@pytest.mark.parametrize("row", SUBSET["models"], ids=row_id)
def test_external_validation_guide_quotes_every_subset_row(row: dict) -> None:
    guide = read("docs/external-validation.md")
    line = f"| {row['name']} | {LABELS[row['kernel_source']]} | {interval(row, 'accuracy')} |"
    assert line in guide
    if row["metrics"]["average_precision"] is not None:
        assert f"| {interval(row, 'average_precision')} |" in guide
    assert json.dumps(row["confusion_matrix"]) in guide


@pytest.mark.parametrize("row", SUBSET["models"], ids=row_id)
def test_readme_quotes_every_subset_row(row: dict) -> None:
    readme = read("README.md")
    label = LABELS[row["kernel_source"]]
    label = label[0].upper() + label[1:] if label[0] != "`" else label
    assert f"| {row['name']} | {label} | {interval(row, 'accuracy')} | {cells(row)} |" in readme


def test_kernel_and_usage_measures_match_the_report() -> None:
    kernel, usage = SUBSET["kernel"], SUBSET["runtime"]["qpu"]
    errors = kernel["test_kernel_error_vs_ideal"]
    calibration = kernel["calibration"]
    quoted = [
        f"{errors['qpu']:.3f} uncorrected, {errors['qpu_calibrated']:.3f} calibrated",
        f"a = {calibration['a']:.3f}, c = {calibration['c']:.4f}, RMSE {calibration['rmse']:.3f}",
        f"{calibration['clipped_test_values']} of the {SUBSET['hardware']['plan']['test']}",
        f"mean {kernel['diagonal_controls_mean']:.3f}",
        f"{usage['execution_seconds']:.1f} s",
        f"{usage['charged_seconds']:.0f} s",
        f"{usage['resource_units']:.2f} resource units",
    ]
    guide = read("docs/external-validation.md")
    assert [text for text in quoted if text not in guide] == []
    for name in ("README.md", "docs/hardware-runs.md"):
        assert f"{usage['execution_seconds']:.1f} s" in read(name)


def test_full_external_set_comparison_matches_its_report() -> None:
    rows = {row["model"]: row["metrics"]["accuracy"] for row in EXTERNAL["models"]}
    expected = f"reach {rows['logistic_regression']:.3f} and {rows['rbf_svm']:.3f} accuracy"
    assert expected in read("docs/external-validation.md")


def test_hardware_runs_marks_the_subset_complete_with_cumulative_usage() -> None:
    guide = read("docs/hardware-runs.md")
    section = guide.split("## External subset (40 rows)", 1)[1].split("\n## ", 1)[0]
    assert "**Status: complete.**" in section
    assert "Status: planned" not in guide
    assert f"`{RUN_ID}`" in section
    usage = evidence_usage()
    execution = sum(run["execution_seconds"] for run in usage)
    charged = sum(run["charged_seconds"] for run in usage)
    units = sum(run["resource_units"] for run in usage)
    circuits = sum(
        job["stop"] - job["start"]
        for path in sorted(ROOT.glob("reports/qpu-evidence*/*/run.json"))
        for job in json.loads(path.read_text(encoding="utf-8"))["jobs"]
    )
    total = f"| **Cumulative** | **{circuits:,}** | **{execution:.1f} s** | **{charged:.0f} s** |"
    assert f"{total} **{units:.2f}** |" in guide
    assert f"about {OPEN_PLAN_SECONDS - charged:.0f} s" in guide
    assert "reports/qpu-evidence-external/" in guide


def test_readme_embeds_the_comparison_chart_and_claims_no_advantage() -> None:
    readme = read("README.md")
    assert "![Accuracy by model on the benchmark test set" in readme
    assert "runtime on the benchmark test set and the subset](docs/model-comparison.svg)" in readme
    assert "The full external set has no published runtime." in readme
    assert "time to classify each set" not in readme
    assert (ROOT / "docs" / "model-comparison.svg").is_file()
    assert "**No quantum advantage is claimed.**" in readme


def test_reproduction_guide_covers_the_subset_and_the_chart_script() -> None:
    guide = read("docs/reproducing.md")
    assert f"scripts/run_qpu_external.py evaluate {RUN_ID}" in guide
    assert f"reports/qpu-evidence-external/{RUN_ID}/" in guide
    assert "scripts/render_comparison_svg.py" in guide
    assert "01 to 07 in order" in guide


def test_reports_guide_documents_every_subset_field() -> None:
    section = read("reports/README.md").split("## External hardware subset file", 1)[1]
    section = section.split("\n## ", 1)[0]
    fields = set(SUBSET) | set(SUBSET["models"][0])
    assert [key for key in sorted(fields) if f"`{key}`" not in section] == []
    assert "reports/qpu-evidence-external/" in section


def test_service_guide_lists_every_route() -> None:
    guide = read("docs/service.md")
    routes = {rule.rule for rule in create_app().url_map.iter_rules()} - {"/static/<path:filename>"}
    assert [route for route in sorted(routes) if f"| `{route}` |" not in guide] == []
    assert "`VIGILANTIA_EXTERNAL_QPU_PATH`" in guide
    assert "`VIGILANTIA_COMPARISON_CHART_PATH`" in guide


def test_architecture_tree_lists_the_repository_files() -> None:
    tree = read("Architecture.md").split("## Repository structure", 1)[1].split("```", 2)[1]
    patterns = (
        "app/*.py",
        "configs/*.json",
        "docs/*.md",
        "docs/*.svg",
        "notebooks/*.ipynb",
        "reports/*.json",
        "scripts/*.py",
        "src/*/*.py",
        "tests/unit/test_*.py",
    )
    files = [path for pattern in patterns for path in ROOT.glob(pattern)]
    names = {path.name for path in files if path.name != "__init__.py"} | {"__init__.py"}
    names |= {path.parent.name + "/" for path in ROOT.glob("reports/qpu-evidence*/*/run.json")}
    assert [name for name in sorted(names) if name not in tree] == []


METRICS = ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "average_precision")


def metric_cells(row: dict) -> str:
    """Every metric cell of a guide table row, with its interval wherever one is published."""
    values = []
    for metric in METRICS:
        value = row["metrics"][metric]
        bounds = (row.get("confidence_intervals") or {}).get(metric)
        if value is None:
            values.append("n/a")
        elif bounds is None:
            values.append(f"{value:.3f}")
        else:
            values.append(interval(row, metric))
    return " | ".join(values)


@pytest.mark.parametrize("report", [SUBSET, EXTERNAL], ids=["subset", "primary"])
def test_external_validation_tables_show_every_published_interval(report: dict) -> None:
    guide = read("docs/external-validation.md")
    for row in report["models"]:
        assert f"| {LABELS[row['kernel_source']]} | {metric_cells(row)} |" in guide, row["name"]
    majority = next(r for r in report["models"] if r["model"] == "majority")
    assert "| 0.000 (0.000–0.000) | 0.000 (0.000–0.000) | 0.000 (0.000–0.000) |" in (
        f"| {metric_cells(majority)} |"
    )


def test_chart_descriptions_match_what_the_chart_draws() -> None:
    guide = read("docs/external-validation.md")
    assert "runtime for the benchmark and this subset" in guide
    assert "accuracy and runtime for the benchmark, the full external set" not in guide
    catalog = read("CATALOG.md")
    assert "runtime to classify the benchmark test set and the 40-row subset" in catalog
    assert "runtime to classify each set" not in catalog


def test_subset_gap_is_described_as_sampling_variation_not_difficulty() -> None:
    for name in ("README.md", "docs/external-validation.md"):
        assert "subset is harder" not in read(name)
    guide = read("docs/external-validation.md")
    assert "consistent with sampling variation" in guide


def test_benchmark_kernel_errors_cite_their_source() -> None:
    guide = read("docs/external-validation.md")
    line = next(x for x in guide.splitlines() if x.startswith("| Test kernel error vs exact"))
    assert "benchmark-protocol.md#ibm-quantum-hardware-full-run" in line
    assert "notebook 04" in line
    assert "### IBM Quantum hardware: full run" in read("docs/benchmark-protocol.md")
    assert "notebook 04 rebuilds" in read("docs/hardware-runs.md")


def test_quota_table_units_are_plain_and_totals_explained() -> None:
    guide = read("docs/hardware-runs.md")
    table = guide.split("| Run | Circuits |", 1)[1].split("\n\n", 1)[0]
    assert " of 10" not in table
    assert "summed from the unrounded recorded usage" in guide


def test_roadmap_leaves_only_submission_and_later_work_open() -> None:
    roadmap = (ROOT / "docs" / "roadmap.md").read_text(encoding="utf-8")
    section = None
    for line in roadmap.splitlines():
        if line.startswith("#"):
            section = line.lstrip("#").strip()
        elif line.startswith("- [ ]"):
            assert section in {"8. Submission", "Research", "Product"}, (section, line)
