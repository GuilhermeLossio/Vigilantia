"""Frozen v2 page-kernel QPU protocol, input rebuilding, evaluation and script; no network."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from qiskit_ibm_runtime.fake_provider import FakeManilaV2

from scripts import run_qpu_v2 as script
from src.evaluation import benchmark_v2_qpu as v2qpu
from src.evaluation import external, external_qpu
from src.evaluation.benchmark_v2 import core, experiment
from src.quantum import hardware, qpu_run

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]
SEARCH = {"angle_maxima": [np.pi / 4], "quantum_c": [1], "folds": 2}


def page_rows(labels: np.ndarray, prefix: str, rng) -> pd.DataFrame:
    rows = pd.DataFrame(
        {
            "row_index": np.arange(len(labels)),
            "group_hash": [f"{prefix}{i:015x}"[-16:] for i in range(len(labels))],
            "is_phishing": labels,
        }
    )
    for name in core.CANDIDATE_FEATURES:
        rows[name] = rng.uniform(1, 5, len(labels))
    for name in core.PAGE_FEATURES:
        # Legitimate pages are larger, as in PhiUSIIL.
        rows[name] = rng.poisson(20, len(labels)) + 400 * (1 - labels)
    return rows


def write_run(runs: Path, run_id: str, bundle: dict, groups: list[str]) -> None:
    with (runs / f"{run_id}.pkl").open("wb") as stream:
        import pickle

        pickle.dump(bundle, stream)
    core.write_json(
        runs / f"{run_id}.json",
        {
            "id": run_id,
            "status": "complete",
            "training_group_hashes": groups,
            "artifact_hashes": {f"{run_id}.pkl": core.digest(runs / f"{run_id}.pkl")},
        },
    )


@pytest.fixture
def frozen(tmp_path):
    """A miniature v2 output with an 8-reference zz model and a 6-row subset protocol."""
    rng = np.random.default_rng(11)
    output = tmp_path / "artifacts/benchmark-v2"
    runs = output / "runs"
    runs.mkdir(parents=True)
    tables = {
        "development": page_rows(np.tile([0, 1], 10), "d", rng),
        "calibration": page_rows(np.tile([0, 1], 4), "c", rng),
        "test": page_rows(np.tile([0, 1], 6), "e", rng),
    }
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
    train = tables["development"].iloc[:8]
    x, y = core.matrix(train, "page"), train.is_phishing.to_numpy()
    quantum, *_ = experiment.search_quantum(x, y, "zz", 42, SEARCH)
    quantum.update(family="quantum", features="page", threshold=0.0)
    write_run(runs, "quantum-zz-page-8-42", quantum, list(train.group_hash))
    classical_runs = []
    for name in ("logistic_regression", "rbf_svm", "random_forest"):
        cfg = {
            "grids": {
                "logistic_regression": {"C": [1]},
                "rbf_svm": {"C": [1], "gamma": ["scale"]},
                "random_forest": {"max_depth": [None], "min_samples_leaf": [1]},
            },
            "folds": 2,
            "forest_trees": 3,
            "n_jobs": 1,
        }
        model, *_ = experiment.search_classical(name, x, y, 42, cfg)
        run_id = f"classical-{name}-page-8-42"
        write_run(
            runs,
            run_id,
            {"model": model, "family": "classical", "features": "page", "threshold": 0.5},
            [],
        )
        classical_runs.append(run_id)
    config = tmp_path / "configs/benchmark-v2.json"
    config.parent.mkdir(parents=True)
    config.write_text("{}\n", encoding="utf-8")
    protocol = copy.deepcopy(v2qpu.load_protocol())
    protocol["benchmark"].update(
        config_sha256=external.protocol_sha256(config),
        test_partition_sha256=core.digest(output / "test.csv"),
        quantum_run="quantum-zz-page-8-42",
        quantum_bundle_sha256=core.digest(runs / "quantum-zz-page-8-42.pkl"),
        classical_runs=classical_runs,
    )
    protocol["subset"].update(size=6, per_class={"legitimate": 3, "phishing": 3})
    protocol["hardware"]["circuits"] = len(hardware.plan_pairs(8, 6))
    protocol["metrics"]["confidence_intervals"]["resamples"] = 50
    path = tmp_path / "configs/benchmark-v2-qpu-subset.json"
    core.write_json(path, protocol)
    return tmp_path, path


def exact_counts(probabilities: list[float]) -> list[dict]:
    """Counts whose all-zero fraction equals each ideal kernel value to 1/1024."""
    counts = []
    for p in probabilities:
        zeros = int(round(p * hardware.SHOTS))
        counts.append({"0000": zeros, "0001": hardware.SHOTS - zeros})
    return counts


def retrieved_record(loaded: v2qpu.Inputs, run_dir: Path, measured=None) -> dict:
    train_states = core.states(loaded.train_angles, "zz")
    ideal_train = core.kernel(train_states, train_states)
    ideal_test = core.kernel(core.states(loaded.test_angles, "zz"), train_states)
    values = measured or [
        ideal_test[s.left, s.right]
        if s.kind == "test"
        else 1.0
        if s.kind == "diagonal"
        else ideal_train[s.left, s.right]
        for s in loaded.plan
    ]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "counts-job1.json").write_text(json.dumps(exact_counts(values)), encoding="utf-8")
    _, record = qpu_run.new_record(
        None,
        "fake",
        loaded.plan,
        False,
        {},
        len(loaded.plan),
        inputs=loaded.inputs,
        prefix="v2page",
    )
    record["status"] = "retrieved"
    record["jobs"] = [{"job_id": "job1", "start": 0, "stop": len(loaded.plan), "status": "DONE"}]
    return record


def test_committed_protocol_matches_configuration_and_tooling():
    protocol = v2qpu.load_protocol()
    assert protocol["hardware"]["circuits"] == len(hardware.plan_pairs(24, 40)) == 1008
    assert protocol["benchmark"]["quantum_run"] == "quantum-zz-page-24-42"
    assert protocol["subset"]["size"] == 40


def test_protocol_refuses_changed_configuration_and_rows(frozen, tmp_path):
    root, path = frozen
    (root / "configs/benchmark-v2.json").write_text('{"changed": 1}\n', encoding="utf-8")
    with pytest.raises(v2qpu.V2QpuError, match="configuration differs"):
        v2qpu.load_protocol(path, root)
    protocol = json.loads(v2qpu.PROTOCOL_PATH.read_text(encoding="utf-8"))
    protocol["compared"] = protocol["compared"][:-1]
    changed = tmp_path / "changed.json"
    core.write_json(changed, protocol)
    with pytest.raises(v2qpu.V2QpuError, match="compared rows"):
        v2qpu.load_protocol(changed, ROOT)


def test_subset_draw_is_deterministic_balanced_and_sorted():
    labels = np.tile([0, 1, 1], 30)
    first = v2qpu.select_subset(labels, {"legitimate": 5, "phishing": 5}, 42)
    assert np.array_equal(first, v2qpu.select_subset(labels, {"legitimate": 5, "phishing": 5}, 42))
    assert np.all(np.diff(first) > 0)
    assert labels[first].sum() == 5
    with pytest.raises(v2qpu.V2QpuError, match="Too few"):
        v2qpu.select_subset(labels, {"legitimate": 31, "phishing": 1}, 42)


def test_inputs_rebuild_bundle_states_and_keep_domains_disjoint(frozen):
    root, path = frozen
    loaded = v2qpu.load_inputs(v2qpu.load_protocol(path, root), root, path)
    assert loaded.train_angles.shape == (8, 4) and loaded.test_angles.shape == (6, 4)
    np.testing.assert_allclose(core.states(loaded.train_angles, "zz"), loaded.bundle["states"])
    assert loaded.labels.sum() == 3
    assert set(loaded.classical) == {"logistic_regression", "rbf_svm", "random_forest"}
    assert loaded.inputs["subset_group_hash"] == list(loaded.rows.group_hash)


def test_inputs_refuse_changed_bundle_partition_or_shared_domain(frozen):
    root, path = frozen
    protocol = v2qpu.load_protocol(path, root)
    output = root / "artifacts/benchmark-v2"
    test = pd.read_csv(output / "test.csv")
    shared = test.copy()
    shared["group_hash"] = pd.read_csv(output / "development.csv").group_hash[: len(test)]
    shared.to_csv(output / "test.csv", index=False)
    changed = copy.deepcopy(protocol)
    changed["benchmark"]["test_partition_sha256"] = core.digest(output / "test.csv")
    with pytest.raises(v2qpu.V2QpuError, match="share domains"):
        v2qpu.load_inputs(changed, root, path)
    with pytest.raises(v2qpu.V2QpuError, match="test partition differs"):
        v2qpu.load_inputs(protocol, root, path)
    test.to_csv(output / "test.csv", index=False)
    (output / "runs/quantum-zz-page-8-42.pkl").write_bytes(b"corrupted")
    with pytest.raises(v2qpu.V2QpuError, match="changed"):
        v2qpu.load_inputs(protocol, root, path)


def test_exact_counts_reproduce_simulation_and_report_has_no_hosts(frozen):
    root, path = frozen
    loaded = v2qpu.load_inputs(v2qpu.load_protocol(path, root), root, path)
    run_dir = root / "run"
    report = v2qpu.evaluate_run(run_dir, retrieved_record(loaded, run_dir), loaded, "test")
    rows = {(r["model"], r["kernel_source"]): r for r in report["models"]}
    assert [key for key in rows] == list(v2qpu.COMPARED)
    simulation = rows["quantum_kernel_svm", "simulation"]["metrics"]
    assert rows["quantum_kernel_svm", "qpu"]["metrics"]["accuracy"] == simulation["accuracy"]
    assert report["kernel"]["test_kernel_error_vs_ideal"]["qpu"] < 1 / hardware.SHOTS
    assert report["kernel"]["calibration"]["valid"]
    assert report["subset"]["class_counts"] == {"legitimate": 3, "phishing": 3}
    assert '"host":' not in json.dumps(report)


def test_evaluation_refuses_other_inputs_and_reports_invalid_calibration(frozen):
    root, path = frozen
    loaded = v2qpu.load_inputs(v2qpu.load_protocol(path, root), root, path)
    run_dir = root / "flat"
    record = retrieved_record(loaded, run_dir, [0.5] * len(loaded.plan))
    report = v2qpu.evaluate_run(run_dir, record, loaded, "test")
    assert not report["kernel"]["calibration"]["valid"]
    assert ("quantum_kernel_svm", "qpu_calibrated") not in {
        (r["model"], r["kernel_source"]) for r in report["models"]
    }
    record["inputs"] = {**record["inputs"], "subset_row_index": []}
    with pytest.raises(v2qpu.V2QpuError, match="other inputs"):
        v2qpu.evaluate_run(run_dir, record, loaded, "test")


def test_rehearsal_runs_the_plan_on_a_local_fake_backend(frozen):
    root, path = frozen
    loaded = v2qpu.load_inputs(v2qpu.load_protocol(path, root), root, path)
    report = script.rehearse(loaded, FakeManilaV2(), root / "rehearsal")
    assert report["source"] == "noisy_simulation"
    assert report["hardware"]["circuits"] == len(loaded.plan)
    assert (root / "rehearsal/summary.json").exists()


def test_script_guards_names_earlier_runs_and_publication(tmp_path, monkeypatch):
    monkeypatch.setattr(qpu_run, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(v2qpu, "REPORT_PATH", tmp_path / "report.json")
    monkeypatch.setattr(v2qpu, "EVIDENCE_DIR", tmp_path / "evidence")
    with pytest.raises(qpu_run.RunError, match="v2page"):
        script.run_dir_for("external-ibm_fez-1")
    with pytest.raises(qpu_run.RunError):
        script.run_dir_for("v2page-../x")
    assert script.earlier_run() is None
    qpu_run.save(tmp_path / "runs/v2page-ibm_fez-1", {"status": "submitted", "jobs": [{}]})
    assert script.earlier_run() == ("v2page-ibm_fez-1", "submitted")
    assert script.published_conflict("v2page-ibm_fez-1") is None
    v2qpu.write({"hardware": {"run_id": "v2page-ibm_fez-0"}}, tmp_path / "report.json")
    assert "v2page-ibm_fez-0" in script.published_conflict("v2page-ibm_fez-1")


def test_evidence_copy_refuses_credentials(tmp_path, monkeypatch):
    monkeypatch.setattr(v2qpu, "EVIDENCE_DIR", tmp_path / "evidence")
    run_dir = tmp_path / "run"
    record = {"run_id": "v2page-ibm_fez-1", "jobs": [{"job_id": "j"}]}
    run_dir.mkdir()
    (run_dir / "counts-j.json").write_text("[]", encoding="utf-8")
    (run_dir / "run.json").write_text('{"token": "x"}', encoding="utf-8")
    with pytest.raises(v2qpu.V2QpuError, match="not copied"):
        script.copy_evidence(run_dir, record)
    (run_dir / "run.json").write_text('{"ok": 1}', encoding="utf-8")
    assert (script.copy_evidence(run_dir, record) / "counts-j.json").exists()


def test_plan_command_needs_no_ibm_connection(monkeypatch, frozen, capsys):
    root, path = frozen
    loaded = v2qpu.load_inputs(v2qpu.load_protocol(path, root), root, path)
    monkeypatch.setattr(script, "load", lambda rows=None: loaded)
    monkeypatch.setattr(script.ibm, "connect", lambda *_: pytest.fail("Connected to IBM"))
    assert script.main(["plan"]) == 0
    assert "No IBM connection was made." in capsys.readouterr().out


def test_reduced_subset_is_a_prefix_of_the_frozen_draw_per_class():
    labels = np.tile([0, 1, 1], 30)
    per_class = {"legitimate": 6, "phishing": 6}
    full = v2qpu.select_subset(labels, per_class, 42)
    assert np.array_equal(full, v2qpu.select_subset(labels, per_class, 42, take=6))
    smaller = v2qpu.select_subset(labels, per_class, 42, take=2)
    assert set(smaller) < set(full)
    assert labels[smaller].sum() == 2 and len(smaller) == 4
    assert set(smaller) < set(v2qpu.select_subset(labels, per_class, 42, take=4))


@pytest.mark.parametrize("rows", [3, 2, 42, 0])
def test_rows_must_be_even_and_within_the_protocol(rows):
    with pytest.raises(v2qpu.V2QpuError, match="even"):
        v2qpu.check_rows(v2qpu.load_protocol(), rows)


def test_largest_rows_that_fit_the_budget_guard():
    protocol = v2qpu.load_protocol()
    assert v2qpu.max_rows(protocol, 24, 223) == 22
    assert v2qpu.max_rows(protocol, 24, 10_000) == 40
    assert v2qpu.max_rows(protocol, 24, 30) is None
    fits = len(hardware.plan_pairs(24, 22))
    assert external_qpu.required_quota(fits) < 223 < external_qpu.required_quota(fits + 48)


def test_reduced_inputs_plan_fewer_circuits_and_report_the_reduction(frozen):
    root, path = frozen
    protocol = v2qpu.load_protocol(path, root)
    full = v2qpu.load_inputs(protocol, root, path)
    reduced = v2qpu.load_inputs(protocol, root, path, rows=4)
    assert set(reduced.rows.group_hash) < set(full.rows.group_hash)
    assert reduced.labels.sum() == 2
    assert len(reduced.plan) == len(hardware.plan_pairs(8, 4)) < len(full.plan)
    run_dir = root / "reduced"
    report = v2qpu.evaluate_run(run_dir, retrieved_record(reduced, run_dir), reduced, "test")
    assert report["subset"]["size"] == 4
    assert any("reduced from 6 to 4" in text for text in report["limitations"])
    assert report["limitations"][0].startswith("4 rows: each row changes accuracy by 25.0")


def test_rows_option_reaches_inputs(monkeypatch, frozen, capsys):
    root, path = frozen
    seen = []

    def fake_load(rows=None):
        seen.append(rows)
        return v2qpu.load_inputs(v2qpu.load_protocol(path, root), root, path, rows=rows)

    monkeypatch.setattr(script, "load", fake_load)
    assert script.main(["plan", "--rows", "4"]) == 0
    assert seen == [4]
    assert "Subset: 4 test rows" in capsys.readouterr().out


def test_published_hardware_report_matches_evidence_protocol_and_docs():
    report = json.loads((ROOT / "reports/benchmark-v2-qpu-subset.json").read_text(encoding="utf-8"))
    run = report["hardware"]["run_id"]
    evidence = ROOT / "reports/qpu-evidence-v2" / run
    record = json.loads((evidence / "run.json").read_text(encoding="utf-8"))
    # The published file comes from the committed protocol and the recorded run.
    assert report["protocol"]["sha256"] == external.protocol_sha256(v2qpu.PROTOCOL_PATH)
    assert record["inputs"] == report["inputs"] and record["run_id"] == run
    assert record["backend"] == report["hardware"]["backend"] == "ibm_fez"
    rows = len(report["inputs"]["subset_row_index"])
    assert report["subset"]["size"] == rows == 22
    plan = [hardware.PairSpec(**spec) for spec in record["plan"]]
    assert plan == hardware.plan_pairs(24, rows)
    counts = []
    for job in sorted(record["jobs"], key=lambda job: job["start"]):
        assert job["status"] == "DONE"
        counts += json.loads((evidence / f"counts-{job['job_id']}.json").read_text("utf-8"))
    assert len(counts) == len(plan) == report["hardware"]["circuits"] == 576
    assert {sum(c.values()) for c in counts} == {hardware.SHOTS}
    assert [job["job_id"] for job in record["jobs"]] == report["hardware"]["job_ids"]
    diagonal = [
        qpu_run.qkernel.kernel_from_counts(c)
        for s, c in zip(plan, counts, strict=True)
        if s.kind == "diagonal"
    ]
    assert np.mean(diagonal) == pytest.approx(report["kernel"]["diagonal_controls_mean"])
    assert [(r["model"], r["kernel_source"]) for r in report["models"]] == list(v2qpu.COMPARED)
    assert any("reduced from 40 to 22" in text for text in report["limitations"])
    assert '"host":' not in json.dumps(report) and not external_qpu.evidence_refusal(
        {path.name: path.read_text(encoding="utf-8") for path in evidence.iterdir()}
    )
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    guide = (ROOT / "docs/benchmark-v2.md").read_text(encoding="utf-8")
    assert run in readme and run in guide
    for row in report["models"]:
        accuracy = row["metrics"]["accuracy"]
        quoted = "100%" if accuracy == 1 else f"{accuracy:.1%}"
        low, high = row["confidence_intervals"]["accuracy"]
        span = f"{low:.1%}".replace("100.0%", "100") + "–" + f"{high:.1%}".replace(".0%", "%")
        for text in (readme, guide):
            assert quoted in text and span.replace("%–", "–") in text
    assert f"{report['hardware']['usage']['charged_seconds']:.0f} seconds" in readme


def test_evaluation_times_every_row_locally_and_takes_qpu_time_from_ibm(frozen):
    root, path = frozen
    loaded = v2qpu.load_inputs(v2qpu.load_protocol(path, root), root, path)
    run_dir = root / "timed"
    record = retrieved_record(loaded, run_dir)
    record["jobs"][0]["metrics"] = {
        "usage": {"qpu_charge_time_seconds": 9, "unit": "resource_units", "value": 0.1},
        "circuits_execution_time_ns": 7_500_000_000,
    }
    report = v2qpu.evaluate_run(run_dir, record, loaded, "test", repeats=3)
    assert report["runtime"]["repeats"] == 3 and "3 repeats" in report["runtime"]["method"]
    for row in report["models"]:
        times = row["runtime_seconds"]
        assert times["fit"] > 0 and times["predict"] > 0
        if row["kernel_source"] == "classical":
            assert times["test_kernel"] is None
        elif row["kernel_source"] == "simulation":
            assert 0 < times["test_kernel"] < 7.5
        else:
            assert times["test_kernel"] == pytest.approx(7.5)


def test_chart_is_rebuilt_exactly_from_the_published_report():
    from xml.etree import ElementTree

    from scripts import render_v2_qpu_svg as chart

    report = json.loads((ROOT / "reports/benchmark-v2-qpu-subset.json").read_text(encoding="utf-8"))
    rendered = chart.render(report)
    assert (ROOT / "docs/benchmark-v2-qpu.svg").read_text(encoding="utf-8") == rendered + "\n"
    root = ElementTree.fromstring(rendered)
    namespace = "{http://www.w3.org/2000/svg}"
    assert "accuracy and runtime" in root.find(f"{namespace}title").text
    description = root.find(f"{namespace}desc").text
    for row in report["models"]:
        assert f"{row['metrics']['accuracy']:.3f}" in description
    assert "153.1 s on ibm_fez" in rendered and "same jobs as uncorrected" in rendered


def test_chart_marks_rows_missing_from_a_report():
    from scripts import render_v2_qpu_svg as chart

    report = json.loads((ROOT / "reports/benchmark-v2-qpu-subset.json").read_text(encoding="utf-8"))
    report["models"] = [r for r in report["models"] if r["kernel_source"] != "qpu_calibrated"]
    rendered = chart.render(report)
    assert rendered.count("not computed in this run") == 2
    assert "same jobs as uncorrected" not in rendered


def test_readme_and_guide_quote_the_published_runtimes():
    from scripts.render_comparison_svg import format_seconds

    report = json.loads((ROOT / "reports/benchmark-v2-qpu-subset.json").read_text(encoding="utf-8"))
    rows = {(r["model"], r["kernel_source"]): r["runtime_seconds"] for r in report["models"]}
    simulated = rows["quantum_kernel_svm", "simulation"]["test_kernel"]
    hardware_time = rows["quantum_kernel_svm", "qpu"]["test_kernel"]
    for name in ("README.md", "docs/benchmark-v2.md"):
        text = (ROOT / name).read_text(encoding="utf-8")
        for times in rows.values():
            for value in times.values():
                assert value is None or format_seconds(value) in text
        assert f"about {hardware_time / simulated / 1e6:.1f} million times longer" in text
        assert "benchmark-v2-qpu.svg" in text
