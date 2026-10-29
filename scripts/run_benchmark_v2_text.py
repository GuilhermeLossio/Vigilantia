"""Post-hackathon v2 text/Qiskit study. CPU-only Statevector and local Aer."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.benchmark_v2_text.experiment import Experiment, load_config  # noqa: E402
from src.evaluation.benchmark_v2_text.report import publish, report  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("prepare", "classical", "statevector", "shots", "report", "publish", "all"),
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs/benchmark-v2-text.json")
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    if args.run_id is None and args.stage not in {"prepare", "all"}:
        parser.error("--run-id is required to resume or publish an existing run")
    run_id = args.run_id or datetime.now(UTC).strftime("portfolio-%Y%m%d-%H%M%S")
    print(f"Local-only study: {run_id}", flush=True)
    try:
        experiment = Experiment(load_config(args.config), run_id)
        stages = (
            ("prepare", "classical", "statevector", "shots", "report")
            if args.stage == "all"
            else (args.stage,)
        )
        for stage in stages:
            if stage == "prepare":
                experiment.prepare()
            elif stage in {"classical", "statevector"}:
                experiment.train(stage)
            elif stage == "shots":
                experiment.shots()
            elif stage == "report":
                report(experiment)
            else:
                publish(experiment)
    except (ValueError, OSError) as error:
        print(f"Text benchmark stopped: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
