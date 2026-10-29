"""Run isolated CPU-only v2 stages: prepare, classical, quantum, report (or all)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.benchmark_v2.experiment import Experiment, load_config  # noqa: E402
from src.evaluation.benchmark_v2.report import build_report  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "classical", "quantum", "report", "all"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/benchmark-v2.json")
    args = parser.parse_args(argv)
    try:
        experiment = Experiment(load_config(args.config))
        stages = (
            ("prepare", "classical", "quantum", "report") if args.stage == "all" else (args.stage,)
        )
        for stage in stages:
            if stage == "prepare":
                experiment.prepare()
            elif stage == "report":
                report = build_report(experiment)
                print(f"report: {report['conclusion']} (internal test only)", flush=True)
            else:
                experiment.train(stage)
    except (ValueError, OSError) as error:
        print(f"Benchmark v2 stopped: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
