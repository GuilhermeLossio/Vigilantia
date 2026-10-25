"""Publish reports/benchmark-replay.json: recorded quantum-kernel SVM test predictions.

Combines the frozen test rows (configs/benchmark-split.json), the simulated evaluation, and
the latest full QPU run, or the run given with --run. Local only; no IBM call.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import replay  # noqa: E402
from src.evaluation.inputs import ARTIFACT_DIR, MANIFEST_PATH  # noqa: E402
from src.quantum.qpu_run import RUNS_DIR  # noqa: E402

OUTPUT = ROOT / "reports" / "benchmark-replay.json"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def qpu_run(run_id: str | None) -> dict | None:
    if run_id is None:
        candidates = sorted(RUNS_DIR.glob("full-*/evaluation.json"))
        if not candidates:
            return None
        run_dir = candidates[-1].parent
    else:
        run_dir = RUNS_DIR / run_id
    record, evaluation = read(run_dir / "run.json"), read(run_dir / "evaluation.json")
    return {
        "run_id": record["run_id"],
        "backend": record["backend"],
        "inputs": record["inputs"],
        "evaluations": evaluation["evaluations"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", help="full QPU run ID; defaults to the latest full run")
    args = parser.parse_args()
    try:
        data = replay.build(
            read(MANIFEST_PATH),
            read(ARTIFACT_DIR / "evaluation-simulation.json"),
            qpu_run(args.run),
        )
    except (FileNotFoundError, KeyError, ValueError) as error:
        print(f"Error: {error}")
        return 1
    replay.write(data, OUTPUT)
    sources = sorted(next(iter(data["samples"]))["predictions"])
    print(f"Wrote {OUTPUT.relative_to(ROOT)}: {len(data['samples'])} samples, sources {sources}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
