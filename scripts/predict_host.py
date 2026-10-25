"""Classify hosts or URLs with the served classical model. Nothing is fetched or visited.

Usage: python scripts/predict_host.py [--model rbf_svm|logistic_regression] HOST_OR_URL ...
Prints one JSON object per input. Invalid inputs are reported and do not stop the others.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inference.predictor import (  # noqa: E402
    LIVE_MODELS,
    SERVED_MODEL,
    PredictionError,
    Predictor,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("inputs", nargs="+", metavar="HOST_OR_URL")
    parser.add_argument("--model", choices=LIVE_MODELS, default=SERVED_MODEL)
    args = parser.parse_args()
    try:
        predictor = Predictor.from_artifacts(args.model)
    except PredictionError as error:
        print(f"Error: {error}")
        return 1
    failures = 0
    for value in args.inputs:
        try:
            print(json.dumps(predictor.predict(value), indent=2))
        except PredictionError as error:
            failures += 1
            print(json.dumps({"input": value, "error": str(error)}))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
