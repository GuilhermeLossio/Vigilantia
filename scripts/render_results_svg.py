"""Render docs/benchmark-results.svg from the recorded benchmark results.

Inputs (each optional except the first; a missing one is drawn as pending):
- reports/benchmark-results.json: published test metrics per model and source.
- artifacts/benchmark/rehearsal-fake_fez/summary.json: noisy-simulation kernel error.
- artifacts/benchmark/qpu-runs/*/evaluation.json: latest pilot and full IBM Quantum runs.

Colors encode the source of each number: gray for classical models, blue for simulation,
orange for IBM Quantum hardware. Projected values use a hatch and are labeled "projected".
"""

from __future__ import annotations

import json
import sys
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reports" / "benchmark-results.json"
REHEARSAL = ROOT / "artifacts" / "benchmark" / "rehearsal-fake_fez" / "summary.json"
RUNS = ROOT / "artifacts" / "benchmark" / "qpu-runs"
OUTPUT = ROOT / "docs" / "benchmark-results.svg"
QPU_LIMIT_SECONDS = 600

WIDTH = 960
LEFT = 300  # start of the plot area; labels sit to its left
PLOT = 520  # plot width
BAR = 18
ROW = 30

STYLE = """
  .surface { fill: var(--surface); }
  .ink { fill: var(--text-primary); }
  .ink2 { fill: var(--text-secondary); }
  .muted { fill: var(--text-muted); }
  .grid { stroke: var(--grid); stroke-width: 1; }
  .ref { stroke: var(--text-secondary); stroke-width: 1.5; stroke-dasharray: 4 3; }
  .classical { fill: var(--neutral); }
  .simulation { fill: var(--series-1); }
  .hardware { fill: var(--series-2); }
  .hatch-line { stroke: var(--series-2); stroke-width: 2; }
  .projected { fill: url(#hatch); stroke: var(--series-2); stroke-width: 1.5; }
  text { font-family: "Segoe UI", Helvetica, Arial, sans-serif; }
  svg {
    --surface: #fcfcfb; --text-primary: #0b0b0b; --text-secondary: #52514e;
    --text-muted: #6f6e69; --grid: #e4e3df; --neutral: #9a9994;
    --series-1: #2a78d6; --series-2: #eb6834;
  }
  @media (prefers-color-scheme: dark) {
    svg {
      --surface: #1a1a19; --text-primary: #f0efec; --text-secondary: #c3c2b7;
      --text-muted: #a3a29b; --grid: #383835; --neutral: #7d7c76;
      --series-1: #3987e5; --series-2: #d95926;
    }
  }
"""


def load_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def latest_run(kind: str) -> dict | None:
    paths = sorted(RUNS.glob(f"{kind}-*/evaluation.json"))
    return load_json(paths[-1]) if paths else None


def bar(x: float, y: float, width: float, css: str) -> str:
    """Horizontal bar anchored at the baseline with a 4px rounded data end."""
    if width <= 0:
        return ""
    r = min(4.0, width / 2, BAR / 2)
    return (
        f'<path class="{css}" d="M{x:.1f},{y:.1f} h{width - r:.1f} '
        f"a{r},{r} 0 0 1 {r},{r} v{BAR - 2 * r:.1f} a{r},{r} 0 0 1 -{r},{r} "
        f'h-{width - r:.1f} z"/>'
    )


def text(
    x: float,
    y: float,
    value: str,
    css: str = "ink",
    size: int = 13,
    anchor: str = "start",
    weight: str = "normal",
) -> str:
    return (
        f'<text class="{css}" x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
        f'text-anchor="{anchor}" font-weight="{weight}">{escape(value)}</text>'
    )


def axis(y_top: float, y_bottom: float, ticks: list[float], scale, fmt) -> list[str]:
    parts = []
    for tick in ticks:
        x = LEFT + scale(tick)
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="{y_top}" x2="{x:.1f}" y2="{y_bottom}"/>')
        parts.append(text(x, y_bottom + 16, fmt(tick), "muted", 11, "middle"))
    return parts


def panel_accuracy(results: dict, y: float) -> tuple[list[str], float]:
    parts = [
        text(40, y, "Test accuracy by model", "ink", 17, weight="600"),
        text(
            40,
            y + 20,
            f"{results['split']['test']} test examples, run {results['run_id']}. "
            "Dashed line: chance level (0.5).",
            "ink2",
            12,
        ),
    ]
    rows = [m for m in results["models"]]
    if not any(m["kernel_source"] in {"qpu", "qpu_calibrated"} for m in rows):
        rows += [
            {"name": "Quantum-kernel SVM", "kernel_source": "qpu", "pending": True},
            {"name": "Quantum-kernel SVM", "kernel_source": "qpu_calibrated", "pending": True},
        ]
    source_label = {
        "classical": "classical",
        "simulation": "exact simulation",
        "qpu": "IBM QPU, uncorrected",
        "qpu_calibrated": "IBM QPU, calibrated",
    }
    css = {
        "classical": "classical",
        "simulation": "simulation",
        "qpu": "hardware",
        "qpu_calibrated": "hardware",
    }
    top = y + 40
    bottom = top + len(rows) * ROW
    scale = lambda v: v * PLOT  # noqa: E731
    parts += axis(top - 4, bottom, [0, 0.25, 0.5, 0.75, 1.0], scale, lambda v: f"{v:.2f}")
    for i, model in enumerate(rows):
        row_y = top + i * ROW
        parts.append(text(LEFT - 12, row_y + 13, model["name"], "ink", 13, "end"))
        parts.append(
            text(LEFT - 12, row_y + 26, source_label[model["kernel_source"]], "muted", 10, "end")
        )
        if model.get("pending"):
            parts.append(
                text(LEFT + 6, row_y + 14, "pending: full QPU run not yet evaluated", "muted", 12)
            )
            continue
        accuracy = model["metrics"]["accuracy"]
        cm = model["confusion_matrix"]
        correct, total = cm[0][0] + cm[1][1], sum(map(sum, cm))
        parts.append(bar(LEFT, row_y, scale(accuracy), css[model["kernel_source"]]))
        parts.append(
            text(
                LEFT + scale(accuracy) + 8,
                row_y + 14,
                f"{accuracy:.3f}  ({correct}/{total} correct)",
                "ink",
                12,
            )
        )
    chance = LEFT + scale(0.5)
    parts.append(
        f'<line class="ref" x1="{chance:.1f}" y1="{top - 8}" x2="{chance:.1f}" y2="{bottom}"/>'
    )
    return parts, bottom + 40


def panel_kernel_error(
    rehearsal: dict | None, pilot: dict | None, full: dict | None, y: float
) -> tuple[list[str], float]:
    parts = [
        text(40, y, "Kernel error against the exact simulation", "ink", 17, weight="600"),
        text(
            40,
            y + 20,
            "Mean absolute difference of measured kernel values, 1,024 shots per circuit.",
            "ink2",
            12,
        ),
    ]
    rows = []
    if rehearsal:
        errors = rehearsal["mean_abs_kernel_error_vs_ideal"]
        rows.append(
            (
                "FakeFez noisy simulation, uncorrected",
                "336 circuits",
                errors["noisy_raw"],
                "simulation",
            )
        )
        if "noisy_calibrated" in errors:
            rows.append(
                (
                    "FakeFez noisy simulation, calibrated",
                    "336 circuits",
                    errors["noisy_calibrated"],
                    "simulation",
                )
            )
    if pilot:
        rows.append(
            (
                f"{pilot['backend']} pilot, uncorrected",
                f"{pilot['circuits']} circuits",
                pilot["mean_abs_kernel_error_vs_ideal"],
                "hardware",
            )
        )
    if full:
        for key, label in (("qpu", "uncorrected"), ("qpu_calibrated", "calibrated")):
            if key in full.get("test_kernel_error_vs_ideal", {}):
                rows.append(
                    (
                        f"{full['backend']} full run, {label}",
                        "288 test pairs",
                        full["test_kernel_error_vs_ideal"][key],
                        "hardware",
                    )
                )
    else:
        rows.append(("Full QPU run", "pending", None, "hardware"))
    top = y + 40
    bottom = top + len(rows) * ROW
    peak = max([value for *_, value, _ in rows if value is not None] + [0.04])
    limit = 0.01 * (int(peak / 0.01) + 1)
    scale = lambda v: v / limit * PLOT  # noqa: E731
    ticks = [round(0.01 * k, 2) for k in range(round(limit / 0.01) + 1)]
    parts += axis(top - 4, bottom, ticks, scale, lambda v: f"{v:.2f}")
    for i, (label, detail, value, css) in enumerate(rows):
        row_y = top + i * ROW
        parts.append(text(LEFT - 12, row_y + 13, label, "ink", 13, "end"))
        parts.append(text(LEFT - 12, row_y + 26, detail, "muted", 10, "end"))
        if value is None:
            parts.append(text(LEFT + 6, row_y + 14, "pending", "muted", 12))
            continue
        parts.append(bar(LEFT, row_y, scale(value), css))
        parts.append(text(LEFT + scale(value) + 8, row_y + 14, f"{value:.3f}", "ink", 12))
    return parts, bottom + 40


def panel_budget(pilot: dict | None, full: dict | None, y: float) -> tuple[list[str], float]:
    parts = [
        text(40, y, "IBM Quantum time budget", "ink", 17, weight="600"),
        text(
            40,
            y + 20,
            f"Open Plan limit: {QPU_LIMIT_SECONDS} s of QPU time per rolling 28 days. "
            "Hatched bars are projections from the pilot.",
            "ink2",
            12,
        ),
    ]
    rows = []
    if pilot:
        usage = pilot["usage"]
        rows.append(
            (
                "Pilot, billed",
                f"{pilot['circuits']} circuits, measured",
                usage["charged_seconds"],
                "hardware",
            )
        )
        projection = pilot.get("projected_full_run", {})
        if not full:
            rows.append(
                (
                    "Full run, execution time",
                    "336 circuits, projected",
                    projection.get("execution_seconds"),
                    "projected",
                )
            )
            rows.append(
                (
                    "Full run, billed upper bound",
                    "336 circuits, projected",
                    projection.get("charged_seconds"),
                    "projected",
                )
            )
    if full:
        rows.append(
            (
                "Full run, billed",
                f"{full['circuits']} circuits, measured",
                full["usage"]["charged_seconds"],
                "hardware",
            )
        )
    if not rows:
        rows.append(("No QPU usage recorded", "pending", None, "hardware"))
    top = y + 40
    bottom = top + len(rows) * ROW
    scale = lambda v: v / QPU_LIMIT_SECONDS * PLOT  # noqa: E731
    parts += axis(top - 4, bottom, [0, 150, 300, 450, 600], scale, lambda v: f"{v:.0f} s")
    for i, (label, detail, value, css) in enumerate(rows):
        row_y = top + i * ROW
        parts.append(text(LEFT - 12, row_y + 13, label, "ink", 13, "end"))
        parts.append(text(LEFT - 12, row_y + 26, detail, "muted", 10, "end"))
        if value is None:
            parts.append(text(LEFT + 6, row_y + 14, "unavailable", "muted", 12))
            continue
        parts.append(bar(LEFT, row_y, max(scale(value), 3), css))
        prefix = "about " if css == "projected" else ""
        parts.append(
            text(LEFT + max(scale(value), 3) + 8, row_y + 14, f"{prefix}{value:.0f} s", "ink", 12)
        )
    limit_x = LEFT + scale(QPU_LIMIT_SECONDS)
    parts.append(
        f'<line class="ref" x1="{limit_x:.1f}" y1="{top - 8}" x2="{limit_x:.1f}" y2="{bottom}"/>'
    )
    return parts, bottom + 40


def legend(y: float) -> list[str]:
    items = [
        ("classical", "Classical model"),
        ("simulation", "Simulation"),
        ("hardware", "IBM Quantum hardware"),
        ("projected", "Projected"),
    ]
    parts, x = [], 40
    for css, label in items:
        parts.append(f'<rect class="{css}" x="{x}" y="{y - 10}" width="14" height="12" rx="2"/>')
        parts.append(text(x + 20, y, label, "ink2", 12))
        x += 30 + len(label) * 7
    return parts


def render() -> str:
    results = load_json(RESULTS)
    if results is None:
        raise SystemExit("reports/benchmark-results.json is missing; run evaluate_benchmark.py.")
    rehearsal, pilot, full = load_json(REHEARSAL), latest_run("pilot"), latest_run("full")
    body = [
        text(40, 42, "Vigilantia benchmark: phishing URL classification", "ink", 22, weight="600"),
        text(
            40,
            64,
            "Classical baselines and a hybrid quantum-kernel SVM (classical training, "
            "quantum classification). Exploratory: one split, 24 training / 12 test examples.",
            "ink2",
            12,
        ),
    ]
    body += legend(92)
    y = 132
    for build in (
        lambda y: panel_accuracy(results, y),
        lambda y: panel_kernel_error(rehearsal, pilot, full, y),
        lambda y: panel_budget(pilot, full, y),
    ):
        parts, y = build(y)
        body += parts
    body.append(
        text(
            40,
            y,
            "No quantum advantage is claimed. With 12 test examples, each example "
            "changes accuracy by about 8.3 percentage points.",
            "muted",
            11,
        )
    )
    height = y + 24
    description = (
        "Bar charts of test accuracy per model, measured kernel error against the exact "
        "simulation, and IBM Quantum time used and projected against the Open Plan limit."
    )
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {height:.0f}" '
            f'width="{WIDTH}" height="{height:.0f}" role="img" aria-labelledby="title desc">',
            '<title id="title">Vigilantia benchmark results</title>',
            f'<desc id="desc">{escape(description)}</desc>',
            f"<style>{STYLE}</style>",
            '<defs><pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" '
            'patternTransform="rotate(45)"><line class="hatch-line" x1="0" y1="0" x2="0" y2="6"/>'
            "</pattern></defs>",
            f'<rect class="surface" width="{WIDTH}" height="{height:.0f}" rx="12"/>',
            *body,
            "</svg>",
        ]
    )


def main() -> int:
    OUTPUT.write_text(render() + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
