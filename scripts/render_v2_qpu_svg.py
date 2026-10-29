"""Render docs/benchmark-v2-qpu.svg: accuracy and runtime of the v2 page-kernel hardware check.

Input: reports/benchmark-v2-qpu-subset.json. Every number in the chart is read from it. The
visual language is that of docs/model-comparison.svg: gray for classical models, blue for
exact simulation, orange for IBM Quantum hardware; dots for fit, diamonds for the test
kernel, squares for prediction, on a log time scale.
"""

from __future__ import annotations

import math
import sys
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.render_comparison_svg import (  # noqa: E402
    CALIBRATED_NOTE,
    KEY_EM,
    LABEL_RIGHT,
    MARGIN,
    PLOT_LEFT,
    PLOT_RIGHT,
    SOURCE_CSS,
    STYLE,
    WIDTH,
    diamond,
    dot,
    format_seconds,
    legend,
    note,
    row_labels,
    square,
    text_width,
    tick_label,
)
from scripts.render_results_svg import load_json, text  # noqa: E402

REPORT = ROOT / "reports" / "benchmark-v2-qpu-subset.json"
OUTPUT = ROOT / "docs" / "benchmark-v2-qpu.svg"
ACCURACY_ROW = 32
RUNTIME_ROW = 40
ACCURACY_DOMAIN = (0.5, 1.0)
CSS = {"classical": "classical", **SOURCE_CSS}

# Rows in display order: (model in the report, kernel source, label, caption).
ROWS = (
    ("logistic_regression", "classical", "Logistic regression", "classical"),
    ("rbf_svm", "classical", "RBF SVM", "classical"),
    ("random_forest", "classical", "Random Forest", "classical"),
    ("quantum_kernel_svm", "simulation", "Quantum-kernel SVM", "exact simulation"),
    ("quantum_kernel_svm", "qpu", "Quantum-kernel SVM", "{backend}, uncorrected"),
    ("quantum_kernel_svm", "qpu_calibrated", "Quantum-kernel SVM", "{backend}, calibrated"),
)


def models(report: dict) -> dict:
    return {(m["model"], m["kernel_source"]): m for m in report["models"]}


def panel_accuracy(report: dict, backend: str, y: float) -> tuple[list[str], float]:
    size = report["subset"]["size"]
    parts = [
        text(MARGIN, y, f"Accuracy on {size} internal-test rows", "ink", 17, weight="600"),
        text(
            MARGIN,
            y + 20,
            "Dot: accuracy. Line: 95% bootstrap interval. Each model uses its frozen v2 "
            f"threshold; one row moves accuracy by {100 / size:.1f} points.",
            "ink2",
            12,
        ),
    ]
    lo, hi = ACCURACY_DOMAIN
    width = PLOT_RIGHT - PLOT_LEFT
    top = y + 50
    bottom = top + len(ROWS) * ACCURACY_ROW

    def scale(v: float) -> float:
        return PLOT_LEFT + (v - lo) / (hi - lo) * width

    for tick in (0.5, 0.625, 0.75, 0.875, 1.0):
        x = scale(tick)
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="{top - 6}" x2="{x:.1f}" y2="{bottom}"/>')
        anchor = "start" if tick == lo else "end" if tick == hi else "middle"
        parts.append(text(x, bottom + 16, f"{tick:.3g}", "muted", 11, anchor))
    rows = models(report)
    for index, (key, source, label, caption) in enumerate(ROWS):
        row_top = top + index * ACCURACY_ROW
        parts += row_labels(LABEL_RIGHT, row_top, label, caption.format(backend=backend))
        model = rows.get((key, source))
        row_y = row_top + 10
        if model is None:
            parts += note(PLOT_LEFT + 4, row_y + 4, "not computed in this run")
            continue
        css = CSS[source]
        accuracy = model["metrics"]["accuracy"]
        low, high = (max(lo, v) for v in model["confidence_intervals"]["accuracy"])
        parts.append(
            f'<line class="ci {css}" x1="{scale(low):.1f}" y1="{row_y:.1f}" '
            f'x2="{scale(high):.1f}" y2="{row_y:.1f}"/>'
        )
        parts.append(dot(scale(accuracy), row_y, css))
        parts.append(text(scale(low) - 10, row_y + 4, f"{accuracy:.3f}", "ink2", 11, "end"))
    return parts, bottom + 44


def runtime_values(report: dict) -> list[float]:
    return [v for m in report["models"] for v in m["runtime_seconds"].values() if v]


def panel_runtime(report: dict, backend: str, y: float) -> tuple[list[str], float]:
    usage = report["hardware"]["usage"]
    parts = [
        text(MARGIN, y, "Runtime to fit and classify the same rows", "ink", 17, weight="600"),
        text(
            MARGIN,
            y + 20,
            f"Seconds per step, log scale. Local steps: median over {report['runtime']['repeats']} "
            "runs on one machine. IBM test kernel: recorded execution of",
            "ink2",
            12,
        ),
        text(
            MARGIN,
            y + 36,
            f"all {report['hardware']['circuits']} circuits, including calibration pairs and "
            f"diagonal controls ({usage['charged_seconds']:.0f} s billed). Queueing is excluded.",
            "ink2",
            12,
        ),
    ]
    key_y = y + 60
    x = MARGIN
    for shape, label in (
        (dot, "fit (quantum fit includes simulating its training kernel)"),
        (diamond, "test kernel"),
        (square, "predict"),
    ):
        parts.append(shape(x + 5, key_y - 4, "key"))
        parts.append(text(x + 16, key_y, label, "ink2", 12))
        x += 16 + text_width(label, 12, KEY_EM) + 18
    values = runtime_values(report)
    low = math.floor(math.log10(min(values)))
    high = math.ceil(math.log10(max(values)))
    width = PLOT_RIGHT - PLOT_LEFT
    top = key_y + 24
    bottom = top + len(ROWS) * RUNTIME_ROW

    def scale(v: float) -> float:
        return PLOT_LEFT + (math.log10(v) - low) / (high - low) * width

    for exponent in range(low, high + 1):
        x = scale(10.0**exponent)
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="{top - 6}" x2="{x:.1f}" y2="{bottom}"/>')
        anchor = "start" if exponent == low else "end" if exponent == high else "middle"
        parts.append(text(x, bottom + 16, tick_label(exponent), "muted", 11, anchor))
    rows = models(report)
    lanes = (("fit", dot, 9), ("test_kernel", diamond, 20), ("predict", square, 31))
    for index, (key, source, label, caption) in enumerate(ROWS):
        row_y = top + index * RUNTIME_ROW
        parts += row_labels(LABEL_RIGHT, row_y + 6, label, caption.format(backend=backend))
        model = rows.get((key, source))
        if model is None:
            parts += note(PLOT_LEFT + 4, row_y + 24, "not computed in this run")
            continue
        css = CSS[source]
        for step, shape, offset in lanes:
            value = model["runtime_seconds"].get(step)
            if not value:
                continue
            cx = scale(value)
            parts.append(shape(cx, row_y + offset, css))
            if step != "test_kernel":
                continue
            if source == "qpu_calibrated":
                annotation = CALIBRATED_NOTE
            elif source == "qpu":
                annotation = f"{format_seconds(value)} on {backend}"
            else:
                annotation = f"{format_seconds(value)} local simulation"
            if cx + 10 + text_width(annotation, 11) <= PLOT_RIGHT:
                parts.append(text(cx + 10, row_y + offset + 4, annotation, "ink2", 11))
            else:
                parts.append(text(cx - 10, row_y + offset + 4, annotation, "ink2", 11, "end"))
    return parts, bottom + 44


def description(report: dict, backend: str) -> str:
    """Text equivalent of every plotted accuracy and the test-kernel times."""
    items = []
    rows = models(report)
    for key, source, label, caption in ROWS:
        model = rows.get((key, source))
        name = f"{label} ({caption.format(backend=backend)})"
        if model is None:
            items.append(f"{name} not computed")
            continue
        low, high = model["confidence_intervals"]["accuracy"]
        item = (
            f"{name} accuracy {model['metrics']['accuracy']:.3f} (95% CI {low:.3f} to {high:.3f})"
        )
        kernel = model["runtime_seconds"].get("test_kernel")
        if kernel and source != "qpu_calibrated":
            item += f", test kernel {format_seconds(kernel)}"
        items.append(item)
    return (
        f"Run {report['hardware']['run_id']} on {report['subset']['size']} internal-test rows: "
        + "; ".join(items)
        + ". The calibrated hardware kernel reuses the uncorrected jobs."
    )


def render(report: dict) -> str:
    backend = report["hardware"]["backend"]
    body = [
        text(
            MARGIN, 42, "Benchmark v2: page kernel on IBM Quantum hardware", "ink", 22, weight="600"
        ),
        text(
            MARGIN,
            64,
            f"Run {report['hardware']['run_id']}. 24-example page-feature models; the quantum "
            "SVM trains on a simulated kernel and hardware only classifies.",
            "ink2",
            12,
        ),
    ]
    body += legend(92)
    y = 132.0
    for build in (panel_accuracy, panel_runtime):
        parts, y = build(report, backend, y)
        body += parts
    body.append(
        text(
            MARGIN,
            y,
            "Rows share no domain with training or calibration data. No quantum advantage is "
            "claimed: a four-qubit kernel is cheap to simulate exactly.",
            "muted",
            11,
        )
    )
    height = y + 24
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {height:.0f}" '
            f'width="{WIDTH}" height="{height:.0f}" role="img" aria-labelledby="title desc">',
            '<title id="title">Benchmark v2 page kernel on IBM Quantum: accuracy and runtime</title>',
            f'<desc id="desc">{escape(description(report, backend))}</desc>',
            f"<style>{STYLE}</style>",
            f'<rect class="surface" width="{WIDTH}" height="{height:.0f}" rx="12"/>',
            *body,
            "</svg>",
        ]
    )


def write(report_path: Path = REPORT, output: Path = OUTPUT) -> Path:
    report = load_json(report_path)
    if report is None:
        raise SystemExit(
            f"{report_path.name} is not published; run scripts/run_qpu_v2.py evaluate."
        )
    output.write_text(render(report) + "\n", encoding="utf-8")
    return output


def main() -> int:
    print(f"Wrote {write().relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
