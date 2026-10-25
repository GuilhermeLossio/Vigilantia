"""Render docs/model-comparison.svg: accuracy and runtime per model across evaluation sets.

Inputs (all published reports; a missing one is drawn as "report not published"):
- reports/benchmark-results.json: benchmark test set (12 rows), runtimes, IBM usage.
- reports/external-validation.json: external primary set, simulation only, 95% intervals.
- reports/external-qpu-subset.json: 40-row external subset run on IBM hardware, runtimes.

Every number in the chart is read from these files. Colors encode the source of each
number, as in docs/benchmark-results.svg: gray for classical models, blue for simulation,
orange for IBM Quantum hardware. Text always uses ink tokens, never a series color.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.render_results_svg import STYLE as BASE_STYLE  # noqa: E402
from scripts.render_results_svg import load_json, text  # noqa: E402

BENCHMARK = ROOT / "reports" / "benchmark-results.json"
EXTERNAL = ROOT / "reports" / "external-validation.json"
SUBSET = ROOT / "reports" / "external-qpu-subset.json"
OUTPUT = ROOT / "docs" / "model-comparison.svg"

WIDTH = 960
MARGIN = 40
LABEL_RIGHT = 212  # row labels end here, right aligned
PLOT_LEFT = 232
PLOT_RIGHT = 924
ACCURACY_ROW = 32
RUNTIME_ROW = 40
ACCURACY_DOMAIN = (0.25, 1.0)
CHANCE = 0.5
KEY_EM = 0.47  # tighter width estimate for the long runtime key labels
# Four of the six subset rows are classical or simulated; only the column was run on hardware.
SUBSET_TITLE = "External hardware subset"
CALIBRATED_NOTE = "same jobs as uncorrected"

# Rows in display order: (model name in the reports, kernel source, source caption).
ROWS = (
    ("Majority baseline", "classical", "classical"),
    ("Logistic regression", "classical", "classical"),
    ("RBF SVM", "classical", "classical"),
    ("Quantum-kernel SVM", "simulation", "exact simulation"),
    ("Quantum-kernel SVM", "qpu", "{backend}, uncorrected"),
    ("Quantum-kernel SVM", "qpu_calibrated", "{backend}, calibrated"),
)
SOURCE_CSS = {
    "classical": "classical",
    "simulation": "simulation",
    "qpu": "hardware",
    "qpu_calibrated": "hardware",
}

# The light neutral is one step darker than in the base chart so that gray marks clear
# 3:1 against the light surface; it is the documented muted ink (#898781).
STYLE = (
    BASE_STYLE
    + """
  svg { --neutral: #898781; }
  @media (prefers-color-scheme: dark) { svg { --neutral: #7d7c76; } }
  .ring { stroke: var(--surface); stroke-width: 2; }
  .axis { stroke: var(--text-muted); stroke-width: 1; }
  .ci { stroke-width: 2; stroke-linecap: round; fill: none; }
  .ci.classical { stroke: var(--neutral); }
  .ci.simulation { stroke: var(--series-1); }
  .ci.hardware { stroke: var(--series-2); }
  .key { fill: var(--text-secondary); }
"""
)


class Report:
    """One published evaluation set, or a placeholder when its file is missing."""

    def __init__(self, data: dict | None):
        self.data = data
        self.models = {(m["name"], m["kernel_source"]): m for m in (data or {}).get("models", [])}

    def model(self, name: str, source: str) -> dict | None:
        return self.models.get((name, source))


def load_reports() -> tuple[Report, Report, Report]:
    data = [load_json(path) for path in (BENCHMARK, EXTERNAL, SUBSET)]
    if not any(data):
        raise SystemExit(
            "No published reports found: reports/benchmark-results.json, "
            "reports/external-validation.json, and reports/external-qpu-subset.json are missing."
        )
    return Report(data[0]), Report(data[1]), Report(data[2])


def backend_name(*reports: Report) -> str:
    for report in reports:
        backend = ((report.data or {}).get("hardware") or {}).get("backend")
        if backend:
            return backend
    return "IBM QPU"


def row_labels(x: float, y: float, name: str, caption: str) -> list[str]:
    return [
        text(x, y + 13, name, "ink", 13, "end"),
        text(x, y + 26, caption, "muted", 10, "end"),
    ]


def text_width(value: str, size: float, em: float = 0.52) -> float:
    """Width estimate for the sans used here (about 0.52 em per glyph, slightly generous).

    Long runs of mostly lowercase text average closer to 0.47 em; pass ``em`` to tighten.
    """
    return len(value) * size * em


def note(x: float, y: float, value: str) -> list[str]:
    """Muted text on a surface-colored backing, so gridlines and reference lines pass behind."""
    width = text_width(value, 11) + 8
    return [
        f'<rect class="surface" x="{x - 4:.1f}" y="{y - 11:.1f}" width="{width:.1f}" '
        'height="15" rx="2"/>',
        text(x, y, value, "muted", 11),
    ]


def dot(x: float, y: float, css: str, r: float = 4.5) -> str:
    return f'<circle class="{css} ring" cx="{x:.1f}" cy="{y:.1f}" r="{r}"/>'


def square(x: float, y: float, css: str, half: float = 4.0) -> str:
    return (
        f'<rect class="{css} ring" x="{x - half:.1f}" y="{y - half:.1f}" '
        f'width="{2 * half}" height="{2 * half}" rx="1"/>'
    )


def diamond(x: float, y: float, css: str, half: float = 5.5) -> str:
    return (
        f'<path class="{css} ring" d="M{x:.1f},{y - half:.1f} l{half},{half} '
        f'l-{half},{half} l-{half},-{half} z"/>'
    )


def clipped_cap(x: float, y: float, css: str, half: float = 4.0) -> str:
    """Open cap marking an interval end that lies outside the axis domain."""
    return (
        f'<line class="ci {css}" x1="{x:.1f}" y1="{y - half:.1f}" '
        f'x2="{x:.1f}" y2="{y + half:.1f}"/>'
    )


def clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def accuracy_columns(benchmark: Report, external: Report, subset: Report) -> list[dict]:
    """The three evaluation sets, with their size and how intervals are reported."""

    def size(data: dict | None, *keys: str) -> int | None:
        for key in keys[:-1]:
            data = (data or {}).get(key)
        return (data or {}).get(keys[-1])

    columns = [
        {
            "report": benchmark,
            "title": "Benchmark test set",
            "n": size(benchmark.data, "split", "test"),
            "note": "no interval published",
        },
        {
            "report": external,
            "title": "External full set",
            "n": size(external.data, "preparation", "primary_rows"),
            "note": "95% CI, simulation only",
        },
        {
            "report": subset,
            "title": SUBSET_TITLE,
            "n": size(subset.data, "subset", "size"),
            "note": "95% CI",
        },
    ]
    for column in columns:
        column["subtitle"] = (
            f"n={column['n']:,} · {column['note']}"
            if column["report"].data and column["n"]
            else "report not published"
        )
    return columns


def panel_accuracy(
    benchmark: Report, external: Report, subset: Report, backend: str, y: float
) -> tuple[list[str], float]:
    parts = [
        text(MARGIN, y, "Accuracy by evaluation set", "ink", 17, weight="600"),
        text(
            MARGIN,
            y + 20,
            "Dot: accuracy. Line: 95% bootstrap interval where the report publishes one. "
            "Dashed line: chance level (0.5). Same trained models in every set.",
            "ink2",
            12,
        ),
    ]
    columns = accuracy_columns(benchmark, external, subset)
    gap = 28
    width = (PLOT_RIGHT - PLOT_LEFT - gap * (len(columns) - 1)) / len(columns)
    lo, hi = ACCURACY_DOMAIN
    header = y + 46
    top = header + 34
    bottom = top + len(ROWS) * ACCURACY_ROW
    for index, (name, _source, caption) in enumerate(ROWS):
        parts += row_labels(
            LABEL_RIGHT, top + index * ACCURACY_ROW, name, caption.format(backend=backend)
        )
    for c, column in enumerate(columns):
        x0 = PLOT_LEFT + c * (width + gap)

        def scale(v: float, x0: float = x0) -> float:
            return x0 + (v - lo) / (hi - lo) * width

        parts.append(text(x0, header, column["title"], "ink", 13, weight="600"))
        parts.append(text(x0, header + 16, column["subtitle"], "muted", 11))
        ticks = [0.25, 0.5, 0.75, 1.0]
        for tick in ticks:
            x = scale(tick)
            parts.append(
                f'<line class="grid" x1="{x:.1f}" y1="{top - 6}" x2="{x:.1f}" y2="{bottom}"/>'
            )
            anchor = "start" if tick == lo else "end" if tick == hi else "middle"
            parts.append(text(x, bottom + 16, f"{tick:.2f}", "muted", 11, anchor))
        chance = scale(CHANCE)
        parts.append(
            f'<line class="ref" x1="{chance:.1f}" y1="{top - 6}" x2="{chance:.1f}" y2="{bottom}"/>'
        )
        report: Report = column["report"]
        for index, (name, source, _) in enumerate(ROWS):
            row_y = top + index * ACCURACY_ROW + 10
            model = report.model(name, source)
            if report.data is None:
                parts += note(x0 + 4, row_y + 4, "report not published")
                continue
            if model is None:
                parts += note(x0 + 4, row_y + 4, "not run on this set")
                continue
            accuracy = model["metrics"]["accuracy"]
            interval = (model.get("confidence_intervals") or {}).get("accuracy")
            css = SOURCE_CSS[source]
            right = scale(accuracy)
            left = scale(accuracy)
            if interval:
                ends = [clamp(v, lo, hi) for v in interval]
                left, right = scale(ends[0]), scale(ends[1])
                parts.append(
                    f'<line class="ci {css}" x1="{left:.1f}" y1="{row_y:.1f}" '
                    f'x2="{right:.1f}" y2="{row_y:.1f}"/>'
                )
                for end, raw in ((left, interval[0]), (right, interval[1])):
                    if raw < lo or raw > hi:
                        parts.append(clipped_cap(end, row_y, css))
            parts.append(dot(scale(clamp(accuracy, lo, hi)), row_y, css))
            label = f"{accuracy:.3f}"
            if right + 8 + text_width(label, 11) <= x0 + width + gap - 6:
                parts.append(text(right + 8, row_y + 4, label, "ink2", 11))
            else:
                parts.append(text(left - 8, row_y + 4, label, "ink2", 11, "end"))
    return parts, bottom + 44


def format_seconds(value: float) -> str:
    if value >= 10:
        return f"{value:.1f} s"
    if value >= 1:
        return f"{value:.2f} s"
    if value >= 1e-3:
        return f"{value * 1e3:.0f} ms" if value >= 1e-2 else f"{value * 1e3:.1f} ms"
    return f"{value * 1e6:.0f} µs"


def tick_label(exponent: int) -> str:
    value = 10.0**exponent
    if exponent >= 0:
        return f"{value:.0f} s"
    if exponent >= -3:
        return f"{value * 1e3:.0f} ms"
    return f"{value * 1e6:.0f} µs"


def runtime_values(report: Report) -> list[float]:
    values = []
    for model in report.models.values():
        values += [v for v in (model.get("runtime_seconds") or {}).values() if v]
    return values


def runtime_subtitle(report: Report, backend: str) -> str:
    if report.data is None:
        return "report not published"
    qpu = (report.data.get("runtime") or {}).get("qpu")
    if not qpu:
        return "no IBM usage recorded"
    return (
        f"{backend}: {qpu['execution_seconds']:.1f} s execution, "
        f"{qpu['charged_seconds']:.0f} s billed"
    )


def panel_runtime(
    benchmark: Report, subset: Report, backend: str, y: float
) -> tuple[list[str], float]:
    parts = [
        text(MARGIN, y, "Runtime to classify the evaluation set", "ink", 17, weight="600"),
        text(
            MARGIN,
            y + 20,
            "Seconds per step, log scale. Local steps: median wall-clock time over repeated "
            "runs on one machine. IBM test kernel:",
            "ink2",
            12,
        ),
        text(
            MARGIN,
            y + 36,
            "IBM's recorded execution time for the run's jobs, which also measured calibration "
            "pairs and diagonal controls. Queueing is excluded.",
            "ink2",
            12,
        ),
    ]
    key_y = y + 60
    x = MARGIN
    for shape, label in (
        (dot, "fit (training; quantum fit includes simulating its training kernel)"),
        (diamond, "test kernel"),
        (square, "predict"),
    ):
        parts.append(shape(x + 5, key_y - 4, "key"))
        parts.append(text(x + 16, key_y, label, "ink2", 12))
        x += 16 + text_width(label, 12, KEY_EM) + 18
    sets = [
        (benchmark, "Benchmark test set", "split", "test"),
        (subset, SUBSET_TITLE, "subset", "size"),
    ]
    values = runtime_values(benchmark) + runtime_values(subset)
    low = math.floor(math.log10(min(values))) if values else -6
    high = math.ceil(math.log10(max(values))) if values else 3
    gap = 32
    width = (PLOT_RIGHT - PLOT_LEFT - gap) / 2
    header = key_y + 30
    top = header + 34
    bottom = top + len(ROWS) * RUNTIME_ROW
    for index, (name, _source, caption) in enumerate(ROWS):
        parts += row_labels(
            LABEL_RIGHT, top + index * RUNTIME_ROW + 6, name, caption.format(backend=backend)
        )
    lanes = (("fit", dot, 9), ("test_kernel", diamond, 20), ("predict", square, 31))
    for c, (report, title, *size_keys) in enumerate(sets):
        x0 = PLOT_LEFT + c * (width + gap)

        def scale(v: float, x0: float = x0) -> float:
            return x0 + (math.log10(v) - low) / (high - low) * width

        n = ((report.data or {}).get(size_keys[0]) or {}).get(size_keys[1])
        heading = f"{title}, n={n:,}" if n else title
        parts.append(text(x0, header, heading, "ink", 13, weight="600"))
        parts.append(text(x0, header + 16, runtime_subtitle(report, backend), "muted", 11))
        for exponent in range(low, high + 1):
            x = scale(10.0**exponent)
            parts.append(
                f'<line class="grid" x1="{x:.1f}" y1="{top - 6}" x2="{x:.1f}" y2="{bottom}"/>'
            )
            if (exponent - low) % 2 == 0:
                anchor = "start" if exponent == low else "end" if exponent == high else "middle"
                parts.append(text(x, bottom + 16, tick_label(exponent), "muted", 11, anchor))
        for index, (name, source, _) in enumerate(ROWS):
            row_y = top + index * RUNTIME_ROW
            model = report.model(name, source)
            if report.data is None or model is None or not model.get("runtime_seconds"):
                message = "report not published" if report.data is None else "not timed"
                parts += note(x0 + 4, row_y + 24, message)
                continue
            css = SOURCE_CSS[source]
            for step, shape, offset in lanes:
                value = model["runtime_seconds"].get(step)
                if not value:
                    continue
                cx = scale(value)
                parts.append(shape(cx, row_y + offset, css))
                if step != "test_kernel":
                    continue
                if source == "qpu_calibrated":
                    # Calibrated kernel is a correction of the same jobs: one run, not two.
                    label = CALIBRATED_NOTE
                elif source == "qpu":
                    label = f"{format_seconds(value)} on {backend}"
                else:
                    label = f"{format_seconds(value)} local simulation"
                if cx + 10 + text_width(label, 11) <= x0 + width + gap - 6:
                    parts.append(text(cx + 10, row_y + offset + 4, label, "ink2", 11))
                else:
                    parts.append(text(cx - 10, row_y + offset + 4, label, "ink2", 11, "end"))
    return parts, bottom + 44


def legend(y: float) -> list[str]:
    parts, x = [], MARGIN
    for css, label in (
        ("classical", "Classical model"),
        ("simulation", "Exact simulation"),
        ("hardware", "IBM Quantum hardware"),
    ):
        parts.append(dot(x + 6, y - 4, css, 5))
        parts.append(text(x + 18, y, label, "ink2", 12))
        x += 18 + text_width(label, 12) + 20
    return parts


def description(benchmark: Report, external: Report, subset: Report, backend: str) -> str:
    """Text equivalent of every plotted accuracy and the hardware test-kernel times."""
    sentences = []
    for column in accuracy_columns(benchmark, external, subset):
        report: Report = column["report"]
        if report.data is None:
            sentences.append(f"{column['title']}: report not published.")
            continue
        items = []
        for name, source, caption in ROWS:
            model = report.model(name, source)
            label = f"{name} ({caption.format(backend=backend)})"
            if model is None:
                items.append(f"{label} not run")
                continue
            item = f"{label} {model['metrics']['accuracy']:.3f}"
            interval = (model.get("confidence_intervals") or {}).get("accuracy")
            if interval:
                item += f" (95% CI {interval[0]:.3f} to {interval[1]:.3f})"
            items.append(item)
        sentences.append(f"{column['title']}, {column['subtitle']}: " + "; ".join(items) + ".")
    for title, report in (("Benchmark test set", benchmark), ("External subset", subset)):
        for source in ("simulation", "qpu"):
            model = report.model("Quantum-kernel SVM", source)
            value = ((model or {}).get("runtime_seconds") or {}).get("test_kernel")
            if value:
                where = "exact simulation" if source == "simulation" else f"{backend} execution"
                sentences.append(f"{title} test kernel, {where}: {format_seconds(value)}.")
        if report.model("Quantum-kernel SVM", "qpu_calibrated"):
            sentences.append(
                f"{title} calibrated test kernel: computed from the same {backend} jobs "
                "as the uncorrected kernel, no extra execution."
            )
    return " ".join(sentences)


def footnotes(subset: Report, benchmark: Report, y: float) -> tuple[list[str], float]:
    lines = []
    size = ((subset.data or {}).get("subset") or {}).get("size")
    test = ((benchmark.data or {}).get("split") or {}).get("test")
    if size and test:
        lines.append(
            f"Each row changes accuracy by {100 / test:.1f} points on the benchmark test set "
            f"and {100 / size:.1f} points on the subset; differences of one or two rows sit "
            "well inside the intervals."
        )
    lines.append(
        "The subset was drawn by a rule frozen before any result. No quantum advantage is claimed."
    )
    parts = [text(MARGIN, y + 18 * i, line, "muted", 11) for i, line in enumerate(lines)]
    return parts, y + 18 * len(lines)


def render() -> str:
    benchmark, external, subset = load_reports()
    backend = backend_name(subset, benchmark)
    body = [
        text(MARGIN, 42, "Vigilantia: accuracy and runtime by model", "ink", 22, weight="600"),
        text(
            MARGIN,
            64,
            "Classical baselines and the hybrid quantum-kernel SVM (trained on an exact "
            "simulated kernel; hardware only classifies), on three evaluation sets.",
            "ink2",
            12,
        ),
    ]
    body += legend(92)
    y = 132.0
    builders: tuple[Callable[[float], tuple[list[str], float]], ...] = (
        lambda y: panel_accuracy(benchmark, external, subset, backend, y),
        lambda y: panel_runtime(benchmark, subset, backend, y),
        lambda y: footnotes(subset, benchmark, y),
    )
    for build in builders:
        parts, y = build(y)
        body += parts
    height = y + 24
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {height:.0f}" '
            f'width="{WIDTH}" height="{height:.0f}" role="img" aria-labelledby="title desc">',
            '<title id="title">Vigilantia model comparison: accuracy and runtime</title>',
            f'<desc id="desc">{escape(description(benchmark, external, subset, backend))}</desc>',
            f"<style>{STYLE}</style>",
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
