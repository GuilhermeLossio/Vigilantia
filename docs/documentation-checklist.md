# Documentation Readiness Checklist

Use before a documentation commit and whenever the design changes. Check an item only after reviewing the final diff; unchecked items do not claim a completed validation. Uncheck every item before starting the next review.

**Last review: 2026-10-25,** on the working tree before the merge to `main`. Every item passed. The checks ran were: `scripts/check_repository.py` (links and anchors), `tests/unit/test_catalog.py`, the Architecture tree compared with `git ls-files`, both charts rendered in light and dark mode (the two diagrams draw their own opaque light background), a scan of the full Git history for credentials, datasets, and local paths, the documentation audits against `reports/*.json`, and a from-scratch run of the seven notebooks with `requirements.txt`.

## Scope and consistency

- [x] README and architecture distinguish implemented files, selected decisions, and proposed components.
- [x] The MVP requires a simulated training kernel, IBM Quantum hardware test kernels, classical comparison, and a local API; exposure and AWS deployment are later phases.
- [x] Samples, shots, circuit definition, matrix handling, and metrics agree with the benchmark protocol.
- [x] Open decisions have explicit resolution criteria and no invented versions, features, backend, or results.
- [x] The [roadmap](roadmap.md) and the single directory tree in [Architecture.md](../Architecture.md#repository-structure) match the checkout; future paths are labeled as planned.
- [x] MIT code licensing is consistent with LICENSE; attribution of both datasets (PhiUSIIL and the external validation dataset) remains CC BY 4.0.

## Navigation and presentation

- [x] Local Markdown links and heading anchors resolve.
- [x] Every new Markdown file or notebook is listed in [CATALOG.md](../CATALOG.md).
- [x] All SVG diagrams, including the generated results and comparison charts, render in light and dark mode with readable labels, accessible descriptions, and no clipping or overlapping connectors.
- [x] README and architecture link the diagrams and the supporting documents, and the [documentation index](README.md) lists every document.
- [x] English text, Unicode directory trees, and formatting are consistent.

## Configuration and evidence

- [x] Dependency installation matches the pinned `requirements*.txt` files and the tested environment in the [reproduction guide](reproducing.md).
- [x] `.env.example` is tracked, contains only commented placeholders, and documents which settings the hardware scripts read.
- [x] `.env` variants and project-local credential files are ignored; no credentials appear in the diff.
- [x] Raw datasets, generated kernels, and local model artifacts are excluded from the documentation commit.
- [x] No private scheduling notes appear in public files or the proposed commit message.
- [x] Simulation, hardware execution, replay, and the external validation and its 40-row hardware subset are identified accurately; pending checks remain pending.
- [x] Every published number matches `reports/benchmark-results.json`, `reports/external-validation.json`, `reports/external-qpu-subset.json`, the frozen configs in `configs/`, or the benchmark protocol, and no host or URL from either dataset appears.

## Suggested checks

Run `git diff --check` and inspect both tracked changes and untracked files with `git status --short`. Check ignore behavior with `git check-ignore --no-index .env .env.local .env.production .env.example`; only the first three paths should be listed. Review Markdown previews and render the SVG files. If the project-docs-generator skill is available, also run its `scripts/validate_docs.py` against the project root; that local tool is not a repository dependency.

This checklist does not validate runtime behavior or benchmark performance. Those checks belong to the [benchmark protocol](benchmark-protocol.md) and [MVP completion criteria](../README.md#mvp-completion-criteria).
