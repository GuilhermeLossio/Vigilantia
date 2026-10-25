# Documentation Readiness Checklist

Use before the initial documentation commit and whenever the design changes. Check an item only after reviewing the final diff; unchecked items do not claim a completed validation.

## Scope and consistency

- [ ] README and architecture distinguish implemented files, selected decisions, and proposed components.
- [ ] The MVP requires a simulated training kernel, IBM Quantum hardware test kernels, classical comparison, and a local API; exposure and AWS deployment are later phases.
- [ ] Samples, shots, circuit definition, matrix handling, and metrics agree with the benchmark protocol.
- [ ] Open decisions have explicit resolution criteria and no invented versions, features, backend, or results.
- [ ] Roadmap and directory trees match the checkout; future paths are labeled as planned.
- [ ] MIT code licensing is consistent with LICENSE; dataset attribution remains CC BY 4.0.

## Navigation and presentation

- [ ] Local Markdown links and heading anchors resolve.
- [ ] All SVG diagrams, including the generated results chart, render in light and dark mode with readable labels, accessible descriptions, and no clipping or overlapping connectors.
- [ ] README and architecture link both diagrams and the supporting documents.
- [ ] English text, Unicode directory trees, and formatting are consistent.

## Configuration and evidence

- [ ] Dependency installation is described as unverified until compatibility is tested and versions are pinned.
- [ ] `.env.example` is tracked, contains only commented placeholders, and states that no loader exists yet.
- [ ] `.env` variants and project-local credential files are ignored; no credentials appear in the diff.
- [ ] Raw datasets, generated kernels, and local model artifacts are excluded from the documentation commit.
- [ ] No private scheduling notes appear in public files or the proposed commit message.
- [ ] Simulation, hardware execution, and replay are identified accurately; pending checks remain pending.

## Suggested checks

Run `git diff --check` and inspect both tracked changes and untracked files with `git status --short`. Check ignore behavior with `git check-ignore --no-index .env .env.local .env.production .env.example`; only the first three paths should be listed. Review Markdown previews and render the SVG files. If the project-docs-generator skill is available, also run its `scripts/validate_docs.py` against the project root; that local tool is not a repository dependency.

This checklist does not validate runtime behavior or benchmark performance. Those checks belong to the [benchmark protocol](benchmark-protocol.md) and [MVP completion criteria](../README.md#mvp-completion-criteria).
