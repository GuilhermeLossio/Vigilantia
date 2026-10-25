# Contributing to Vigilantia

Start with the [README](README.md), the [documentation catalog](CATALOG.md), the [architecture](Architecture.md), and the [benchmark protocol](docs/benchmark-protocol.md). The [development guide](docs/development.md) runs each pipeline script.

## Change workflow

Create a short-lived branch from `develop` with a descriptive prefix such as `docs/`, `feat/`, or `fix/`. Target `develop` in your PR; promote reviewed changes from `develop` to `main` through a separate PR. Keep each change focused and describe the problem, resulting behavior, validation performed, and remaining limitations. Update documentation in the same change as implementation or experimental protocol changes.

Do not mark a proposed feature complete until evidence exists. Changes to sample selection, features, preprocessing, shots, circuits, or model settings must be versioned and documented before the final test set is evaluated. Do not overwrite a previous run's results with a different configuration. The [external validation protocol](docs/external-validation.md) is frozen in `configs/external-validation.json`; never edit it or tune anything after seeing external results.

## Local checks

For documentation changes, follow the [readiness checklist](docs/documentation-checklist.md), check relative links and diagram rendering, and run `git diff --check`. Install `requirements-dev.txt`, then run `python -m ruff check .`, `python -m ruff format --check .`, `python -m pytest`, and `python scripts/check_repository.py`. Unit tests must accompany new code; see [tests/README.md](tests/README.md). See [CI and branch protection](docs/ci-and-branch-protection.md) for setup, required checks, and remote activation status.

Dependencies are pinned in the `requirements*.txt` files; the [reproduction guide](docs/reproducing.md#tested-environment) records the tested environment. Use an isolated environment and update the pins and the tested environment together. The IBM Quantum settings in `.env.example` are read by the hardware scripts; see [hardware runs](docs/hardware-runs.md#configure-ibm-quantum-access). The other entries are proposals.

## Data and credentials

Follow [data/README.md](data/README.md). Keep downloaded datasets (`data/raw/`, `data/external/`), model binaries, local credentials, and real personal or breach records out of commits. Use fictitious exposure examples only when that later module is implemented. Retain the attribution of both datasets independently of the repository's MIT license, and never publish hosts or URLs from them.

Never place tokens in notebooks, outputs, logs, screenshots, or PR descriptions. Hardware work must target the intended Open Plan instance and respect verified remaining quota; do not switch to paid execution automatically. Record reproducibility metadata without account secrets.

## Review expectations

Reviewers should be able to distinguish measured results from proposals, reproduce published experiments from recorded inputs and configuration, and understand any incomplete validation. Contributions are distributed under the repository's [MIT license](LICENSE), while third-party code and data retain their respective licenses.
