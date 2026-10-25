# Contributing to Vigilantia

Vigilantia is in the design stage. Start with the [README](README.md), [architecture](Architecture.md), and [benchmark protocol](docs/benchmark-protocol.md).

## Change workflow

Create a short-lived branch with a descriptive prefix such as `docs/`, `feat/`, or `fix/`. Keep each change focused and submit a pull request describing the problem, resulting behavior, validation performed, and remaining limitations. Update documentation in the same change as implementation or experimental protocol changes.

Do not mark a proposed feature complete until evidence exists. Changes to sample selection, features, preprocessing, shots, circuits, or model settings must be versioned and documented before the final test set is evaluated. Do not overwrite a previous run's results with a different configuration.

## Local checks

For documentation changes, follow the [readiness checklist](docs/documentation-checklist.md), check relative links and diagram rendering, and run `git diff --check`. No application test suite exists yet. When adding code, include checks appropriate to the behavior and document the commands; do not claim unrun checks passed.

Dependencies are currently an unpinned proposal. Use an isolated environment and validate compatibility before recording a supported Python version or reproducible installation. Proposed `.env.example` settings require an explicit loader in future application code.

## Data and credentials

Follow [data/README.md](data/README.md). Keep downloaded datasets, model binaries, local credentials, and real personal or breach records out of commits. Use fictitious exposure examples only when that later module is implemented. Retain dataset attribution independently of the repository's MIT license.

Never place tokens in notebooks, outputs, logs, screenshots, or PR descriptions. Hardware work must target the intended Open Plan instance and respect verified remaining quota; do not switch to paid execution automatically. Record reproducibility metadata without account secrets.

## Review expectations

Reviewers should be able to distinguish measured results from proposals, reproduce published experiments from recorded inputs and configuration, and understand any incomplete validation. Contributions are distributed under the repository's [MIT license](LICENSE), while third-party code and data retain their respective licenses.
