# CI and Branch Protection

## Current status

The workflow, test tooling, unit tests for repository checks, and an importable branch ruleset are committed-file proposals. They become available remotely only after publication. A JSON file in the repository does **not** activate branch protection. Remote enforcement has not been configured or verified by this change: the local GitHub CLI could not access this private repository's administration API.

## Python checks

[Python CI](../.github/workflows/ci.yml) runs on pushes to `develop` and `main`, pull requests targeting either branch, and manual dispatch. It has no path filters, so documentation-only PRs still produce the required check. Python 3.12 and 3.13 are the CI targets for tooling; this is not yet a compatibility guarantee for the full QML runtime.

The lint job runs Ruff lint and format checks. Each Python matrix job installs [requirements-dev.txt](../requirements-dev.txt), runs `pip check`, Pytest, and the offline repository checks. The stable **CI required** job succeeds only when lint and the whole test matrix succeed. Failure, cancellation, or a skipped dependency cannot produce a passing gate.

The workflow uses hosted runners, read-only repository permissions, SHA-pinned actions, checkout without persisted credentials, concurrency cancellation, and timeouts. It uses `pull_request`, not privileged `pull_request_target`. No cloud secrets, hardware jobs, deployment, or runtime model downloads are involved. See GitHub's [secure-use guidance](https://docs.github.com/en/actions/reference/security/secure-use).

## Local commands

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe scripts/check_repository.py
```

On macOS/Linux, replace `.\.venv\Scripts\python.exe` with `.venv/bin/python`. The repository checker validates local inline Markdown file targets and commented environment settings; it does not validate heading anchors, remote URLs, rendered layout, or detect every possible secret. Continue the [documentation checklist](documentation-checklist.md).

## Activate protection after publishing CI

1. Publish the workflow and `develop`, then let CI complete successfully. Inspect existing rules before adding another ruleset.
2. With repository administrator access, open **Settings > Rules > Rulesets > New ruleset > Import a ruleset** and import [.github/rulesets/protected-branches.json](../.github/rulesets/protected-branches.json).
3. Confirm both `main` and `develop` are targeted and enforcement is Active. Select GitHub Actions as the source for **CI required** if available after its first run.
4. Verify with a feature-branch PR that a failing check blocks merging, an unresolved review thread blocks merging, and a successful up-to-date PR can merge. Verify force-push and deletion restrictions in the effective rules without attempting destructive operations.

The rules require PRs, resolved review discussions, the latest base changes, and **CI required**; they block force-push and deletion with no bypass actors. Required approvals are zero to support a solo maintainer, who cannot approve their own PR. Raise this to one when another reviewer is available. Work on short-lived branches targeting `develop`; promote `develop` to `main` through a PR. Do not work directly on protected branches once enforcement is active.

The repository is private. GitHub plan eligibility must be checked before enabling rulesets; see [ruleset availability and creation](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/creating-rulesets-for-a-repository). Publishing workflow YAML alone does not block direct pushes or merges.

## Dependency updates

[Dependabot](../.github/dependabot.yml) proposes weekly GitHub Actions and Python dependency updates targeting `develop`. GitHub reads this configuration from the default branch, so it must reach `main` to activate. Updates must pass the same CI and PR process; no automatic merging is configured. Runtime dependencies remain an unpinned design-stage proposal; only direct test/lint tool versions are pinned here.
