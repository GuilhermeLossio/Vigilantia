# Tests

Run `python -m pytest` after installing `requirements-dev.txt`. Tests in `unit/` cover repository checks, dataset acquisition, the split, features and scaling, the quantum kernel and hardware tooling, QPU runs with stubbed IBM services, models and metrics, runtimes, inference, the Flask service, the requirements files, the recorded QPU evidence, and notebook hygiene, including negative cases. Downloader tests use small synthetic ZIPs and mocked HTTP responses; they never contact UCI. No test contacts IBM Quantum.

Add tests of Python behavior under `tests/unit/test_<component>.py` with each implementation change. Keep unit tests deterministic and independent of network, AWS, IBM credentials, and QPU execution. Use small fixtures and mock external service boundaries. Register new markers in `pyproject.toml` before using them.

The notebooks are end-to-end replication checks, run manually from a fresh environment as described in the [reproduction guide](../docs/reproducing.md); unit tests only check their structure and outputs. QPU benchmarks are separate manual experiments, not unit tests.

No coverage percentage is claimed before application code exists. Pytest's no-tests-collected exit code is not suppressed, and failing/skipped CI jobs cannot satisfy the required gate.
