# Tests

Run `python -m pytest` after installing `requirements-dev.txt`. Tests in `unit/` cover the offline repository validation helpers, including negative cases. They are infrastructure tests, not evidence of application or model correctness.

Add tests of Python behavior under `tests/unit/test_<component>.py` with each implementation change. Keep unit tests deterministic and independent of network, AWS, IBM credentials, and QPU execution. Use small fixtures and mock external service boundaries. Register new markers in `pyproject.toml` before using them.

Future application tests should cover target mapping, domain split isolation, training-only preprocessing, feature order, circuit behavior in local simulation, and API validation. Install the relevant runtime dependencies in CI when those tests are introduced. QPU benchmarks are separate manual experiments, not unit tests.

No coverage percentage is claimed before application code exists. Pytest's no-tests-collected exit code is not suppressed, and failing/skipped CI jobs cannot satisfy the required gate.
