"""python-dotenv .env reading and the IBM access check credentials; no network or API calls."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from src.config import env_file

pytestmark = pytest.mark.unit
SPEC = importlib.util.spec_from_file_location(
    "check_ibm_access", Path(__file__).resolve().parents[2] / "scripts" / "check_ibm_access.py"
)
check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check)


def test_env_file_handles_comments_quotes_and_export(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    lines = [
        "# comment",
        "IBM_QUANTUM_TOKEN=abc=def",
        'export IBM_QUANTUM_INSTANCE="crn:v1:x"',
        "IBM_QUANTUM_BACKEND='ibm_fez'",
        "# IBM_QUANTUM_BACKEND=commented",
        "EMPTY=",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert env_file.merged(path, {}) == {
        "IBM_QUANTUM_TOKEN": "abc=def",
        "IBM_QUANTUM_INSTANCE": "crn:v1:x",
        "IBM_QUANTUM_BACKEND": "ibm_fez",
        "EMPTY": "",
    }


def test_process_environment_overrides_the_file(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("IBM_QUANTUM_TOKEN=file\nIBM_QUANTUM_BACKEND=ibm_fez\n", encoding="utf-8")
    values = env_file.merged(path, {"IBM_QUANTUM_TOKEN": "session", "IBM_QUANTUM_BACKEND": "  "})
    assert values["IBM_QUANTUM_TOKEN"] == "session"
    assert values["IBM_QUANTUM_BACKEND"] == "ibm_fez"


def test_missing_file_is_not_an_error(tmp_path: Path) -> None:
    assert env_file.merged(tmp_path / "absent.env", {"A": "1"}) == {"A": "1"}


def test_saved_account_is_used_without_a_token() -> None:
    assert check.service_kwargs({}) == {}
    assert check.service_kwargs({"IBM_QUANTUM_TOKEN": "  ", "IBM_QUANTUM_INSTANCE": "crn"}) == {}


def test_credentials_are_passed_explicitly() -> None:
    assert check.service_kwargs({"IBM_QUANTUM_TOKEN": "t"}) == {
        "channel": "ibm_quantum_platform",
        "token": "t",
    }
    kwargs = check.service_kwargs({"IBM_QUANTUM_TOKEN": "t", "IBM_QUANTUM_INSTANCE": " crn "})
    assert kwargs["instance"] == "crn"


def test_script_reads_the_repository_env_file() -> None:
    assert check.ENV_PATH == Path(__file__).resolve().parents[2] / ".env"
