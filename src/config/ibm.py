"""IBM Quantum connection settings shared by the access check and the QPU run script."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from src.config import env_file

CHANNEL = "ibm_quantum_platform"
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"


def settings(path: Path = ENV_PATH) -> dict[str, str]:
    """`.env` values overlaid by terminal-session variables."""
    return env_file.merged(path)


def service_kwargs(values: Mapping[str, str]) -> dict[str, str]:
    """Explicit credentials from the settings, or none to use the saved account."""
    token = values.get("IBM_QUANTUM_TOKEN", "").strip()
    instance = values.get("IBM_QUANTUM_INSTANCE", "").strip()
    if not token:
        return {}
    kwargs = {"channel": CHANNEL, "token": token}
    if instance:
        kwargs["instance"] = instance
    return kwargs


def connect(values: Mapping[str, str]):
    """Open a QiskitRuntimeService. This contacts IBM; callers decide when to do it."""
    from qiskit_ibm_runtime import QiskitRuntimeService

    return QiskitRuntimeService(**service_kwargs(values))
