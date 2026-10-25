"""Read the local .env file with python-dotenv, without modifying os.environ.

Variables already set in the terminal session take precedence over the file. Values are
never printed or logged here.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from dotenv import dotenv_values


def merged(path: Path, environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """File values overlaid by non-empty process variables; a missing file is not an error."""
    file_values = {key: value for key, value in dotenv_values(path).items() if value is not None}
    process = os.environ if environ is None else environ
    return {**file_values, **{k: v for k, v in process.items() if v.strip()}}
