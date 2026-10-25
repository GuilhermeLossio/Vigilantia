"""CLI reporting checks without remote requests or real dataset writes."""

import pytest

from scripts import download_dataset as downloader

pytestmark = pytest.mark.unit


def test_main_reports_count_warnings(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    manifest = {
        "inspection": {
            "rows": 2,
            "labels": {"0": 1, "1": 1},
            "column_count": 3,
            "warnings": ["Observed 2 rows; UCI publishes 235795."],
        },
        "files": {"csv": {"sha256": "test-hash"}},
    }
    monkeypatch.setattr(downloader, "acquire", lambda: (manifest, False))
    assert downloader.main() == 0
    output = capsys.readouterr()
    assert "Acquisition complete" in output.out
    assert "WARNING: Observed 2 rows" in output.err


def test_main_returns_nonzero_on_failure(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    def fail():
        raise downloader.AcquisitionError("Existing acquisition is inconsistent")

    monkeypatch.setattr(downloader, "acquire", fail)
    assert downloader.main() == 1
    assert "Acquisition failed" in capsys.readouterr().err
