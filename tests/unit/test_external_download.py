"""External dataset acquisition with synthetic CSVs and HTTP doubles; never contacts Mendeley."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from urllib.error import HTTPError

import pytest

from scripts import download_external as downloader

pytestmark = pytest.mark.unit
CSV = (
    b"url,length_url,status\nhttps://a.example/,18,legitimate\nhttp://b.example/login,22,phishing\n"
)


class Response(io.BytesIO):
    def __init__(self, data: bytes, advertised_length: int | None = None) -> None:
        super().__init__(data)
        self.headers = {
            "Content-Length": str(len(data) if advertised_length is None else advertised_length)
        }

    def geturl(self) -> str:
        return "https://files.example/object"


def write_config(tmp_path: Path, content: bytes = CSV, **changes) -> Path:
    dataset = {
        "title": "Synthetic",
        "doi": "10.0000/synthetic",
        "version": 1,
        "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "file": "synthetic.csv",
        "download_url": "https://data.example/file",
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "expected_rows": 2,
        "expected_labels": {"legitimate": 1, "phishing": 1},
        "columns_used": ["url", "status"],
        "label_map": {"phishing": 1, "legitimate": 0},
    } | changes
    path = tmp_path / "external-validation.json"
    path.write_text(json.dumps({"dataset": dataset}), encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def no_network_or_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        downloader, "urlopen", lambda *a, **kw: pytest.fail("Unit test hit the network")
    )
    monkeypatch.setattr(downloader.time, "sleep", lambda seconds: None)


def serve(monkeypatch: pytest.MonkeyPatch, content: bytes) -> list[str]:
    calls: list[str] = []

    def respond(request, timeout):
        calls.append(request.full_url)
        return Response(content)

    monkeypatch.setattr(downloader, "urlopen", respond)
    return calls


def test_verified_download_writes_file_and_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = serve(monkeypatch, CSV)
    destination = tmp_path / "external" / "snapshot"
    manifest, reused = downloader.acquire(destination, write_config(tmp_path))
    assert not reused and calls == ["https://data.example/file"]
    assert (destination / "synthetic.csv").read_bytes() == CSV
    assert manifest["file"]["sha256"] == hashlib.sha256(CSV).hexdigest()
    assert manifest["inspection"] == {
        "columns": 3,
        "rows": 2,
        "labels": {"legitimate": 1, "phishing": 1},
    }
    assert manifest["final_download_host"] == "files.example"
    assert manifest["license"]["name"] == "CC BY 4.0" and manifest["doi"] == "10.0000/synthetic"
    assert manifest["acquired_at_utc"].endswith("+00:00")
    saved = json.loads((destination / downloader.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert saved == manifest
    assert not list((tmp_path / "external").glob(".hannousse-*"))


def test_intact_snapshot_is_reused_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    serve(monkeypatch, CSV)
    config = write_config(tmp_path)
    destination = tmp_path / "snapshot"
    original, _ = downloader.acquire(destination, config)
    monkeypatch.setattr(downloader, "urlopen", lambda *a, **kw: pytest.fail("Re-downloaded"))
    manifest, reused = downloader.acquire(destination, config)
    assert reused and manifest == original


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"sha256": "0" * 64}, "SHA-256 mismatch"),
        ({"size_bytes": 1}, "Size mismatch"),
        ({"expected_rows": 3}, "protocol differs"),
        ({"expected_labels": {"legitimate": 2, "phishing": 0}}, "protocol differs"),
        ({"columns_used": ["url", "status", "absent"]}, "lacks required columns"),
        ({"label_map": {"phishing": 1}}, "outside the protocol's label map"),
    ],
)
def test_mismatch_fails_and_publishes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changes: dict, message: str
) -> None:
    serve(monkeypatch, CSV)
    config = write_config(tmp_path, **changes)
    parent = tmp_path / "external"
    with pytest.raises(downloader.AcquisitionError, match=message):
        downloader.acquire(parent / "snapshot", config)
    assert list(parent.iterdir()) == []


def test_changed_snapshot_is_rejected_and_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    serve(monkeypatch, CSV)
    config = write_config(tmp_path)
    destination = tmp_path / "snapshot"
    downloader.acquire(destination, config)
    changed = destination / "synthetic.csv"
    changed.write_bytes(CSV.replace(b"18", b"19"))
    with pytest.raises(downloader.AcquisitionError, match="incomplete or inconsistent"):
        downloader.acquire(destination, config)
    assert changed.read_bytes() == CSV.replace(b"18", b"19")


def test_partial_snapshot_without_manifest_is_rejected(tmp_path: Path) -> None:
    destination = tmp_path / "snapshot"
    destination.mkdir()
    (destination / "synthetic.csv").write_bytes(CSV)
    with pytest.raises(downloader.AcquisitionError, match="incomplete or inconsistent"):
        downloader.acquire(destination, write_config(tmp_path))


def test_transient_errors_retry_then_fail_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def unavailable(*args, **kwargs):
        calls.append(1)
        raise HTTPError("https://data.example/file", 503, "unavailable", {}, None)

    monkeypatch.setattr(downloader, "urlopen", unavailable)
    parent = tmp_path / "external"
    with pytest.raises(downloader.AcquisitionError, match="after 3 attempts"):
        downloader.acquire(parent / "snapshot", write_config(tmp_path))
    assert len(calls) == 3 and list(parent.iterdir()) == []


def test_short_body_is_retried_and_definitive_errors_are_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    responses = [Response(b"abc", advertised_length=100), Response(CSV)]
    monkeypatch.setattr(downloader, "urlopen", lambda *a, **kw: responses.pop(0))
    downloader.acquire(tmp_path / "ok", write_config(tmp_path))
    assert (tmp_path / "ok" / "synthetic.csv").read_bytes() == CSV

    calls = []

    def missing(*args, **kwargs):
        calls.append(1)
        raise HTTPError("https://data.example/file", 404, "not found", {}, None)

    monkeypatch.setattr(downloader, "urlopen", missing)
    with pytest.raises(downloader.AcquisitionError, match="HTTP 404"):
        downloader.acquire(tmp_path / "missing", write_config(tmp_path))
    assert calls == [1]


def test_protocol_identity_matches_the_frozen_config() -> None:
    dataset = downloader.load_dataset_config()
    assert dataset["file"] == "dataset_B_05_2020.csv"
    assert dataset["size_bytes"] == 3661166
    assert downloader.DEFAULT_DESTINATION.parts[-2:] == ("external", "hannousse-2020")


@pytest.mark.parametrize(
    "content", [b"\xef\xbb\xbf" + CSV, CSV.replace(b"\n", b"\r\n")], ids=["bom", "crlf"]
)
def test_bom_and_crlf_are_accepted(tmp_path: Path, content: bytes) -> None:
    path = tmp_path / "synthetic.csv"
    path.write_bytes(content)
    dataset = downloader.load_dataset_config(write_config(tmp_path, content=content))
    assert downloader.inspect_csv(path, dataset)["rows"] == 2


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (CSV + b"https://c.example/,legitimate\n", "wrong width"),
        (CSV.replace(b"a.example", b"a.\xffexample"), "CSV parsing failed"),
    ],
    ids=["short-row", "invalid-utf8"],
)
def test_malformed_csv_is_rejected(tmp_path: Path, content: bytes, message: str) -> None:
    path = tmp_path / "synthetic.csv"
    path.write_bytes(content)
    dataset = downloader.load_dataset_config(write_config(tmp_path, content=content))
    with pytest.raises(downloader.AcquisitionError, match=message):
        downloader.inspect_csv(path, dataset)
