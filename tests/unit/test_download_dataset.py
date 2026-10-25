"""Acquisition behavior using tiny archives and HTTP doubles; never contacts UCI."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from urllib.error import HTTPError
from zipfile import ZIP_STORED, ZipFile

import pytest

from scripts import download_dataset as downloader

pytestmark = pytest.mark.unit
VALID_CSV = b"URL,Domain,label\nhttps://a.invalid,a.invalid,1\nhttps://b.invalid,,0\n"


def archive_bytes(members: dict[str, bytes] | None = None) -> bytes:
    buffer = io.BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_STORED) as archive:
        for name, data in (
            members if members is not None else {downloader.CSV_NAME: VALID_CSV}
        ).items():
            archive.writestr(name, data)
    return buffer.getvalue()


class Response(io.BytesIO):
    def __init__(self, data: bytes, advertised_length: int | None = None) -> None:
        super().__init__(data)
        self.headers = {
            "Content-Length": str(len(data) if advertised_length is None else advertised_length)
        }
        self.read_sizes: list[int] = []

    def geturl(self) -> str:
        return downloader.DOWNLOAD_URL

    def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        return super().read(size)


@pytest.fixture(autouse=True)
def no_network_or_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args, **kwargs):
        pytest.fail("Unit test attempted an unmocked network call")

    monkeypatch.setattr(downloader, "urlopen", forbidden)
    monkeypatch.setattr(downloader.time, "sleep", lambda seconds: None)


def mock_archive(monkeypatch: pytest.MonkeyPatch, content: bytes) -> Response:
    response = Response(content)
    monkeypatch.setattr(downloader, "urlopen", lambda *args, **kwargs: response)
    return response


def test_complete_acquisition_preserves_original_files_and_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = archive_bytes()
    response = mock_archive(monkeypatch, content)
    destination = tmp_path / "raw"
    manifest, reused = downloader.acquire(destination)
    assert not reused
    assert set(p.name for p in destination.iterdir()) == {
        downloader.ZIP_NAME,
        downloader.CSV_NAME,
        downloader.MANIFEST_NAME,
    }
    assert (destination / downloader.ZIP_NAME).read_bytes() == content
    assert (destination / downloader.CSV_NAME).read_bytes() == VALID_CSV
    assert manifest["files"]["csv"]["sha256"] == hashlib.sha256(VALID_CSV).hexdigest()
    assert manifest["files"]["archive"]["bytes"] == len(content)
    assert manifest["source_version"] == "unavailable"
    assert manifest["acquired_at_utc"].endswith("+00:00")
    assert manifest["final_download_url"] == downloader.DOWNLOAD_URL
    assert manifest["inspection"]["labels"] == {"0": 1, "1": 1}
    assert manifest["inspection"]["empty_cells_by_column"]["Domain"] == 1
    assert manifest["inspection"]["warnings"]
    assert response.read_sizes and set(response.read_sizes) == {downloader.CHUNK_SIZE}
    assert json.loads((destination / downloader.MANIFEST_NAME).read_text()) == manifest
    assert not list(tmp_path.glob(".phiusiil-*"))


def test_reuse_does_not_download_or_change_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_archive(monkeypatch, archive_bytes())
    destination = tmp_path / "raw"
    original, _ = downloader.acquire(destination)
    original_bytes = (destination / downloader.MANIFEST_NAME).read_bytes()
    monkeypatch.setattr(
        downloader, "urlopen", lambda *a, **kw: pytest.fail("Reused data hit network")
    )
    manifest, reused = downloader.acquire(destination)
    assert reused and manifest == original
    assert (destination / downloader.MANIFEST_NAME).read_bytes() == original_bytes


@pytest.mark.parametrize(
    "name", [downloader.CSV_NAME, downloader.ZIP_NAME, downloader.MANIFEST_NAME]
)
def test_changed_files_are_preserved_and_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    mock_archive(monkeypatch, archive_bytes())
    destination = tmp_path / "raw"
    downloader.acquire(destination)
    changed = destination / name
    changed.write_bytes(b"changed")
    with pytest.raises(downloader.AcquisitionError, match="incomplete or inconsistent"):
        downloader.acquire(destination)
    assert changed.read_bytes() == b"changed"


def test_incomplete_snapshot_is_not_overwritten(tmp_path: Path) -> None:
    destination = tmp_path / "raw"
    destination.mkdir()
    (destination / downloader.CSV_NAME).write_bytes(VALID_CSV)
    with pytest.raises(downloader.AcquisitionError, match="incomplete or inconsistent"):
        downloader.acquire(destination)
    assert list(destination.iterdir()) == [destination / downloader.CSV_NAME]


def test_edited_manifest_counts_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_archive(monkeypatch, archive_bytes())
    destination = tmp_path / "raw"
    manifest, _ = downloader.acquire(destination)
    manifest["inspection"]["rows"] = 999
    path = destination / downloader.MANIFEST_NAME
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(downloader.AcquisitionError, match="incomplete or inconsistent"):
        downloader.acquire(destination)


def test_timeout_is_retried_and_restarts_partial_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class InterruptedResponse(Response):
        def read(self, size: int = -1) -> bytes:
            if self.tell():
                raise TimeoutError("interrupted stream")
            return super().read(3)

    calls = []
    content = archive_bytes()

    def open_response(request, timeout):
        calls.append(timeout)
        return InterruptedResponse(content) if len(calls) == 1 else Response(content)

    monkeypatch.setattr(downloader, "urlopen", open_response)
    destination = tmp_path / "raw"
    downloader.acquire(destination)
    assert calls == [downloader.TIMEOUT_SECONDS] * 2
    assert (destination / downloader.ZIP_NAME).read_bytes() == content


@pytest.mark.parametrize("failure", ["timeout", "503", "short"])
def test_transient_failures_stop_after_three_attempts_without_publishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    calls = []

    def open_response(*args, **kwargs):
        calls.append(1)
        if failure == "timeout":
            raise TimeoutError("unavailable")
        if failure == "503":
            raise HTTPError(downloader.DOWNLOAD_URL, 503, "unavailable", {}, None)
        return Response(b"short", advertised_length=100)

    monkeypatch.setattr(downloader, "urlopen", open_response)
    with pytest.raises(downloader.AcquisitionError, match="after 3 attempts"):
        downloader.acquire(tmp_path / "raw")
    assert len(calls) == 3
    assert list(tmp_path.iterdir()) == []


def test_definitive_http_error_is_not_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def missing(*args, **kwargs):
        calls.append(1)
        raise HTTPError(downloader.DOWNLOAD_URL, 404, "not found", {}, None)

    monkeypatch.setattr(downloader, "urlopen", missing)
    with pytest.raises(downloader.AcquisitionError, match="HTTP 404"):
        downloader.acquire(tmp_path / "raw")
    assert len(calls) == 1
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "content",
    [
        b"not a ZIP",
        archive_bytes({"other.csv": VALID_CSV}),
        archive_bytes({downloader.CSV_NAME: VALID_CSV, "nested/" + downloader.CSV_NAME: VALID_CSV}),
        archive_bytes().replace(b"https://a.invalid", b"https://x.invalid", 1),
    ],
    ids=["invalid", "missing", "ambiguous", "crc-corruption"],
)
def test_bad_archives_do_not_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, content: bytes
) -> None:
    mock_archive(monkeypatch, content)
    with pytest.raises(downloader.AcquisitionError):
        downloader.acquire(tmp_path / "raw")
    assert list(tmp_path.iterdir()) == []


def test_unrelated_archive_members_are_not_extracted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_archive(
        monkeypatch, archive_bytes({downloader.CSV_NAME: VALID_CSV, "../escape.txt": b"no"})
    )
    downloader.acquire(tmp_path / "raw")
    assert not (tmp_path / "escape.txt").exists()
    assert len(list((tmp_path / "raw").iterdir())) == 3


@pytest.mark.parametrize(
    "csv_bytes",
    [
        b"URL,Domain,label,label\na,a,1,1\n",
        b"URL,Domain\na,a\n",
        b"URL,Domain,label\na,a,2\n",
        b"URL,Domain,label\na,a,\n",
        b"URL,Domain,label\na,1\n",
        b"URL,Domain,label\na,a,1,extra\n",
        b'URL,Domain,label\n"unterminated,a,1\n',
        b"URL,Domain,label\n",
        b"",
    ],
    ids=[
        "duplicate",
        "missing-column",
        "invalid-label",
        "empty-label",
        "short-row",
        "long-row",
        "malformed-quote",
        "header-only",
        "empty-file",
    ],
)
def test_invalid_csv_does_not_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, csv_bytes: bytes
) -> None:
    mock_archive(monkeypatch, archive_bytes({downloader.CSV_NAME: csv_bytes}))
    with pytest.raises(downloader.AcquisitionError):
        downloader.acquire(tmp_path / "raw")
    assert list(tmp_path.iterdir()) == []


def test_matching_counts_have_no_warning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(downloader, "EXPECTED_COUNTS", {"rows": 2, "labels": {"0": 1, "1": 1}})
    csv_path = tmp_path / "dataset.csv"
    csv_path.write_bytes(VALID_CSV)
    report = downloader.inspect_csv(csv_path)
    assert report["counts_match_published"]
    assert not report["warnings"]


def test_new_destination_during_download_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "raw"

    def competing_download(path):
        destination.mkdir()
        (destination / "existing.txt").write_text("preserve", encoding="utf-8")
        path.write_bytes(archive_bytes())
        return downloader.DOWNLOAD_URL

    monkeypatch.setattr(downloader, "download_archive", competing_download)
    with pytest.raises(downloader.AcquisitionError, match="appeared during download"):
        downloader.acquire(destination)
    assert (destination / "existing.txt").read_text() == "preserve"
    assert len(list(destination.iterdir())) == 1
