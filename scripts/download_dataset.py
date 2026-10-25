"""Acquire and validate the original PhiUSIIL archive using only the standard library."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from http.client import IncompleteRead
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile

SOURCE_PAGE = "https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset"
DOWNLOAD_URL = (
    "https://archive.ics.uci.edu/static/public/967/phiusiil%2Bphishing%2Burl%2Bdataset.zip"
)
CSV_NAME = "PhiUSIIL_Phishing_URL_Dataset.csv"
ZIP_NAME = "phiusiil.zip"
MANIFEST_NAME = "acquisition.json"
DEFAULT_DESTINATION = Path(__file__).resolve().parents[1] / "data" / "raw"
EXPECTED_COUNTS = {"rows": 235795, "labels": {"0": 100945, "1": 134850}}
TIMEOUT_SECONDS = 30
MAX_ATTEMPTS = 3
CHUNK_SIZE = 1024 * 1024
TRANSIENT_HTTP_CODES = {408, 429, 500, 502, 503, 504}


class AcquisitionError(Exception):
    """An acquisition failed validation or cannot safely reuse existing files."""


def file_record(path: Path) -> dict:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"filename": path.name, "bytes": path.stat().st_size, "sha256": digest}


def download_archive(path: Path) -> str:
    """Stream to a staging file; restart bounded transient failures from byte zero."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            request = Request(
                DOWNLOAD_URL, headers={"User-Agent": "Vigilantia-dataset-acquisition/1"}
            )
            with urlopen(request, timeout=TIMEOUT_SECONDS) as response, path.open("wb") as target:
                length = response.headers.get("Content-Length")
                received = 0
                while chunk := response.read(CHUNK_SIZE):
                    target.write(chunk)
                    received += len(chunk)
                if length is not None and received != int(length):
                    raise IncompleteRead(b"", max(0, int(length) - received))
                return response.geturl()
        except HTTPError as error:
            error.close()
            if error.code not in TRANSIENT_HTTP_CODES:
                raise AcquisitionError(
                    f"Download failed: HTTP {error.code} (not retried)."
                ) from error
            reason = f"HTTP {error.code}"
        except (URLError, TimeoutError, ConnectionError, IncompleteRead) as error:
            reason = str(error)
        if attempt == MAX_ATTEMPTS:
            raise AcquisitionError(f"Download failed after {MAX_ATTEMPTS} attempts: {reason}")
        print(f"Download attempt {attempt} failed ({reason}); retrying.", file=sys.stderr)
        time.sleep(2 ** (attempt - 1))
    raise AssertionError("Unreachable retry state")


def extract_csv(archive: Path, destination: Path) -> str:
    """Verify archive CRCs and copy only the unique expected member to a fixed path."""
    try:
        with ZipFile(archive) as zipped:
            bad_member = zipped.testzip()
            if bad_member is not None:
                raise AcquisitionError(f"ZIP integrity check failed for {bad_member}.")
            matches = [
                member
                for member in zipped.infolist()
                if not member.is_dir() and PurePosixPath(member.filename).name == CSV_NAME
            ]
            if len(matches) != 1:
                raise AcquisitionError("ZIP must contain exactly one expected CSV member.")
            member = matches[0]
            # Never use extractall or trust the ZIP member's path on the filesystem.
            with zipped.open(member) as source, destination.open("xb") as target:
                shutil.copyfileobj(source, target, length=CHUNK_SIZE)
            return member.filename
    except (BadZipFile, RuntimeError, NotImplementedError) as error:
        raise AcquisitionError(f"Invalid or unsupported ZIP: {error}") from error


def inspect_csv(path: Path) -> dict:
    """Stream structural checks and counts without modifying or selecting records."""
    previous_limit = csv.field_size_limit()
    csv.field_size_limit(10 * CHUNK_SIZE)
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream, strict=True)
            header = next(reader, None)
            if not header or any(not name.strip() for name in header):
                raise AcquisitionError("CSV has an empty header or unnamed columns.")
            if len(set(header)) != len(header):
                raise AcquisitionError("CSV contains duplicate column names.")
            if not {"URL", "Domain", "label"}.issubset(header):
                raise AcquisitionError("CSV must contain URL, Domain and label columns.")
            label_index = header.index("label")
            empty = dict.fromkeys(header, 0)
            labels: Counter[str] = Counter({"0": 0, "1": 0})
            total = 0
            for row in reader:
                if len(row) != len(header):
                    raise AcquisitionError(
                        f"CSV row ending at line {reader.line_num} has wrong width."
                    )
                if row[label_index] not in {"0", "1"}:
                    raise AcquisitionError(f"CSV has an invalid label at line {reader.line_num}.")
                labels[row[label_index]] += 1
                total += 1
                for name, value in zip(header, row, strict=True):
                    if not value.strip():
                        empty[name] += 1
            if total == 0:
                raise AcquisitionError("CSV contains no data rows.")
    except (csv.Error, UnicodeError) as error:
        raise AcquisitionError(f"CSV parsing failed: {error}") from error
    finally:
        csv.field_size_limit(previous_limit)
    warnings = []
    if total != EXPECTED_COUNTS["rows"]:
        warnings.append(f"Observed {total} rows; UCI publishes {EXPECTED_COUNTS['rows']}.")
    for label, expected in EXPECTED_COUNTS["labels"].items():
        if labels[label] != expected:
            warnings.append(f"Label {label}: observed {labels[label]}; UCI publishes {expected}.")
    return {
        "columns": header,
        "column_count": len(header),
        "rows": total,
        "labels": dict(labels),
        "empty_cells_by_column": empty,
        "published_counts": EXPECTED_COUNTS,
        "counts_match_published": not warnings,
        "warnings": warnings,
    }


def reuse_existing(destination: Path) -> dict:
    """Only reuse a complete snapshot with matching file identities and manifest schema."""
    advice = "Preserve/move the existing data/raw snapshot aside before a fresh acquisition."
    try:
        manifest = json.loads((destination / MANIFEST_NAME).read_text(encoding="utf-8"))
        if manifest["schema_version"] != 1 or manifest["dataset_id"] != 967:
            raise ValueError("unsupported manifest")
        for key, name in (("archive", ZIP_NAME), ("csv", CSV_NAME)):
            if manifest["files"][key] != file_record(destination / name):
                raise ValueError(f"checksum, name or size mismatch for {name}")
        # Validate report coherence too, rather than trusting edited manifest statistics.
        if manifest["inspection"] != inspect_csv(destination / CSV_NAME):
            raise ValueError("inspection metadata mismatch")
        if manifest["source_url"] != DOWNLOAD_URL or not manifest["acquired_at_utc"]:
            raise ValueError("missing or unexpected acquisition metadata")
    except (OSError, ValueError, KeyError, TypeError, AcquisitionError) as error:
        raise AcquisitionError(
            f"Existing acquisition is incomplete or inconsistent. {advice}"
        ) from error
    return manifest


def acquire(destination: Path = DEFAULT_DESTINATION) -> tuple[dict, bool]:
    """Validate in a sibling staging directory, then publish the entire snapshot at once."""
    if destination.exists():
        return reuse_existing(destination), True
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".phiusiil-", dir=destination.parent) as temporary:
        staged = Path(temporary) / "snapshot"
        staged.mkdir()
        archive, csv_path = staged / ZIP_NAME, staged / CSV_NAME
        print("Downloading the original PhiUSIIL archive from UCI...", flush=True)
        final_url = download_archive(archive)
        member = extract_csv(archive, csv_path)
        inspection = inspect_csv(csv_path)
        manifest = {
            "schema_version": 1,
            "dataset_id": 967,
            "dataset_name": "PhiUSIIL Phishing URL (Website)",
            "source_page": SOURCE_PAGE,
            "source_url": DOWNLOAD_URL,
            "final_download_url": final_url,
            "source_version": "unavailable",
            "acquired_at_utc": datetime.now(UTC).isoformat(),
            "retrieval_method": "original UCI ZIP via Python urllib",
            "license": {"name": "CC BY 4.0", "url": "https://creativecommons.org/licenses/by/4.0/"},
            "attribution": "Prasad, A. & Chandra, S. (2024). PhiUSIIL. UCI Machine Learning Repository.",
            "checksum_note": "Locally computed identity hashes; no official reference hash provided.",
            "csv_archive_member": member,
            "files": {"archive": file_record(archive), "csv": file_record(csv_path)},
            "inspection": inspection,
        }
        (staged / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        if destination.exists():
            raise AcquisitionError(
                "Destination appeared during download; existing files preserved."
            )
        # Same-filesystem directory rename: no valid-looking partially published snapshot.
        staged.rename(destination)
    return manifest, False


def main() -> int:
    try:
        manifest, reused = acquire()
    except (AcquisitionError, OSError, ValueError) as error:
        print(f"Acquisition failed: {error}", file=sys.stderr)
        return 1
    report = manifest["inspection"]
    print(
        "Verified existing snapshot; no download performed." if reused else "Acquisition complete."
    )
    print(f"Rows: {report['rows']}; labels: {report['labels']}; columns: {report['column_count']}")
    print(f"Manifest: {DEFAULT_DESTINATION / MANIFEST_NAME}")
    print(f"CSV SHA-256: {manifest['files']['csv']['sha256']}")
    for warning in report["warnings"]:
        print(f"WARNING: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
