"""Acquire and verify the external validation CSV using only the standard library.

The file, its size, and its SHA-256 are fixed by configs/external-validation.json. The
download is staged in a sibling directory and published only after every check passes;
a mismatch fails without leaving a valid-looking snapshot in data/external/.
"""

from __future__ import annotations

import csv
import io
import json
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from http.client import IncompleteRead
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.download_dataset import (  # noqa: E402
    CHUNK_SIZE,
    MAX_ATTEMPTS,
    TIMEOUT_SECONDS,
    TRANSIENT_HTTP_CODES,
    AcquisitionError,
    file_record,
)

CONFIG_PATH = ROOT / "configs" / "external-validation.json"
DEFAULT_DESTINATION = ROOT / "data" / "external" / "hannousse-2020"
MANIFEST_NAME = "acquisition.json"
ATTRIBUTION = (
    "Hannousse, A. and Yahiouche, S. (2020). Web page phishing detection, version 2. "
    "Mendeley Data. DOI 10.17632/c2gw7fy2j4.2. CC BY 4.0."
)


def load_dataset_config(path: Path = CONFIG_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))["dataset"]


def download_file(url: str, path: Path) -> str:
    """Stream to a staging file; restart bounded transient failures from byte zero.

    Mirrors scripts/download_dataset.download_archive; apply retry fixes to both.
    """
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            request = Request(url, headers={"User-Agent": "Vigilantia-dataset-acquisition/1"})
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


def verify_identity(record: dict, dataset: dict) -> None:
    if record["bytes"] != dataset["size_bytes"]:
        raise AcquisitionError(
            f"Size mismatch: {record['bytes']} bytes, expected {dataset['size_bytes']}."
        )
    if record["sha256"] != dataset["sha256"]:
        raise AcquisitionError("SHA-256 mismatch with configs/external-validation.json.")


def inspect_csv(path: Path, dataset: dict) -> dict:
    """Count rows and labels; require the columns and counts the protocol names."""
    try:
        text = path.read_text(encoding="utf-8-sig")
        reader = csv.reader(io.StringIO(text, newline=""), strict=True)
        header = next(reader, None) or []
        missing = [name for name in dataset["columns_used"] if name not in header]
        if missing:
            raise AcquisitionError(f"CSV lacks required columns: {missing}.")
        status = header.index("status")
        labels: Counter[str] = Counter()
        rows = 0
        for row in reader:
            if len(row) != len(header):
                raise AcquisitionError(f"CSV row ending at line {reader.line_num} has wrong width.")
            labels[row[status]] += 1
            rows += 1
    except (csv.Error, UnicodeError) as error:
        raise AcquisitionError(f"CSV parsing failed: {error}") from error
    if set(labels) - set(dataset["label_map"]):
        raise AcquisitionError("CSV has a status value outside the protocol's label map.")
    if rows != dataset["expected_rows"] or dict(labels) != dataset["expected_labels"]:
        raise AcquisitionError(f"Observed {rows} rows and labels {dict(labels)}; protocol differs.")
    return {"columns": len(header), "rows": rows, "labels": dict(sorted(labels.items()))}


def reuse_existing(destination: Path, dataset: dict) -> dict:
    """Only reuse a complete snapshot whose file still matches the protocol identity."""
    advice = "Move the existing data/external snapshot aside before a fresh acquisition."
    try:
        manifest = json.loads((destination / MANIFEST_NAME).read_text(encoding="utf-8"))
        record = file_record(destination / dataset["file"])
        verify_identity(record, dataset)
        if manifest["schema_version"] != 1 or manifest["file"] != record:
            raise ValueError("manifest does not describe the file on disk")
        if manifest["doi"] != dataset["doi"] or not manifest["acquired_at_utc"]:
            raise ValueError("missing or unexpected acquisition metadata")
        if manifest["inspection"] != inspect_csv(destination / dataset["file"], dataset):
            raise ValueError("inspection metadata mismatch")
    except (OSError, ValueError, KeyError, TypeError, AcquisitionError) as error:
        raise AcquisitionError(
            f"Existing acquisition is incomplete or inconsistent ({error}). {advice}"
        ) from error
    return manifest


def acquire(
    destination: Path = DEFAULT_DESTINATION, config_path: Path = CONFIG_PATH
) -> tuple[dict, bool]:
    """Return the acquisition manifest and whether an existing snapshot was reused."""
    dataset = load_dataset_config(config_path)
    if destination.exists():
        return reuse_existing(destination, dataset), True
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".hannousse-", dir=destination.parent) as temporary:
        staged = Path(temporary) / "snapshot"
        staged.mkdir()
        csv_path = staged / dataset["file"]
        print("Downloading the external validation CSV from Mendeley Data...", flush=True)
        final_url = download_file(dataset["download_url"], csv_path)
        record = file_record(csv_path)
        verify_identity(record, dataset)
        manifest = {
            "schema_version": 1,
            "title": dataset["title"],
            "doi": dataset["doi"],
            "version": dataset["version"],
            "source_url": dataset["download_url"],
            "final_download_host": final_url.split("/")[2] if "://" in final_url else None,
            "acquired_at_utc": datetime.now(UTC).isoformat(),
            "retrieval_method": "Mendeley Data public file via Python urllib",
            "license": {"name": dataset["license"], "url": dataset["license_url"]},
            "attribution": ATTRIBUTION,
            "file": record,
            "inspection": inspect_csv(csv_path, dataset),
        }
        (staged / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        if destination.exists():
            raise AcquisitionError(
                "Destination appeared during download; existing files preserved."
            )
        staged.rename(destination)
    return manifest, False


def main() -> int:
    try:
        manifest, reused = acquire()
    except (AcquisitionError, OSError, ValueError, KeyError) as error:
        print(f"Acquisition failed: {error}", file=sys.stderr)
        return 1
    report = manifest["inspection"]
    print(
        "Verified existing snapshot; no download performed." if reused else "Acquisition complete."
    )
    print(f"Rows: {report['rows']}; labels: {report['labels']}; columns: {report['columns']}")
    print(f"Manifest: {(DEFAULT_DESTINATION / MANIFEST_NAME).relative_to(ROOT).as_posix()}")
    print(f"CSV SHA-256: {manifest['file']['sha256']} (matches the protocol)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
