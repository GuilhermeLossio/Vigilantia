# PhiUSIIL Data Guide

**Acquisition status: locally verified.** The original UCI ZIP and CSV were downloaded and structurally checked on October 25, 2026. The acquisition manifest records checksums, source metadata, and observed counts. Downloaded files remain local and are not included in the repository; run the command below on a fresh checkout.

## Source and attribution

Prasad, A. & Chandra, S. (2024). *PhiUSIIL Phishing URL (Website)*. UCI Machine Learning Repository. [Dataset page](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset). [Associated publication](https://doi.org/10.1016/j.cose.2023.103545).

The source lists 235,795 instances, 54 features, 134,850 legitimate examples, and 100,945 phishing examples. The local CSV has 235,795 data rows and 56 columns, including `FILENAME` and `label`; its class counts match the source. Original labels are `1 = legitimate`, `0 = phishing`; the modeling target is `is_phishing = 1 - label`. The downloader validates original labels but does not transform them.

The dataset is distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Retain attribution, the license link, and an indication of modifications when sharing derived data. The repository's MIT license does not replace the dataset license.

## Acquisition and storage

From the repository root, run:

```powershell
python scripts/download_dataset.py
```

The script uses only the Python standard library; it does not need pandas, Qiskit, or `requirements.txt`. When running from another directory, pass the script's absolute or relative path. Its output is always resolved relative to the repository, not the current working directory.

The script streams the original UCI ZIP with a 30-second socket timeout and at most three attempts for transient network failures or HTTP 408/429/500/502/503/504 responses. Other HTTP failures stop immediately. Retries restart the staged download rather than appending to partial content. The timeout applies to network operations, not to a guaranteed total runtime.

It verifies archive CRCs, extracts only the unique expected CSV member to a fixed filename, and checks UTF-8 CSV structure, duplicate/empty column names, required `URL`/`Domain`/`label` columns, row widths, and binary labels. Empty or whitespace-only cells are counted per column; textual missing-value sentinels such as `NA` are not inferred as missing. Counts differing from UCI produce warnings in the manifest and terminal, rather than discarding an otherwise structurally valid snapshot.

Successful output under `data/raw/`:

- `phiusiil.zip`: original downloaded archive, unchanged.
- `PhiUSIIL_Phishing_URL_Dataset.csv`: original extracted CSV, unchanged.
- `acquisition.json`: acquisition timestamp in UTC, source/final download URLs, attribution, license, source-version availability, file sizes and SHA-256 hashes, column names, row/label counts, empty-cell counts, and discrepancy warnings.

Acquisition and validation occur in a temporary sibling directory. The complete directory is published only after successful checks; failures remove staged files. Run the same command again to verify both hashes and inspection statistics and reuse the snapshot without downloading. Existing incomplete or inconsistent snapshots cause an error and are never silently replaced. Preserve/move the existing `data/raw/` aside before requesting a fresh acquisition. Staging directories are also ignored by Git in case the process is forcibly interrupted.

The source version is recorded as `unavailable`; neither an HTTP timestamp nor acquisition time is presented as a dataset version. SHA-256 values identify the files acquired locally and are not verification against a publisher-provided reference hash.

Alternatively, `ucimlrepo.fetch_ucirepo(id=967)` is UCI's documented retrieval interface, but it is not used by this script. Serialized API retrieval and original downloads are different snapshot formats and can have different checksums.

## Verified local snapshot

The verified snapshot contains 235,795 rows: 100,945 with label `0` and 134,850 with label `1`. All 56 columns have zero empty/whitespace-only cells. No published-count discrepancies were found.

- Archive: 15,376,007 bytes; SHA-256 `0a639fd03aea6308c5b1c10c92aa23c2ce1505447a9137271865cd0badc9a59a`.
- CSV: 56,854,345 bytes; SHA-256 `a236549cd369cd80bd478ff8e1779cbf44c58d5c3f79f7a51a1adbed7d06d1c6`.

These values describe this acquisition, not a permanent guarantee about future UCI downloads. Feature-type/provenance analysis, URL deduplication, domain validation, feature selection, and train/test splits remain pending.

For a locally downloaded CSV, PowerShell can calculate the checksum with:

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath data/raw/PhiUSIIL_Phishing_URL_Dataset.csv
```

Run the downloader before using that checksum command on a fresh checkout. Store future transformed datasets and split manifests under `data/processed/`; both raw and processed data directories are ignored by Git. The acquisition manifest is also local and ignored. Publish only reviewed attribution and non-sensitive reproducibility metadata with reports. Do not commit the full CSV or follow the suspicious URLs in it.

## Required records before the benchmark

- Snapshot checksum and acquisition metadata: **verified locally**.
- Structural CSV checks, original labels, row/class counts, and empty cells: **verified locally**.
- Feature types/provenance and modeling-target transformation: **pending**.
- Deduplication, invalid-host exclusions, and Public Suffix List configuration: **pending**.
- Domain-disjoint 24/12 split and seed 42 manifest: **pending**.
- Four ordered numerical features and training-fitted scaler: **pending**.

See the [benchmark protocol](../docs/benchmark-protocol.md) for selection and leakage controls and the [README](../README.md) for project status.
