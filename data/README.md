# PhiUSIIL Data Guide

**Acquisition status: pending.** No dataset snapshot, checksum, local schema validation, or label-count verification is present yet.

## Source and attribution

Prasad, A. & Chandra, S. (2024). *PhiUSIIL Phishing URL (Website)*. UCI Machine Learning Repository. [Dataset page](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset). [Associated publication](https://doi.org/10.1016/j.cose.2023.103545).

The source lists 235,795 instances, 54 features, 134,850 legitimate examples, and 100,945 phishing examples. Original labels are `1 = legitimate`, `0 = phishing`; the modeling target is `is_phishing = 1 - label`. These are published values, not locally verified counts.

The dataset is distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Retain attribution, the license link, and an indication of modifications when sharing derived data. The repository's MIT license does not replace the dataset license.

## Acquisition and storage

Download the archive from the UCI dataset page, retaining the original archive and extracted CSV under `data/raw/`. Alternatively, `ucimlrepo.fetch_ucirepo(id=967)` is the documented retrieval interface; if used, preserve the retrieved table and record its serialization and library version. A serialized retrieval and an original download are different snapshot formats and can have different checksums.

Record source URL, acquisition time in UTC, source version when available, filenames, byte sizes, SHA-256 checksums, and retrieval method in a future acquisition manifest. Inspect column names/types, missing values, duplicate URLs, labels, and counts before sampling. Source version information that is unavailable must be marked unavailable, not invented.

For a locally downloaded CSV, PowerShell can calculate the checksum with:

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath data/raw/PhiUSIIL_Phishing_URL_Dataset.csv
```

The path is a planned local location; download and create it before using the command. Store transformed datasets and split manifests under `data/processed/`; both raw and processed data directories are ignored by Git. Publish only reviewed attribution and non-sensitive reproducibility metadata with reports. Do not commit the full CSV or follow the suspicious URLs in it.

## Required records before the benchmark

- Snapshot checksum and acquisition metadata: **pending**.
- Local schema, label mapping, and counts: **pending**.
- Deduplication, invalid-host exclusions, and Public Suffix List configuration: **pending**.
- Domain-disjoint 24/12 split and seed 42 manifest: **pending**.
- Four ordered numerical features and training-fitted scaler: **pending**.

See the [benchmark protocol](../docs/benchmark-protocol.md) for selection and leakage controls and the [README](../README.md) for project status.
