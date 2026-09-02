# Data Governance and Reproducibility Rules

These rules apply to every dataset in the SS1 research pipeline.

## Raw data is immutable

- Treat `data/raw/` as read-only after download.
- Do not clean, rename, deduplicate or overwrite raw files.
- Write all derived data to `data/interim/` or `data/processed/`.
- Re-run the raw-data audit whenever a raw file is added or replaced.

## Provenance register

Maintain [dataset-provenance.json](../data/metadata/dataset-provenance.json) for every dataset. Before using a dataset in analysis, its entry must include:

- source and stable download location;
- citation and version;
- licence or other access/use terms;
- download date;
- intended role in the research pipeline.

The Essays dataset currently has unresolved provenance fields. Do not mark its preparation complete until these are verified.

## Raw-data audit

Start the project-local watcher from the project root while working with raw data:

```bash
make watch-data
```

The watcher runs an initial audit and refreshes [raw-data-inventory.json](../data/metadata/raw-data-inventory.json) after CSV or TSV files are added or changed. It stops with `Ctrl+C` or when its terminal closes. The inventory records the row count, schema, file size, modification time and SHA-256 checksum for each raw CSV/TSV; comparing checksums confirms that raw source files have not changed.

For a one-off audit, run `make audit-data`.

## Identifiers and model features

Keep source identifiers only for auditing, grouping, duplicate checks and leakage checks. Do not use author IDs, Reddit usernames, thread IDs, timestamps, subreddit names or comparable identifiers as model features.

## Data minimisation and responsible handling

- Do not attempt to re-identify authors or link records to external profiles.
- Do not publish raw responses, usernames or unusually distinctive examples in reports unless the source terms and ethics review explicitly permit it.
- Treat free text as potentially containing personal, offensive or distressing material; limit access to the research team and report aggregate findings by default.

## Reproducible transformations

- Every transformation must be performed by a versioned script or notebook.
- Preserve raw text in a separate field when creating cleaned model text.
- Record all filtering and label-aggregation decisions.
- Set and record a random seed before creating any train, validation and test split.
- Freeze a split once model comparisons begin; do not tune against the held-out test set.

## Data-quality reporting

Before modelling each dataset, save a short report covering schema validation, missingness, duplicates, text quality, label distributions, exclusions and potential leakage risks.
