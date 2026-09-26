# PromptShield

Compact multilingual prompt-injection detection project. Tasks 1–2 implement a CPU-only
dataset pipeline, TF-IDF/logistic regression and an unmodified external DeBERTa baseline.
Real internal evaluation and a PINT public-example evaluation are available.
Detection is one layer of defense in depth, never a guaranteed defense.

## Setup and build

Use CPython 3.14.7 and uv. Dependencies are declared in `pyproject.toml` and resolved in
`uv.lock`; CPU model and evaluation dependencies are included.

```sh
uv python install
uv sync --locked
uv run promptshield-data fetch
uv run promptshield-data build
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
```

On the current Windows workstation, uv and Python were installed locally under `.tools/`
because C: is full. Start a PowerShell session in this repository and run
`. ./scripts/activate-local.ps1` before the commands above. The helper uses D: for caches
and temporary files. Other machines with uv already on PATH do not need it.
Pyright includes its own current Node runtime; it does not rely on the old system Node.

The first fetch needs network access. Subsequent builds are offline. Fetch verifies SHA-256
and refuses to replace existing mismatched raw bytes. Public source version, selection,
licensing ambiguity and annotation policy are documented in `data/SOURCES.md`.
The curated JSONL sources are small and versioned; raw downloads and processed artifacts
are ignored by Git. PINT is evaluation-only; AgentDojo is not implemented.

## Baselines and evaluation

Run `uv run python -m promptshield.external` to acquire checksum-pinned PINT example
data and the official public DeBERTa snapshot. The snapshot path and source metadata
are written to `artifacts/external_status.json`. Then run:

```sh
uv run promptshield-baselines --output artifacts/baselines/my-run --deberta-snapshot .cache/huggingface/hub/models--protectai--deberta-v3-base-prompt-injection-v2/snapshots/90c9989b1a342275dd0d1a95aad283c04e075671 --pint data/raw/pint/0efab3f463eae9c823130d8faffb71b2e7c06e63/example-dataset.yaml --pint-sha256 df068b9a4ff72483f493add6be6242c6aa777df756bd61462aa0e13645cffa90
```

Use a new output directory for each run. The shared `Detector` interface provides
`score`, `predict`, configurable threshold, batches and model identity. TF-IDF fits
training only and uses a validation-selected threshold. DeBERTa uses fixed 0.5,
unmodified weights and no calibration. Russian results measure cross-lingual
generalization of an English model. No custom transformer has been trained.

Completed run: `artifacts/baselines/task2-complete/`. Each detector has identity,
threshold and latency JSON plus validation/internal_test/pint metrics JSON,
predictions JSONL and targeted errors JSON. TF-IDF's trusted local serialized model
is `tfidf/model.joblib`; never load an untrusted joblib file. Local MLflow records are
under `mlruns/`; no tracking server is needed. Generated data/models/results stay
Git ignored. [BASELINE_REPORT.md](BASELINE_REPORT.md) records results, source links,
taxonomy, limitations and reproducibility details. The public eight-row PINT example
is a smoke evaluation, not the full PINT score. Lightweight tests require no downloads.

## Outputs and reproducibility

`data/processed/v1/` contains `all.parquet`, `train.parquet`, `validation.parquet`,
`test.parquet`, `report.json`, and `manifest.json`. `DATA_CARD.md` is generated from the
same processed records. The manifest records source hashes, source row provenance,
annotation hashes, dependency lock/code hashes, Python version, Git commit (null before
the initial commit), split configuration, exact sample assignments and artifact hashes.

Repeat the same build to verify byte equality with the existing output. To compare an
independent build, use:

```sh
uv run promptshield-data build --output data/processed/reproduction --card data/REPRO_CARD.md
```

Builds refuse to overwrite different existing artifacts. A changed source/configuration,
implementation or dependency lock requires a new output version directory. The logical
fingerprint excludes the Git commit so committing unchanged source files does not change
the dataset identity; the manifest still records the execution commit. Parquet byte
reproducibility is verified on the supported local environment, not asserted across all
operating systems or dependency upgrades.

## Input contract

Canonical JSONL, CSV and Parquet contain exactly these fields:

`id`, `text`, `label`, `injection_mode`, `language`, `content_role`, `source`, `group_id`,
`metadata`.

`promptshield.schema.read_records` validates them strictly. CSV encodes `metadata` as a
JSON object and a null injection mode as an empty cell. Produced Parquet uses nullable
UTF-8 columns and a JSON-encoded metadata column for stable interoperability; the reader
also accepts native Parquet metadata structs. IDs/text are never silently coerced.
Blank fields, invalid taxonomy, missing/extra columns and contradictory annotations fail.

Use `data/dataset.json` as the build configuration example. Canonical inputs use adapter
`canonical`; the sole public adapter `deepset_reviewed` checks an explicit allowlist against
the pinned upstream rows and labels. A source checksum must match before ingestion.
New input paths must stay inside the repository. Group IDs are globally scoped: assign
translations, paraphrases, templates and known attack-family variants the same group.
Unknown language/role/mode must be marked unknown rather than inferred from keywords.

Exact and normalized duplicates are merged with member provenance; conflicting duplicate
labels fail. Lexical near matches are retained but conservatively grouped. The exhaustive
near audit supports at most 5,000 input records and fails explicitly above that limit;
larger corpora require a separate scalable audit design, not unaudited chunking.

All splits are reserved once built. Never tune models, thresholds, calibration or
augmentation against test data. This 100-record starter is too small for reliable detector
evaluation, particularly Russian subgroups and low-FPR metrics. Read `DATA_CARD.md` before use.
