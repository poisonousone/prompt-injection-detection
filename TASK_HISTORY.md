# PromptShield — Task History

Historical task log.

Do not use this file as the primary description of current state.
Current state belongs in PROJECT_STATE.md.

---

## Initialization

Date: YYYY-MM-DD

Repository context files created.

No implementation completed.

Next task: Task 1 — Repository + Dataset Pipeline.

---

## Task 1 — Repository + Dataset Pipeline

Date: 2026-09-25

State: completed in working tree; no Git commits exist yet. No model trained.

Major files: `pyproject.toml`, `uv.lock`, `.python-version`, `.gitattributes`, `.gitignore`,
`src/promptshield/{schema,dataset,pipeline}.py`, `tests/`, `data/dataset.json`,
`data/curated/*.jsonl`, `data/SOURCES.md`, `DATA_CARD.md`, `README.md`,
`scripts/activate-local.ps1`, `.github/workflows/checks.yml`, project context files.

Implemented strict three-format ingestion, pinned allowlist/provenance, duplicate handling,
near audit, group-aware balanced splitting, Parquet outputs and generated reporting.

Commands/checks: local uv/Python installation and stable dependency resolution; targeted
schema/dataset and pipeline tests; `uv sync --locked`; `uv run ruff check .`;
`uv run ruff format --check .`; `uv run pyright`; full `uv run pytest -q --tb=short`
(34 passed); `uv run promptshield-data fetch`; two full dataset builds and SHA-256 comparison
of all six outputs. Ruff/Pyright pass; raw checksum and group isolation verified.

Measured: 100 samples (57 benign, 43 injection), English 68 / Russian 32, hard negatives 32,
protected groups 41. Splits 70/16/14. Exact/normalized-only duplicates 0/0; near warnings 0.
All six artifacts byte-identical across builds. Fingerprint:
`7d653ab1a682f5570392bc379c07e25c97db6ad40c94d1f8e042b5a205008768`.
Primary artifacts: `data/processed/v1/`; public revision and source hashes in manifest/card.

Known issues: small convenience sample; Russian is authored translation data; independent
review and natural Russian data needed; conflicting upstream license fields; lexical audit
cannot establish semantic cleanliness. C: full; toolchain/cache/temp installed on D:.
Initial pytest/tempfile owner-only directories failed under Windows sandbox; an external
test-run permission request was rejected. Workspace directories inheriting existing ACLs
resolved the issue; final tests/builds passed inside the sandbox. Remote CI not executed.

Next expected task: user-supplied Task 2 (anticipated TF-IDF + logistic regression baseline),
with data limitations reviewed before drawing ML conclusions. Not implemented.
