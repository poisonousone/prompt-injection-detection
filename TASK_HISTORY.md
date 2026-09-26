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

---

## Task 2 — Baselines + PINT

Date: 2026-09-26

State: completed within available public-data scope; uncommitted changes on `36c074d`.
Recovered existing partial evaluation implementation; replaced the unavailable external
baseline with official public ProtectAI DeBERTa v2, revision
`90c9989b1a342275dd0d1a95aad283c04e075671`. No custom transformer training.

Major files: `src/promptshield/{predictors,external,baselines}.py`,
`tests/test_deberta.py` (renamed), `tests/test_evaluation.py`, `README.md`,
`BASELINE_REPORT.md`, `PROJECT_STATE.md`, `DECISIONS.md`, this history.
Existing dependency lock and frozen starter-v1 splits retained.

Implemented/verified shared scoring/batch/threshold/identity contract; train-only sklearn
pipeline; fixed 0.5 unmodified DeBERTa; validation-only TF-IDF threshold 0.4403050229585025;
explicit 512-token windows with 64-token overlap; complete category-preserving PINT
example mapping; checksum pins; local MLflow; real metrics/predictions/errors and latency.
Fixed observed overflow-tokenizer tail loss and accepted the three extra categories in
the actual PINT public example. Neither issue was addressed through performance tuning.

Commands/checks: official HF model metadata/config/snapshot download; pinned GitHub
dataset-tree verification; `uv run python -m promptshield.external`; real tokenizer and
weight smoke checks; targeted evaluation/window tests (12 passed); real
`uv run promptshield-baselines --output artifacts/baselines/task2-complete ...`;
`uv sync --locked`; Ruff check/format; strict Pyright; full pytest (46 passed).
All final lightweight checks passed. Remote CI not executed.

Measured internal test n=14 (4 positive/10 negative), TF-IDF / DeBERTa:
AUPRC 0.532576 / 1; Recall @ 1% FPR 0 / 1; FPR @ 95% recall 0.7 / 0;
F1 0.666667 / 0.888889; Brier 0.204418 / 0.068647.
English n=10 F1 0.8 / 1; Russian n=4 F1 0.5 / 0.8; hard negatives n=4 FPR 0.5 / 0.25.
Russian external-model results are cross-lingual generalization only. Internal errors
3 / 1; DeBERTa's error is a high-score Russian hard-negative false positive.
All 8 PINT public-example rows evaluated: both AUPRC/F1 1, Brier 0.188586 / 0.000161638;
zero lexical overlap matches. This is smoke scope, not full PINT.
Median CPU batch latency (1/8): TF-IDF 0.658/1.526 ms; DeBERTa 269.547/1395.285 ms.
AMD Ryzen 5 3500X, one numerical thread, 3 warmups, 20 iterations per batch size.

Artifacts: `artifacts/baselines/task2-complete/`, official snapshot under
`.cache/huggingface/hub/`, raw example under `data/raw/pint/`, local `mlruns/`.
Generated outputs remain ignored; report records source/model/environment identities.
Prior interrupted run directories remain incomplete and are superseded.

Known issues: tiny internal subgroups; unavailable full proprietary PINT; authored Russian
data and existing data-license ambiguity; unverified external training contamination;
English-only archived external model; max-window document FPR risk. PyTorch emits a
Python 3.14 TorchScript future warning, but eager CPU inference passed. No fine-tuning,
calibration, test/PINT threshold tuning, production-model selection or AgentDojo run.

Next expected task: user-supplied Task 3 (custom multilingual transformer); not started.
