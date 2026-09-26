# PromptShield — Current Project State

Last updated: 2026-09-25

## Current phase

Task 1 — Repository + Dataset Pipeline completed. No model has been trained.
Implementation is in the working tree; the repository has no commits yet.

## Completed tasks

Task 1: strict JSONL/CSV/Parquet ingestion, pinned public-source allowlist, immutable raw
checksums, provenance, exact/normalized deduplication, near-duplicate audit, deterministic
group-aware splits, Parquet artifacts, reproducibility manifest and generated data card.

Ruff check/format, strict Pyright and all 34 lightweight tests pass locally.
Lightweight GitHub Actions configured; remote CI has not run.

## Active technical baseline

- Runtime: CPython 3.14.7
- Package manager: uv 0.12.19; exact resolution in `uv.lock`; CPU-only dependencies
- Primary languages: English + Russian
- Primary task: benign vs prompt_injection
- External detector baseline: ProtectAI DeBERTa-v3-base Prompt Injection v2
- External detector holdout: PINT
- External system benchmark: AgentDojo

## Current dataset

- Version: `starter-v1`; config: `data/dataset.json`.
- Fingerprint: `7d653ab1a682f5570392bc379c07e25c97db6ad40c94d1f8e042b5a205008768`.
- Public source: `deepset/prompt-injections` at
  `4f61ecb038e9c3fb77e21034b22511b523772cdd`, upstream train only.
- Raw: `data/raw/deepset/train.parquet`; 546 upstream rows, 36 selected, 510 excluded.
- Versioned inputs: `data/curated/deepset_review.jsonl`, `data/curated/bilingual.jsonl`.
- Processed: `data/processed/v1/{all,train,validation,test}.parquet`.
- Reports: `data/processed/v1/{report,manifest}.json`, `DATA_CARD.md`, `data/SOURCES.md`.
- Total: 100 records; 57 benign / 43 injection; 68 English / 32 Russian;
  32 identifiable hard negatives; 41 protected groups.
- Splits: train 70, validation 16, test 14. Each contains both classes, both languages
  and hard negatives. Validation/test each have 4 hard negatives; Russian counts are 6/4.
- Duplicates: 0 exact, 0 normalized-only; 0 lexical near warnings at configured thresholds.
- Group/normalized-text isolation passed. Two full builds produced byte-identical artifacts,
  including manifests; raw checksum remained unchanged.
- Raw/processed artifacts are Git ignored; curated inputs and lockfile are intended for Git.

## Current models

None.

## Current selected production model

None.

## Current operating point

None.

## Current experiment artifacts

None.

## Current external benchmark status

- PINT: not integrated
- AgentDojo: not integrated

## Known issues / limitations

- Small starter corpus, insufficient for reliable model or low-FPR evaluation.
- Russian is authored translation data; natural Russian data and independent human review
  remain needed. Public selection is a convenience allowlist.
- Upstream license fields conflict (apache-2.0 vs cc-by-4.0); clarify before redistribution.
- Public role/mode unknown; lexical grouping cannot guarantee semantic decontamination.
- C: is full on this workstation. Local toolchain/cache/temp paths use D: through
  `scripts/activate-local.ps1`. Workspace fixtures/staging inherit directory permissions
  to support the Windows sandbox; all final checks passed inside it.

## Next expected task

User-supplied Task 2, expected to be the character n-gram TF-IDF + logistic regression
baseline. Review data limitations before interpreting results. Not started.
