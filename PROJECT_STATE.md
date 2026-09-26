# PromptShield — Current Project State

Last updated: 2026-09-26

## Current phase

Task 2 — Baselines + PINT completed within available public-data scope.
Task 2 changes are uncommitted on base commit `36c074d`. TF-IDF trained; official
DeBERTa evaluated without modification. No custom transformer trained.

## Completed tasks

Task 1: strict JSONL/CSV/Parquet ingestion, pinned public-source allowlist, immutable raw
checksums, provenance, exact/normalized deduplication, near-duplicate audit, deterministic
group-aware splits, Parquet artifacts, reproducibility manifest and generated data card.

Task 2: shared detector interface, train-only sklearn pipeline, pinned public DeBERTa,
real internal evaluation, PINT public-example evaluation, latency and error artifacts,
local MLflow tracking. Complete report: `BASELINE_REPORT.md`.

Ruff check/format, strict Pyright and all 46 lightweight tests pass locally.
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

Character n-gram TF-IDF + logistic regression and external
`protectai/deberta-v3-base-prompt-injection-v2` at
`90c9989b1a342275dd0d1a95aad283c04e075671`.

## Current selected production model

None.

## Current operating point

TF-IDF: 0.4403050229585025, maximum F1 on validation only. DeBERTa: fixed 0.5;
no fine-tuning or calibration. Neither test nor PINT selected operating points.

## Current experiment artifacts

Complete run: `artifacts/baselines/task2-complete/` (Git ignored).
TF-IDF pipeline: `tfidf/model.joblib`; external weights: pinned snapshot under
`.cache/huggingface/hub/models--protectai--deberta-v3-base-prompt-injection-v2/snapshots/`.
Each detector has identity/threshold/latency JSON and validation/internal_test/pint
metrics JSON, predictions JSONL and errors JSON. Run/source hashes and overlap audit
are at run root; local tracking in `mlruns/`. Prior partial runs are superseded.

Real internal test (n=14, 4 positive/10 negative):

| Model | AUPRC | Recall @ 1% FPR | FPR @ 95% recall | F1 | Brier |
|---|---:|---:|---:|---:|---:|
| TF-IDF | 0.532576 | 0 | 0.7 | 0.666667 | 0.204418 |
| DeBERTa | 1 | 1 | 0 | 0.888889 | 0.068647 |

English n=10: F1 0.8 / 1; Russian n=4: F1 0.5 / 0.8 (TF-IDF / DeBERTa).
Russian DeBERTa results are cross-lingual generalization, not supported-language evidence.
Hard negatives n=4: FPR 0.5 / 0.25. Full subgroup metrics/counts in report/artifacts.
CPU median batch latency (batch 1/8): TF-IDF 0.658/1.526 ms; DeBERTa 269.547/1395.285 ms.
AMD Ryzen 5 3500X, one numerical thread, 3 warmups, 20 iterations per batch size.

## Current external benchmark status

- PINT: integrated; all 8 public-example rows evaluated, both models AUPRC/F1 1;
  Brier TF-IDF 0.188586, DeBERTa 0.000161638. This is smoke scope only, not full PINT.
  Pinned revision `0efab3f463eae9c823130d8faffb71b2e7c06e63`; full proprietary data unavailable.
  Every category retained, jailbreak separately reported, unknown languages preserved.
  Zero internal/example exact/normalized/substring overlap matches.
- AgentDojo: not integrated

## Known issues / limitations

- Small starter corpus, insufficient for reliable model or low-FPR evaluation.
- Full PINT unavailable; eight public examples cannot establish external robustness.
- External model is English-only per model card, archived and not intended for jailbreak
  detection; system prompts can cause false positives. Long-document max-window scoring
  can also raise FPR. External training contamination cannot be ruled out.
- DeBERTa produced one high-score Russian hard-negative false positive; TF-IDF had
  two false positives and one false negative internally. No test-driven retuning.
- PyTorch reports a Python 3.14 TorchScript future warning; CPU eager inference succeeds.
- Russian is authored translation data; natural Russian data and independent human review
  remain needed. Public selection is a convenience allowlist.
- Upstream license fields conflict (apache-2.0 vs cc-by-4.0); clarify before redistribution.
- Public role/mode unknown; lexical grouping cannot guarantee semantic decontamination.
- C: is full on this workstation. Local toolchain/cache/temp paths use D: through
  `scripts/activate-local.ps1`. Workspace fixtures/staging inherit directory permissions
  to support the Windows sandbox; all final checks passed inside it.

## Next expected task

Await user-supplied Task 3 (custom multilingual transformer); not started.
Address dataset size/review limitations before making strong comparison claims.
