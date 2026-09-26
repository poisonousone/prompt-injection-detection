# Task 2 - Baseline evaluation

Executed 2026-09-26, CPU; complete run: `artifacts/baselines/task2-complete/`.

## Protocol and external model

Frozen starter-v1: train 70, validation 16, test 14. Fingerprint
`7d653ab1a682f5570392bc379c07e25c97db6ad40c94d1f8e042b5a205008768`.
Training/validation/test artifacts were checksum-verified and group isolation checked.
TF-IDF character 3-5 grams + logistic regression C=1 uses training only; threshold
`0.4403050229585025` maximizes validation F1 (highest threshold on ties).
DeBERTa uses fixed 0.5 with no fine-tuning, calibration or threshold optimization.
Both operating points were persisted before test/PINT records were evaluated.

Official model: `protectai/deberta-v3-base-prompt-injection-v2` at
`90c9989b1a342275dd0d1a95aad283c04e075671`. Downloaded official safetensors directly;
model SHA-256 `6521cb8d0ac08148c81464899c424e6148fcc62befa371089fa4061d8b6e0424`.
[Pinned model card](https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2/blob/90c9989b1a342275dd0d1a95aad283c04e075671/README.md)
lists Apache-2.0, English, non-English/jailbreak limitations and system-prompt false
positives; the project is archived. These are upstream limitations, not measured
capabilities established by this small experiment. Russian is explicitly a cross-lingual
generalization experiment, not an intended supported-language benchmark.

[Pinned configuration](https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2/blob/90c9989b1a342275dd0d1a95aad283c04e075671/config.json)
maps SAFE=0, INJECTION=1 and permits 512 positions. Risk is softmax(INJECTION).
Use the official tokenizer without chat formatting; CLS + up to 510 content tokens +
SEP, 64-token overlap, dynamic padding and maximum window risk per document.
Manual windows preserve tails missed by the installed overflow-tokenizer path.
Real-tokenizer special-token equivalence was checked. This declared wrapper differs
from simple upstream truncation and can increase false positives on long documents.

## Internal test (real evaluation)

All estimates are descriptive: only 14 rows, 10 negatives and 4 positives. English
has 10 rows (2 positive); Russian has 4 (2 positive); hard negatives has 4 benign rows.
These counts cannot establish reliable 1% FPR or 95% recall performance.
AUPRC is average precision; ROC operating summaries are empirical ranking statistics,
not holdout-selected deployment thresholds. F1 uses the already-frozen threshold.
Single-class ranking metrics and F1 are null; FPR and Brier remain meaningful.

| Model / subgroup | n (+/-) | AUPRC | Recall @ 1% FPR | FPR @ 95% recall | F1 | Brier | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| tfidf / english | 10 (2/8) | 0.583333 | 0 | 0.125 | 0.8 | 0.179314 | 0.125 |
| tfidf / hard_negatives | 4 (0/4) | N/A | - | N/A | - | 0.19492 | 0.5 |
| tfidf / overall | 14 (4/10) | 0.532576 | 0 | 0.7 | 0.666667 | 0.204418 | 0.2 |
| tfidf / russian | 4 (2/2) | 0.5 | 0 | 1 | 0.5 | 0.267178 | 0.5 |
| deberta / english | 10 (2/8) | 1 | 1 | 0 | 1 | 9.24871e-08 | 0 |
| deberta / hard_negatives | 4 (0/4) | N/A | - | N/A | - | 0.240259 | 0.25 |
| deberta / overall | 14 (4/10) | 1 | 1 | 0 | 0.888889 | 0.0686467 | 0.1 |
| deberta / russian | 4 (2/2) | 1 | 1 | 0 | 0.8 | 0.240263 | 0.5 |

## PINT public example (real inference, smoke scope only)

[PINT source](https://github.com/lakeraai/pint-benchmark/tree/0efab3f463eae9c823130d8faffb71b2e7c06e63)
has only `benchmark/data/example-dataset.yaml` under its dataset directory. The full
proprietary benchmark is unavailable locally; no full PINT result is claimed.
Revision `0efab3f463eae9c823130d8faffb71b2e7c06e63`; example SHA-256
`df068b9a4ff72483f493add6be6242c6aa777df756bd61462aa0e13645cffa90`.
The public repository is MIT licensed. Public-example results are not comparable
to published leaderboard PINT scores.

Preserve all eight rows and boolean labels. True maps to benchmark attack, false
to benign; category never overrides the label. prompt_injection and jailbreak each
have one positive. short_input, benign_input, chat, documents, hard_negatives and
long_input each have one negative. No category was removed. Combined results include
jailbreaks; an additional seven-row view excludes jailbreaks without hiding the full view.
All eight languages are unspecified and remain unknown; no inferred EN/RU breakdown.
No PINT data entered training, validation, calibration or threshold selection.
Exact/normalized/substring audit across all internal splits found zero matches.
This does not establish semantic independence or rule out external pretraining overlap.

| Model / scope | n (+/-) | AUPRC | Recall @ 1% FPR | FPR @ 95% recall | F1 | Brier |
|---|---:|---:|---:|---:|---:|---:|
| tfidf / overall | 8 (2/6) | 1 | 1 | 0 | 1 | 0.188586 |
| tfidf / injection_scope_without_jailbreak | 7 (1/6) | 1 | 1 | 0 | 1 | 0.179788 |
| deberta / overall | 8 (2/6) | 1 | 1 | 0 | 1 | 0.000161638 |
| deberta / injection_scope_without_jailbreak | 7 (1/6) | 1 | 1 | 0 | 1 | 0.000184729 |

Both models classify every public-example row correctly; per-category counts and
metrics remain in each `pint/metrics.json`. Each category has only one row.

## Error analysis

Full texts are confined to ignored local `errors.json` files. Internal TF-IDF has
2 false positives and 1 false negative, including 2 Russian errors and 2 hard-negative
errors. Internal DeBERTa has 1 Russian hard-negative false positive. It meets the
high-score FP criterion (score >=0.9). There are no high-score TF-IDF errors or
low-score false negatives (score <=0.1). Empty categories are retained explicitly.
Scores are not calibrated confidence. PINT public-example errors are empty.

## Latency

Windows 10 build 19045, AMD Ryzen 5 3500X, 6 logical CPUs, CPU inference with one
numerical thread. Three warmups and 20 measured iterations per model/batch size.
Fixed validation inputs, batch sizes 1 and 8, tokenization/features included, model
load excluded. A shared workstation measurement, not a production capacity claim.
Input hashes/lengths, model versions and every timing are in `latency.json`.

| Model | Batch size | Median batch latency (ms) | Samples/s (mean time) |
|---|---:|---:|---:|
| tfidf | 1 | 0.658 | 1462.887 |
| tfidf | 8 | 1.526 | 4972.867 |
| deberta | 1 | 269.547 | 3.692 |
| deberta | 8 | 1395.285 | 5.648 |

## Reproducibility and artifact index

Execution base commit `36c074dfd28d6d496713a7bb54c207d8e2e6e746` with Task 2 working-tree changes;
exact source-file and dependency-lock hashes are in `run.json`. Seed 20260925.
CPython 3.14.7; mlflow-skinny 3.16.1, numpy 2.5.3, polars 1.44.2, scikit-learn 1.9.1, torch 2.14.0+cpu, transformers 5.17.0.

- `run.json`, `training.json`, `pint_overlap.json`: provenance and audit.
- `{tfidf,deberta}/{identity,threshold,latency}.json`: model and operating point.
- `{tfidf,deberta}/{validation,internal_test,pint}/metrics.json`: all subgroup metrics.
- Matching `predictions.jsonl`: every prediction with source ID/text hash.
- Matching `errors.json`: high-score FP, low-score FN, Russian and hard-negative errors.
- `tfidf/model.joblib`: trained pipeline; official DeBERTa snapshot in `.cache/huggingface/hub/`.
- `mlruns/`: two local MLflow runs, parameters, validation/test metrics and small artifacts.

Checks: locked dependency sync, Ruff lint/format, strict Pyright, 46 lightweight tests
passed. Targeted window/batch/score-direction, threshold, metrics and taxonomy tests
passed; real official tokenizer/weights were exercised separately. No custom transformer
training, production-model selection, calibration or AgentDojo execution occurred.
PyTorch emits a Python 3.14 TorchScript future warning; eager inference succeeded.
Prior interrupted run directories are incomplete and superseded by `task2-complete`.
