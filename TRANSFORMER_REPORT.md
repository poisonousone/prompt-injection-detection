# Task 3 — Custom multilingual transformer

Executed 2026-09-27 on base commit `0233cf0b9b127b214aa795bbce883f37334341a7`,
with uncommitted Task 3 code. Model selection rationale and primary sources:
[ADR](docs/adr/model-selection.md). No test/PINT-driven retuning was performed.

## Execution and reproducibility

Full fine-tuning of `distilbert/distilbert-base-multilingual-cased`, revision
`45c032ab32cc946ad88a166f7cb282f58c753c2e`, completed on all 70 starter training rows.
This is a real pretrained-model experiment, not the separate toy smoke. It is not
a statistically adequate large-corpus study. Classifier parameter count: 135,326,210.

Dataset fingerprint:
`7d653ab1a682f5570392bc379c07e25c97db6ad40c94d1f8e042b5a205008768`.
Splits are unchanged: train 70, validation 16, test 14. PINT is the eight public
examples only, revision `0efab3f463eae9c823130d8faffb71b2e7c06e63`, SHA-256
`df068b9a4ff72483f493add6be6242c6aa777df756bd61462aa0e13645cffa90`.

Hardware: AMD Ryzen 5 3500X, CPU, two numerical threads; installed PyTorch has no CUDA.
Python 3.14.7, torch 2.14.0+cpu, Transformers 5.17.0, MLflow-skinny 3.16.1.
The existing `uv.lock` resolution was retained. Configuration:
`configs/transformer.json` (seed 20260925, LR 2e-5, batch 4, accumulation 2,
max length 256, weight decay 0.01, linear scheduler/10% warmup, patience 2, fp32).
Measured training/validation/checkpoint phase: 43.460 seconds, excluding initial
weight download/load and final evaluation. No inference latency benchmark was added.

| Epoch | Training loss | Validation recall at <=1% empirical FPR | Validation AUPRC | Validation Brier |
|---|---:|---:|---:|---:|
| 1 (selected) | 0.690276 | 0.125 | 0.683631 | 0.243222 |
| 2 | 0.638252 | 0.125 | 0.666588 | 0.239301 |
| 3 | 0.585432 | 0 | 0.578752 | 0.236183 |

Epoch 1 won the prespecified validation ordering. Its threshold is
`0.5215609669685364`; validation has 8 positives/8 negatives, 1 TP and 0 FP.
Selection stopped after two non-improving epochs, also the configured three-epoch cap.
Eight validation negatives cannot establish a reliable 1% FPR operating point.

Artifact directory: `artifacts/transformer/task3-real/model/`.
Model version: `a74926ae21c0aa7a1ec917fdafd0dae2e4361711f2b47d51bbf1ebbb1c6b318c`.
The selected checkpoint loaded offline in a separate evaluation process; weights,
tokenizer, config, labels and metadata are all local to that directory.
`frozen.json` contains their checksums and the dataset-manifest checksum.
Training metadata includes code hashes, Git dirty state, lock hash, configuration,
hardware, packages, counts, selection result and tracking identity.

MLflow local experiment `promptshield-transformer`, ID `866129714858236231`,
run `8672a61f959341b4a78385b237da70cc`, status FINISHED, under `mlruns/`.
Separate offline toy smoke: `artifacts/transformer/task3-smoke/`, experiment
`promptshield-transformer-smoke`, ID `788857031564388795`,
run `14db6d62b90b46fe81b7b43cd37dea89`. Smoke metrics are not benchmark results.

## Frozen final comparison

Existing TF-IDF/DeBERTa results are reused from `artifacts/baselines/task2-complete/`,
not rerun or retuned. Their dataset fingerprint matches this run. DeBERTa revision:
`90c9989b1a342275dd0d1a95aad283c04e075671`. TF-IDF uses its prior validation-F1
threshold `0.4403050229585025`; DeBERTa retains fixed 0.5. Different threshold
objectives mean F1 comparisons are descriptive, not equal-operating-point comparisons.
ROC-derived columns below are ranking summaries, not thresholds fitted on holdouts.

Internal test: n=14, 4 positives/10 negatives.

| Model | Recall @ 1% FPR | FPR @ 95% recall | AUPRC | F1 | Brier | Frozen-threshold recall / FPR |
|---|---:|---:|---:|---:|---:|---|
| Character TF-IDF | 0 | 0.7 | 0.532576 | 0.666667 | 0.204418 | 0.75 / 0.2 |
| External DeBERTa | 1 | 0 | 1 | 0.888889 | 0.068647 | 1 / 0.1 |
| Custom multilingual | 0 | 0.2 | 0.525000 | 0 | 0.236534 | 0 / 0.1 |

Custom model produced four false negatives and one false positive. No threshold,
checkpoint or configuration was changed after seeing these outcomes.

| Internal subgroup | Model | n (positive/negative) | Recall @ 1% FPR | FPR @ 95% recall | AUPRC | Frozen recall / FPR |
|---|---|---|---:|---:|---:|---|
| English | TF-IDF | 10 (2/8) | 0 | 0.125 | 0.583333 | 1 / 0.125 |
| English | DeBERTa | 10 (2/8) | 1 | 0 | 1 | 1 / 0 |
| English | Custom | 10 (2/8) | 0 | 0.25 | 0.416667 | 0 / 0.125 |
| Russian | TF-IDF | 4 (2/2) | 0 | 1 | 0.5 | 0.5 / 0.5 |
| Russian | DeBERTa | 4 (2/2) | 1 | 0 | 1 | 1 / 0.5 |
| Russian | Custom | 4 (2/2) | 1 | 0 | 1 | 0 / 0 |
| Hard negatives | TF-IDF | 4 (0/4) | — | — | — | — / 0.5 |
| Hard negatives | DeBERTa | 4 (0/4) | — | — | — | — / 0.25 |
| Hard negatives | Custom | 4 (0/4) | — | — | — | — / 0.25 |

Single-class ranking metrics are undefined. The custom Russian ranking is perfect
on just four rows but both positives fall below the frozen threshold; this does not
demonstrate reliable Russian detection. External DeBERTa Russian results represent
cross-lingual generalization of its English detector.

PINT **public-example smoke scope**, n=8, 2 attacks (one injection, one jailbreak)/6 benign:

| Model | Recall @ 1% FPR | FPR @ 95% recall | AUPRC | F1 | Brier | Frozen recall / FPR |
|---|---:|---:|---:|---:|---:|---|
| Character TF-IDF | 1 | 0 | 1 | 1 | 0.188586 | 1 / 0 |
| External DeBERTa | 1 | 0 | 1 | 1 | 0.000162 | 1 / 0 |
| Custom multilingual | 0 | 0.166667 | 0.583333 | 0 | 0.237446 | 0 / 0 |

All original categories are retained and separately reported. Excluding jailbreak,
custom AUPRC is 0.5 on n=7 (1 positive/6 negatives). Injection and jailbreak recall
are each 0 on one row; hard-negative FPR is 0 on one row. Explicit English/Russian
counts are both 0 because the example has no language annotations; no languages
were inferred. Lexical overlap audit found 0 matches. Full PINT was not executed.

## Limits and next work

The real training pipeline, offline smoke, local tracking, early stopping,
best-checkpoint save/reload and final evaluation were executed. CUDA/mixed precision
branches were not executed; CPU rejects unsupported precision requests. Tests exercise
selection ties, partial accumulation batches, early stopping, local tracking, offline
reload, batching, integrity guards and smoke/holdout separation without downloads.
Final local checks: Ruff check/format pass, strict Pyright 0 errors, full lightweight
pytest 55 passed (128 upstream MLflow deprecation warnings). Remote CI not executed.

The starter corpus is too small for strong comparisons. Natural Russian data,
independent annotation review, license clarification and larger low-FPR validation
remain necessary. Custom truncation can miss late instructions and differs from
DeBERTa's max-window policy. Probabilities are uncalibrated; detection remains one
defense layer. No production model was selected. Do not use these holdout outcomes
for further tuning. Await the next user-supplied task (anticipated calibration and
threshold validation); no future task was implemented.
