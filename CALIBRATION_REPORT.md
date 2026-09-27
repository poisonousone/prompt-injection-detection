# Task 4 — Calibration, operating profiles and error analysis

Executed 2026-09-27 on the existing Task 3 checkpoint. No classifier retraining,
new classifier, test-example edits or post-test retuning. All 16 validation and 14
test rows evaluated. Results are descriptive starter-data results, not production
qualification. The test had already been evaluated in Task 3; this is a frozen
follow-up evaluation, not a newly unseen holdout. PINT and external baselines were
not recalibrated or reevaluated in Task 4.

## Method and frozen selection

One scalar **temperature scaling** parameter, following
[Guo et al. (2017)](https://proceedings.mlr.press/v70/guo17a.html), minimizes validation
binary negative log likelihood. For raw injection probability p, calibrated probability
is sigmoid(logit(p) / T). Binary softmax log odds recover the classifier logit difference
up to floating-point rounding; probabilities are clipped to [1e-12, 1-1e-12].
Positive temperature preserves score ordering apart from numerical saturation/ties.
Temperature bounds [0.05, 20] include the identity T=1. The convex objective in 1/T
is solved by 80 derivative-bisection iterations with explicit boundary handling.

Fitted **T = 0.18332022330585276**, an interior solution. Scores become more dispersed.
This does not establish that individual probabilities are reliable. The same small
validation split selected the checkpoint, fit calibration and selected thresholds;
validation diagnostics are optimistic resubstitution measurements. No calibration
algorithm sweep was performed. The NLL objective does not guarantee lower Brier or ECE.

Config: `configs/calibration.json`. Recall targets were declared before fitting:
`default` 0.80 and `high_security` 0.95. For each, select the threshold minimizing
empirical validation FPR subject to minimum recall; break ties by highest threshold.
The decision is `calibrated_score >= threshold`. These targets are workflow choices,
not risk guarantees; no FPR guarantee is attached to either profile.

| Profile | Frozen threshold | Validation TP / positives | FP / negatives | Recall | FPR |
|---|---:|---:|---:|---:|---:|
| default | 0.39916462741208897 | 7 / 8 | 5 / 8 | 0.875 | 0.625 |
| high_security | 0.3230774532502201 | 8 / 8 | 6 / 8 | 1.000 | 0.750 |

Artifacts were saved before a separate test process. Model files and dataset manifest
were checked against Task 3 seals; fit loaded only the validation split. Test evaluation
verified the calibration/profile checksums and all internal split hashes and isolation.
Existing output directories cannot be overwritten. No fit step reads test or PINT.

## Brier and reliability

Five fixed equal-width bins, [lower, upper), with 1 included in the last bin.
ECE here is the sample-weighted absolute gap between mean injection probability and
observed positive fraction; it is sensitive to binning and these very small counts.

| Split | n | Brier before | Brier after | ECE before | ECE after |
|---|---:|---:|---:|---:|---:|
| Validation (fitted data) | 16 | 0.243222 | 0.233063 | 0.015321 | 0.099769 |
| Final test | 14 | 0.236534 | 0.197017 | 0.197184 | 0.343071 |

Both Brier scores improve, but both ECE values worsen. Before calibration every score
is in the middle bin, hiding variation within it; the tiny pre-fit validation ECE is
not strong calibration evidence. Do not claim uniformly improved calibration.

| Split / stage | Bin | n | Mean probability | Observed positive fraction |
|---|---|---:|---:|---:|
| Validation before | [0.4, 0.6) | 16 | 0.484679 | 0.500000 |
| Validation after | [0.0, 0.2) | 1 | 0.181218 | 0.000000 |
| Validation after | [0.2, 0.4) | 5 | 0.339064 | 0.400000 |
| Validation after | [0.4, 0.6) | 9 | 0.474893 | 0.555556 |
| Validation after | [0.6, 0.8) | 1 | 0.615560 | 1.000000 |
| Test before | [0.4, 0.6) | 14 | 0.482899 | 0.285714 |
| Test after | [0.2, 0.4) | 8 | 0.329945 | 0.000000 |
| Test after | [0.4, 0.6) | 5 | 0.495233 | 0.800000 |
| Test after | [0.6, 0.8) | 1 | 0.639597 | 0.000000 |

Unlisted bins are empty (n=0, means/fractions null), not perfectly calibrated.
Machine-readable artifacts preserve every bin. One-row bins support no generalization.

## Final frozen test results

Test n=14: 4 injection / 10 benign. AUPRC uses average precision. Ranking summaries
are descriptive ROC metrics, not test-selected deployment thresholds.

| Profile | AUPRC | Recall @ 1% FPR | FPR @ 95% recall | F1 | Brier | Recall | FPR | TP / FP / FN / TN |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| default | 0.525000 | 0 | 0.200000 | 0.800000 | 0.197017 | 1 | 0.200000 | 4 / 2 / 0 / 8 |
| high_security | 0.525000 | 0 | 0.200000 | 0.500000 | 0.197017 | 1 | 0.800000 | 4 / 8 / 0 / 2 |

The ranking metrics are unchanged from the raw classifier. Task 3's original threshold
had 4 FN / 1 FP and F1=0. Task 4's decision changes come from the declared recall-based
threshold policy; monotonic temperature scaling does not improve ranking or make the
classifier intrinsically more discriminative. Neither profile is promoted to production.

| Subgroup | n (positive / negative) | AUPRC | Recall @ 1% FPR | FPR @ 95% recall | Brier | default F1 / recall / FPR | high_security F1 / recall / FPR |
|---|---|---:|---:|---:|---:|---|---|
| English | 10 (2 / 8) | 0.416667 | 0 | 0.25 | 0.181423 | 0.666667 / 1 / 0.25 | 0.400000 / 1 / 0.75 |
| Russian | 4 (2 / 2) | 1.000000 | 1 | 0 | 0.236002 | 1.000000 / 1 / 0 | 0.666667 / 1 / 1 |
| Hard negatives | 4 (0 / 4) | null | null | null | 0.251555 | null / null / 0.50 | null / null / 1.00 |

Hard negatives overlap language subgroups. Russian consists of authored translations;
four samples do not establish Russian-language capability. Ten benign test samples
cannot substantiate a 1% FPR claim. Recall 1 here means only four attacks detected.

## Error analysis (descriptive manual grouping)

All final errors were inspected; no test samples or classifier behavior were patched.
Categories below describe text patterns, not proven causal explanations of model internals.

| Recurring pattern | default FP | high_security FP | Evidence IDs |
|---|---:|---:|---|
| Legitimate summarization instructions | 1 | 2 | curated-10-en, curated-10-ru |
| Legitimate spreadsheet instructions using “ignore” | 1 | 2 | curated-11-en, curated-11-ru |
| Ordinary information requests / topic query | 0 | 4 | deepset-train-0006, 0008, 0016, 0031 |

The two legitimate instruction patterns account for all default errors and all four
hard-negative errors under high_security. English gardening summarization scores
0.639597; English spreadsheet filtering scores 0.584641. These are benign instructions
within their legitimate role, even when they contain imperative language or “ignore”.

The lower high_security threshold additionally flags restaurant, stock-market and
apartment information requests plus a short trade-agreement topic query (scores
0.325322–0.373556). This broader overblocking suggests weak score separation; it does
not support adding keyword exceptions or changing the threshold using this test set.

English failures: default 2 FP / 0 FN among n=10 (8 benign / 2 injection);
high_security 6 FP / 0 FN. Russian failures: default 0 FP / 0 FN among n=4
(2 benign / 2 injection); high_security 2 FP / 0 FN. Those two Russian failures are
the translated legitimate gardening and spreadsheet instructions (0.335565 / 0.377793).
The translated pairs show different scores, but there are only two benign pairs here.

High-confidence definition fixed in config: FP score >=0.9, FN score <=0.1.
Both profiles have **0 high-confidence FP and 0 high-confidence FN**. Both also have
zero FN of any confidence in this test, so no recurring final FN category is supported.
Absence of such errors in 14 examples is not evidence of reliable confidence or safety.

## Reproducibility and artifacts

- Implementation: `src/promptshield/calibration.py`; tests: `tests/test_calibration.py`.
- Fit: `uv run python -m promptshield.calibration fit --config configs/calibration.json --output artifacts/calibration/task4-real`.
- Final test: `uv run python -m promptshield.calibration evaluate --output artifacts/calibration/task4-real`.
- Calibration: `artifacts/calibration/task4-real/calibration.json`.
- Profiles/config/seal: `profiles.json`, `frozen.json` in the same directory.
- Diagnostics: `validation.json`, `final_test/metrics.json`, `final_test/predictions.jsonl`,
  `final_test/{default,high_security}_errors.json`. Generated artifacts remain Git ignored.
- Freeze time: `2026-09-27T11:08:03.743523+00:00` (before final test).
- Model: `artifacts/transformer/task3-real/model`, selected epoch 1; original weights unchanged.
  Version `a74926ae21c0aa7a1ec917fdafd0dae2e4361711f2b47d51bbf1ebbb1c6b318c`.
- Dataset fingerprint: `7d653ab1a682f5570392bc379c07e25c97db6ad40c94d1f8e042b5a205008768`.
- Base Git: `0233cf0b9b127b214aa795bbce883f37334341a7`, dirty working tree with prior Task 3 changes.
  Frozen artifact includes source/code/lock hashes, Python version and configuration.
- Local MLflow: experiment `promptshield-calibration`, run `377c53650dc14eea987f9fab70e3fb44`.
- CPU inference, existing locked CPython 3.14.7 environment; no dependency changes.

Validation: eight new targeted tests passed, covering calibration math, ties, reliability
bins, invalid inputs, fit without train/test files, frozen-profile tampering and overwrite
rejection. Full lightweight checks are recorded in `TASK_HISTORY.md`.

## Unresolved issues and next task

The starter data, reuse of validation for checkpoint/calibration/profile selection,
previously observed test, high FPR, worsening binned ECE, authored Russian, source-license
ambiguity and first-256-token truncation remain limitations. No strong calibration or
security claim, production promotion, new classifier training or future roadmap work.
Next: await the user-supplied task; a larger independently reviewed evaluation protocol
is needed before making stronger deployment claims. Do not retune on these holdouts.
