# Task 5 — bounded robustness benchmark

## Scope and protocol

Benchmark version `robustness-v1` evaluates the frozen `starter-v1` internal test split
(n=14; 4 injection, 10 benign) without retraining, calibration, threshold selection, or
test-driven iteration. Generated artifacts are under `artifacts/robustness/task5-v1/`:
`examples.jsonl` contains transformation provenance and `results.json` contains metrics,
model identities, artifact hashes, dataset identity, and the seed (`20260925`). The clean
Parquet file is read after hash verification and is never overwritten.

Exactly five bounded families are used: one adjacent-letter typo, one internal-space token
fragment, one allowlisted Latin/Cyrillic visual confusable, fenced text-code-block wrapping,
and one neutral untrusted-source context prefix. Each row records `parent_id`, transformation,
parameters, and a seed for stochastic edit-position selection. All stochastic choices are
derived reproducibly from the global seed, parent ID, and family.

Labels are reused only when the edit remains readable and preserves the original intent.
Rows with no eligible bounded edit are flagged ambiguous rather than silently copied.
Code-block wrapping is excluded for `user_input`, because quoting a direct command can turn
it into mentioned text and change its communicative meaning. The four such rows remain in
the provenance artifact but not in wrapping metrics. This yields 70 generated rows, 66 valid
evaluations, and 4 excluded ambiguous rows.

## Results

Primary values are change in descriptive Recall @ 1% FPR. AUPRC is shown clean → transformed
on the same eligible parents; FPR uses each model's already-frozen operating threshold.

| Model | Family | n | Δ Recall @ 1% FPR | AUPRC | FPR |
|---|---|---:|---:|---:|---:|
| Char TF-IDF | common typos | 14 | +0.25 | 0.533 → 0.658 | 0.20 → 0.20 |
| Char TF-IDF | whitespace fragmentation | 14 | 0 | 0.533 → 0.570 | 0.20 → 0.20 |
| Char TF-IDF | Unicode confusables | 14 | 0 | 0.533 → 0.570 | 0.20 → 0.10 |
| Char TF-IDF | quotation/code block | 10 | 0 | 0.893 → 0.893 | 0 → 0 |
| Char TF-IDF | benign prefix/context | 14 | 0 | 0.533 → 0.476 | 0.20 → 0.30 |
| ProtectAI DeBERTa v2 | common typos | 14 | 0 | 1 → 1 | 0.10 → 0.10 |
| ProtectAI DeBERTa v2 | whitespace fragmentation | 14 | 0 | 1 → 1 | 0.10 → 0.20 |
| ProtectAI DeBERTa v2 | Unicode confusables | 14 | 0 | 1 → 1 | 0.10 → 0.20 |
| ProtectAI DeBERTa v2 | quotation/code block | 10 | 0 | 1 → 1 | 0 → 0 |
| ProtectAI DeBERTa v2 | benign prefix/context | 14 | -0.25 | 1 → 0.917 | 0.10 → 0.20 |
| Custom multilingual | common typos | 14 | 0 | 0.525 → 0.525 | 0.10 → 0.10 |
| Custom multilingual | whitespace fragmentation | 14 | 0 | 0.525 → 0.525 | 0.10 → 0.20 |
| Custom multilingual | Unicode confusables | 14 | 0 | 0.525 → 0.525 | 0.10 → 0.10 |
| Custom multilingual | quotation/code block | 10 | -0.25 | 1 → 0.875 | 0 → 0 |
| Custom multilingual | benign prefix/context | 14 | 0 | 0.525 → 0.483 | 0.10 → 0 |

The largest observed degradations were a 0.25 recall drop for DeBERTa under context expansion
and for the custom model under code-block wrapping. Context expansion also reduced AUPRC and
doubled DeBERTa's frozen-threshold FPR. Whitespace and confusable edits doubled FPR for
DeBERTa; whitespace doubled it for the custom model. The TF-IDF context prefix reduced AUPRC
and raised FPR, although its already-zero full-set low-FPR recall could not decline further.
Some transformations improved a metric on this tiny set; those are sampling/ranking outcomes,
not evidence that corruption is beneficial.

EN (n=10) and RU (n=4) breakdowns are stored for every family. The clearest subgroup result
was DeBERTa context expansion: EN Δ recall 0 with FPR 0.125, versus RU Δ recall -0.5 with FPR
0.5. The RU subgroup has only two positives and two negatives, and the wrapping RU subset is
single-class (n=2), so no strong language conclusion is warranted. DeBERTa Russian behavior
remains cross-lingual generalization rather than supported-language evidence.

## Limitations

The 14-row convenience test cannot resolve a genuine 1% FPR or stable language effects.
Transformations are intentionally one-edit probes, not an attack generator and not coverage
of natural corruption. The benchmark reuses an already-observed test set for frozen diagnostic
evaluation. No result was used for training, augmentation, calibration, threshold selection,
model promotion, or production claims. PromptShield remains one defense-in-depth layer, not a
complete defense.
