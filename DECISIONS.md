# PromptShield — Durable Decisions

This file records durable ML and architectural decisions that future agents need to understand.

Do not record trivial implementation details.

---

## D001 — Project scope

Status: accepted

PromptShield is intentionally a compact single-service ML/Applied AI project rather than an enterprise distributed platform.

Primary focus:

- multilingual prompt-injection detection;
- evaluation quality;
- robustness;
- security/utility trade-offs;
- reproducibility.

Technologies without a concrete requirement should not be introduced merely for portfolio breadth.

---

## D002 — Primary task taxonomy

Status: accepted

Primary task:

- benign
- prompt_injection

Primary languages:

- English
- Russian

Direct and indirect injection are metadata/subtypes.

Jailbreak detection is not merged into the primary task without explicit taxonomy mapping.

---

## D003 — External evaluation boundaries

Status: accepted

- PINT is external holdout only.
- PINT must not be used for training, threshold selection, or hyperparameter tuning.
- AgentDojo is a system-level external evaluation.
- ProtectAI DeBERTa-v3-base Prompt Injection v2 is an external model baseline.

---

## D004 — Compact, explicitly scoped starter data

Status: accepted (Task 1, 2026-09-25)

Use one revision/checksum-pinned public source (`deepset/prompt-injections`, upstream
train only) through an explicit reviewed allowlist. Do not equate every upstream
positive with prompt injection: ambiguous and jailbreak-style/unreviewed examples
remain excluded. Preserve upstream text, labels, row identity and unknown context.
Document the conflicting upstream license declarations rather than inventing a resolution.

A small AI-authored bilingual supplement supplies Russian examples, indirect modes and
identifiable hard negatives. Mark authorship and translation pairs explicitly; do not
describe it as natural or independently human-reviewed data. This is a starter dataset,
not a statistically sufficient benchmark. No external holdout data is used.

## D005 — Protected components precede splitting

Status: accepted (Task 1, 2026-09-25)

Globally scoped group IDs protect translations, templates and known attack families.
Union group links from exact/normalized duplicates and lexical near matches before
deduplication and splitting. NFKC/casefold/whitespace normalization is a comparison key
only; keep the selected original text and duplicate-member provenance. Conflicting
duplicate labels fail. Near matches remain and generate audit records.

Assign whole components largest-first, breaking ordering ties with a seeded SHA-256 key.
Choose the split minimizing normalized squared deficit changes for sample, label,
language and hard-negative counts. Balance is approximate; group isolation takes priority
over exact ratios. Validate group and normalized-text isolation programmatically.
Persist seed, ratios, threshold, assignments and hashes. Never select a seed based
on model/test performance. The bounded exhaustive lexical audit fails above 5,000 inputs;
larger datasets require a reviewed scalable approach.

## D006 — Reproducible data artifacts and staged dependencies

Status: accepted (Task 1, 2026-09-25)

Raw inputs are checksum-verified and never overwritten. Publish complete processed builds
separately and reject replacement of different existing builds. Version small curated
inputs/configuration and `uv.lock`; keep downloads and generated Parquet out of Git.
Use LF for versioned source data so checkout newline conversion cannot break checksums.

Parquet stores canonical string fields and JSON-encoded metadata; the in-memory schema
uses a strict JSON object. CSV uses JSON metadata and an empty mode cell for null.
Manifests record source/annotation/code/lock hashes, runtime, Git identity and artifact
checksums. Dataset identity excludes execution Git commit, but the manifest records it.

Install only Task 1 runtime and quality-tool dependencies now. ML/training/serving
dependencies are deferred until their tasks, retaining Python 3.14 and CPU support.

## D007 — Baseline operating points and external-model scope

Status: accepted (Task 2, 2026-09-26)

Use the existing frozen starter-v1 splits unchanged. Character TF-IDF (3–5 grams)
and logistic regression (C=1, liblinear) fit training only. Select its threshold by
maximum validation F1, breaking ties toward the highest threshold. Persist operating
points before test/PINT evaluation. ROC-derived Recall @ 1% FPR and FPR @ 95% Recall
are descriptive ranking summaries, never thresholds for deployment. AUPRC is average
precision (step integration). Single-class subgroup ranking metrics/F1 are null;
report counts, FPR and Brier instead. Small subgroups support no strong conclusions.

The external detector is `protectai/deberta-v3-base-prompt-injection-v2`, official
revision `90c9989b1a342275dd0d1a95aad283c04e075671`, unmodified weights. Use fixed 0.5
threshold, softmax INJECTION score, no fine-tuning or calibration. Russian evaluation
is explicitly cross-lingual generalization, not supported-language validation.
Use 512-token windows including CLS/SEP, 64-token overlap and maximum window score
to cover long documents; this wrapper can raise document-level false positives.
Explicit token slicing avoids observed backend overflow tail loss. This is a declared
wrapper policy, not a claim of native long-context support or the upstream PINT setup.

## D008 — PINT taxonomy, availability and contamination reporting

Status: accepted (Task 2, 2026-09-26)

Pin `lakeraai/pint-benchmark` at `0efab3f463eae9c823130d8faffb71b2e7c06e63`
and checksum raw bytes. Its public repository contains only the eight-row example;
evaluate it as `public_example_smoke`, never as the full proprietary PINT benchmark.
Keep every row, original category and boolean target: true maps to benchmark attack,
false to benign. Combined PINT results include jailbreaks and are not pure injection
metrics. Report all categories separately and an additional view excluding jailbreaks.
Supported categories include prompt_injection, jailbreak, hard_negatives, chat,
documents, short_input, benign_input and long_input; unknown categories fail explicitly.
Only explicitly labeled hard_negatives count as such; missing language stays unknown.
Never train, tune thresholds or calibrate on PINT. Audit exact, normalized and long
substring overlaps against all internal splits, report matches and retain rows rather
than hiding contamination. This lexical audit cannot prove semantic independence or
exclude overlap with the external detector's unavailable complete training corpus.
