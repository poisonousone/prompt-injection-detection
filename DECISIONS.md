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
