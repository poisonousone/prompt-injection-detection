# Dataset sources and annotation policy

## Public source

- Attribution: deepset, `deepset/prompt-injections`, hosted on Hugging Face.
- Repository: https://huggingface.co/datasets/deepset/prompt-injections
- Revision: `4f61ecb038e9c3fb77e21034b22511b523772cdd`.
- File: `data/train-00000-of-00001-9564e8b05b4757ab.parquet`.
- Raw local path: `data/raw/deepset/train.parquet` (immutable, Git ignored).
- SHA-256: `2e10bc7ab30f542c97e4e83e2a5683000b5057d25ec10908784c631d44124c04`.
- Measured upstream training rows: 546. Reviewed selection: 36; excluded: 510.
- Upstream test split is not downloaded, mixed into development, or evaluated.

The pinned upstream README declares `apache-2.0` at top level and `cc-by-4.0` inside
`dataset_info`. This conflict is unresolved; do not assert a single verified license.
Both declarations and attribution are retained here. Before publishing a redistributed
dataset, clarify the license and include the required license/attribution materials.

`curated/deepset_review.jsonl` is the explicit allowlist. Text is unchanged, labels map
0 -> benign and 1 -> prompt_injection. Added annotations and IDs are PromptShield's
AI-assisted review, not upstream labels. Each record identifies its zero-based upstream
row, split and revision. English was assessed from the selected text; content role and
injection mode are unknown because upstream supplies no trustworthy context for them.

Selection deliberately favors clear benign queries and explicit instruction overrides.
Roleplay-only, potentially legitimate instruction-only, harmful-request/jailbreak-style,
ambiguous and unreviewed rows are excluded rather than silently treated as injection.
The 510 excluded records are **not** all claimed to be jailbreaks. The allowlist is a
convenience sample, not a representative random sample or a new benchmark.

All selected positive rows share `family:control_override`, also used by related authored
examples. Selected benign rows have source-row groups; no known translated or derivative
benign rows from upstream are selected. Lexical audit adds conservative links where found.

## Authored supplement

`curated/bilingual.jsonl` contains 64 AI-authored records: 32 paired English/Russian
concepts, 32 injection examples and 32 hard negatives. It was authored for this task;
there was no independent human review and no model-generated augmentation job or training.
No content was derived from PINT, AgentDojo or an external evaluation result.

Hard negatives cover quoted attacks, injection discussion/security research, security
documentation, inert source-code strings, test fixtures, legitimate instructions,
system-prompt discussion, and benign keyword use. The metadata identifies category,
translation pair, authorship, family/topic, and interpretation context.

Injection examples address a reading assistant and attempt to cross the authority boundary
of their declared role. The source-code example that addresses the assistant is an indirect
injection; inert example strings/test assertions are benign in their stated context.
Direct samples explicitly attempt to override governing instructions. Mock tool names,
mock credentials and the reserved `example.invalid` domain are inert; nothing is executed.

Translations, related templates and attack families share a group. The four records in each
authored family/topic stay together even if their words differ. Text-only intent is inherently
context-sensitive, so these labels require review before substantive ML claims.
