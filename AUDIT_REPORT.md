# Task 8 — Final audit

Audited 2026-09-27 on base `0233cf0`, with the pre-existing uncommitted Task 3–7 work
preserved. No real training, threshold retuning, new model or benchmark was introduced.
Historical measured results remain unchanged. Verification used local artifacts as evidence.

## ML findings

| Check | Evidence and finding |
|---|---|
| Split/group leakage | Revalidated frozen Parquet hashes, IDs, protected groups and normalized text across train/validation/test; passed |
| Duplicate contamination | Recomputed exact/normalized dedup and configured lexical near audit over all 100 rows: zero duplicates and zero near warnings |
| PINT contamination | Rechecked all internal splits against checksum-pinned eight-row PINT example: zero exact/normalized/long-substring matches |
| Labels and score direction | Injection=1 in sklearn/custom; DeBERTa explicitly resolves INJECTION from SAFE/INJECTION mapping; larger scores mean higher risk |
| Train/test separation | TF-IDF fits train only; custom training reads train/validation; thresholds freeze before final evaluation; no PINT fitting path |
| Thresholds | Recomputed TF-IDF validation F1 threshold; inspected custom security-point checkpoint selection; recomputed both calibration profiles from saved validation scores |
| Calibration | Verified classifier and calibration seals; recomputed temperature fit, profiles, before/after reliability, Brier, ECE and final test metrics |
| Metrics/subgroups | Recomputed six original internal/PINT reports from saved predictions, including every subgroup; checked IDs, labels, text hashes and decisions against source records |
| Robustness | Checked matched parent counts, stored deltas and custom model hashes; empty language/eligible sets exposed a reporting crash, now fixed with regression coverage |
| AgentDojo | Recomputed ASR and clean/attacked utility denominators from saved episodes; clean security sentinel excluded; artifact says `real_llm_evaluation=not_executed` |
| Comparisons | README distinguishes ranking from operating points, uncalibrated serving from custom calibration, different input policies and scripted smoke from actual LLM evidence |

No demonstrated leakage, score-direction or metric-calculation error was found in the
reported historical experiments. This is not a proof of semantic decontamination or an
independent reconstruction of every past human decision. Training/configuration provenance
and code show validation-based selection; previously observed test results cannot become
an unseen holdout again. The 16 validation rows were reused for checkpoint selection,
calibration and thresholds. Explicitly retain that limitation.

The external model's complete training corpus is unavailable; contamination with public
internal examples cannot be ruled out. Russian data is entirely authored, translated and
not independently reviewed. The eight PINT examples cannot establish full external quality.
Robustness uses a previously observed test and bounded edits. No methodology was changed
to improve any result.

## Engineering verification

- Fresh isolated `.cache/task8-venv`: `uv sync --locked --offline --group agentdojo`
  installed 121 packages from the populated cache on CPython 3.14.7. This verifies a new
  environment, not a cold network download or a Linux installation.
- `uv lock --check --offline` passed. Dependencies remain unchanged; declared runtime
  packages support implemented data/model/tracking/serving paths, with SentencePiece
  supporting the selected tokenizer and AgentDojo isolated in its optional group.
- Dataset fetch verified the existing immutable source. Current-code build to
  `data/processed/task8-audit` passed: 100 records, 70/16/14 splits. Four Parquet files and
  `report.json` are byte-identical to the historical build. Historical manifests were not
  overwritten. The new fingerprint is
  `57cf663df57ec68746396251b2d51f440cc79d18c123f2672a31bc25f305c9aa`;
  it differs because implementation/lock hashes are part of dataset identity.
- Targeted robustness tests: 4 passed. Complete lightweight suite in the fresh environment:
  **76 passed, 185 upstream deprecation warnings**, 41.25 seconds. No real encoder training
  or external benchmark job was run by the suite.
- Ruff lint and format checks pass; strict Pyright: 0 errors, 0 warnings.
- Fresh-environment real-model Uvicorn startup succeeded. Health, single scan, two-input
  batch and metrics returned HTTP 200. Model identity matches the Task 7 artifact:
  `4fc6e6ff38900cf703a2e8da0fb3a941c2e8a76a9d648eb614c3e8e29ada695c`.
- Versioned/pending source inventory contained no files over 1 MB, raw/processed Parquet,
  model binaries or temporary logs. Reachable Git history had no blobs over 1 MB. Small
  curated JSONL inputs are intentionally versioned. Generated artifacts, superseded partial
  runs, caches and logs remain ignored and excluded from Docker context.
- No hardcoded Windows drive paths found in source/config/docs, and no matches for checked
  private-key/AWS/GitHub/OpenAI credential patterns. These bounded checks are not a guarantee
  that every possible secret format is absent. Benchmark account strings are mock fixtures.
- Local Markdown links checked. Stale introductory claims were replaced. Reproduction now
  documents optional dependencies, fixed relative artifact prerequisites, immutable existing
  builds and historical versus rebuilt dataset fingerprints.

## Fixed defects

1. `Dockerfile`: enable `set -eu` in model acquisition so a failed download cannot be
   masked by a successful cleanup/chown. No image build result is claimed.
2. `robustness.py`: report absent language groups as `no_samples` and wholly ineligible
   transformation families as `no_valid_samples`, instead of crashing or evaluating an
   empty list. Regression test uses an English-only input with no eligible surface edits.
   Existing n=14 historical results are unaffected.

## Remaining verification boundary

Neither Docker nor Podman is installed. Full/default and weightless image builds and
container startup remain **not executed**. Remote GitHub Actions is configured but has
not been observed. These prevent claiming complete container/CI verification. Run the
commands in [reproduction](docs/REPRODUCTION.md) on a Docker-enabled host; no infrastructure
installation or next ML experiment was started by this audit.

Historical artifact roots: `artifacts/baselines/task2-complete`,
`artifacts/transformer/task3-real`, `artifacts/calibration/task4-real`,
`artifacts/robustness/task5-v1`, `artifacts/agentdojo/task6-final`.
Task 8 local recomputation summary: `artifacts/task8-audit.json`; data smoke card:
`artifacts/task8-data-card.md`. They are Git ignored. Versioned reports and README contain
only values checked against these available local artifacts; a fresh clone does not contain
the model weights or historical prediction files.
