# Task 3: multilingual encoder selection

Accepted 2026-09-27, before custom-model holdout evaluation.

## Candidate comparison

This is a capability/cost comparison, not three trained experiments. All candidates
have English and Russian pretraining coverage and native Transformers encoder
implementations. Current library support does not imply ongoing upstream pretraining.
Hugging Face API metadata was checked on the decision date.

| Candidate | Parameters / layers | Context | License | Cost and maintenance evidence |
|---|---|---|---|---|
| `distilbert/distilbert-base-multilingual-cased` | ~134M encoder; Hub masked-LM artifact 135,445,755 / 6 | 512 tokens | Apache-2.0 | Smallest candidate; 104-language Wikipedia distillation. Official checkpoint last modified 2024-05-06; current native Transformers support. |
| `FacebookAI/xlm-roberta-base` | ~278M; Hub masked-LM artifact 278,885,778 / 12 | 512 usable tokens (514 positions including offset) | MIT | 100-language CommonCrawl pretraining. More layers and larger embeddings increase CPU/memory cost. Official checkpoint last modified 2024-02-19; current native Transformers support. |
| `microsoft/mdeberta-v3-base` | ~276M (86M backbone + 190M embeddings, rounded upstream figures) / 12 | 512 tokens | MIT | 100-language CC100 pretraining, disentangled attention. Larger embedding/optimizer footprint; official repository lacks safetensors metadata. Last modified 2023-04-06; native Transformers support, but this exact stack was not exercised. |

Model revisions observed respectively:
`45c032ab32cc946ad88a166f7cb282f58c753c2e`,
`e73636d4f797dec63c3081bb6ed5c7b0bb3f2089`,
`a0484667b22365f84929a935b5e50a51f71f159d`.

Parameter figures include different heads; the actual trained classifier count is
recorded in its training metadata. Cost comparisons are architectural estimates,
not measured cross-model latency. Russian coverage is not evidence of Russian
prompt-injection performance. None of these cards establishes a recent model-release
cadence; they are mature checkpoints supported by a maintained library.

## Decision and operating point

Select multilingual DistilBERT, pinned to the revision above, with a newly initialized
two-class head and full encoder fine-tuning. Its lower CPU cost fits this compact
project. XLM-R and mDeBERTa offer greater capacity at higher cost; no claim is made
that the selected model is more accurate. Do not train extra models for table size.

Retain the exact existing Python 3.14.7 / torch 2.14.0+cpu / Transformers 5.17.0 /
MLflow-skinny 3.16.1 lock. No new packages are needed. Selected-model compatibility
is checked by actual forward/backward, serialization and offline reload, rather than
assuming compatibility from a card. CUDA mixed precision is implemented but cannot
be verified with the installed CPU-only build; unsupported configurations fail.

The preregistered starter experiment is `configs/transformer.json`: seed 20260925,
three epochs maximum, AdamW 2e-5, batch 4, accumulation 2, weight decay 0.01,
256-token maximum, linear schedule, 10% warmup, patience 2, fp32 CPU, two threads.
Train on all 70 frozen training rows. No hyperparameter sweep. Select checkpoints
by maximum validation recall at an empirical FPR <=1%, then AUPRC, then lower
Brier; retain the earliest exact tie. Within an epoch choose the highest threshold
among equally good feasible thresholds. The 16-row validation set cannot estimate
1% FPR reliably: this constraint effectively requires zero observed false positives.
Report this limitation; it is not a deployed false-positive guarantee.

Use the same first-256-token truncation during training and inference. This can miss
late injections and differs from the external baseline's max-window wrapper. It is
fixed before holdout evaluation, recorded in artifacts and not changed from errors.
Calibration and long-document robustness improvements are future tasks.

## Experiment boundaries and artifact contract

Training reads train/validation only, checks their hashes and group isolation, and
logs local MLflow parameters, epoch validation metrics and small provenance artifacts.
Best-checkpoint files are overwritten only inside a new run. The optional
`every_epoch` checkpoint policy also retains each epoch; no optimizer-resume
contract is promised. Weights, tokenizer, config, label map and training metadata
are sufficient for offline inference without the source snapshot or MLflow store.
`frozen.json` seals those files and the dataset manifest before a separate final
evaluation command can read the internal test or PINT. Never change configuration
based on those results. Final evaluation refuses changed artifacts or reruns over
an existing evaluation directory. These are workflow guards, not tamper-proof access
controls. Earlier baseline test metrics were already public in repository context;
they were not used to choose this training configuration.

The offline CPU smoke uses a tiny randomly initialized encoder and fixed toy
tokenizer in a distinct MLflow experiment. It exercises training and persistence;
it cannot be evaluated or described as a real multilingual pretrained experiment.
PINT remains the eight public examples at the Task 2 revision, with original
categories/jailbreak reporting and overlap audit; full proprietary PINT is unavailable.

## Primary sources

- [DistilBERT card](https://huggingface.co/distilbert/distilbert-base-multilingual-cased)
- [XLM-R card](https://huggingface.co/FacebookAI/xlm-roberta-base)
- [mDeBERTa card](https://huggingface.co/microsoft/mdeberta-v3-base)
- [Maintained DistilBERT implementation documentation](https://huggingface.co/docs/transformers/model_doc/distilbert)

Actual outcomes and environment limitations belong in `TRANSFORMER_REPORT.md` and
`PROJECT_STATE.md`; this choice is not a production-model promotion.
