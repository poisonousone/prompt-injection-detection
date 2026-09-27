"""Validation-selected multilingual training and a separate frozen evaluation command."""

# pyright: reportMissingTypeStubs=false
import argparse
import importlib.metadata
import json
import math
import os
import platform
import random
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from promptshield.baselines import processor_name, save_evaluation, write_json
from promptshield.dataset import canonical_json, digest, verify_splits
from promptshield.evaluation import Case, check_scores, metrics
from promptshield.external import PINT_REVISION, PINT_SHA256, overlap_audit, read_pint
from promptshield.predictors import Detector, validate_inputs
from promptshield.schema import Sample, read_records

MODEL_ID = "distilbert/distilbert-base-multilingual-cased"
MODEL_REVISION = "45c032ab32cc946ad88a166f7cb282f58c753c2e"
LABELS = {0: "benign", 1: "prompt_injection"}


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    seed: int = Field(default=20260925, ge=0, le=2**32 - 1)
    learning_rate: float = Field(default=2e-5, gt=0)
    batch_size: int = Field(default=4, ge=1)
    max_length: int = Field(default=256, ge=8, le=512)
    weight_decay: float = Field(default=0.01, ge=0)
    scheduler: Literal["linear", "cosine", "constant"] = "linear"
    warmup_ratio: float = Field(default=0.1, ge=0, lt=1)
    gradient_accumulation: int = Field(default=2, ge=1)
    epochs: int = Field(default=3, ge=1)
    patience: int = Field(default=2, ge=1)
    checkpoint_policy: Literal["best", "every_epoch"] = "best"
    precision: Literal["fp32", "fp16", "bf16"] = "fp32"
    device: Literal["cpu", "cuda"] = "cpu"
    threads: int = Field(default=2, ge=1)
    smoke: bool = False


def security_threshold(labels: Sequence[int], scores: Sequence[float]) -> dict[str, Any]:
    """Maximize validation recall subject to empirical FPR <= 1%; no interpolation."""
    check_scores(labels, scores)
    if set(labels) != {0, 1}:
        raise ValueError("Selection requires both validation classes")
    candidates = sorted({0.0, 1.0, *scores})
    eligible = [
        (float(m["recall"]), t, m)
        for t in candidates
        if (m := metrics(labels, scores, t))["fpr"] <= 0.01
    ]
    if not eligible:
        raise ValueError("No probability threshold meets validation FPR constraint")
    _, threshold, summary = max(eligible, key=lambda item: (item[0], item[1]))
    return {
        "threshold": threshold,
        "selection_split": "validation",
        "objective": "max_recall_at_empirical_fpr_le_0.01",
        "tie_break": "highest_threshold",
        "metrics": summary,
    }


def selection_key(selection: dict[str, Any]) -> tuple[float, float, float]:
    m = selection["metrics"]
    return float(m["recall"]), float(m["auprc"]), -float(m["brier_score"])


def file_hashes(directory: Path) -> dict[str, str]:
    return {p.name: digest(p.read_bytes()) for p in sorted(directory.iterdir()) if p.is_file()}


def load_split(dataset: Path, manifest: dict[str, Any], name: str) -> list[Sample]:
    path = dataset / f"{name}.parquet"
    if digest(path.read_bytes()) != manifest["artifacts"][path.name]:
        raise ValueError(f"Frozen {name} artifact mismatch")
    rows = read_records(path)
    if sorted(r.id for r in rows) != sorted(manifest["assignments"][name]):
        raise ValueError(f"Frozen {name} assignment mismatch")
    return rows


class TransformerDetector(Detector):
    def __init__(self, artifact: Path, device: str = "cpu", threads: int = 2) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch: Any = torch
        torch.set_num_threads(threads)
        metadata = json.loads((artifact / "training.json").read_bytes())
        self.metadata: dict[str, Any] = metadata
        self.max_length = metadata["config"]["max_length"]
        self.device = device
        tokenizer_api: Any = AutoTokenizer
        model_api: Any = AutoModelForSequenceClassification
        self.tokenizer: Any = tokenizer_api.from_pretrained(
            str(artifact), local_files_only=True, trust_remote_code=False
        )
        self.model: Any = model_api.from_pretrained(
            str(artifact), local_files_only=True, trust_remote_code=False, use_safetensors=True
        ).to(device)
        if dict(self.model.config.id2label) != LABELS:
            raise ValueError("Unexpected custom model label mapping")
        self.model.eval()
        super().__init__(
            {
                "model_id": metadata["model_id"],
                "version": metadata["version"],
                "smoke": metadata["config"]["smoke"],
                "max_length": self.max_length,
                "long_input_policy": "truncate_tail",
            },
            metadata["selection"]["threshold"],
        )

    def score(self, texts: Sequence[str], batch_size: int = 16) -> list[float]:
        validate_inputs(texts, batch_size)
        return score_model(
            self.model, self.tokenizer, texts, self.max_length, batch_size, self.device
        )


def score_model(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    max_length: int,
    batch_size: int,
    device: str,
) -> list[float]:
    import torch

    model.eval()
    scores: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            inputs = tokenizer(
                list(texts[start : start + batch_size]),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            probabilities: Any = torch.softmax(model(**inputs).logits.float(), dim=-1)
            scores.extend(float(v) for v in probabilities[:, 1].tolist())
    return scores


def initialize(config: TrainingConfig, root: Path) -> tuple[Any, Any]:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DistilBertConfig,
        DistilBertForSequenceClassification,
        PreTrainedTokenizerFast,
    )

    if config.smoke:
        from tokenizers import Tokenizer, models, pre_tokenizers, processors

        # Offline random toy encoder. Never mislabeled as the pretrained multilingual model.
        vocab = {
            word: i
            for i, word in enumerate(
                [
                    "[PAD]",
                    "[UNK]",
                    "[CLS]",
                    "[SEP]",
                    "[MASK]",
                    "hello",
                    "ignore",
                    "привет",
                    "игнорируй",
                ]
            )
        }
        model_api: Any = models
        pre_api: Any = pre_tokenizers
        processor_api: Any = processors
        config_api: Any = DistilBertConfig
        backend: Any = Tokenizer(model_api.WordLevel(vocab, unk_token="[UNK]"))
        backend.pre_tokenizer = pre_api.Whitespace()
        backend.post_processor = processor_api.TemplateProcessing(
            single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
        )
        tokenizer: Any = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]",
            model_input_names=["input_ids", "attention_mask"],
        )
        factory: Any = DistilBertForSequenceClassification
        model = factory(
            config_api(
                vocab_size=len(vocab),
                dim=16,
                hidden_dim=32,
                n_layers=1,
                n_heads=2,
                max_position_embeddings=512,
                num_labels=2,
                id2label=LABELS,
                label2id={v: k for k, v in LABELS.items()},
            )
        )
        return model, tokenizer
    tokenizer_factory: Any = AutoTokenizer
    model_factory: Any = AutoModelForSequenceClassification
    kwargs = dict(
        revision=MODEL_REVISION,
        cache_dir=str(root / ".cache/huggingface/hub"),
        local_files_only=True,
        trust_remote_code=False,
    )
    tokenizer = tokenizer_factory.from_pretrained(MODEL_ID, **kwargs)
    model = model_factory.from_pretrained(
        MODEL_ID,
        **kwargs,
        use_safetensors=True,
        num_labels=2,
        id2label=LABELS,
        label2id={v: k for k, v in LABELS.items()},
    )
    return model, tokenizer


def train(root: Path, dataset: Path, output: Path, config: TrainingConfig) -> dict[str, Any]:
    import torch
    import transformers

    transformer_api: Any = transformers
    torch_api: Any = torch

    root, dataset, output = root.resolve(), dataset.resolve(), output.resolve()
    if not output.is_relative_to(root / "artifacts") or output.exists():
        raise ValueError("Choose a new output directory under artifacts/")
    if config.smoke and (config.device != "cpu" or config.precision != "fp32"):
        raise ValueError("Smoke requires CPU fp32")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable in this environment")
    if config.precision != "fp32" and config.device != "cuda":
        raise ValueError("Mixed precision requires CUDA; CPU uses fp32")
    if config.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("CUDA device does not support bf16")
    manifest = json.loads((dataset / "manifest.json").read_bytes())
    # Only training and validation are read here; holdouts belong to evaluate_frozen.
    rows = load_split(dataset, manifest, "train")
    validation = load_split(dataset, manifest, "validation")
    verify_splits({"train": rows, "validation": validation})
    if set(r.label for r in rows) != set(LABELS.values()):
        raise ValueError("Training requires both classes")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch_api.manual_seed(config.seed)
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    model, tokenizer = initialize(config, root)
    model.to(config.device)
    output.mkdir(parents=True)
    artifact = output / "model"
    artifact.mkdir()
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True
    )
    metadata: dict[str, Any] = {
        "config": config.model_dump(),
        "dataset_fingerprint": manifest["fingerprint"],
        "dataset_manifest_sha256": digest((dataset / "manifest.json").read_bytes()),
        "split_hashes": manifest["artifacts"],
        "git_commit": git.stdout.strip(),
        "git_dirty": bool(status.stdout.strip()),
        "lock_sha256": digest((root / "uv.lock").read_bytes()),
        "code_hashes": {
            p.name: digest(p.read_bytes()) for p in (root / "src/promptshield").glob("*.py")
        },
        "model_id": "promptshield/toy-smoke" if config.smoke else MODEL_ID,
        "revision": None if config.smoke else MODEL_REVISION,
        "label_mapping": LABELS,
        "long_input_policy": "truncate_tail",
        "checkpoint_selection": (
            "validation recall at empirical FPR<=1%, then AUPRC, "
            "then lower Brier; earliest exact tie"
        ),
        "python": platform.python_version(),
        "packages": {
            n: importlib.metadata.version(n) for n in ("torch", "transformers", "mlflow-skinny")
        },
        "hardware": {
            "processor": processor_name(),
            "device": config.device,
            "cuda": torch.version.cuda,
            "threads": config.threads,
        },
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "train_count": len(rows),
        "validation_count": len(validation),
    }
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    steps_per_epoch = math.ceil(
        math.ceil(len(rows) / config.batch_size) / config.gradient_accumulation
    )
    total_steps = steps_per_epoch * config.epochs
    scheduler_api: Any = transformer_api.get_scheduler
    scheduler = scheduler_api(
        config.scheduler,
        optimizer=optimizer,
        num_warmup_steps=int(total_steps * config.warmup_ratio),
        num_training_steps=total_steps,
    )
    scaler: Any = torch.amp.GradScaler("cuda", enabled=config.precision == "fp16")
    dtype = torch.float16 if config.precision == "fp16" else torch.bfloat16
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    import mlflow

    tracking: Any = mlflow
    tracking.set_tracking_uri((root / "mlruns").as_uri())
    tracking.set_experiment(
        "promptshield-transformer-smoke" if config.smoke else "promptshield-transformer"
    )
    started = time.perf_counter()
    best: tuple[float, float, float] | None = None
    stale = 0
    history: list[dict[str, Any]] = []
    rng = random.Random(config.seed)
    with tracking.start_run(run_name=output.name) as run:
        metadata["mlflow"] = {
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "tracking_uri": (root / "mlruns").as_uri(),
        }
        tracking.log_params(
            {
                **config.model_dump(),
                "model_id": metadata["model_id"],
                "revision": metadata["revision"],
                "git_commit": metadata["git_commit"],
                "dataset_fingerprint": manifest["fingerprint"],
            }
        )
        for epoch in range(1, config.epochs + 1):
            model.train()
            indices = list(range(len(rows)))
            rng.shuffle(indices)
            batches = [
                indices[i : i + config.batch_size]
                for i in range(0, len(indices), config.batch_size)
            ]
            loss_sum = 0.0
            for group_start in range(0, len(batches), config.gradient_accumulation):
                group = batches[group_start : group_start + config.gradient_accumulation]
                group_count = sum(len(b) for b in group)
                optimizer.zero_grad(set_to_none=True)
                for batch in group:
                    inputs = tokenizer(
                        [rows[i].text for i in batch],
                        padding=True,
                        truncation=True,
                        max_length=config.max_length,
                        return_tensors="pt",
                    ).to(config.device)
                    labels = torch.tensor(
                        [int(rows[i].label == "prompt_injection") for i in batch],
                        device=config.device,
                    )
                    with torch.autocast(
                        device_type=config.device, dtype=dtype, enabled=config.precision != "fp32"
                    ):
                        loss = model(**inputs, labels=labels).loss
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite training loss")
                    loss_sum += float(loss.detach()) * len(batch)
                    # Weight by samples, including the final partial accumulation group.
                    scaler.scale(loss * len(batch) / group_count).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= previous_scale:
                    scheduler.step()
            scores = score_model(
                model,
                tokenizer,
                [r.text for r in validation],
                config.max_length,
                config.batch_size,
                config.device,
            )
            selected = security_threshold(
                [int(r.label == "prompt_injection") for r in validation], scores
            )
            key = selection_key(selected)
            history.append(
                {"epoch": epoch, "train_loss": loss_sum / len(rows), "selection": selected}
            )
            if config.checkpoint_policy == "every_epoch":
                checkpoint = output / "checkpoints" / f"epoch-{epoch}"
                checkpoint.mkdir(parents=True)
                model.save_pretrained(str(checkpoint), safe_serialization=True)
                tokenizer.save_pretrained(str(checkpoint))
                write_json(
                    checkpoint / "training.json",
                    {
                        **metadata,
                        "selected_epoch": epoch,
                        "selection": selected,
                        "version": digest(canonical_json(file_hashes(checkpoint))),
                    },
                )
            tracking.log_metrics(
                {
                    "train_loss": loss_sum / len(rows),
                    **{
                        f"validation_{k}": float(v)
                        for k, v in selected["metrics"].items()
                        if type(v) in {int, float}
                    },
                },
                step=epoch,
            )
            if best is None or key > best:
                best, stale = key, 0
                model.save_pretrained(str(artifact), safe_serialization=True)
                tokenizer.save_pretrained(str(artifact))
                metadata.update(selected_epoch=epoch, selection=selected)
            else:
                stale += 1
            print(
                json.dumps(
                    {"epoch": epoch, "train_loss": loss_sum / len(rows), "selection_key": key}
                ),
                flush=True,
            )
            if stale >= config.patience:
                break
        metadata.update(
            elapsed_seconds=time.perf_counter() - started, epochs_completed=len(history)
        )
        metadata["version"] = digest(canonical_json(file_hashes(artifact)))
        write_json(artifact / "training.json", metadata)
        write_json(output / "history.json", history)
        write_json(
            output / "frozen.json",
            {
                "status": "smoke_only" if config.smoke else "selection_frozen",
                "model_files": file_hashes(artifact),
                "metadata": metadata,
            },
        )
        # Small provenance/selection artifacts only; weights stay in the local model directory.
        for path in (artifact / "training.json", output / "history.json", output / "frozen.json"):
            tracking.log_artifact(str(path))
    return metadata


def evaluate_frozen(root: Path, dataset: Path, output: Path) -> dict[str, Any]:
    frozen = json.loads((output / "frozen.json").read_bytes())
    if frozen["status"] != "selection_frozen":
        raise ValueError("Smoke artifacts cannot be evaluated as real experiments")
    if file_hashes(output / "model") != frozen["model_files"]:
        raise ValueError("Frozen model artifact mismatch")
    manifest = json.loads((dataset / "manifest.json").read_bytes())
    if (
        digest((dataset / "manifest.json").read_bytes())
        != frozen["metadata"]["dataset_manifest_sha256"]
    ):
        raise ValueError("Frozen dataset manifest mismatch")
    evaluation = output / "evaluation"
    if evaluation.exists():
        raise ValueError("Final evaluation already exists; do not overwrite")
    splits = {n: load_split(dataset, manifest, n) for n in ("train", "validation", "test")}
    verify_splits(splits)
    pint_path = root / "data/raw/pint" / PINT_REVISION / "example-dataset.yaml"
    pint = read_pint(pint_path, PINT_SHA256)
    evaluation.mkdir()
    model = TransformerDetector(output / "model")
    internal = {n: [Case.from_sample(r) for r in rows] for n, rows in splits.items()}
    write_json(evaluation / "pint_overlap.json", overlap_audit(internal, pint))
    results = {
        "internal_test": save_evaluation(
            model, internal["test"], evaluation / "internal_test", "frozen_internal_test"
        ),
        "pint": save_evaluation(model, pint, evaluation / "pint", "public_example_smoke"),
    }
    write_json(
        evaluation / "run.json",
        {
            "frozen_sha256": digest((output / "frozen.json").read_bytes()),
            "pint_revision": PINT_REVISION,
            "pint_sha256": PINT_SHA256,
            "results": results,
        },
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "evaluate"])
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/v1"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.command == "evaluate":
        if args.config or args.smoke:
            parser.error("Evaluation accepts only a frozen artifact and dataset")
        evaluate_frozen(Path.cwd(), args.dataset, args.output)
    else:
        values: dict[str, Any] = json.loads(args.config.read_bytes()) if args.config else {}
        if args.smoke:
            values.update(smoke=True, device="cpu", precision="fp32", epochs=1, max_length=32)
        train(Path.cwd(), args.dataset, args.output, TrainingConfig.model_validate(values))


if __name__ == "__main__":
    main()
