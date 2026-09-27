"""Validation-only temperature scaling and sealed, one-pass test evaluation."""

import argparse
import json
import math
import os
import platform
import subprocess
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from promptshield.baselines import write_json
from promptshield.dataset import canonical_json, digest, verify_splits
from promptshield.evaluation import Case, check_scores, error_analysis, evaluate, metrics
from promptshield.training import TransformerDetector, file_hashes, load_split


class CalibrationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    temperature_min: float = Field(default=0.05, gt=0)
    temperature_max: float = Field(default=20.0, gt=0)
    reliability_bins: int = Field(default=5, ge=2, le=100)
    high_confidence: float = Field(default=0.9, gt=0.5, lt=1)
    default_min_recall: float = Field(default=0.8, gt=0, le=1)
    high_security_min_recall: float = Field(default=0.95, gt=0, le=1)

    @model_validator(mode="after")
    def ordered(self) -> CalibrationConfig:
        if not self.temperature_min <= 1 <= self.temperature_max:
            raise ValueError("Temperature bounds must contain identity (1)")
        if self.high_security_min_recall < self.default_min_recall:
            raise ValueError("High security recall must be at least default recall")
        return self


def log_odds(scores: Sequence[float]) -> list[float]:
    check_scores([0] * len(scores), scores)
    # Binary softmax log odds equal the two-logit difference, up to float rounding.
    clipped = [min(1 - 1e-12, max(1e-12, p)) for p in scores]
    return [math.log(p) - math.log1p(-p) for p in clipped]


def sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-value)) if value >= 0 else math.exp(value) / (1 + math.exp(value))


def calibrate(scores: Sequence[float], temperature: float) -> list[float]:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    return [sigmoid(z / temperature) for z in log_odds(scores)]


def fit_temperature(
    labels: Sequence[int], scores: Sequence[float], config: CalibrationConfig
) -> dict[str, Any]:
    check_scores(labels, scores)
    if set(labels) != {0, 1}:
        raise ValueError("Calibration requires both validation classes")
    logits = log_odds(scores)
    low, high = 1 / config.temperature_max, 1 / config.temperature_min

    def derivative(inverse: float) -> float:
        return sum(z * (sigmoid(z * inverse) - y) for y, z in zip(labels, logits, strict=True))

    # NLL is convex in inverse temperature: bracket its monotonic derivative.
    if all(z == 0 for z in logits):
        inverse = 1.0
    elif derivative(low) >= 0:
        inverse = low
    elif derivative(high) <= 0:
        inverse = high
    else:
        for _ in range(80):
            middle = (low + high) / 2
            if derivative(middle) < 0:
                low = middle
            else:
                high = middle
        inverse = (low + high) / 2
    return {
        "method": "temperature_scaling",
        "temperature": 1 / inverse,
        "objective": "binary_negative_log_likelihood",
        "fit_split": "validation",
        "n": len(labels),
        "positive": sum(labels),
        "at_bound": inverse in {1 / config.temperature_max, 1 / config.temperature_min},
        "probability_clip": 1e-12,
        "optimizer": "convex_inverse_temperature_derivative_bisection_80_steps",
    }


def select_profile(
    labels: Sequence[int], scores: Sequence[float], min_recall: float
) -> dict[str, Any]:
    check_scores(labels, scores)
    if set(labels) != {0, 1} or not 0 < min_recall <= 1:
        raise ValueError("Profile requires both classes and recall in (0, 1]")
    candidates = [metrics(labels, scores, t) for t in sorted({0.0, *scores})]
    selected = min(
        (m for m in candidates if m["recall"] >= min_recall),
        key=lambda m: (m["fpr"], -m["threshold"]),
    )
    return {
        "threshold": selected["threshold"],
        "min_recall": min_recall,
        "rule": "minimize_empirical_fpr_subject_to_min_recall",
        "tie_break": "highest_threshold",
        "comparison": "score >= threshold",
        "selection_split": "validation",
        "validation_metrics": selected,
    }


def reliability(labels: Sequence[int], scores: Sequence[float], bins: int) -> dict[str, Any]:
    check_scores(labels, scores)
    if bins < 2:
        raise ValueError("At least two bins required")
    rows: list[dict[str, Any]] = []
    ece = 0.0
    for b in range(bins):
        indices = [i for i, p in enumerate(scores) if min(int(p * bins), bins - 1) == b]
        mean = sum(scores[i] for i in indices) / len(indices) if indices else None
        rate = sum(labels[i] for i in indices) / len(indices) if indices else None
        if mean is not None and rate is not None:
            ece += len(indices) / len(labels) * abs(mean - rate)
        rows.append(
            {
                "lower": b / bins,
                "upper": (b + 1) / bins,
                "n": len(indices),
                "mean_probability": mean,
                "positive_fraction": rate,
            }
        )
    return {
        "n": len(labels),
        "brier_score": metrics(labels, scores, 0.5)["brier_score"],
        "ece": ece,
        "binning": "equal_width_left_closed_last_includes_one",
        "bins": rows,
    }


def verify_inputs(model_run: Path, dataset: Path) -> dict[str, Any]:
    frozen = json.loads((model_run / "frozen.json").read_bytes())
    if frozen["status"] != "selection_frozen" or frozen["metadata"]["config"]["smoke"]:
        raise ValueError("Real frozen model required")
    if file_hashes(model_run / "model") != frozen["model_files"]:
        raise ValueError("Frozen model artifact mismatch")
    if (
        digest((dataset / "manifest.json").read_bytes())
        != frozen["metadata"]["dataset_manifest_sha256"]
    ):
        raise ValueError("Frozen dataset manifest mismatch")
    return frozen


def fit(
    root: Path, model_run: Path, dataset: Path, output: Path, config: CalibrationConfig
) -> dict[str, Any]:
    if output.exists() or not output.resolve().is_relative_to(root.resolve() / "artifacts"):
        raise ValueError("Choose a new output under artifacts/")
    source = verify_inputs(model_run, dataset)
    manifest = json.loads((dataset / "manifest.json").read_bytes())
    # No train/test/PINT records or predictions enter this command.
    validation = load_split(dataset, manifest, "validation")
    cases = [Case.from_sample(r) for r in validation]
    raw = TransformerDetector(model_run / "model").score([r.text for r in cases])
    labels = [r.target for r in cases]
    scaling = fit_temperature(labels, raw, config)
    calibrated = calibrate(raw, scaling["temperature"])
    profiles = {
        name: select_profile(labels, calibrated, target)
        for name, target in (
            ("default", config.default_min_recall),
            ("high_security", config.high_security_min_recall),
        )
    }
    output.mkdir(parents=True)
    write_json(output / "calibration.json", scaling)
    write_json(output / "profiles.json", profiles)
    write_json(
        output / "validation.json",
        {
            "scope": "resubstitution_validation_reused_for_checkpoint_calibration_thresholds",
            "before": reliability(labels, raw, config.reliability_bins),
            "after": reliability(labels, calibrated, config.reliability_bins),
            "predictions": [
                {"id": r.id, "target": r.target, "raw": p, "calibrated": q}
                for r, p, q in zip(cases, raw, calibrated, strict=True)
            ],
        },
    )
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True
    )
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    import mlflow

    tracking: Any = mlflow
    tracking.set_tracking_uri((root.resolve() / "mlruns").as_uri())
    tracking.set_experiment("promptshield-calibration")
    with tracking.start_run(run_name=output.name) as run:
        tracking.log_params(
            {
                **config.model_dump(),
                "method": scaling["method"],
                "model_version": source["metadata"]["version"],
                "dataset_fingerprint": manifest["fingerprint"],
                "git_commit": git.stdout.strip(),
            }
        )
        tracking.log_metrics(
            {
                "temperature": scaling["temperature"],
                "validation_brier_before": reliability(labels, raw, config.reliability_bins)[
                    "brier_score"
                ],
                "validation_brier_after": reliability(labels, calibrated, config.reliability_bins)[
                    "brier_score"
                ],
            }
        )
        for name in ("calibration.json", "profiles.json"):
            tracking.log_artifact(str(output / name))
        tracking_id = run.info.run_id
    frozen = {
        "status": "calibration_and_profiles_frozen",
        "created_at": datetime.now(UTC).isoformat(),
        "config": config.model_dump(),
        "source_frozen_sha256": digest((model_run / "frozen.json").read_bytes()),
        "model_version": source["metadata"]["version"],
        "dataset_fingerprint": manifest["fingerprint"],
        "files": file_hashes(output),
        "code_hashes": {
            p.name: digest(p.read_bytes()) for p in (root / "src/promptshield").glob("*.py")
        },
        "lock_sha256": digest((root / "uv.lock").read_bytes()),
        "classifier_retrained": False,
        "git_commit": git.stdout.strip(),
        "git_dirty": bool(status.stdout.strip()),
        "python": platform.python_version(),
        "mlflow_run_id": tracking_id,
    }
    write_json(output / "frozen.json", frozen)
    return frozen


def final_evaluation(model_run: Path, dataset: Path, output: Path) -> dict[str, Any]:
    frozen = json.loads((output / "frozen.json").read_bytes())
    if frozen["status"] != "calibration_and_profiles_frozen":
        raise ValueError("Calibration must be frozen")
    for name, expected in frozen["files"].items():
        if digest((output / name).read_bytes()) != expected:
            raise ValueError("Frozen calibration artifact mismatch")
    if digest((model_run / "frozen.json").read_bytes()) != frozen["source_frozen_sha256"]:
        raise ValueError("Source model freeze mismatch")
    verify_inputs(model_run, dataset)
    destination = output / "final_test"
    if destination.exists():
        raise ValueError("Final test evaluation already exists; do not overwrite or retune")
    config = CalibrationConfig.model_validate(frozen["config"])
    scaling = json.loads((output / "calibration.json").read_bytes())
    profiles = json.loads((output / "profiles.json").read_bytes())
    manifest = json.loads((dataset / "manifest.json").read_bytes())
    splits = {n: load_split(dataset, manifest, n) for n in ("train", "validation", "test")}
    verify_splits(splits)
    cases = [Case.from_sample(r) for r in splits["test"]]
    destination.mkdir()
    raw = TransformerDetector(model_run / "model").score([r.text for r in cases])
    calibrated = calibrate(raw, scaling["temperature"])
    labels = [r.target for r in cases]
    results = {
        "scope": "previously_observed_task3_test_frozen_task4_followup",
        "frozen_sha256": digest((output / "frozen.json").read_bytes()),
        "before": reliability(labels, raw, config.reliability_bins),
        "after": reliability(labels, calibrated, config.reliability_bins),
        "profiles": {
            name: evaluate(cases, calibrated, p["threshold"]) for name, p in profiles.items()
        },
    }
    write_json(destination / "metrics.json", results)
    with (destination / "predictions.jsonl").open("wb") as stream:
        for row, p, q in zip(cases, raw, calibrated, strict=True):
            stream.write(
                canonical_json(
                    {
                        **asdict(row),
                        "raw_score": p,
                        "calibrated_score": q,
                        "decisions": {
                            name: int(q >= v["threshold"]) for name, v in profiles.items()
                        },
                    }
                )
                + b"\n"
            )
    for name, profile in profiles.items():
        errors = error_analysis(cases, calibrated, profile["threshold"])
        all_errors = errors["all_errors"]
        errors["english_errors"] = [e for e in all_errors if e["language"] == "en"]
        errors["high_confidence_definition"] = (
            f"FP calibrated score >= {config.high_confidence}; "
            f"FN <= {1 - config.high_confidence}; not a confidence guarantee"
        )
        errors["high_confidence_false_positives"] = [
            e
            for e in all_errors
            if e["error"] == "false_positive" and e["risk_score"] >= config.high_confidence
        ]
        errors["high_confidence_false_negatives"] = [
            e
            for e in all_errors
            if e["error"] == "false_negative" and e["risk_score"] <= 1 - config.high_confidence
        ]
        write_json(destination / f"{name}_errors.json", errors)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["fit", "evaluate"])
    parser.add_argument("--model-run", type=Path, default=Path("artifacts/transformer/task3-real"))
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/v1"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    if args.command == "fit":
        config = (
            CalibrationConfig.model_validate_json(args.config.read_bytes())
            if args.config
            else CalibrationConfig()
        )
        fit(Path.cwd(), args.model_run, args.dataset, args.output, config)
    else:
        if args.config:
            parser.error("Evaluation uses the frozen config only")
        final_evaluation(args.model_run, args.dataset, args.output)


if __name__ == "__main__":
    main()
