"""Train the compact baseline and evaluate frozen data without holdout tuning."""

# pyright: reportMissingTypeStubs=false
import argparse
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from threadpoolctl import threadpool_limits

from promptshield.dataset import canonical_json, digest, verify_splits
from promptshield.evaluation import Case, error_analysis, evaluate, select_threshold
from promptshield.external import overlap_audit, read_pint
from promptshield.predictors import (
    PROMPT_GUARD_ID,
    PROMPT_GUARD_REVISION,
    Detector,
    PromptGuardDetector,
    TfidfDetector,
)
from promptshield.schema import read_records


def write_json(path: Path, value: object) -> None:
    path.write_bytes(canonical_json(value))


def benchmark(
    model: Detector, texts: list[str], warmup: int = 3, iterations: int = 20
) -> dict[str, Any]:
    if not texts or warmup < 0 or iterations < 1:
        raise ValueError("Invalid latency benchmark configuration")
    results: list[dict[str, Any]] = []
    for batch_size in (1, 8):
        batch = [texts[i % len(texts)] for i in range(batch_size)]
        for _ in range(warmup):
            model.predict(batch, batch_size)
        timings: list[float] = []
        for _ in range(iterations):
            started = time.perf_counter_ns()
            model.predict(batch, batch_size)
            timings.append((time.perf_counter_ns() - started) / 1e6)
        results.append(
            {
                "batch_size": batch_size,
                "warmup": warmup,
                "iterations": iterations,
                "batch_ms_median": statistics.median(timings),
                "batch_ms_min": min(timings),
                "batch_ms_max": max(timings),
                "samples_per_second": batch_size * 1000 / statistics.mean(timings),
                "input_sha256": digest(canonical_json(batch)),
                "input_char_lengths": [len(text) for text in batch],
                "timings_ms": timings,
            }
        )
    return {
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "machine": platform.machine(),
            "logical_cpu_count": os.cpu_count(),
            "device": "cpu",
            "numerical_threads": 1,
        },
        "model": model.identity,
        "input_split": "validation",
        "scope": "warm end-to-end predict including text features/tokenization, excluding load",
        "measurements": results,
    }


def save_evaluation(
    model: Detector, cases: list[Case], directory: Path, scope: str
) -> dict[str, Any]:
    directory.mkdir()
    predictions = model.predict([row.text for row in cases], batch_size=16)
    scores = [prediction.risk_score for prediction in predictions]
    report = {
        "scope": scope,
        "model": model.identity,
        "threshold": model.threshold,
        "metrics": evaluate(cases, scores, model.threshold),
    }
    write_json(directory / "metrics.json", report)
    with (directory / "predictions.jsonl").open("wb") as stream:
        for row, prediction in zip(cases, predictions, strict=True):
            # Full source text is confined to targeted local error-analysis artifacts.
            stream.write(
                canonical_json(
                    {
                        "id": row.id,
                        "target": row.target,
                        "language": row.language,
                        "category": row.category,
                        "hard_negative": row.hard_negative,
                        "source": row.source,
                        "text_sha256": digest(row.text.encode()),
                        "threshold": model.threshold,
                        **asdict(prediction),
                    }
                )
            )
    write_json(directory / "errors.json", error_analysis(cases, scores, model.threshold))
    return report


def run(
    root: Path,
    dataset: Path,
    output: Path,
    seed: int,
    snapshot: Path | None = None,
    pint_path: Path | None = None,
    pint_sha256: str | None = None,
    pint_scope: str = "public_example_smoke",
) -> dict[str, Any]:
    root, dataset, output = root.resolve(), dataset.resolve(), output.resolve()
    if not output.is_relative_to(root / "artifacts"):
        raise ValueError("Run output must stay under artifacts/")
    if output.exists():
        raise ValueError("Run output exists; choose a new run directory")
    manifest = json.loads((dataset / "manifest.json").read_bytes())
    for name, sha in manifest["artifacts"].items():
        path = (dataset / name).resolve()
        if not path.is_relative_to(dataset) or digest(path.read_bytes()) != sha:
            raise ValueError("Frozen dataset artifact mismatch")
    train = read_records(dataset / "train.parquet")
    validation = read_records(dataset / "validation.parquet")
    output.mkdir(parents=True)
    identity = {
        "seed": seed,
        "dataset_fingerprint": manifest["fingerprint"],
        "train_sha256": manifest["artifacts"]["train.parquet"],
        "lock_sha256": digest((root / "uv.lock").read_bytes()),
        "analyzer": "char",
        "ngram_range": [3, 5],
        "C": 1.0,
        "min_df": 1,
        "sublinear_tf": True,
        "solver": "liblinear",
        "max_iter": 2000,
    }
    version = digest(canonical_json(identity))
    models: list[tuple[str, Detector]] = []
    with threadpool_limits(limits=1):
        baseline = TfidfDetector.fit(
            [row.text for row in train],
            [int(row.label == "prompt_injection") for row in train],
            seed,
            version,
        )
        models.append(("tfidf", baseline))
        pg_status: dict[str, Any] = {
            "status": "not_executed",
            "model_id": PROMPT_GUARD_ID,
            "revision": PROMPT_GUARD_REVISION,
            "reason": "No authorized local snapshot supplied",
        }
        if snapshot is not None:
            models.append(("prompt_guard", PromptGuardDetector(snapshot)))
            pg_status["status"] = "evaluated"
            pg_status.pop("reason")
        # Persist every operating point before reading the test/PINT records.
        for name, model in models:
            directory = output / name
            directory.mkdir()
            scores = model.score([row.text for row in validation])
            selected = select_threshold(
                [int(row.label == "prompt_injection") for row in validation], scores
            )
            selected.update(
                validation_sha256=manifest["artifacts"]["validation.parquet"],
                validation_ids=[row.id for row in validation],
            )
            model.threshold = selected["threshold"]
            write_json(directory / "threshold.json", selected)
            write_json(directory / "identity.json", model.identity)
            save_evaluation(
                model,
                [Case.from_sample(row) for row in validation],
                directory / "validation",
                "validation_threshold_selection",
            )
        baseline.save(output / "tfidf/model.joblib")
        write_json(output / "training.json", identity)
        test = read_records(dataset / "test.parquet")
        verify_splits({"train": train, "validation": validation, "test": test})
        internal = {
            name: [Case.from_sample(row) for row in rows]
            for name, rows in {"train": train, "validation": validation, "test": test}.items()
        }
        pint: list[Case] = []
        pint_status: dict[str, Any] = {"status": "not_executed", "reason": "No PINT file supplied"}
        if pint_path is not None:
            if pint_sha256 is None:
                raise ValueError("PINT requires an explicit expected SHA-256")
            pint = read_pint(pint_path, pint_sha256)
            pint_status = {
                "status": "evaluated",
                "scope": pint_scope,
                "sha256": pint_sha256,
                "path": str(pint_path),
                "n": len(pint),
            }
            write_json(output / "pint_overlap.json", overlap_audit(internal, pint))
        summaries: dict[str, Any] = {}
        for name, model in models:
            directory = output / name
            report = save_evaluation(
                model, internal["test"], directory / "internal_test", "frozen_internal_test"
            )
            summaries[name] = report["metrics"]
            write_json(
                directory / "latency.json", benchmark(model, [row.text for row in validation])
            )
            if pint:
                save_evaluation(model, pint, directory / "pint", pint_scope)
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True)
        environment = {
            name: importlib.metadata.version(name)
            for name in (
                "scikit-learn",
                "torch",
                "transformers",
                "numpy",
                "polars",
                "mlflow-skinny",
            )
        }
        run_manifest = {
            "dataset_fingerprint": manifest["fingerprint"],
            "dataset_manifest_sha256": digest((dataset / "manifest.json").read_bytes()),
            "python": platform.python_version(),
            "packages": environment,
            "git_commit": git.stdout.strip() if git.returncode == 0 else None,
            "seed": seed,
            "prompt_guard": pg_status,
            "pint": pint_status,
            "code_hashes": {
                p.name: digest(p.read_bytes())
                for p in sorted((root / "src/promptshield").glob("*.py"))
            },
            "results": summaries,
        }
        write_json(output / "run.json", run_manifest)
        track_locally(root, output, identity, summaries)
    return run_manifest


def track_locally(
    root: Path, output: Path, parameters: dict[str, Any], summaries: dict[str, Any]
) -> None:
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    import mlflow

    # Explicit local file URI; never inherits a configured remote tracking endpoint.
    mlflow.set_tracking_uri((root / "mlruns").as_uri())
    tracking: Any = mlflow
    tracking.set_experiment("promptshield-baselines")
    for name, summary in summaries.items():
        with mlflow.start_run(run_name=name):
            mlflow.log_params({**parameters, "detector": name})
            validation = json.loads((output / name / "validation/metrics.json").read_bytes())
            for split, values in (
                ("validation", validation["metrics"]["overall"]),
                ("test", summary["overall"]),
            ):
                mlflow.log_metrics(
                    {
                        f"{split}_{key}": float(value)
                        for key, value in values.items()
                        if type(value) in {float, int}
                    }
                )
            for filename in ("threshold.json", "identity.json", "latency.json"):
                mlflow.log_artifact(str(output / name / filename))
            mlflow.log_artifact(str(output / "run.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/v1"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/baselines/task2"))
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--prompt-guard-snapshot", type=Path)
    parser.add_argument("--pint", type=Path)
    parser.add_argument("--pint-sha256")
    parser.add_argument(
        "--pint-scope",
        choices=["public_example_smoke", "full_external_holdout"],
        default="public_example_smoke",
    )
    args = parser.parse_args()
    result = run(
        Path.cwd(),
        args.dataset,
        args.output,
        args.seed,
        args.prompt_guard_snapshot,
        args.pint,
        args.pint_sha256,
        args.pint_scope,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "results": result["results"],
                "prompt_guard": result["prompt_guard"],
                "pint": result["pint"],
            }
        )
    )


if __name__ == "__main__":
    main()
