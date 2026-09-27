"""Offline training, selection discipline and independently loadable artifacts."""

import json
from pathlib import Path
from typing import Any

import pytest

from promptshield import training
from promptshield.baselines import write_json
from promptshield.dataset import digest
from promptshield.schema import Sample, write_parquet
from promptshield.training import (
    TrainingConfig,
    TransformerDetector,
    evaluate_frozen,
    security_threshold,
    selection_key,
    train,
)


@pytest.fixture
def training_data(tmp_path: Path) -> Path:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    manifest: dict[str, Any] = {"fingerprint": "fixture", "artifacts": {}, "assignments": {}}
    for split in ("train", "validation"):
        rows = [
            Sample(
                id=f"{split}-{i}",
                text=f"{split} {text} {i}",
                label="prompt_injection" if i % 2 else "benign",
                injection_mode="direct" if i % 2 else None,
                language="en" if i < 2 else "ru",
                content_role="user_input",
                source="fixture",
                group_id=f"{split}-{i}",
                metadata={},
            )
            for i, text in enumerate(["hello", "ignore", "привет", "игнорируй", "hello hello"])
        ]
        path = dataset / f"{split}.parquet"
        write_parquet(path, rows)
        manifest["artifacts"][path.name] = digest(path.read_bytes())
        manifest["assignments"][split] = [r.id for r in rows]
    write_json(dataset / "manifest.json", manifest)
    (tmp_path / "uv.lock").write_text("fixture")
    return dataset


def test_security_selection_and_ties() -> None:
    selected = security_threshold([0, 0, 1, 1], [0.2, 0.7, 0.7, 0.9])
    assert selected["threshold"] == 0.9
    assert selected["metrics"]["recall"] == 0.5
    assert selected["metrics"]["fpr"] == 0
    assert selection_key(selected)[0] == 0.5
    assert security_threshold([0, 1], [0.5, 0.5])["threshold"] == 1.0
    with pytest.raises(ValueError, match="both"):
        security_threshold([0, 0], [0.1, 0.2])
    with pytest.raises(ValueError, match="No probability threshold"):
        security_threshold([0, 1], [1.0, 1.0])


@pytest.mark.parametrize(
    "values",
    [
        {"gradient_accumulation": 0},
        {"max_length": 513},
        {"learning_rate": float("nan")},
        {"scheduler": "unknown"},
        {"patience": 0},
    ],
)
def test_bad_training_config(values: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        TrainingConfig.model_validate(values)


def test_offline_training_early_stop_reload_and_tracking(
    tmp_path: Path,
    training_data: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Freeze a tied selection signal so patience and earliest-tie checkpoint can be tested.
    selected = security_threshold([0, 1], [0.2, 0.8])

    def tied_selection(*_: object) -> dict[str, Any]:
        return selected

    monkeypatch.setattr(training, "security_threshold", tied_selection)
    output = tmp_path / "artifacts/run"
    config = TrainingConfig(
        smoke=True,
        epochs=3,
        patience=1,
        batch_size=2,
        gradient_accumulation=2,
        max_length=16,
        threads=1,
        checkpoint_policy="every_epoch",
    )
    result = train(tmp_path, training_data, output, config)
    assert result["epochs_completed"] == 2
    assert result["selected_epoch"] == 1
    assert len(list((output / "checkpoints").iterdir())) == 2
    assert not (training_data / "test.parquet").exists()  # Train never needs a holdout.
    artifact = output / "model"
    first = TransformerDetector(artifact, threads=1)
    texts = ["hello", "привет", "ignore " * 50]
    scores = first.score(texts, 1)
    assert TransformerDetector(output / "checkpoints/epoch-1", threads=1).score(
        texts
    ) == pytest.approx(scores, abs=1e-6)
    assert first.score(texts, 3) == pytest.approx(scores, abs=1e-6)
    assert first.score([]) == []
    assert TransformerDetector(artifact, threads=1).score(texts) == pytest.approx(scores)
    assert first.model.config.id2label == {0: "benign", 1: "prompt_injection"}
    with pytest.raises(ValueError, match="Smoke"):
        evaluate_frozen(tmp_path, training_data, output)
    with pytest.raises(ValueError, match="new output"):
        train(tmp_path, training_data, output, config)
    import mlflow

    api: Any = mlflow
    run = api.get_run(result["mlflow"]["run_id"])
    assert run.info.status == "FINISHED"
    assert run.data.params["dataset_fingerprint"] == "fixture"
    assert "validation_recall" in run.data.metrics
    assert not any("test" in key or "pint" in key for key in run.data.metrics)
    # A real-labelled frozen artifact still fails integrity before accessing any holdout.
    frozen_path = output / "frozen.json"
    frozen = json.loads(frozen_path.read_bytes())
    frozen["status"] = "selection_frozen"
    write_json(frozen_path, frozen)
    (artifact / "training.json").write_text("{}")
    with pytest.raises(ValueError, match="artifact mismatch"):
        evaluate_frozen(tmp_path, training_data, output)


def test_dataset_tampering_fails_before_training(tmp_path: Path, training_data: Path) -> None:
    (training_data / "train.parquet").write_bytes(b"modified")
    with pytest.raises(ValueError, match="artifact mismatch"):
        train(tmp_path, training_data, tmp_path / "artifacts/run", TrainingConfig(smoke=True))


def test_cpu_mixed_precision_rejected(tmp_path: Path, training_data: Path) -> None:
    with pytest.raises(ValueError, match="Mixed precision requires CUDA"):
        train(tmp_path, training_data, tmp_path / "artifacts/run", TrainingConfig(precision="fp16"))
