"""Calibration math, validation isolation, ties and frozen artifact enforcement."""

import json
import math
from pathlib import Path
from typing import Any

import pytest

from promptshield import calibration
from promptshield.baselines import write_json
from promptshield.calibration import (
    CalibrationConfig,
    calibrate,
    final_evaluation,
    fit,
    fit_temperature,
    reliability,
    select_profile,
)
from promptshield.dataset import digest
from promptshield.schema import Sample, write_parquet
from promptshield.training import file_hashes


def test_temperature_nll_ranking_and_boundaries() -> None:
    labels = [0, 1, 0, 1]
    scores = [0.01, 0.99, 0.8, 0.2]
    fitted = fit_temperature(labels, scores, CalibrationConfig())
    adjusted = calibrate(scores, fitted["temperature"])

    def nll(values: list[float]) -> float:
        return -sum(math.log(p if y else 1 - p) for y, p in zip(labels, values, strict=True))

    assert fitted["temperature"] > 1
    assert nll(adjusted) < nll(scores)
    assert sorted(range(4), key=adjusted.__getitem__) == sorted(range(4), key=scores.__getitem__)
    assert calibrate(scores, 1) == pytest.approx(scores)
    assert fit_temperature(labels, [0.5] * 4, CalibrationConfig())["temperature"] == 1
    bound = fit_temperature([0, 1], [0.1, 0.9], CalibrationConfig())
    assert bound["temperature"] == 0.05 and bound["at_bound"]
    assert all(math.isfinite(p) for p in calibrate([0, 1], 0.05))


def test_profiles_preserve_ties_and_recall() -> None:
    labels, scores = [0, 0, 1, 1, 1, 1], [0.2, 0.8, 0.3, 0.8, 0.9, 0.95]
    default = select_profile(labels, scores, 0.5)
    strict = select_profile(labels, scores, 0.95)
    assert default["threshold"] == 0.9
    assert default["validation_metrics"]["fpr"] == 0
    assert strict["threshold"] == 0.3
    assert strict["validation_metrics"]["recall"] == 1
    assert select_profile([0, 1], [0.5, 0.5], 1)["validation_metrics"]["fpr"] == 1


def test_reliability_empty_bins_and_endpoints() -> None:
    result = reliability([0, 1, 1], [0, 0.5, 1], 5)
    assert [b["n"] for b in result["bins"]] == [1, 0, 1, 0, 1]
    assert result["bins"][1]["positive_fraction"] is None
    assert result["brier_score"] == pytest.approx(0.25 / 3)
    assert result["ece"] == pytest.approx(0.5 / 3)


@pytest.mark.parametrize("scores", [[float("nan"), 0.2], [-0.1, 0.2], []])
def test_invalid_scores(scores: list[float]) -> None:
    with pytest.raises(ValueError):
        calibrate(scores, 1)


def test_invalid_config_and_single_class() -> None:
    with pytest.raises(ValueError):
        CalibrationConfig(default_min_recall=0.99)
    with pytest.raises(ValueError):
        CalibrationConfig(temperature_min=2)
    with pytest.raises(ValueError, match="both"):
        fit_temperature([1, 1], [0.3, 0.5], CalibrationConfig())
    with pytest.raises(ValueError):
        calibrate([0.5], 0)


def test_validation_only_fit_and_frozen_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    manifest: dict[str, Any] = {"fingerprint": "fixture", "artifacts": {}, "assignments": {}}
    pending: dict[str, bytes] = {}
    for split in ("train", "validation", "test"):
        rows = [
            Sample(
                id=f"{split}-{i}",
                text=f"{split} example {i}",
                label="prompt_injection" if i else "benign",
                injection_mode="direct" if i else None,
                language="en",
                content_role="user_input",
                source="fixture",
                group_id=f"{split}-{i}",
                metadata={},
            )
            for i in range(2)
        ]
        path = dataset / f"{split}.parquet"
        write_parquet(path, rows)
        manifest["artifacts"][path.name] = digest(path.read_bytes())
        manifest["assignments"][split] = [r.id for r in rows]
        if split != "validation":
            pending[split] = path.read_bytes()
            path.unlink()  # Calibration succeeds with no training or test file available.
    write_json(dataset / "manifest.json", manifest)
    model_run = tmp_path / "model_run"
    (model_run / "model").mkdir(parents=True)
    (model_run / "model/fixture").write_text("immutable weights")
    write_json(
        model_run / "frozen.json",
        {
            "status": "selection_frozen",
            "model_files": file_hashes(model_run / "model"),
            "metadata": {
                "config": {"smoke": False},
                "version": "fixture",
                "dataset_manifest_sha256": digest((dataset / "manifest.json").read_bytes()),
            },
        },
    )
    (tmp_path / "uv.lock").write_text("fixture")

    class FakeDetector:
        def __init__(self, _: Path) -> None:
            pass

        def score(self, texts: list[str]) -> list[float]:
            return [0.2 if t.endswith("0") else 0.8 for t in texts]

    monkeypatch.setattr(calibration, "TransformerDetector", FakeDetector)
    output = tmp_path / "artifacts/calibration"
    fit(tmp_path, model_run, dataset, output, CalibrationConfig())
    with pytest.raises(ValueError, match="new output"):
        fit(tmp_path, model_run, dataset, output, CalibrationConfig())
    original = (output / "profiles.json").read_bytes()
    (output / "profiles.json").write_text("{}")
    with pytest.raises(ValueError, match="calibration artifact mismatch"):
        final_evaluation(model_run, dataset, output)
    (output / "profiles.json").write_bytes(original)
    for split, data in pending.items():
        (dataset / f"{split}.parquet").write_bytes(data)
    result = final_evaluation(model_run, dataset, output)
    assert result["profiles"]["default"]["overall"]["n"] == 2
    assert result["profiles"]["high_security"]["overall"]["recall"] == 1
    assert json.loads((output / "frozen.json").read_bytes())["classifier_retrained"] is False
    with pytest.raises(ValueError, match="already exists"):
        final_evaluation(model_run, dataset, output)
