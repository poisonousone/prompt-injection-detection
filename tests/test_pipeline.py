import json
from pathlib import Path

import polars as pl
import pytest

from promptshield.dataset import canonical_json, digest, statistics, verify_splits
from promptshield.pipeline import Config, Input, build, fetch, ingest
from promptshield.schema import Sample, read_records


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "data" / "raw").mkdir(parents=True)
    rows = [
        Sample(
            id=str(i),
            text=f"Distinct sample {i}",
            label="benign",
            injection_mode=None,
            language="en",
            content_role="other",
            source="fixture",
            group_id=str(i),
            metadata={},
        )
        for i in range(12)
    ]
    source = tmp_path / "data/raw/input.jsonl"
    source.write_text("".join(row.model_dump_json() + "\n" for row in rows), encoding="utf-8")
    config = {
        "version": "test-v1",
        "seed": 42,
        "ratios": [0.7, 0.15, 0.15],
        "near_threshold": 1.0,
        "limitations": [],
        "inputs": [
            {
                "path": "data/raw/input.jsonl",
                "sha256": digest(source.read_bytes()),
                "adapter": "canonical",
            }
        ],
    }
    (tmp_path / "data/dataset.json").write_bytes(canonical_json(config))
    (tmp_path / "uv.lock").write_text("fixture lock", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("fixture project", encoding="utf-8")
    return tmp_path


def test_reproducible_build_artifacts_and_raw_immutability(workspace: Path) -> None:
    source = workspace / "data/raw/input.jsonl"
    original = source.read_bytes()
    config = workspace / "data/dataset.json"
    first = workspace / "data/processed/first"
    second = workspace / "data/processed/second"
    card = workspace / "DATA_CARD.md"
    result = build(workspace, config, first, card)
    assert result == build(workspace, config, second, card)
    assert result == build(workspace, config, first, card)
    assert {p.name: p.read_bytes() for p in first.iterdir()} == {
        p.name: p.read_bytes() for p in second.iterdir()
    }
    assert source.read_bytes() == original
    splits = {
        name: read_records(first / f"{name}.parquet") for name in ("train", "validation", "test")
    }
    verify_splits(splits)
    report = json.loads((first / "report.json").read_bytes())
    assert report["overall"] == statistics(read_records(first / "all.parquet"))
    assert sum(len(rows) for rows in splits.values()) == 12
    assert str(result["fingerprint"]) in card.read_text(encoding="utf-8")
    manifest = json.loads((first / "manifest.json").read_bytes())
    for name, sha in manifest["artifacts"].items():
        assert digest((first / name).read_bytes()) == sha


def test_raw_checksum_change_fails(workspace: Path) -> None:
    source = workspace / "data/raw/input.jsonl"
    source.write_text("changed", encoding="utf-8")
    config_path = workspace / "data/dataset.json"
    with pytest.raises(ValueError, match="checksum"):
        build(workspace, config_path, workspace / "data/processed/test", workspace / "DATA_CARD.md")
    with pytest.raises(ValueError, match="checksum"):
        fetch(workspace, Config.model_validate_json(config_path.read_bytes()))
    assert source.read_text() == "changed"


def test_build_cannot_write_into_raw(workspace: Path) -> None:
    with pytest.raises(ValueError, match="version directory"):
        build(
            workspace,
            workspace / "data/dataset.json",
            workspace / "data/raw/output",
            workspace / "DATA_CARD.md",
        )
    with pytest.raises(ValueError, match="outside raw"):
        build(
            workspace,
            workspace / "data/dataset.json",
            workspace / "data/processed/output",
            workspace / "data/raw/card.md",
        )


def test_changed_existing_output_is_not_overwritten(workspace: Path) -> None:
    output = workspace / "data/processed/test"
    config = workspace / "data/dataset.json"
    card = workspace / "DATA_CARD.md"
    build(workspace, config, output, card)
    (output / "report.json").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="Existing build differs"):
        build(workspace, config, output, card)
    assert (output / "report.json").read_text() == "changed"


def test_reviewed_public_adapter_verifies_annotation(workspace: Path) -> None:
    raw = workspace / "data/raw/upstream.parquet"
    pl.DataFrame({"text": ["A benign question"], "label": [0]}).write_parquet(raw)
    annotation = Sample(
        id="review-0",
        text="A benign question",
        label="benign",
        injection_mode=None,
        language="en",
        content_role="unknown",
        source="public-fixture",
        group_id="review-0",
        metadata={"upstream_row": 0},
    )
    path = workspace / "data/review.jsonl"
    path.write_text(annotation.model_dump_json() + "\n", encoding="utf-8")
    item = Input(
        path="data/raw/upstream.parquet",
        sha256=digest(raw.read_bytes()),
        adapter="deepset_reviewed",
        annotations="data/review.jsonl",
    )
    rows, details = ingest(workspace, item)
    assert rows[0].text == annotation.text
    assert details["selected_rows"] == 1
    assert "provenance" in rows[0].metadata
    changed = annotation.model_dump() | {"text": "Changed source text"}
    path.write_text(json.dumps(changed) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="annotation mismatch"):
        ingest(workspace, item)
