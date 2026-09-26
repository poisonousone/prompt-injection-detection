import csv
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from promptshield.dataset import normalize, prepare, split_records, verify_splits
from promptshield.schema import Sample, read_records, write_parquet


def sample(identifier: str = "a", **changes: Any) -> Sample:
    values: dict[str, Any] = {
        "id": identifier,
        "text": f"Text for {identifier}",
        "label": "benign",
        "injection_mode": None,
        "language": "en",
        "content_role": "other",
        "source": "test",
        "group_id": identifier,
        "metadata": {},
    }
    return Sample.model_validate(values | changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"id": 7},
        {"text": " \n"},
        {"label": "jailbreak"},
        {"language": "de"},
        {"content_role": "system"},
        {"group_id": ""},
        {"source": ""},
        {"metadata": []},
        {"unexpected": True},
        {"injection_mode": "direct"},
        {"label": "prompt_injection", "injection_mode": None},
        {"metadata": {"hard_negative": "true"}},
        {
            "label": "prompt_injection",
            "injection_mode": "unknown",
            "metadata": {"hard_negative": True},
        },
    ],
)
def test_schema_rejects_invalid(changes: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        sample(**changes)


def test_missing_field_rejected() -> None:
    values = sample().model_dump()
    del values["language"]
    with pytest.raises(ValidationError):
        Sample.model_validate(values)


def test_duplicate_csv_headers_fail(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    fields = list(Sample.model_fields) + ["text"]
    path.write_text(",".join(fields) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="CSV header"):
        read_records(path)


@pytest.mark.parametrize("extension", ["jsonl", "csv", "parquet"])
def test_ingestion_rejects_bad_label(tmp_path: Path, extension: str) -> None:
    import polars as pl

    path = tmp_path / f"bad.{extension}"
    values = sample().model_dump() | {"label": "jailbreak"}
    if extension == "jsonl":
        path.write_text(json.dumps(values) + "\n", encoding="utf-8")
    else:
        values["metadata"] = "{}"
        if extension == "parquet":
            pl.DataFrame([values]).write_parquet(path)
        else:
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(Sample.model_fields))
                writer.writeheader()
                writer.writerow(values)
    with pytest.raises(ValueError, match="Invalid record"):
        read_records(path)


@pytest.mark.parametrize("extension", ["jsonl", "csv", "parquet"])
def test_lossless_ingestion(tmp_path: Path, extension: str) -> None:
    original = sample(
        text='  Пароль, system\n"quoted"  ',
        language="mixed",
        metadata={"hard_negative": True, "nested": [1, None, "данные"]},
    )
    path = tmp_path / f"sample.{extension}"
    if extension == "jsonl":
        path.write_text(original.model_dump_json() + "\n", encoding="utf-8")
    elif extension == "parquet":
        write_parquet(path, [original])
    else:
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(Sample.model_fields))
            writer.writeheader()
            writer.writerow({**original.model_dump(), "metadata": json.dumps(original.metadata)})
    assert read_records(path) == [original]


def test_exact_and_normalized_duplicates_preserve_text_and_provenance() -> None:
    records = [
        sample("a", text="Hello  WORLD"),
        sample("b", text="Hello  WORLD"),
        sample("c", text=" hello world\n", metadata={"hard_negative": True}),
        sample("d", text="Independent text", group_id="c"),
    ]
    processed, audit = prepare(records)
    assert audit["exact_duplicates_removed"] == 1
    assert audit["normalized_only_duplicates_removed"] == 1
    assert len(processed) == 2
    assert processed[0].text == "Hello  WORLD"
    assert processed[0].metadata["hard_negative"] is True
    members = processed[0].metadata["duplicate_members"]
    assert isinstance(members, list) and len(members) == 3
    assert processed[0].group_id == processed[1].group_id
    assert normalize("ＡＢＣ\tD") == "abc d"


def test_conflicting_duplicate_labels_fail() -> None:
    with pytest.raises(ValueError, match="Conflicting duplicate labels"):
        prepare(
            [
                sample("a"),
                sample("b", text="Text for a", label="prompt_injection", injection_mode="direct"),
            ]
        )


def test_duplicate_ids_fail() -> None:
    with pytest.raises(ValueError, match="Duplicate sample IDs"):
        prepare([sample(), sample(text="other")])


def test_near_duplicates_are_retained_but_grouped() -> None:
    text = "The assistant should summarize the supplied document using the requested format."
    processed, audit = prepare([sample("a", text=text), sample("b", text=text + " Extra.")])
    assert len(processed) == 2
    assert audit["near_duplicate_warning_count"] == 1
    assert processed[0].group_id == processed[1].group_id


def test_deterministic_group_split_and_seed() -> None:
    records = [sample(str(i), group_id=f"family-{i // 2}") for i in range(60)]
    first = split_records(records, seed=17)
    assert first == split_records(list(reversed(records)), seed=17)
    assert first != split_records(records, seed=18)
    assert sorted(row.id for rows in first.values() for row in rows) == sorted(
        r.id for r in records
    )
    verify_splits(first)
    owners: dict[str, str] = {}
    for name, rows in first.items():
        assert rows
        for row in rows:
            assert owners.setdefault(row.group_id, name) == name


@pytest.mark.parametrize("changes", [{"group_id": "a"}, {"text": " TEXT FOR A "}])
def test_leakage_verifier_rejects_cross_split(changes: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="Leakage"):
        verify_splits({"train": [sample("a")], "test": [sample("b", **changes)]})


def test_split_requires_groups_and_valid_ratios() -> None:
    with pytest.raises(ValueError, match="three protected groups"):
        split_records([sample()], 1)
    with pytest.raises(ValueError, match="ratios"):
        split_records([sample()], 1, (0.7, 0.2, 0.2))
