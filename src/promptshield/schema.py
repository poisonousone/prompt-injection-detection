"""Strict canonical records and lossless file transport."""

import csv
import json
from pathlib import Path
from typing import Literal

import polars as pl
from pydantic import BaseModel, ConfigDict, JsonValue, field_validator, model_validator


class Sample(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True, allow_inf_nan=False)

    id: str
    text: str
    label: Literal["benign", "prompt_injection"]
    injection_mode: Literal["direct", "indirect", "unknown"] | None
    language: Literal["en", "ru", "mixed", "other", "unknown"]
    content_role: Literal[
        "user_input",
        "retrieved_document",
        "tool_output",
        "email",
        "webpage",
        "code",
        "other",
        "unknown",
    ]
    source: str
    group_id: str
    metadata: dict[str, JsonValue]

    @field_validator("id", "text", "source", "group_id")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @model_validator(mode="after")
    def consistent_annotations(self) -> Sample:
        if (self.label == "benign") != (self.injection_mode is None):
            raise ValueError("benign requires null mode; injection requires a non-null mode")
        if "hard_negative" in self.metadata:
            value = self.metadata["hard_negative"]
            if not isinstance(value, bool):
                raise ValueError("hard_negative must be a boolean")
            if value and self.label != "benign":
                raise ValueError("hard_negative requires benign label")
        return self


def read_records(path: Path) -> list[Sample]:
    """CSV/Parquet encode metadata as JSON; CSV empty mode means null only."""
    records: list[Sample] = []
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            raw = [json.loads(line) for line in stream]
    elif path.suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames
            if (
                fields is None
                or len(fields) != len(Sample.model_fields)
                or set(fields) != set(Sample.model_fields)
            ):
                raise ValueError(f"Invalid CSV header: {path}")
            raw = list(reader)
        for row in raw:
            row["metadata"] = json.loads(row["metadata"])
            if row["injection_mode"] == "":
                row["injection_mode"] = None
    elif path.suffix == ".parquet":
        raw = pl.read_parquet(path).to_dicts()
        for row in raw:
            if isinstance(row.get("metadata"), str):
                row["metadata"] = json.loads(row["metadata"])
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")
    for index, row in enumerate(raw, 1):
        try:
            records.append(Sample.model_validate(row))
        except ValueError as error:
            # Do not include potentially sensitive source text in errors.
            raise ValueError(f"Invalid record at {path}:{index}") from error
    return records


def write_parquet(path: Path, records: list[Sample]) -> None:
    rows = [
        {
            **record.model_dump(),
            "metadata": json.dumps(record.metadata, ensure_ascii=False, sort_keys=True),
        }
        for record in records
    ]
    schema = dict.fromkeys(Sample.model_fields, pl.String)
    pl.DataFrame(rows, schema=schema).write_parquet(path)
