"""Pinned source acquisition and reproducible dataset builds."""

import argparse
import json
import platform
import shutil
import subprocess
import urllib.request
import uuid
from pathlib import Path
from typing import Literal

import polars as pl
from pydantic import BaseModel, ConfigDict, JsonValue

from promptshield.dataset import canonical_json, digest, prepare, split_records, statistics
from promptshield.schema import Sample, read_records, write_parquet


class Input(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    path: str
    sha256: str
    adapter: Literal["canonical", "deepset_reviewed"]
    url: str | None = None
    annotations: str | None = None


class Config(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    version: str
    seed: int
    ratios: list[float]
    near_threshold: float
    inputs: list[Input]
    limitations: list[str]


def checked_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Input paths must stay inside the repository")
    return path


def read_config(path: Path) -> Config:
    return Config.model_validate_json(path.read_bytes())


def fetch(root: Path, config: Config) -> None:
    """Never overwrite raw files; verify bytes before publishing a download."""
    for item in config.inputs:
        path = checked_path(root, item.path)
        if path.exists():
            if digest(path.read_bytes()) != item.sha256:
                raise ValueError(f"Source checksum mismatch: {item.path}")
            continue
        if item.url is None or not item.url.startswith("https://huggingface.co/"):
            raise ValueError(f"Missing local source: {item.path}")
        with urllib.request.urlopen(item.url, timeout=60) as response:
            data = response.read()
        if digest(data) != item.sha256:
            raise ValueError(f"Download checksum mismatch: {item.path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)


def ingest(root: Path, item: Input) -> tuple[list[Sample], dict[str, JsonValue]]:
    path = checked_path(root, item.path)
    if digest(path.read_bytes()) != item.sha256:
        raise ValueError(f"Source checksum mismatch: {item.path}")
    details: dict[str, JsonValue] = {"path": item.path, "sha256": item.sha256, "url": item.url}
    records: list[Sample]
    if item.adapter == "canonical":
        records = read_records(path)
        source_rows = list(range(len(records)))
    else:
        if item.annotations is None:
            raise ValueError("Reviewed adapter requires annotations")
        annotation_path = checked_path(root, item.annotations)
        annotations = read_records(annotation_path)
        raw = pl.read_parquet(path).to_dicts()
        records = []
        source_rows: list[int] = []
        for annotation in annotations:
            index = annotation.metadata.get("upstream_row")
            if type(index) is not int or not 0 <= index < len(raw):
                raise ValueError("Invalid upstream row index")
            row = raw[index]
            expected_label = 0 if annotation.label == "benign" else 1
            if (
                row.get("text") != annotation.text
                or type(row.get("label")) is not int
                or row["label"] != expected_label
            ):
                raise ValueError(f"Upstream annotation mismatch at row {index}")
            records.append(annotation)
            source_rows.append(index)
        if len(set(source_rows)) != len(source_rows):
            raise ValueError("Repeated upstream annotation")
        details.update(
            {
                "upstream_rows": len(raw),
                "selected_rows": len(records),
                "excluded_unreviewed_rows": len(raw) - len(records),
                "annotations": item.annotations,
                "annotations_sha256": digest(annotation_path.read_bytes()),
            }
        )
    enriched: list[Sample] = []
    for record, index in zip(records, source_rows, strict=True):
        if "provenance" in record.metadata:
            raise ValueError("provenance is reserved for ingestion")
        metadata = dict(record.metadata)
        metadata["provenance"] = {
            "path": item.path,
            "sha256": item.sha256,
            "row_zero_based": index,
            "url": item.url,
        }
        enriched.append(Sample.model_validate({**record.model_dump(), "metadata": metadata}))
    return enriched, details


def render_card(config: Config, fingerprint: str, report: dict[str, JsonValue]) -> str:
    audit = report["audit"]
    splits = report["splits"]
    assert isinstance(audit, dict) and isinstance(splits, dict)
    compact = {
        "overall": report["overall"],
        "split_sizes": report["split_sizes"],
        "exact_duplicates_removed": audit["exact_duplicates_removed"],
        "normalized_only_duplicates_removed": audit["normalized_only_duplicates_removed"],
        "near_duplicate_warning_count": audit["near_duplicate_warning_count"],
        "group_leakage_check": report["group_leakage_check"],
    }
    table = "\n\n| Split | Benign | Injection | English | Russian | Hard negatives |\n"
    table += "|---|---:|---:|---:|---:|---:|\n"
    for name in ("train", "validation", "test"):
        values = splits[name]
        assert isinstance(values, dict)
        labels, languages = values["label"], values["language"]
        assert isinstance(labels, dict) and isinstance(languages, dict)
        table += (
            f"| {name} | {labels.get('benign', 0)} | "
            f"{labels.get('prompt_injection', 0)} | {languages.get('en', 0)} | "
            f"{languages.get('ru', 0)} | {values['hard_negative_count']} |\n"
        )
    text = (
        "# PromptShield dataset card\n\n"
        f"Version: `{config.version}`. Fingerprint: `{fingerprint}`.\n\n"
        "Generated by `promptshield-data build`; all counts below come from processed records.\n\n"
        "## Scope and provenance\n\n"
        "Compact English/Russian starter corpus for benign vs prompt_injection. "
        "This is pipeline validation data, not evidence of detector quality. "
        "No model was trained.\n\n"
        "Public source: [deepset/prompt-injections]"
        "(https://huggingface.co/datasets/deepset/prompt-injections/tree/"
        "4f61ecb038e9c3fb77e21034b22511b523772cdd), upstream train only. "
        "The committed review file selects clear English examples, preserving text and labels. "
        "Unreviewed, ambiguous and jailbreak-style records are excluded rather than relabeled. "
        "Original content role and attack mode are unknown. See `data/SOURCES.md`.\n\n"
        "The authored supplement uses paired English/Russian examples, explicit contexts, "
        "attack families and benign topics. It is synthetic and has not received independent "
        "human annotation review. Hard negatives are marked in metadata.\n\n"
        "## Processing and splits\n\n"
        "Raw files are checksum pinned and never overwritten. Exact and NFKC/casefold/whitespace "
        "duplicate keys merge records while preserving representative text and member provenance. "
        "Conflicting labels fail the build. Translation/template/family groups are global. "
        "Character 5-gram Jaccard and containment warnings additionally join protected groups "
        "without deleting near duplicates. The audit is lexical, not a semantic guarantee.\n\n"
        f"Seed: {config.seed}; requested train/validation/test ratios: {config.ratios}; "
        f"near Jaccard threshold: {config.near_threshold}; containment threshold: 0.9 "
        "with at least 40 distinct shingles. Largest groups are allocated first, minimizing "
        "normalized squared deficits for sample, class, language and hard-negative counts; "
        "seeded hashes break ordering ties. Balance is approximate and ratios may differ. "
        "Group and normalized-text isolation is checked. "
        "Test data is reserved for final evaluation, never model/threshold/calibration selection. "
        "PINT and AgentDojo are not ingested.\n\n"
        "## Measured statistics\n\n```json\n"
        + json.dumps(compact, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n```"
        + table
        + "\nFull subgroup counts, duplicate members and suspicious pairs are in "
        "`report.json`; exact assignments and artifact hashes are in `manifest.json`.\n"
        + "\n## Limitations\n\n"
    )
    return text + "\n".join(f"- {issue}" for issue in config.limitations) + "\n"


def build(root: Path, config_path: Path, output: Path, card: Path) -> dict[str, JsonValue]:
    config = read_config(config_path)
    root, output, card = root.resolve(), output.resolve(), card.resolve()
    raw_root = root / "data" / "raw"
    processed_root = root / "data" / "processed"
    if not output.is_relative_to(processed_root) or output == processed_root:
        raise ValueError("Build output must be a version directory under data/processed")
    if card.is_relative_to(raw_root) or card.is_relative_to(output):
        raise ValueError("Data card must be outside raw data and the build directory")
    protected = [checked_path(root, item.path) for item in config.inputs]
    protected += [
        checked_path(root, item.annotations)
        for item in config.inputs
        if item.annotations is not None
    ]
    protected.append(config_path.resolve())
    if any(path == card or path.is_relative_to(output) for path in protected):
        raise ValueError("Output would overwrite an input")
    records: list[Sample] = []
    sources: list[JsonValue] = []
    for item in config.inputs:
        rows, source = ingest(root, item)
        records.extend(rows)
        sources.append(source)
    prepared, audit = prepare(records, config.near_threshold)
    if len(config.ratios) != 3:
        raise ValueError("Three split ratios required")
    ratios = (config.ratios[0], config.ratios[1], config.ratios[2])
    splits = split_records(prepared, config.seed, ratios)
    code_paths = sorted((root / "src" / "promptshield").glob("*.py"))
    code_paths += [root / "uv.lock", root / "pyproject.toml"]
    implementation = {
        path.relative_to(root).as_posix(): digest(path.read_bytes()) for path in code_paths
    }
    identity = {
        "config": config.model_dump(),
        "sources": sources,
        "implementation": implementation,
        "records": [row.model_dump() for row in prepared],
    }
    fingerprint = digest(canonical_json(identity))
    report: dict[str, JsonValue] = {
        "overall": statistics(prepared),
        "splits": {name: statistics(rows) for name, rows in splits.items()},
        "split_sizes": {name: len(rows) for name, rows in splits.items()},
        "audit": audit,
        "group_leakage_check": "passed",
    }
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    manifest = {
        "version": config.version,
        "fingerprint": fingerprint,
        "config": config.model_dump(),
        "sources": sources,
        "implementation": implementation,
        "python": platform.python_version(),
        "git_commit": result.stdout.strip() if result.returncode == 0 else None,
        "assignments": {name: [row.id for row in rows] for name, rows in splits.items()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # Inherit the workspace ACL: Windows restricted-token users can lose access
    # to directories created by tempfile with its owner-only (0700) ACL.
    stage = output.parent / f".build-{uuid.uuid4().hex}"
    stage.mkdir()
    try:
        write_parquet(stage / "all.parquet", prepared)
        for name, rows in splits.items():
            write_parquet(stage / f"{name}.parquet", rows)
        (stage / "report.json").write_bytes(canonical_json(report))
        manifest["artifacts"] = {
            path.name: digest(path.read_bytes()) for path in sorted(stage.iterdir())
        }
        (stage / "manifest.json").write_bytes(canonical_json(manifest))
        if output.exists():
            if {p.name: p.read_bytes() for p in stage.iterdir()} != {
                p.name: p.read_bytes() for p in output.iterdir()
            }:
                raise ValueError("Existing build differs; choose a new version directory")
        else:
            # Publish the complete directory atomically on the same filesystem.
            stage.rename(output)
    finally:
        if stage.exists():
            if stage.resolve().parent != output.parent:
                raise ValueError("Unexpected staging directory location")
            shutil.rmtree(stage)
    card.write_text(render_card(config, fingerprint, report), encoding="utf-8")
    return {
        "fingerprint": fingerprint,
        "sample_count": len(prepared),
        "split_sizes": report["split_sizes"],
        "group_leakage_check": "passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["fetch", "build"])
    parser.add_argument("--config", type=Path, default=Path("data/dataset.json"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/v1"))
    parser.add_argument("--card", type=Path, default=Path("DATA_CARD.md"))
    args = parser.parse_args()
    root = Path.cwd()
    if args.command == "fetch":
        fetch(root, read_config(args.config))
    else:
        print(json.dumps(build(root, args.config, args.output, args.card), sort_keys=True))


if __name__ == "__main__":
    main()
