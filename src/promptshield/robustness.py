"""Bounded, reproducible robustness transformations and frozen-model evaluation."""

# pyright: reportMissingTypeStubs=false
import argparse
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from promptshield.baselines import write_json
from promptshield.dataset import canonical_json, digest
from promptshield.evaluation import Case, evaluate
from promptshield.predictors import DebertaDetector, Detector, TfidfDetector
from promptshield.schema import Sample, read_records
from promptshield.training import TransformerDetector, file_hashes

BENCHMARK_VERSION = "robustness-v1"
TRANSFORMATIONS = (
    "common_typos",
    "whitespace_fragmentation",
    "unicode_confusables",
    "quotation_code_block",
    "benign_prefix_context",
)
WORD = re.compile(r"(?u)\b[^\W\d_]{5,}\b")
CONFUSABLES = {
    "a": "а",
    "c": "с",
    "e": "е",
    "o": "о",
    "p": "р",
    "x": "х",
    "y": "у",
    "а": "a",
    "с": "c",
    "е": "e",
    "о": "o",
    "р": "p",
    "х": "x",
    "у": "y",
    "A": "А",
    "C": "С",
    "E": "Е",
    "O": "О",
    "P": "Р",
    "X": "Х",
    "А": "A",
    "С": "C",
    "Е": "E",
    "О": "O",
    "Р": "P",
    "Х": "X",
}


@dataclass(frozen=True)
class RobustnessExample:
    id: str
    parent_id: str
    text: str
    target: int
    language: str
    hard_negative: bool
    source: str
    transformation: str
    parameters: dict[str, Any]
    seed: int | None
    semantic_validity: Literal["valid", "ambiguous"]
    exclusion_reason: str | None = None


def _words(text: str) -> list[re.Match[str]]:
    return list(WORD.finditer(text))


def transform(sample: Sample, family: str, seed: int) -> RobustnessExample:
    """Apply one bounded edit; ambiguous/inapplicable rows are retained but excluded."""
    rng = random.Random(f"{seed}:{sample.id}:{family}")
    text, params, valid, reason = sample.text, {}, True, None
    words = _words(text)
    if family == "common_typos":
        eligible = [m for m in words if len(m.group()) >= 5]
        if eligible:
            match = rng.choice(eligible)
            word = match.group()
            index = rng.randrange(1, len(word) - 2)
            changed = word[:index] + word[index + 1] + word[index] + word[index + 2 :]
            text = text[: match.start()] + changed + text[match.end() :]
            params = {
                "operation": "transpose_adjacent",
                "word_index": words.index(match),
                "character_index": index,
            }
        else:
            valid, reason = False, "no_alphabetic_word_with_five_characters"
    elif family == "whitespace_fragmentation":
        eligible = [m for m in words if len(m.group()) >= 6]
        if eligible:
            match = rng.choice(eligible)
            word = match.group()
            index = len(word) // 2
            text = text[: match.start()] + word[:index] + " " + word[index:] + text[match.end() :]
            params = {
                "operation": "single_internal_space",
                "word_index": words.index(match),
                "character_index": index,
            }
        else:
            valid, reason = False, "no_alphabetic_word_with_six_characters"
    elif family == "unicode_confusables":
        positions = [i for i, character in enumerate(text) if character in CONFUSABLES]
        if positions:
            index = rng.choice(positions)
            original = text[index]
            text = text[:index] + CONFUSABLES[original] + text[index + 1 :]
            params = {
                "operation": "single_visual_confusable",
                "character_index": index,
                "from": original,
                "to": CONFUSABLES[original],
            }
        else:
            valid, reason = False, "no_allowlisted_visual_confusable"
    elif family == "quotation_code_block":
        # A direct user command can become a mere quotation; do not silently reuse its label.
        if sample.content_role == "user_input":
            valid, reason = False, "wrapping_direct_user_input_may_change_communicative_intent"
        else:
            text = f"```text\n{text}\n```"
            params = {"wrapper": "fenced_text_code_block"}
    elif family == "benign_prefix_context":
        prefix = (
            "Контекст: ниже приведено содержимое из ненадёжного источника.\n\n"
            if sample.language == "ru"
            else "Context: the following content came from an untrusted source.\n\n"
        )
        text = prefix + text
        params = {
            "prefix_language": "ru" if sample.language == "ru" else "en",
            "prefix_kind": "untrusted_source_context",
        }
    else:
        raise ValueError(f"Unknown transformation: {family}")
    return RobustnessExample(
        id=f"{sample.id}::{family}",
        parent_id=sample.id,
        text=text,
        target=int(sample.label == "prompt_injection"),
        language=sample.language,
        hard_negative=sample.metadata.get("hard_negative") is True,
        source=sample.source,
        transformation=family,
        parameters=params,
        seed=seed if family in TRANSFORMATIONS[:3] else None,
        semantic_validity="valid" if valid else "ambiguous",
        exclusion_reason=reason,
    )


def build_examples(rows: list[Sample], seed: int) -> list[RobustnessExample]:
    return [transform(row, family, seed) for family in TRANSFORMATIONS for row in rows]


def _cases(rows: list[Sample]) -> list[Case]:
    return [Case.from_sample(row) for row in rows]


def _robust_cases(rows: list[RobustnessExample]) -> list[Case]:
    return [
        Case(
            row.id,
            row.text,
            row.target,
            row.language,
            row.hard_negative,
            row.transformation,
            row.source,
        )
        for row in rows
    ]


def _metric_view(values: dict[str, Any]) -> dict[str, Any]:
    if values["n"] == 0:
        return {"n": 0, "status": "no_samples"}
    return {
        key: values[key]
        for key in ("n", "positive", "negative", "auprc", "recall_at_1pct_fpr", "fpr")
    }


def compare(
    model: Detector, clean: list[Sample], transformed: list[RobustnessExample]
) -> dict[str, Any]:
    valid = [row for row in transformed if row.semantic_validity == "valid"]
    by_id = {row.id: row for row in clean}
    families: dict[str, Any] = {}
    for family in TRANSFORMATIONS:
        changed = [row for row in valid if row.transformation == family]
        parents = [by_id[row.parent_id] for row in changed]
        if not changed:
            families[family] = {
                "included": 0,
                "excluded_ambiguous": len(clean),
                "status": "no_valid_samples",
                "groups": {},
            }
            continue
        clean_metrics = evaluate(
            _cases(parents), model.score([row.text for row in parents]), model.threshold
        )
        changed_metrics = evaluate(
            _robust_cases(changed), model.score([row.text for row in changed]), model.threshold
        )
        groups: dict[str, Any] = {}
        for group in ("overall", "english", "russian"):
            before, after = clean_metrics[group], changed_metrics[group]
            groups[group] = {
                "clean": _metric_view(before),
                "transformed": _metric_view(after),
                "delta_recall_at_1pct_fpr": (
                    after["recall_at_1pct_fpr"] - before["recall_at_1pct_fpr"]
                    if before.get("recall_at_1pct_fpr") is not None
                    and after.get("recall_at_1pct_fpr") is not None
                    else None
                ),
            }
        families[family] = {
            "included": len(changed),
            "excluded_ambiguous": len(clean) - len(changed),
            "groups": groups,
        }
    return {"model": model.identity, "threshold": model.threshold, "families": families}


def run(
    root: Path, dataset: Path, output: Path, seed: int, deberta_snapshot: Path
) -> dict[str, Any]:
    root, dataset, output = root.resolve(), dataset.resolve(), output.resolve()
    if output.exists() or not output.is_relative_to(root / "artifacts"):
        raise ValueError("Choose a new output directory under artifacts/")
    manifest = json.loads((dataset / "manifest.json").read_bytes())
    test_path = dataset / "test.parquet"
    if digest(test_path.read_bytes()) != manifest["artifacts"]["test.parquet"]:
        raise ValueError("Frozen test artifact mismatch")
    rows = read_records(test_path)
    examples = build_examples(rows, seed)
    output.mkdir(parents=True)
    with (output / "examples.jsonl").open("wb") as stream:
        for example in examples:
            stream.write(canonical_json(asdict(example)))
    tfidf = TfidfDetector.load(root / "artifacts/baselines/task2-complete/tfidf/model.joblib")
    deberta = DebertaDetector(deberta_snapshot)
    custom_path = root / "artifacts/transformer/task3-real/model"
    custom = TransformerDetector(custom_path)
    models = {"char_tfidf": tfidf, "protectai_deberta_v2": deberta, "custom_transformer": custom}
    results = {name: compare(model, rows, examples) for name, model in models.items()}
    summary = {
        "benchmark_version": BENCHMARK_VERSION,
        "seed": seed,
        "dataset_fingerprint": manifest["fingerprint"],
        "test_sha256": manifest["artifacts"]["test.parquet"],
        "validity_rule": (
            "Reuse labels only for bounded surface edits that preserve readable intent. "
            "Exclude transformations lacking an eligible edit and code-block wrapping of "
            "user_input because quoting may change communicative intent."
        ),
        "transformation_families": list(TRANSFORMATIONS),
        "example_counts": {
            "generated": len(examples),
            "valid": sum(e.semantic_validity == "valid" for e in examples),
            "ambiguous_excluded": sum(e.semantic_validity == "ambiguous" for e in examples),
        },
        "model_artifacts": {
            "tfidf": "artifacts/baselines/task2-complete/tfidf/model.joblib",
            "deberta_snapshot": str(deberta_snapshot),
            "custom_files": file_hashes(custom_path),
        },
        "results": results,
        "retraining": False,
    }
    write_json(output / "results.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/v1"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--deberta-snapshot", type=Path, required=True)
    args = parser.parse_args()
    result = run(Path.cwd(), args.dataset, args.output, args.seed, args.deberta_snapshot)
    print(json.dumps({"output": str(args.output), "counts": result["example_counts"]}))


if __name__ == "__main__":
    main()
