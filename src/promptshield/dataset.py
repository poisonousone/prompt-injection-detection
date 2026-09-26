"""Deterministic deduplication, conservative grouping and split verification."""

import hashlib
import json
import unicodedata
from collections import Counter
from itertools import combinations

from pydantic import JsonValue

from promptshield.schema import Sample


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        + "\n"
    ).encode("utf-8")


def normalize(text: str) -> str:
    """Comparison key only; the stored text is never rewritten."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


class Groups:
    def __init__(self, ids: list[str]) -> None:
        self.parent = {key: key for key in ids}

    def root(self, key: str) -> str:
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def join(self, left: str, right: str) -> None:
        a, b = sorted((self.root(left), self.root(right)))
        self.parent[b] = a


def prepare(
    records: list[Sample],
    near_threshold: float = 0.8,
) -> tuple[list[Sample], dict[str, JsonValue]]:
    if not 0 < near_threshold <= 1:
        raise ValueError("near_threshold must be in (0, 1]")
    if not records or len(records) > 5000:
        raise ValueError(
            "Exhaustive near audit supports 1..5000 records; larger inputs need review"
        )
    ordered = sorted(records, key=lambda row: row.id)
    if len({row.id for row in ordered}) != len(ordered):
        raise ValueError("Duplicate sample IDs")
    groups = Groups([row.group_id for row in ordered])
    buckets: dict[str, list[Sample]] = {}
    for row in ordered:
        buckets.setdefault(normalize(row.text), []).append(row)
    exact_count = len(ordered) - len({row.text for row in ordered})
    total_duplicates = len(ordered) - len(buckets)
    for bucket in buckets.values():
        if len({row.label for row in bucket}) > 1:
            raise ValueError(f"Conflicting duplicate labels: {[row.id for row in bucket]}")
        for row in bucket:
            groups.join(bucket[0].group_id, row.group_id)

    representatives = [bucket[0] for bucket in buckets.values()]
    shingles = {
        row.id: {
            normalize(row.text)[i : i + 5] for i in range(max(1, len(normalize(row.text)) - 4))
        }
        for row in representatives
    }
    warnings: list[JsonValue] = []
    for left, right in combinations(representatives, 2):
        a, b = shingles[left.id], shingles[right.id]
        similarity = len(a & b) / len(a | b)
        # Containment catches appended/prefixed attack templates missed by Jaccard.
        containment = len(a & b) / min(len(a), len(b))
        if similarity >= near_threshold or (min(len(a), len(b)) >= 40 and containment >= 0.9):
            warnings.append(
                {
                    "left": left.id,
                    "right": right.id,
                    "jaccard": round(similarity, 6),
                    "containment": round(containment, 6),
                }
            )
            groups.join(left.group_id, right.group_id)

    result: list[Sample] = []
    duplicate_members: list[JsonValue] = []
    for bucket in buckets.values():
        first = bucket[0]
        metadata = dict(first.metadata)
        if len(bucket) > 1:
            members: list[JsonValue] = [
                {
                    "id": row.id,
                    "source": row.source,
                    "group_id": row.group_id,
                    "text_sha256": digest(row.text.encode()),
                    "metadata": row.metadata,
                }
                for row in bucket
            ]
            metadata["duplicate_members"] = members
            metadata["hard_negative"] = any(
                row.metadata.get("hard_negative") is True for row in bucket
            )
            duplicate_members.append({"kept": first.id, "members": [r.id for r in bucket]})
        metadata["original_group_id"] = first.group_id
        result.append(
            Sample.model_validate(
                {
                    **first.model_dump(),
                    "metadata": metadata,
                    "group_id": groups.root(first.group_id),
                }
            )
        )
    return result, {
        "input_count": len(ordered),
        "exact_duplicates_removed": exact_count,
        "normalized_only_duplicates_removed": total_duplicates - exact_count,
        "duplicate_groups": duplicate_members,
        "near_duplicate_warnings": warnings,
        "near_duplicate_warning_count": len(warnings),
    }


def split_records(
    records: list[Sample],
    seed: int,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> dict[str, list[Sample]]:
    if any(not 0 < ratio < 1 for ratio in ratios) or abs(sum(ratios) - 1) > 1e-9:
        raise ValueError("Split ratios must be positive and sum to one")
    grouped: dict[str, list[Sample]] = {}
    for row in sorted(records, key=lambda row: row.id):
        grouped.setdefault(row.group_id, []).append(row)
    if len(grouped) < 3:
        raise ValueError("At least three protected groups required")
    splits: dict[str, list[Sample]] = {"train": [], "validation": [], "test": []}
    # Largest components first; seeded hash breaks ties without interpreter RNG dependence.
    keys = sorted(grouped, key=lambda key: (-len(grouped[key]), digest(f"{seed}:{key}".encode())))

    def features(rows: list[Sample]) -> Counter[str]:
        counts: Counter[str] = Counter()
        for row in rows:
            counts.update(["samples", f"label:{row.label}", f"language:{row.language}"])
            if row.metadata.get("hard_negative") is True:
                counts["hard_negative"] += 1
        return counts

    totals = features(records)
    targets = {
        name: {feature: count * ratio for feature, count in totals.items()}
        for name, ratio in zip(splits, ratios, strict=True)
    }
    allocated = {name: Counter[str]() for name in splits}
    for index, key in enumerate(keys):
        empty = [name for name, rows in splits.items() if not rows]
        choices = empty if len(keys) - index == len(empty) else list(splits)
        contribution = features(grouped[key])

        def cost_change(name: str, contribution: Counter[str] = contribution) -> float:
            # Change in normalized squared deficit, balancing sample counts,
            # labels, languages and hard negatives without breaking any group.
            return sum(
                count
                * (2 * allocated[name][feature] + count - 2 * targets[name][feature])
                / (targets[name][feature] + 1)
                for feature, count in contribution.items()
            )

        destination = min(choices, key=cost_change)
        splits[destination].extend(grouped[key])
        allocated[destination].update(contribution)
    for rows in splits.values():
        rows.sort(key=lambda row: row.id)
    verify_splits(splits)
    return splits


def verify_splits(splits: dict[str, list[Sample]]) -> None:
    seen_groups: dict[str, str] = {}
    seen_texts: dict[str, str] = {}
    seen_ids: set[str] = set()
    for name, rows in splits.items():
        for row in rows:
            if row.id in seen_ids:
                raise ValueError(f"Duplicate split ID: {row.id}")
            seen_ids.add(row.id)
            for key, seen in ((row.group_id, seen_groups), (normalize(row.text), seen_texts)):
                if key in seen and seen[key] != name:
                    raise ValueError(f"Leakage across {seen[key]} and {name}: {row.id}")
                seen[key] = name


def statistics(records: list[Sample]) -> dict[str, JsonValue]:
    report: dict[str, JsonValue] = {
        "sample_count": len(records),
        "protected_groups": len({r.group_id for r in records}),
        "hard_negative_count": sum(r.metadata.get("hard_negative") is True for r in records),
    }
    for field in ("label", "language", "source", "content_role", "injection_mode"):
        report[field] = dict(
            sorted(
                Counter(
                    "null" if getattr(row, field) is None else str(getattr(row, field))
                    for row in records
                ).items()
            )
        )
    return report
