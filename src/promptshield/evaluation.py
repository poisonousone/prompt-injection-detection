"""Validation-only operating points and descriptive detector metrics."""

# pyright: reportMissingTypeStubs=false
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn import metrics as sklearn_metrics

from promptshield.schema import Sample

metric_api: Any = sklearn_metrics  # Public sklearn metrics are incompletely typed.


@dataclass(frozen=True)
class Case:
    id: str
    text: str
    target: int
    language: str
    hard_negative: bool
    category: str
    source: str

    @classmethod
    def from_sample(cls, row: Sample) -> Case:
        return cls(
            row.id,
            row.text,
            int(row.label == "prompt_injection"),
            row.language,
            row.metadata.get("hard_negative") is True,
            "internal",
            row.source,
        )


def check_scores(labels: Sequence[int], scores: Sequence[float]) -> None:
    if len(labels) != len(scores) or not labels or not set(labels) <= {0, 1}:
        raise ValueError("Nonempty aligned binary labels required")
    if any(not math.isfinite(score) or not 0 <= score <= 1 for score in scores):
        raise ValueError("Scores must be finite probabilities")


def select_threshold(labels: Sequence[int], scores: Sequence[float]) -> dict[str, Any]:
    """Caller supplies validation only; maximize F1, prefer higher threshold on ties."""
    check_scores(labels, scores)
    if set(labels) != {0, 1}:
        raise ValueError("Threshold selection requires both validation classes")
    candidates = sorted({0.0, 0.5, 1.0, *scores})
    best = max(
        candidates, key=lambda t: (float(metric_api.f1_score(labels, [s >= t for s in scores])), t)
    )
    return {
        "threshold": best,
        "selection_split": "validation",
        "objective": "max_f1",
        "tie_break": "highest_threshold",
        "sample_count": len(labels),
        "validation_f1": float(metric_api.f1_score(labels, [s >= best for s in scores])),
    }


def metrics(labels: Sequence[int], scores: Sequence[float], threshold: float) -> dict[str, Any]:
    check_scores(labels, scores)
    predictions = [int(score >= threshold) for score in scores]
    positive = sum(labels)
    negative = len(labels) - positive
    tp = sum(y == p == 1 for y, p in zip(labels, predictions, strict=True))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions, strict=True))
    result: dict[str, Any] = {
        "n": len(labels),
        "positive": positive,
        "negative": negative,
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": positive - tp,
        "tn": negative - fp,
        "auprc": None,
        "recall_at_1pct_fpr": None,
        "fpr_at_95pct_recall": None,
        "f1": None,
        "brier_score": float(metric_api.brier_score_loss(labels, scores)),
        "fpr": fp / negative if negative else None,
        "recall": tp / positive if positive else None,
        "warnings": [],
        "roc_summary_scope": "descriptive_only_not_deployment_thresholds",
    }
    if positive and negative:
        fpr, tpr, _ = metric_api.roc_curve(labels, scores, drop_intermediate=False)
        result.update(
            auprc=float(metric_api.average_precision_score(labels, scores)),
            f1=float(metric_api.f1_score(labels, predictions)),
            recall_at_1pct_fpr=float(np.max(tpr[fpr <= 0.01])),
            fpr_at_95pct_recall=float(np.min(fpr[tpr >= 0.95])),
        )
    else:
        result["warnings"].append(
            "Single-class subgroup: ranking metrics and F1 are not informative"
        )
    if negative < 100:
        result["warnings"].append("Fewer than 100 negatives: cannot resolve 1% FPR reliably")
    if positive < 20:
        result["warnings"].append("Fewer than 20 positives: 95% recall is too coarsely resolved")
    if len(labels) < 30:
        result["warnings"].append("Very small subgroup: descriptive results only")
    return result


def evaluate(cases: list[Case], scores: list[float], threshold: float) -> dict[str, Any]:
    check_scores([row.target for row in cases], scores)
    groups = {
        "overall": list(range(len(cases))),
        "english": [i for i, row in enumerate(cases) if row.language == "en"],
        "russian": [i for i, row in enumerate(cases) if row.language == "ru"],
        "hard_negatives": [i for i, row in enumerate(cases) if row.hard_negative],
    }
    for category in sorted({row.category for row in cases} - {"internal"}):
        groups[f"category:{category}"] = [
            i for i, row in enumerate(cases) if row.category == category
        ]
    if any(row.category == "jailbreak" for row in cases):
        groups["injection_scope_without_jailbreak"] = [
            i for i, row in enumerate(cases) if row.category != "jailbreak"
        ]
    return {
        name: metrics([cases[i].target for i in indices], [scores[i] for i in indices], threshold)
        if indices
        else {"n": 0, "status": "no_samples"}
        for name, indices in groups.items()
    }


def error_analysis(cases: list[Case], scores: list[float], threshold: float) -> dict[str, Any]:
    errors: list[dict[str, Any]] = [
        {
            "id": row.id,
            "text": row.text,
            "target": row.target,
            "risk_score": score,
            "language": row.language,
            "category": row.category,
            "source": row.source,
            "hard_negative": row.hard_negative,
            "error": "false_positive" if row.target == 0 else "false_negative",
        }
        for row, score in zip(cases, scores, strict=True)
        if int(score >= threshold) != row.target
    ]
    return {
        "high_confidence_definition": "FP score >=0.9; FN score <=0.1; not calibrated confidence",
        "high_confidence_false_positives": [
            e for e in errors if e["error"] == "false_positive" and e["risk_score"] >= 0.9
        ],
        "high_confidence_false_negatives": [
            e for e in errors if e["error"] == "false_negative" and e["risk_score"] <= 0.1
        ],
        "russian_errors": [e for e in errors if e["language"] == "ru"],
        "hard_negative_errors": [e for e in errors if e["hard_negative"]],
        "all_errors": errors,
    }
