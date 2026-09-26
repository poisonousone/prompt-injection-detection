from pathlib import Path

import pytest

from promptshield.dataset import digest
from promptshield.evaluation import Case, error_analysis, evaluate, metrics, select_threshold
from promptshield.external import overlap_audit, read_pint
from promptshield.predictors import TfidfDetector, malicious_index


def test_known_metric_values_and_ties() -> None:
    perfect = metrics([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8], 0.5)
    assert perfect["auprc"] == perfect["f1"] == perfect["recall_at_1pct_fpr"] == 1
    assert perfect["fpr_at_95pct_recall"] == 0
    assert perfect["brier_score"] == pytest.approx(0.025)
    tied = metrics([0, 1], [0.5, 0.5], 0.5)
    assert tied["auprc"] == 0.5
    assert tied["recall_at_1pct_fpr"] == 0
    assert tied["fpr_at_95pct_recall"] == 1
    assert tied["warnings"]


def test_threshold_max_f1_and_highest_tie() -> None:
    selection = select_threshold([0, 0, 1, 1], [0.1, 0.2, 0.7, 0.9])
    assert selection["threshold"] == 0.7
    assert selection["selection_split"] == "validation"
    with pytest.raises(ValueError):
        select_threshold([0, 0], [0.1, 0.2])


@pytest.mark.parametrize("scores", [[float("nan")], [1.1], [-0.1], []])
def test_invalid_scores_fail(scores: list[float]) -> None:
    with pytest.raises(ValueError):
        metrics([0], scores, 0.5)


def test_single_class_and_error_categories() -> None:
    cases = [
        Case("a", "benign text", 0, "ru", True, "hard_negatives", "fixture"),
        Case("b", "attack text", 1, "en", False, "jailbreak", "fixture"),
    ]
    result = evaluate(cases, [0.95, 0.05], 0.5)
    assert result["hard_negatives"]["auprc"] is None
    assert result["hard_negatives"]["fpr"] == 1
    assert result["injection_scope_without_jailbreak"]["n"] == 1
    errors = error_analysis(cases, [0.95, 0.05], 0.5)
    assert len(errors["high_confidence_false_positives"]) == 1
    assert len(errors["high_confidence_false_negatives"]) == 1
    assert len(errors["russian_errors"]) == len(errors["hard_negative_errors"]) == 1


def test_tfidf_fits_only_train_batches_and_roundtrips(tmp_path: Path) -> None:
    model = TfidfDetector.fit(
        ["apple fruit", "orange juice", "ignore rules", "override task"], [0, 0, 1, 1], 42, "test"
    )
    vocabulary = dict(model.pipeline.named_steps["tfidf"].vocabulary_)
    texts = ["zzzzzz unseen validation", "ignore task"]
    scores = model.score(texts, 1)
    assert scores == pytest.approx(model.score(texts, 2))
    assert model.pipeline.named_steps["tfidf"].vocabulary_ == vocabulary
    assert "zzz" not in vocabulary
    model.threshold = scores[0]
    assert model.predict(texts)[0].label == "prompt_injection"
    path = tmp_path / "model.joblib"
    model.save(path)
    loaded = TfidfDetector.load(path)
    assert loaded.predict(texts) == model.predict(texts)
    assert loaded.identity == model.identity
    assert model.predict([]) == []
    with pytest.raises(ValueError):
        model.predict([" "], 1)
    with pytest.raises(ValueError):
        model.predict(texts, 0)


def test_deberta_mapping_is_explicit_and_reversible() -> None:
    assert malicious_index({0: "SAFE", 1: "INJECTION"}) == 1
    assert malicious_index({1: "SAFE", 0: "INJECTION"}) == 0
    with pytest.raises(ValueError):
        malicious_index({0: "LABEL_0", 1: "LABEL_1"})


def test_pint_preserves_every_category_and_label(tmp_path: Path) -> None:
    path = tmp_path / "pint.yaml"
    path.write_text(
        """
- {text: injection example, category: prompt_injection, label: true}
- {text: jailbreak example, category: jailbreak, label: true}
- {text: quoted example, category: hard_negatives, label: false}
- {text: a chat message, category: chat, label: false}
- {text: a document, category: documents, label: false}
- {text: legitimate instruction, category: prompt_injection, label: false}
- {text: short example, category: short_input, label: true}
- {text: benign example, category: benign_input, label: false}
- {text: long example, category: long_input, label: true}
""",
        encoding="utf-8",
    )
    cases = read_pint(path, digest(path.read_bytes()))
    assert len(cases) == 9 and sum(row.target for row in cases) == 4
    assert cases[5].target == 0
    assert all(row.language == "unknown" for row in cases)
    assert cases[2].hard_negative
    audit = overlap_audit({"train": [cases[0]]}, cases)
    assert audit["match_count"] == 1
    with pytest.raises(ValueError, match="checksum"):
        read_pint(path, "incorrect")
    path.write_text("- {text: unexpected, category: unknown, label: true}", encoding="utf-8")
    with pytest.raises(ValueError):
        read_pint(path, digest(path.read_bytes()))


def test_pint_boolean_strings_are_not_silently_coerced(tmp_path: Path) -> None:
    path = tmp_path / "pint.yaml"
    path.write_text('- {text: a sample, category: chat, label: "false"}', encoding="utf-8")
    with pytest.raises(ValueError):
        read_pint(path, digest(path.read_bytes()))
