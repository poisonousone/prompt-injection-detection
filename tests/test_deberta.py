"""Offline mechanics tests; these are not real DeBERTa model evaluations."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

from promptshield.predictors import DebertaDetector, Detector


class WindowModel:
    def __init__(self) -> None:
        self.lengths: list[int] = []

    def __call__(self, input_ids: torch.Tensor, **kwargs: Any) -> SimpleNamespace:
        self.lengths.append(input_ids.shape[1])
        malicious = (input_ids == 4).any(dim=1).float() * 10
        return SimpleNamespace(logits=torch.stack([torch.zeros_like(malicious), malicious], dim=1))


def test_overflow_tail_coverage_batch_order_and_score_direction() -> None:
    model_api: Any = models
    pre_tokenizer_api: Any = pre_tokenizers
    processor_api: Any = processors
    raw = Tokenizer(
        model_api.WordLevel(
            {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "word": 3, "trigger": 4}, unk_token="[UNK]"
        )
    )
    raw.pre_tokenizer = pre_tokenizer_api.Whitespace()
    raw.post_processor = processor_api.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)]
    )
    model = DebertaDetector.__new__(DebertaDetector)
    Detector.__init__(model, {"model_id": "offline-mechanics-fixture"}, threshold=0.9)
    model.tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw, pad_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]"
    )
    model.model = WindowModel()
    model.torch = torch
    model.positive_index = 1
    texts = ["word " * 1100 + "trigger", "word word", "trigger"]
    predictions = model.predict(texts, batch_size=2)
    assert [row.label for row in predictions] == ["prompt_injection", "benign", "prompt_injection"]
    assert [row.risk_score for row in predictions] == pytest.approx(model.score(texts, 1))
    assert max(model.model.lengths) <= 512
    assert model.predict([]) == []
