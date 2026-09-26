"""Two small detectors with a shared, higher-is-riskier prediction contract."""

# pyright: reportMissingTypeStubs=false
import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
from pydantic import JsonValue
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from promptshield.dataset import digest

serialization: Any = joblib  # Public serialization functions are incompletely typed.

PROMPT_GUARD_ID = "meta-llama/Llama-Prompt-Guard-2-86M"
PROMPT_GUARD_REVISION = "a8ded8e697ce7c355e395a0df51f94adb4a2fd27"


@dataclass(frozen=True)
class Prediction:
    risk_score: float
    label: str


class Detector(ABC):
    def __init__(self, identity: dict[str, JsonValue], threshold: float = 0.5) -> None:
        self.identity = identity
        self.threshold = threshold

    @abstractmethod
    def score(self, texts: Sequence[str], batch_size: int = 16) -> list[float]:
        """Return one finite [0, 1] risk score per input, preserving order."""

    def predict(self, texts: Sequence[str], batch_size: int = 16) -> list[Prediction]:
        validate_inputs(texts, batch_size)
        if not math.isfinite(self.threshold) or not 0 <= self.threshold <= 1:
            raise ValueError("Threshold must be finite and in [0, 1]")
        scores = self.score(texts, batch_size)
        if len(scores) != len(texts) or any(
            not math.isfinite(s) or not 0 <= s <= 1 for s in scores
        ):
            raise ValueError("Detector violated the score contract")
        return [
            Prediction(s, "prompt_injection" if s >= self.threshold else "benign") for s in scores
        ]


def validate_inputs(texts: Sequence[object], batch_size: int) -> None:
    if isinstance(texts, str) or any(
        not isinstance(text, str) or not text.strip() for text in texts
    ):
        raise ValueError("Expected a sequence of nonblank strings")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")


class TfidfDetector(Detector):
    def __init__(self, pipeline: Any, version: str, threshold: float = 0.5) -> None:
        super().__init__(
            {"model_id": "promptshield/char-tfidf-logreg", "version": version}, threshold
        )
        self.pipeline = pipeline

    @classmethod
    def fit(
        cls, texts: Sequence[str], labels: Sequence[int], seed: int, version: str
    ) -> TfidfDetector:
        validate_inputs(texts, 1)
        if len(texts) != len(labels) or set(labels) != {0, 1}:
            raise ValueError("Training requires aligned binary labels and both classes")
        # Fixed before test/PINT evaluation. No learned preprocessing outside fit(train).
        pipeline: Any = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        analyzer="char",
                        ngram_range=(3, 5),
                        min_df=1,
                        sublinear_tf=True,
                        lowercase=True,
                    ),
                ),
                (
                    "logreg",
                    LogisticRegression(C=1.0, max_iter=2000, solver="liblinear", random_state=seed),
                ),
            ]
        )
        pipeline.fit(list(texts), list(labels))
        return cls(pipeline, version)

    def score(self, texts: Sequence[str], batch_size: int = 16) -> list[float]:
        validate_inputs(texts, batch_size)
        classes = list(self.pipeline.classes_)
        if classes != [0, 1]:
            raise ValueError("Unexpected sklearn class mapping")
        scores: list[float] = []
        for start in range(0, len(texts), batch_size):
            probabilities: Any = self.pipeline.predict_proba(
                list(texts[start : start + batch_size])
            )
            scores.extend(float(row[1]) for row in probabilities)
        return scores

    def save(self, path: Path) -> None:
        serialization.dump(
            {"pipeline": self.pipeline, "identity": self.identity, "threshold": self.threshold},
            path,
        )

    @classmethod
    def load(cls, path: Path) -> TfidfDetector:
        """Load only trusted locally produced artifacts (joblib can execute code)."""
        data: Any = serialization.load(path)
        return cls(data["pipeline"], data["identity"]["version"], data["threshold"])


def malicious_index(id2label: dict[int, str]) -> int:
    labels = {key: value.upper() for key, value in id2label.items()}
    if len(labels) != 2 or set(labels.values()) != {"BENIGN", "MALICIOUS"}:
        raise ValueError(f"Unexpected Prompt Guard class mapping: {labels}")
    return next(key for key, label in labels.items() if label == "MALICIOUS")


class PromptGuardDetector(Detector):
    def __init__(self, snapshot: Path, threshold: float = 0.5, threads: int = 1) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if threads < 1:
            raise ValueError("threads must be positive")
        torch.set_num_threads(threads)
        self.torch: Any = torch
        tokenizer_factory: Any = AutoTokenizer
        model_factory: Any = AutoModelForSequenceClassification
        self.tokenizer: Any = tokenizer_factory.from_pretrained(
            str(snapshot), local_files_only=True, trust_remote_code=False, use_fast=True
        )
        self.model: Any = model_factory.from_pretrained(
            str(snapshot), local_files_only=True, trust_remote_code=False, use_safetensors=True
        )
        self.model.to("cpu")
        self.model.eval()
        if not self.tokenizer.is_fast:
            raise ValueError("Fast tokenizer required for overflow-to-document mapping")
        self.positive_index = malicious_index(dict(self.model.config.id2label))
        files: dict[str, JsonValue] = {
            path.name: digest(path.read_bytes())
            for path in sorted(snapshot.iterdir())
            if path.is_file() and path.suffix in {".json", ".safetensors", ".model"}
        }
        super().__init__(
            {
                "model_id": PROMPT_GUARD_ID,
                "requested_revision": PROMPT_GUARD_REVISION,
                "version": digest(str(sorted(files.items())).encode()),
                "files": files,
                "device": "cpu",
                "threads": threads,
                "max_tokens": 512,
                "stride": 64,
                "aggregation": "max",
                "score": "softmax(MALICIOUS), temperature=1",
            },
            threshold,
        )

    def score(self, texts: Sequence[str], batch_size: int = 16) -> list[float]:
        validate_inputs(texts, batch_size)
        scores: list[float] = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            encoded: Any = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                stride=64,
                return_overflowing_tokens=True,
                return_tensors="pt",
            )
            owners: list[int] = encoded.pop("overflow_to_sample_mapping").tolist()
            maxima: list[float] = [0.0] * len(batch)
            for offset in range(0, len(owners), batch_size):
                inputs = {
                    key: value[offset : offset + batch_size] for key, value in encoded.items()
                }
                with self.torch.inference_mode():
                    logits = self.model(**inputs).logits
                    risks: list[float] = self.torch.softmax(logits, dim=-1)[
                        :, self.positive_index
                    ].tolist()
                for owner, risk in zip(owners[offset : offset + batch_size], risks, strict=True):
                    maxima[owner] = max(maxima[owner], float(risk))
            scores.extend(maxima)
        return scores
