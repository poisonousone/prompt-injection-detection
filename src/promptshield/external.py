"""Pinned external artifacts and explicit PINT benchmark taxonomy."""

# pyright: reportMissingTypeStubs=false
import json
import urllib.request
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from promptshield.dataset import canonical_json, digest, normalize
from promptshield.evaluation import Case
from promptshield.predictors import DEBERTA_ID, DEBERTA_REVISION

PINT_REVISION = "0efab3f463eae9c823130d8faffb71b2e7c06e63"
PINT_SHA256 = "df068b9a4ff72483f493add6be6242c6aa777df756bd61462aa0e13645cffa90"
PINT_URL = (
    f"https://raw.githubusercontent.com/lakeraai/pint-benchmark/{PINT_REVISION}/"
    "benchmark/data/example-dataset.yaml"
)


class PintRow(BaseModel):
    model_config = ConfigDict(strict=True, extra="allow")
    text: str = Field(min_length=1)
    label: bool
    category: Literal[
        "prompt_injection",
        "jailbreak",
        "hard_negatives",
        "chat",
        "documents",
        "short_input",
        "benign_input",
        "long_input",
    ]
    language: str = "unknown"


def acquire(root: Path) -> dict[str, Any]:
    """Fetch the public PINT example and the pinned official public model."""
    import huggingface_hub
    from huggingface_hub.errors import GatedRepoError

    directory = root / "data/raw/pint" / PINT_REVISION
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "example-dataset.yaml"
    if not path.exists():
        with urllib.request.urlopen(PINT_URL, timeout=60) as response:
            data = response.read()
        if digest(data) != PINT_SHA256:
            raise ValueError("Downloaded PINT checksum mismatch")
        with path.open("xb") as stream:
            stream.write(data)
    if digest(path.read_bytes()) != PINT_SHA256:
        raise ValueError("Existing immutable PINT checksum mismatch")
    status: dict[str, Any] = {
        "pint": {
            "status": "public_example_only",
            "path": path.relative_to(root).as_posix(),
            "sha256": digest(path.read_bytes()),
            "repository": "lakeraai/pint-benchmark",
            "revision": PINT_REVISION,
            "url": PINT_URL,
            "full_benchmark": "not_available_locally; proprietary data not in public repository",
        },
        "deberta": {"model_id": DEBERTA_ID, "revision": DEBERTA_REVISION},
    }
    try:
        hub: Any = huggingface_hub
        snapshot = hub.snapshot_download(
            DEBERTA_ID,
            revision=DEBERTA_REVISION,
            cache_dir=str(root / ".cache/huggingface/hub"),
            allow_patterns=[
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "added_tokens.json",
                "spm.model",
                "model.safetensors",
                "README.md",
                "LICENSE",
            ],
            max_workers=2,
        )
        status["deberta"].update(status="available", snapshot=snapshot)
    except GatedRepoError:
        status["deberta"].update(
            status="blocked_gated_access",
            reason="Official repository denied access; authorized HF session required",
        )
    except OSError as error:
        status["deberta"].update(status="download_failed", reason=type(error).__name__)
    target = root / "artifacts/external_status.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(canonical_json(status))
    return status


def read_pint(path: Path, expected_sha256: str) -> list[Case]:
    data = path.read_bytes()
    if digest(data) != expected_sha256:
        raise ValueError("PINT checksum mismatch")
    rows = TypeAdapter(list[PintRow]).validate_python(yaml.safe_load(data))
    if not rows:
        raise ValueError("PINT input must be a nonempty YAML list")
    cases: list[Case] = []
    for index, row in enumerate(rows):
        text, label, category, language = row.text, row.label, row.category, row.language
        if not text.strip():
            raise ValueError(f"Blank PINT row {index}")
        # Boolean targets are retained even where category alone suggests another label.
        # Combined PINT target is 'benchmark attack', not pure injection taxonomy.
        cases.append(
            Case(
                f"pint-{index:06}",
                text,
                int(label),
                language,
                category == "hard_negatives" and not label,
                str(category),
                "PINT",
            )
        )
    return cases


def overlap_audit(internal: dict[str, list[Case]], external: list[Case]) -> dict[str, Any]:
    pairs: list[dict[str, Any]] = []
    for split, rows in internal.items():
        for left in rows:
            a = normalize(left.text)
            for right in external:
                b = normalize(right.text)
                exact = left.text == right.text
                normalized = a == b
                # A bounded containment check additionally catches embedded fixture payloads.
                contained = min(len(a), len(b)) >= 40 and (a in b or b in a)
                if normalized or contained:
                    pairs.append(
                        {
                            "internal_id": left.id,
                            "external_id": right.id,
                            "internal_split": split,
                            "exact": exact,
                            "normalized": normalized,
                            "containment": contained,
                        }
                    )
    return {
        "method": "exact/NFKC-normalized/substring containment >=40 characters",
        "matches": pairs,
        "match_count": len(pairs),
        "policy": "report contamination; retain every external row; do not retrain or retune",
    }


if __name__ == "__main__":
    print(json.dumps(acquire(Path.cwd()), indent=2))
