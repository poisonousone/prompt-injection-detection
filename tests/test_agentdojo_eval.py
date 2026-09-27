"""Lightweight integration checks; no downloads, real models or provider clients."""

# pyright: reportMissingTypeStubs=false
import socket
from collections.abc import Sequence
from copy import deepcopy
from typing import Any

import pytest

pytest.importorskip("agentdojo")

from agentdojo.functions_runtime import FunctionCall
from agentdojo.types import ChatMessage

from promptshield.agentdojo_eval import BLOCKED, ToolOutputDefense, no_network, run_case, summarize
from promptshield.predictors import Detector


class StubDetector(Detector):
    def __init__(self, score: float = 1.0) -> None:
        super().__init__({"model_id": "test-only"})
        self.value = score
        self.batches: list[list[str]] = []

    def score(self, texts: Sequence[str], batch_size: int = 16) -> list[float]:
        self.batches.append(list(texts))
        return [self.value] * len(texts)


def test_defense_batches_new_outputs_preserves_roles_and_input() -> None:
    messages: list[ChatMessage] = [
        {"role": "system", "content": [{"type": "text", "content": "system"}]},
        {"role": "user", "content": [{"type": "text", "content": "user"}]},
        {"role": "assistant", "content": None, "tool_calls": []},
    ]
    for i in range(2):
        messages.append(
            {
                "role": "tool",
                "content": [{"type": "text", "content": f"result {i}"}],
                "tool_call": FunctionCall(function="read_file", args={}, id=str(i)),
                "tool_call_id": str(i),
                "error": "untrusted error",
            }
        )
    original = deepcopy(messages)
    detector = StubDetector()
    defense = ToolOutputDefense(detector)
    _, _, _, filtered, _ = defense.query("user", None, messages=messages)
    assert messages == original
    assert filtered[:3] == original[:3]
    assert detector.batches == [["result 0\nuntrusted error", "result 1\nuntrusted error"]]
    for before, after in zip(original[3:], filtered[3:], strict=True):
        assert after["content"] == [{"type": "text", "content": BLOCKED}]
        assert before["role"] == "tool"
        assert after["role"] == "tool"
        assert before["tool_call"] == after["tool_call"]
        assert after["error"] is None
    assert len(defense.events) == 2
    # The next iteration must not rescan historical outputs.
    defense.query("user", None, messages=[*filtered, original[2]])
    assert len(detector.batches) == 1


def test_score_contract_failure_is_not_silently_allowed() -> None:
    detector = StubDetector(float("nan"))
    with no_network(), pytest.raises(ValueError, match="score contract"):
        run_case(detector, True)


def test_official_task_metrics_and_fresh_sandbox() -> None:
    with no_network():
        baseline = summarize([run_case(None, False), run_case(None, True)])
        blocking = summarize([run_case(StubDetector(), False), run_case(StubDetector(), True)])
        passing = summarize([run_case(StubDetector(0), False), run_case(StubDetector(0), True)])
    assert baseline["clean_utility_success_rate"] == 1
    assert baseline["attack_success_rate"] == 1
    assert baseline["attacked_utility_success_rate"] == 0
    assert blocking["clean_utility_success_rate"] == 0
    assert blocking["attack_success_rate"] == 0
    assert passing["attack_success_rate"] == 1
    assert passing["clean_utility_success_rate"] == 1
    assert baseline["episodes"][0]["attack_success"] is None
    assert all(row["fresh_environment_verified"] for row in blocking["episodes"])


def test_network_is_denied_and_restored() -> None:
    original: Any = socket.create_connection
    with no_network(), pytest.raises(RuntimeError, match="Network disabled"):
        socket.create_connection(("example.com", 443))
    assert socket.create_connection is original
