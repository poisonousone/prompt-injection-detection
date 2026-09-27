"""Small local warm-latency sanity check for the Task 7 HTTP application."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path

from fastapi.testclient import TestClient

from promptshield.service import ServiceSettings, create_app


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(fraction * len(ordered) + 0.999999) - 1))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 1 or args.warmups < 0:
        raise ValueError("iterations must be positive and warmups nonnegative")

    app = create_app(ServiceSettings(model_path=args.model_path, warmup=True))
    payload = {"text": "Summarize this routine project status update."}
    latencies: list[float] = []
    with TestClient(app) as client:
        for _ in range(args.warmups):
            response = client.post("/v1/scan", json=payload)
            response.raise_for_status()
        for _ in range(args.iterations):
            started = time.perf_counter()
            response = client.post("/v1/scan", json=payload)
            response.raise_for_status()
            latencies.append((time.perf_counter() - started) * 1000)

    result = {
        "scope": "local_warm_sequential_in_process_http_sanity_not_production_capacity",
        "client": "fastapi_testclient",
        "model_path": str(args.model_path),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "logical_cpu_count": os.cpu_count(),
        "threads": 1,
        "batch_size": 1,
        "warmups": args.warmups,
        "iterations": args.iterations,
        "text_characters": len(payload["text"]),
        "p50_ms": statistics.median(latencies),
        "p95_ms": percentile(latencies, 0.95),
        "minimum_ms": min(latencies),
        "maximum_ms": max(latencies),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
