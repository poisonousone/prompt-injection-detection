# Task 7 — Production-like service

Date: 2026-09-27

## Scope and selected detector

The service exposes the frozen external detector
`protectai/deberta-v3-base-prompt-injection-v2` at revision
`90c9989b1a342275dd0d1a95aad283c04e075671`. It uses the existing unmodified
weights, maximum-over-window scoring and fixed threshold 0.5. The sole accepted
operating profile is `default`; no calibration or test-driven threshold selection was
introduced. This follows the user's Task 7 instruction to use DeBERTa and its stronger
frozen internal result, but does not turn the small evaluation into production validation.
Detection remains one defense-in-depth signal.

## HTTP and lifecycle

- `POST /v1/scan`: one text and optional `default` profile.
- `POST /v1/scan/batch`: 1–32 texts in one detector call.
- `GET /health`: readiness plus model ID/version.
- `GET /metrics`: Prometheus text exposition.

The FastAPI lifespan loads one detector per process. CPU calls reuse that object and are
serialized because the current deployment is a one-worker CPU service. Optional startup
warmup is enabled by default and is excluded from request/decision metrics. Inputs are
strict strings, stripped and limited to 10,000 characters. Unknown fields, malformed
JSON, blank/oversized text, invalid profiles and oversized/empty batches return 422.
Validation responses and telemetry omit raw inputs; Uvicorn access logging is disabled in
the container.

Metrics cover requests, request latency, error responses, decisions, risk-score and
input-length histograms, and model identity. They are process-local and reset on restart;
they cannot measure online precision or recall without labels.

## Local verification and latency

The pinned local snapshot was loaded through a real Uvicorn process. `/health`, single
scan, two-item batch scan and `/metrics` returned successfully. Model identity version:
`4fc6e6ff38900cf703a2e8da0fb3a941c2e8a76a9d648eb614c3e8e29ada695c`.

Latency command:

```powershell
uv run python scripts/benchmark_service.py `
  --model-path .cache/huggingface/hub/models--protectai--deberta-v3-base-prompt-injection-v2/snapshots/90c9989b1a342275dd0d1a95aad283c04e075671 `
  --iterations 20 --warmups 3 --output artifacts/serving/task7/latency.json
```

Configuration: Windows 10 10.0.19045, CPython 3.14.7, AMD Ryzen 5 3500X (6 logical
CPUs), eager CPU inference with one numerical thread, batch size 1, 45-character benign
input, three explicit warmups after startup warmup, 20 sequential requests through
FastAPI TestClient. Startup/model load is excluded.

| Measure | Result |
|---|---:|
| p50 | 263.110 ms |
| p95 | 271.341 ms |
| minimum | 257.758 ms |
| maximum | 294.214 ms |

This measures warm in-process HTTP application latency on one workstation. It is not a
concurrency, throughput, container, network or production-scale benchmark.

## Container and CI

The multi-stage Dockerfile installs the locked runtime only, runs as an unprivileged user,
disables access logs, and defaults to downloading the exact model revision into the image.
`.dockerignore` admits only package/build sources, excluding datasets, MLflow, artifacts,
caches and checkpoints. The model itself is about 738 MB and PyTorch dominates the image;
this is an inherent limitation of the selected CPU detector.

Routine CI performs locked sync, Ruff, strict Pyright, all lightweight pytest/API tests,
and a weightless Docker packaging build. It does not train, run PINT/AgentDojo, or download
the large model. A normal Docker build remains the self-contained runnable form.

Docker was not installed on the Task 7 workstation, so neither the full image build nor a
container run was executed locally. Remote GitHub Actions also has not yet run. The
Dockerfile/CI configuration is implemented but those execution claims remain unverified.

## Deployment limitations

The selected external model is archived, primarily English, CPU-heavy, uncalibrated and
not a jailbreak detector. Russian behavior is cross-lingual generalization only. Window
maximum aggregation can raise long-document false positives. The starter evaluation is
too small for reliable low-FPR, subgroup or production-readiness conclusions. Metrics are
single-process and ephemeral, requests are serialized, there is no authentication/TLS or
rate limiting, and no GPU path was exercised.
