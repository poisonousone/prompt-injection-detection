# Reproduction

Run from the repository root with CPython 3.14.7 and uv 0.12.19. `pyproject.toml`
and `uv.lock` define the environment. CPU operation is supported. On the original Windows
workstation, first dot-source `scripts/activate-local.ps1`; other machines do not need it.

## Installation and lightweight verification

```sh
uv python install
uv sync --locked --group agentdojo
uv run --locked --group agentdojo ruff check .
uv run --locked --group agentdojo ruff format --check .
uv run --locked --group agentdojo pyright
uv run --locked --group agentdojo pytest -q
```

No real weights, datasets, provider credentials or training are needed by these checks.
Tests use fixtures, stub detectors and a tiny random encoder. AgentDojo is an optional
locked group, required for its integration tests and the complete Pyright check.
A plain `uv sync --locked` is sufficient for the service/training runtime; AgentDojo tests
skip when that optional package is absent. Initial installation needs network or a populated
uv cache. Task 8 verified a new environment from the populated cache, not a cold download.

## Lightweight dataset workflow

```sh
uv run --locked promptshield-data fetch
uv run --locked promptshield-data build --card data/REPRO_CARD.md
```

Fetch downloads the checksum-pinned public Parquet only if absent. It never overwrites
mismatched raw data. Build reads the reviewed allowlist and versioned bilingual JSONL;
it writes `data/processed/v1` on a clean checkout. The alternate card keeps the historical
`DATA_CARD.md` unchanged. See `data/SOURCES.md` for provenance and license ambiguity.

If `data/processed/v1` already exists, preserve it. To verify current code separately:

```sh
uv run --locked promptshield-data build --output data/processed/reproduction --card data/REPRO_CARD.md
```

The manifest fingerprint includes implementation and lockfile hashes. Current builds
therefore differ from the historical experiment fingerprint
`7d653ab1a682f5570392bc379c07e25c97db6ad40c94d1f8e042b5a205008768`.
Task 8 verified byte-identical four Parquet files and `report.json`, including split
assignments. Do not substitute a new manifest into sealed historical experiments.
Pass `--dataset data/processed/reproduction` to training/evaluation when using that build.
Builds reject overwriting any different output. A repeat is byte-identical only with the
same code, lock, inputs, runtime and execution commit. Cross-platform equality is not promised.

## Manual baseline training and external evaluation

The following commands download the frozen DeBERTa model and eight public PINT examples,
then fit TF-IDF and evaluate both detectors. They are optional for installation verification.
Use the named output below only on a clean checkout; every run requires a new directory.

```sh
uv run --locked python -m promptshield.external
uv run --locked promptshield-baselines --output artifacts/baselines/task2-complete --deberta-snapshot .cache/huggingface/hub/models--protectai--deberta-v3-base-prompt-injection-v2/snapshots/90c9989b1a342275dd0d1a95aad283c04e075671 --pint data/raw/pint/0efab3f463eae9c823130d8faffb71b2e7c06e63/example-dataset.yaml --pint-sha256 df068b9a4ff72483f493add6be6242c6aa777df756bd61462aa0e13645cffa90
```

Inspect `artifacts/external_status.json` for acquisition status. TF-IDF-only execution
can omit the external arguments. Frozen thresholds precede test/PINT evaluation. The
public PINT example is not the full benchmark. Only load trusted local `model.joblib` files.

## Manual custom training and frozen evaluation

First download the exact multilingual base model (run this in `uv run python`):

```python
from huggingface_hub import snapshot_download

snapshot_download(
    "distilbert/distilbert-base-multilingual-cased",
    revision="45c032ab32cc946ad88a166f7cb282f58c753c2e",
    cache_dir=".cache/huggingface/hub",
    allow_patterns=[
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
        "model.safetensors",
    ],
)
```

```sh
uv run --locked promptshield-train train --config configs/transformer.json --output artifacts/transformer/task3-real
uv run --locked promptshield-train evaluate --output artifacts/transformer/task3-real
```

Evaluate only after selection is final; it requires the acquired PINT example. Training
reads train/validation only, uses CPU fp32 and records local MLflow metadata. Config controls
seed, learning rate, batching, accumulation, epochs and checkpoint retention. The sealed
`model/` is independently loadable; optimizer-state resume is not implemented. GPU execution
is unverified and the lock selects CPU PyTorch. Do not iterate based on test/PINT results.

An optional offline toy command needs built data but no base-model download:

```sh
uv run --locked promptshield-train train --smoke --output artifacts/transformer/toy-smoke
```

It is not a multilingual training result and cannot pass final-evaluation guards.

## Manual calibration and diagnostics

With the preceding custom run frozen:

```sh
uv run --locked python -m promptshield.calibration fit --config configs/calibration.json --output artifacts/calibration/task4-real
uv run --locked python -m promptshield.calibration evaluate --output artifacts/calibration/task4-real
uv run --locked promptshield-robustness --output artifacts/robustness/reproduction --deberta-snapshot .cache/huggingface/hub/models--protectai--deberta-v3-base-prompt-injection-v2/snapshots/90c9989b1a342275dd0d1a95aad283c04e075671
uv run --locked --group agentdojo python -m promptshield.agentdojo_eval --output artifacts/agentdojo/reproduction
```

Calibration accepts `--model-run` for an alternate custom run. Robustness and AgentDojo
currently expect the historical relative artifact locations used above (AgentDojo also
expects `data/processed/v1`). They are offline manual inference jobs requiring the existing
weights; AgentDojo uses a scripted oracle and no providers. No real LLM benchmark command
is implemented. Do not run paid evaluation without a separately scoped protocol.

Outputs are ignored under `artifacts/` and `mlruns/`. Keep historical runs immutable.
Predictions are JSONL; metrics, thresholds, configurations and seals are JSON. Local
error-analysis files can contain source text. Small versioned reports summarize real runs;
large historical model/prediction artifacts are not distributed with Git.

## API and Docker

After `python -m promptshield.external` successfully acquires DeBERTa:

```sh
uv run --locked uvicorn promptshield.service:app --host 127.0.0.1 --port 8000 --no-access-log
```

Example POST bodies: `/v1/scan`: `{"text":"Hello","operating_profile":"default"}`;
`/v1/scan/batch`: `{"texts":["Hello","Привет"]}`. Health is at `/health` and aggregate
metrics at `/metrics`. Settings: `PROMPTSHIELD_MODEL_PATH`, `PROMPTSHIELD_THREADS`,
`PROMPTSHIELD_BATCH_SIZE`, `PROMPTSHIELD_WARMUP=true|false`. Use one worker.

```sh
docker build -t promptshield:latest .
docker run --rm -p 127.0.0.1:8000:8000 promptshield:latest
```

The default image downloads and embeds the pinned model. CI uses
`docker build --build-arg INCLUDE_MODEL=false --tag promptshield:ci .` for packaging only;
that image needs a pinned snapshot mounted at `/model` to start. Container build/run is
unverified locally because Docker/Podman are unavailable. CI is configured but not remotely
observed. Do not treat configuration as a successful container test.
