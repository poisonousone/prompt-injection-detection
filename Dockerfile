# syntax=docker/dockerfile:1.7
FROM ghcr.io/astral-sh/uv:0.12.19 AS uv
FROM python:3.14.7-slim-bookworm

COPY --from=uv /uv /uvx /bin/
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PROMPTSHIELD_MODEL_PATH=/model \
    PROMPTSHIELD_THREADS=1 \
    PROMPTSHIELD_BATCH_SIZE=8 \
    PROMPTSHIELD_WARMUP=true

WORKDIR /app
RUN useradd --create-home --uid 10001 app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --locked --no-dev --no-group agentdojo --no-editable

# The default image is self-contained. CI may set false to validate packaging without
# repeatedly downloading the 700+ MB pinned public model during lightweight checks.
ARG INCLUDE_MODEL=true
RUN set -eu; if [ "$INCLUDE_MODEL" = "true" ]; then \
      HF_HOME=/tmp/huggingface .venv/bin/python -c \
      "from huggingface_hub import snapshot_download; snapshot_download('protectai/deberta-v3-base-prompt-injection-v2', revision='90c9989b1a342275dd0d1a95aad283c04e075671', local_dir='/model', allow_patterns=['added_tokens.json','config.json','model.safetensors','spm.model','tokenizer.json','tokenizer_config.json','special_tokens_map.json'])"; \
      rm -rf /tmp/huggingface /model/.cache; \
    else mkdir -p /model; fi && chown -R app:app /model

USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"
CMD ["uvicorn", "promptshield.service:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
