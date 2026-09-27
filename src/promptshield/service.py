"""FastAPI service for the frozen ProtectAI DeBERTa detector."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import AsyncGenerator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, cast

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import JSONResponse

from promptshield.predictors import DEBERTA_ID, DEBERTA_REVISION, DebertaDetector, Detector

MAX_TEXT_LENGTH = 10_000
MAX_BATCH_SIZE = 32
DEFAULT_BATCH_SIZE = 8
DEFAULT_SNAPSHOT = Path(
    ".cache/huggingface/hub/"
    "models--protectai--deberta-v3-base-prompt-injection-v2/"
    f"snapshots/{DEBERTA_REVISION}"
)
Profile = Literal["default"]
Text = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_TEXT_LENGTH),
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ScanRequest(StrictModel):
    text: Text
    operating_profile: Profile = "default"


class BatchScanRequest(StrictModel):
    texts: list[Text] = Field(min_length=1, max_length=MAX_BATCH_SIZE)
    operating_profile: Profile = "default"


class ScanResponse(StrictModel):
    label: Literal["benign", "prompt_injection"]
    risk_score: float
    operating_profile: Profile
    model_version: str


class BatchScanResponse(StrictModel):
    items: list[ScanResponse]
    operating_profile: Profile
    model_version: str


class HealthResponse(StrictModel):
    status: Literal["ok"]
    model_id: str
    model_version: str


@dataclass(frozen=True)
class ServiceSettings:
    model_path: Path = DEFAULT_SNAPSHOT
    threads: int = 1
    batch_size: int = DEFAULT_BATCH_SIZE
    warmup: bool = True

    @classmethod
    def from_environment(cls) -> ServiceSettings:
        warmup = os.getenv("PROMPTSHIELD_WARMUP", "true").lower()
        if warmup not in {"true", "false"}:
            raise ValueError("PROMPTSHIELD_WARMUP must be true or false")
        settings = cls(
            model_path=Path(os.getenv("PROMPTSHIELD_MODEL_PATH", str(DEFAULT_SNAPSHOT))),
            threads=int(os.getenv("PROMPTSHIELD_THREADS", "1")),
            batch_size=int(os.getenv("PROMPTSHIELD_BATCH_SIZE", str(DEFAULT_BATCH_SIZE))),
            warmup=warmup == "true",
        )
        if settings.threads < 1:
            raise ValueError("PROMPTSHIELD_THREADS must be positive")
        if not 1 <= settings.batch_size <= MAX_BATCH_SIZE:
            raise ValueError(f"PROMPTSHIELD_BATCH_SIZE must be in [1, {MAX_BATCH_SIZE}]")
        return settings


class RuntimeMetrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "promptshield_http_requests_total",
            "HTTP requests completed by route and status.",
            ("method", "route", "status"),
            registry=self.registry,
        )
        self.latency = Histogram(
            "promptshield_http_request_duration_seconds",
            "HTTP request latency by route.",
            ("method", "route"),
            registry=self.registry,
        )
        self.errors = Counter(
            "promptshield_http_errors_total",
            "HTTP error responses by route and status.",
            ("method", "route", "status"),
            registry=self.registry,
        )
        self.decisions = Counter(
            "promptshield_decisions_total",
            "Detector decisions by label and operating profile.",
            ("label", "operating_profile"),
            registry=self.registry,
        )
        self.risk_scores = Histogram(
            "promptshield_risk_score",
            "Distribution of detector risk scores.",
            buckets=(0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0),
            registry=self.registry,
        )
        self.input_lengths = Histogram(
            "promptshield_input_length_characters",
            "Distribution of input lengths in characters.",
            buckets=(16, 64, 256, 1024, 4096, MAX_TEXT_LENGTH),
            registry=self.registry,
        )
        self.model_info = Gauge(
            "promptshield_model_info",
            "Loaded detector identity.",
            ("model_id", "model_version", "revision"),
            registry=self.registry,
        )

    def observe_predictions(
        self, texts: Sequence[str], labels: Sequence[str], scores: Sequence[float], profile: str
    ) -> None:
        for text, label, score in zip(texts, labels, scores, strict=True):
            self.input_lengths.observe(len(text))
            self.risk_scores.observe(score)
            self.decisions.labels(label=label, operating_profile=profile).inc()


class InferenceService:
    """Own one reusable model instance and serialize its CPU inference calls."""

    def __init__(self, detector: Detector, metrics: RuntimeMetrics, batch_size: int) -> None:
        self.detector = detector
        self.metrics = metrics
        self.batch_size = batch_size
        self._lock = threading.Lock()
        version = detector.identity.get("version")
        if not isinstance(version, str) or not version:
            raise ValueError("Detector identity must include a string version")
        self.model_version = version

    def predict(self, texts: Sequence[str], profile: Profile) -> list[ScanResponse]:
        with self._lock:
            predictions = self.detector.predict(texts, batch_size=self.batch_size)
        labels = [prediction.label for prediction in predictions]
        scores = [prediction.risk_score for prediction in predictions]
        self.metrics.observe_predictions(texts, labels, scores, profile)
        return [
            ScanResponse(
                label=cast(Literal["benign", "prompt_injection"], prediction.label),
                risk_score=prediction.risk_score,
                operating_profile=profile,
                model_version=self.model_version,
            )
            for prediction in predictions
        ]


DetectorLoader = Callable[[ServiceSettings], Detector]


def load_detector(settings: ServiceSettings) -> Detector:
    return DebertaDetector(settings.model_path, threshold=0.5, threads=settings.threads)


def _route_label(request: Request) -> str:
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    return path if isinstance(path, str) else "unmatched"


def create_app(
    settings: ServiceSettings | None = None, detector_loader: DetectorLoader = load_detector
) -> FastAPI:
    configured = settings or ServiceSettings.from_environment()
    metrics = RuntimeMetrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        detector = detector_loader(configured)
        runtime = InferenceService(detector, metrics, configured.batch_size)
        if configured.warmup:
            await run_in_threadpool(
                detector.predict, ["Routine service warmup."], configured.batch_size
            )
        identity = detector.identity
        metrics.model_info.labels(
            model_id=str(identity.get("model_id", "unknown")),
            model_version=runtime.model_version,
            revision=str(identity.get("requested_revision", "unknown")),
        ).set(1)
        app.state.runtime = runtime
        yield

    app = FastAPI(
        title="PromptShield",
        version="0.1.0",
        description="Prompt-injection risk scoring as one defense-in-depth signal.",
        lifespan=lifespan,
    )

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [
            {key: value for key, value in error.items() if key != "input"} for error in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": jsonable_encoder(errors)})

    @app.middleware("http")
    async def instrument(request: Request, call_next: RequestResponseEndpoint) -> Response:
        started = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            route = _route_label(request)
            status_text = str(status)
            metrics.requests.labels(request.method, route, status_text).inc()
            metrics.latency.labels(request.method, route).observe(time.perf_counter() - started)
            if status >= 400:
                metrics.errors.labels(request.method, route, status_text).inc()

    def runtime() -> InferenceService:
        value = getattr(app.state, "runtime", None)
        if not isinstance(value, InferenceService):
            raise HTTPException(status_code=503, detail="Model is not ready")
        return value

    @app.post("/v1/scan", response_model=ScanResponse)
    async def scan(payload: ScanRequest) -> ScanResponse:
        results = await run_in_threadpool(
            runtime().predict, [payload.text], payload.operating_profile
        )
        return results[0]

    @app.post("/v1/scan/batch", response_model=BatchScanResponse)
    async def scan_batch(payload: BatchScanRequest) -> BatchScanResponse:
        service = runtime()
        results = await run_in_threadpool(service.predict, payload.texts, payload.operating_profile)
        return BatchScanResponse(
            items=results,
            operating_profile=payload.operating_profile,
            model_version=service.model_version,
        )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        service = runtime()
        return HealthResponse(status="ok", model_id=DEBERTA_ID, model_version=service.model_version)

    @app.get("/metrics", include_in_schema=False)
    async def prometheus_metrics() -> Response:
        return Response(generate_latest(metrics.registry), media_type="text/plain; version=0.0.4")

    return app


app = create_app()
