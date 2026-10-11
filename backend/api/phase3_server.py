"""Phase 3 FastAPI service for constrained, ranked route generation."""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from functools import partial
import json
import logging
import os
from pathlib import Path
import secrets
from threading import Lock
import time
from typing import Any, Protocol
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, model_validator

from backend.climb_path.schema import Hold, SUPPORTED_GRADES
from backend.climb_path.structured_inference import (
    GenerationResult,
    NoValidRouteError,
    Phase3RouteService,
)


LOGGER = logging.getLogger("climb_path.phase3")


class RouteService(Protocol):
    model_version: str
    data_version: str

    def metadata(self) -> dict[str, Any]: ...

    def generate(self, **kwargs) -> GenerationResult: ...


class Phase3GenerateRequest(BaseModel):
    grade: str
    start_holds: list[str] | None = None
    end_holds: list[str] | None = None
    start_hold: tuple[int, int] | None = None
    end_hold: tuple[int, int] | None = None
    candidate_count: int = Field(3, ge=1, le=3)
    temperature: float = Field(0.8, ge=0.2, le=1.3)
    top_p: float = Field(0.95, ge=0.5, le=1.0)
    seed: int | None = Field(None, ge=0, le=2**31 - 1)

    @field_validator("grade")
    @classmethod
    def validate_grade(cls, value: str) -> str:
        if value not in SUPPORTED_GRADES:
            raise ValueError(f"grade must be one of {', '.join(SUPPORTED_GRADES)}")
        return value

    @field_validator("start_holds", "end_holds")
    @classmethod
    def validate_hold_labels(cls, values: list[str] | None) -> list[str] | None:
        if values is None:
            return None
        return [Hold.from_label(value).label for value in values]

    @field_validator("start_hold", "end_hold")
    @classmethod
    def validate_legacy_hold(
        cls, value: tuple[int, int] | None
    ) -> tuple[int, int] | None:
        if value is not None:
            Hold.from_pair(value)
        return value

    @model_validator(mode="after")
    def validate_endpoints(self) -> "Phase3GenerateRequest":
        if (self.start_holds is None) == (self.start_hold is None):
            raise ValueError("provide exactly one of start_holds or start_hold")
        if (self.end_holds is None) == (self.end_hold is None):
            raise ValueError("provide exactly one of end_holds or end_hold")
        starts, ends = self.endpoint_labels()
        if not 1 <= len(starts) <= 2 or not 1 <= len(ends) <= 2:
            raise ValueError("one or two start and end holds are required")
        if len(set(starts + ends)) != len(starts + ends):
            raise ValueError("start and end holds must be unique")
        return self

    def endpoint_labels(self) -> tuple[list[str], list[str]]:
        starts = (
            list(self.start_holds)
            if self.start_holds is not None
            else [Hold.from_pair(self.start_hold).label]  # type: ignore[arg-type]
        )
        ends = (
            list(self.end_holds)
            if self.end_holds is not None
            else [Hold.from_pair(self.end_hold).label]  # type: ignore[arg-type]
        )
        return starts, ends


class DiagnosticsResponse(BaseModel):
    valid: bool
    hard_rejections: list[str]
    warnings: list[str]
    middle_hold_count: int
    target_middle_holds: int
    length_delta: int
    move_distances: list[float]
    maximum_move_distance: float
    reachability_violations: int
    downward_moves: int
    novel_route: bool


class Phase3CandidateResponse(BaseModel):
    rank: int
    source: str
    score: float
    model_log_probability: float | None
    source_route_id: str | None
    retrieval_similarity: float
    grade: str
    start_holds: list[str]
    middle_holds: list[str]
    end_holds: list[str]
    holds: list[str]
    path: list[list[int]]
    diagnostics: DiagnosticsResponse


class Phase3GenerateResponse(BaseModel):
    request_id: str
    model_version: str
    data_version: str
    seed: int
    used_fallback: bool
    generated_attempts: int
    rejected_attempts: int
    candidates: list[Phase3CandidateResponse]


class SlidingWindowRateLimiter:
    def __init__(self, requests_per_minute: int) -> None:
        self.limit = requests_per_minute
        self.events: dict[str, deque[float]] = defaultdict(deque)
        self.lock = Lock()

    def allow(self, key: str, now: float | None = None) -> bool:
        timestamp = time.monotonic() if now is None else now
        with self.lock:
            events = self.events[key]
            while events and timestamp - events[0] >= 60:
                events.popleft()
            if len(events) >= self.limit:
                return False
            events.append(timestamp)
            return True


def _allowed_origins() -> list[str]:
    value = os.environ.get(
        "CLIMB_ALLOWED_ORIGINS",
        "http://localhost:3000,http://localhost:5173",
    )
    return [origin.strip() for origin in value.split(",") if origin.strip()]


def _load_service() -> Phase3RouteService:
    contract_dir = Path(os.environ.get("CLIMB_CONTRACT_DIR", "/app/data_contract/v1"))
    return Phase3RouteService.from_paths(
        model_dir=os.environ.get(
            "CLIMB_PHASE3_MODEL_DIR", "/models/phase2/structured-v1/final"
        ),
        train_path=contract_dir / "train.jsonl",
        board_holds_path=contract_dir / "board_holds.json",
        manifest_path=contract_dir / "manifest.json",
        model_version=os.environ.get("CLIMB_MODEL_VERSION", "structured-v1"),
    )


def create_phase3_app(service: RouteService | None = None) -> FastAPI:
    request_timeout = float(os.environ.get("CLIMB_REQUEST_TIMEOUT_SECONDS", "45"))
    limiter = SlidingWindowRateLimiter(
        int(os.environ.get("CLIMB_REQUESTS_PER_MINUTE", "30"))
    )

    @asynccontextmanager
    async def lifespan(web_app: FastAPI):
        web_app.state.service = service
        web_app.state.load_error = None
        if web_app.state.service is None:
            try:
                web_app.state.service = _load_service()
            except Exception as exc:
                web_app.state.load_error = type(exc).__name__
        yield
        web_app.state.service = None

    web_app = FastAPI(
        title="Climb Path Generator Phase 3 API",
        version="0.3.0",
        lifespan=lifespan,
    )
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=_allowed_origins(),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @web_app.get("/")
    async def root():
        return {"name": "Climb Path Generator Phase 3 API", "version": "0.3.0"}

    @web_app.get("/api/v1/health/live")
    async def health_live():
        return {"status": "healthy"}

    @web_app.get("/api/v1/health/ready")
    async def health_ready():
        if web_app.state.service is None:
            raise HTTPException(status_code=503, detail="Model is not ready")
        return {"status": "ready"}

    @web_app.get("/api/v1/model")
    async def model_metadata():
        active_service: RouteService | None = web_app.state.service
        if active_service is None:
            raise HTTPException(status_code=503, detail="Model is not ready")
        return active_service.metadata()

    @web_app.post(
        "/api/v1/routes/generate", response_model=Phase3GenerateResponse
    )
    async def generate_route(payload: Phase3GenerateRequest, request: Request):
        active_service: RouteService | None = web_app.state.service
        if active_service is None:
            raise HTTPException(status_code=503, detail="Model is not ready")
        client_key = request.client.host if request.client else "unknown"
        if not limiter.allow(client_key):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

        request_id = str(uuid4())
        seed = payload.seed if payload.seed is not None else secrets.randbelow(2**31)
        starts, ends = payload.endpoint_labels()
        started = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                run_in_threadpool(
                    partial(
                        active_service.generate,
                        grade=payload.grade,
                        start_holds=starts,
                        end_holds=ends,
                        candidate_count=payload.candidate_count,
                        temperature=payload.temperature,
                        top_p=payload.top_p,
                        seed=seed,
                    )
                ),
                timeout=request_timeout,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None
        except NoValidRouteError:
            raise HTTPException(
                status_code=422, detail="No route passed the generation constraints"
            ) from None
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Route generation timed out") from None
        except Exception:
            LOGGER.exception(
                json.dumps({"event": "generation_failed", "request_id": request_id})
            )
            raise HTTPException(status_code=500, detail="Route generation failed") from None

        candidates: list[Phase3CandidateResponse] = []
        for rank, candidate in enumerate(result.candidates, start=1):
            diagnostics = candidate.diagnostics.as_dict()
            candidates.append(
                Phase3CandidateResponse(
                    rank=rank,
                    source=candidate.source,
                    score=round(candidate.score, 6),
                    model_log_probability=(
                        round(candidate.model_log_probability, 6)
                        if candidate.model_log_probability is not None
                        else None
                    ),
                    source_route_id=candidate.source_route_id,
                    retrieval_similarity=round(candidate.retrieval_similarity, 6),
                    grade=candidate.route.grade,
                    start_holds=[hold.label for hold in candidate.route.start_holds],
                    middle_holds=[hold.label for hold in candidate.route.middle_holds],
                    end_holds=[hold.label for hold in candidate.route.end_holds],
                    holds=[hold.label for hold in candidate.route.holds],
                    path=candidate.route.as_pairs(),
                    diagnostics=DiagnosticsResponse(**diagnostics),
                )
            )

        LOGGER.info(
            json.dumps(
                {
                    "event": "generation_completed",
                    "request_id": request_id,
                    "grade": payload.grade,
                    "candidate_count": len(candidates),
                    "used_fallback": result.used_fallback,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                },
                separators=(",", ":"),
            )
        )
        return Phase3GenerateResponse(
            request_id=request_id,
            model_version=active_service.model_version,
            data_version=active_service.data_version,
            seed=result.seed,
            used_fallback=result.used_fallback,
            generated_attempts=result.generated_attempts,
            rejected_attempts=result.rejected_attempts,
            candidates=candidates,
        )

    return web_app


app = create_phase3_app()
