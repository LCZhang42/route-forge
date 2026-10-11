"""FastAPI service for the Modal-hosted DistilGPT-2 baseline."""

from __future__ import annotations

from contextlib import asynccontextmanager
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, model_validator

from backend.climb_path.inference import DistilGPT2Generator
from backend.climb_path.schema import Hold, SUPPORTED_GRADES
from backend.models.valid_holds import load_valid_holds_from_dataset


class GenerateRequest(BaseModel):
    grade: str = Field(..., description="Requested climbing grade")
    start_hold: tuple[int, int] = Field((5, 4), description="Start hold [x, y]")
    end_hold: tuple[int, int] = Field((5, 17), description="End hold [x, y]")
    candidate_count: int = Field(1, ge=1, le=3)
    temperature: float = Field(0.8, ge=0.1, le=1.5)
    top_p: float = Field(0.95, gt=0.0, le=1.0)
    seed: int | None = Field(None, ge=0)

    @field_validator("grade")
    @classmethod
    def validate_grade(cls, value: str) -> str:
        if value not in SUPPORTED_GRADES:
            raise ValueError(f"grade must be one of {', '.join(SUPPORTED_GRADES)}")
        return value

    @field_validator("start_hold", "end_hold")
    @classmethod
    def validate_hold(cls, value: tuple[int, int]) -> tuple[int, int]:
        Hold.from_pair(value)
        return value

    @model_validator(mode="after")
    def validate_different_endpoints(self) -> "GenerateRequest":
        if self.start_hold == self.end_hold:
            raise ValueError("start_hold and end_hold must be different")
        return self


class CandidateResponse(BaseModel):
    path: list[list[int]]
    grade: str
    num_holds: int
    warnings: list[str]
    seed: int | None = None


class GenerateResponse(BaseModel):
    path: list[list[int]]
    grade: str
    num_holds: int
    warnings: list[str]
    candidates: list[CandidateResponse]
    model_version: str


def _allowed_origins() -> list[str]:
    configured = os.environ.get(
        "CLIMB_ALLOWED_ORIGINS",
        "http://localhost:3000,http://localhost:5173",
    )
    return [origin.strip() for origin in configured.split(",") if origin.strip()]


def create_app(
    model_dir: str | Path | None = None,
    valid_holds_csv: str | Path | None = None,
) -> FastAPI:
    configured_model_dir = Path(
        model_dir or os.environ.get("CLIMB_MODEL_DIR", "/models/distilgpt2-baseline/final")
    )
    configured_holds_csv = Path(
        valid_holds_csv
        or os.environ.get(
            "CLIMB_VALID_HOLDS_CSV",
            "/app/data_reordered/moonboard_train_quality_filtered.csv",
        )
    )

    @asynccontextmanager
    async def lifespan(web_app: FastAPI):
        web_app.state.generator = None
        web_app.state.load_error = None
        if configured_model_dir.is_dir():
            try:
                holds = None
                if configured_holds_csv.is_file():
                    holds = load_valid_holds_from_dataset(str(configured_holds_csv))
                web_app.state.generator = DistilGPT2Generator(
                    configured_model_dir,
                    valid_holds=holds,
                )
            except Exception as exc:  # readiness exposes only the error type
                web_app.state.load_error = type(exc).__name__
        else:
            web_app.state.load_error = "model_not_found"
        yield
        web_app.state.generator = None

    web_app = FastAPI(
        title="Climb Path Generator API",
        version="0.2.0",
        lifespan=lifespan,
    )
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=_allowed_origins(),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Authorization"],
    )

    @web_app.get("/")
    async def root():
        return {"name": "Climb Path Generator API", "version": "0.2.0"}

    @web_app.get("/api/health/live")
    @web_app.get("/api/health")
    async def health_live():
        return {"status": "healthy"}

    @web_app.get("/api/health/ready")
    async def health_ready():
        if web_app.state.generator is None:
            raise HTTPException(status_code=503, detail="Model is not ready")
        return {"status": "ready"}

    @web_app.get("/api/grades")
    async def get_grades():
        return {"grades": list(SUPPORTED_GRADES)}

    @web_app.get("/api/v1/model")
    async def model_metadata():
        return {
            "model": "distilgpt2-baseline",
            "model_loaded": web_app.state.generator is not None,
            "supported_grades": list(SUPPORTED_GRADES),
            "board": "MoonBoard 2016",
            "endpoint_cardinality": {"start": 1, "end": 1},
        }

    @web_app.post("/api/generate", response_model=GenerateResponse)
    @web_app.post("/api/v1/routes/generate", response_model=GenerateResponse)
    async def generate_path(request: GenerateRequest):
        generator: DistilGPT2Generator | None = web_app.state.generator
        if generator is None:
            raise HTTPException(status_code=503, detail="Model is not ready")

        try:
            candidates = generator.generate(
                grade=request.grade,
                start_hold=Hold.from_pair(request.start_hold),
                end_hold=Hold.from_pair(request.end_hold),
                candidate_count=request.candidate_count,
                temperature=request.temperature,
                top_p=request.top_p,
                seed=request.seed,
            )
        except Exception:
            raise HTTPException(status_code=500, detail="Route generation failed") from None

        if not candidates:
            raise HTTPException(status_code=422, detail="No parseable route was generated")

        responses = [
            CandidateResponse(
                path=candidate.route.as_pairs(),
                grade=candidate.route.grade,
                num_holds=len(candidate.route.holds),
                warnings=list(candidate.warnings),
                seed=request.seed,
            )
            for candidate in candidates
        ]
        first = responses[0]
        return GenerateResponse(
            path=first.path,
            grade=first.grade,
            num_holds=first.num_holds,
            warnings=first.warnings,
            candidates=responses,
            model_version="distilgpt2-baseline",
        )

    return web_app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
