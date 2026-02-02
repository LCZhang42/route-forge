"""
FastAPI server for Climb Path Generator.

Provides REST API endpoints for generating climbing paths.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Tuple, Optional
import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from models.tokenizer import ClimbPathTokenizer
from models.climb_transformer import ClimbPathTransformerWithGeneration
from models.logits_processor import (
    ClimbPathLogitsProcessor,
    MinHoldsLogitsProcessor,
    MaxHoldsLogitsProcessor,
    ValidHoldsLogitsProcessor,
    NoRepeatHoldsLogitsProcessor,
)
from models.valid_holds import load_valid_holds_from_dataset
from training.generate_paths import load_model, generate_climb_path


app = FastAPI(title="Climb Path Generator API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    grade: str = Field(..., description="Climbing grade (e.g., '7A', '6B+', '8A')")
    temperature: float = Field(1.0, ge=0.1, le=2.0, description="Sampling temperature")
    min_holds: int = Field(3, ge=1, le=50, description="Minimum number of holds")
    max_holds: int = Field(30, ge=1, le=50, description="Maximum number of holds")
    use_constraints: bool = Field(True, description="Enable constraint enforcement")
    valid_holds_only: bool = Field(True, description="Only generate valid MoonBoard 2016 holds")


class PathResponse(BaseModel):
    path: List[List[int]]
    grade: str
    num_holds: int
    quality_score: Optional[float] = None


model = None
tokenizer = None
device = None
valid_holds = None


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global model, tokenizer, device, valid_holds
    
    checkpoint_path = Path(__file__).parent.parent.parent / "checkpoints" / "climb_path_cpu" / "best.pt"
    
    if not checkpoint_path.exists():
        print(f"Warning: Model checkpoint not found at {checkpoint_path}")
        print("API will start but generation will fail until model is available")
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")
    
    model, config = load_model(str(checkpoint_path), device)
    tokenizer = ClimbPathTokenizer()
    
    # Load valid holds from dataset
    train_csv_path = Path(__file__).parent.parent.parent / "data" / "moonboard_train_quality.csv"
    if train_csv_path.exists():
        print("Loading valid MoonBoard 2016 hold positions...")
        valid_holds = load_valid_holds_from_dataset(str(train_csv_path))
        print(f"Loaded {len(valid_holds)} valid holds")
    else:
        print(f"Warning: Training data not found at {train_csv_path}")
        print("Valid holds constraint will be disabled")
    
    print("Model loaded successfully!")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Climb Path Generator API",
        "version": "1.0.0",
        "status": "running" if model is not None else "model not loaded"
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.get("/api/grades")
async def get_grades():
    """Get list of available climbing grades."""
    if tokenizer is None:
        tokenizer_temp = ClimbPathTokenizer()
        return {"grades": tokenizer_temp.GRADES}
    return {"grades": tokenizer.GRADES}


@app.post("/api/generate", response_model=PathResponse)
async def generate_path(request: GenerateRequest):
    """Generate a climbing path."""
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure the model checkpoint exists."
        )
    
    if request.grade not in tokenizer.GRADES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid grade '{request.grade}'. Valid grades: {', '.join(tokenizer.GRADES)}"
        )
    
    try:
        # Use valid holds constraint if requested and available
        holds_constraint = valid_holds if request.valid_holds_only else None
        
        grade, holds = generate_climb_path(
            model=model,
            tokenizer=tokenizer,
            grade=request.grade,
            device=device,
            temperature=request.temperature,
            min_holds=request.min_holds,
            max_holds=request.max_holds,
            use_constraints=request.use_constraints,
            valid_holds=holds_constraint,
        )
        
        path = [[int(x), int(y)] for x, y in holds]
        
        return PathResponse(
            path=path,
            grade=grade,
            num_holds=len(holds),
            quality_score=None
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating path: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
