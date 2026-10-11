"""Modal training and inference deployment for the DistilGPT-2 baseline.

Model weights and the Hugging Face cache live only in Modal Volumes. The public
base model does not require a Hugging Face secret.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import modal


APP_NAME = "climb-path"
PROJECT_ROOT = Path(__file__).resolve().parent
REMOTE_ROOT = "/app"
MODEL_VOLUME_ROOT = "/models"
HF_CACHE_ROOT = "/cache/huggingface"
BASELINE_NAME = "distilgpt2-baseline"
BASELINE_DIR = f"{MODEL_VOLUME_ROOT}/{BASELINE_NAME}"

DEPENDENCIES = (
    "torch==2.10.0",
    "transformers==5.0.0",
    "datasets==4.5.0",
    "accelerate==1.12.0",
    "pandas==3.0.0",
    "numpy==2.4.1",
    "fastapi==0.128.0",
    "pydantic==2.12.5",
    "uvicorn==0.40.0",
    "tensorboard==2.20.0",
    "tqdm==4.67.1",
)

runtime_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(*DEPENDENCIES)
    .add_local_dir(PROJECT_ROOT / "backend", f"{REMOTE_ROOT}/backend", copy=True)
    .add_local_dir(
        PROJECT_ROOT / "data_reordered",
        f"{REMOTE_ROOT}/data_reordered",
        copy=True,
    )
)

app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name("climb-path-models", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("climb-path-hf-cache", create_if_missing=True)


@app.function(
    image=runtime_image,
    gpu="T4",
    timeout=2 * 60 * 60,
    retries=2,
    volumes={
        MODEL_VOLUME_ROOT: model_volume,
        HF_CACHE_ROOT: hf_cache_volume,
    },
)
def train_baseline(
    num_epochs: int = 3,
    smoke: bool = False,
) -> dict[str, str | int | bool]:
    """Fine-tune DistilGPT-2 and persist the model in a Modal Volume."""

    env = os.environ.copy()
    env.update(
        {
            "HF_HOME": HF_CACHE_ROOT,
            "TRANSFORMERS_CACHE": HF_CACHE_ROOT,
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONPATH": REMOTE_ROOT,
        }
    )
    output_dir = f"{MODEL_VOLUME_ROOT}/distilgpt2-smoke" if smoke else BASELINE_DIR
    checkpoint_steps = "5" if smoke else "100"
    command = [
        sys.executable,
        f"{REMOTE_ROOT}/backend/training/finetune_huggingface.py",
        "--train_csv",
        f"{REMOTE_ROOT}/data_reordered/moonboard_train_quality_filtered.csv",
        "--val_csv",
        f"{REMOTE_ROOT}/data_reordered/moonboard_val_quality.csv",
        "--model_name",
        "distilbert/distilgpt2",
        "--output_dir",
        output_dir,
        "--num_epochs",
        str(1 if smoke else num_epochs),
        "--batch_size",
        "16",
        "--eval_batch_size",
        "32",
        "--max_length",
        "96",
        "--learning_rate",
        "5e-5",
        "--warmup_steps",
        "100",
        "--save_steps",
        checkpoint_steps,
        "--eval_steps",
        checkpoint_steps,
        "--logging_steps",
        "20",
        "--seed",
        "42",
        "--fp16",
    ]
    if smoke:
        command.extend(["--max_train_samples", "128", "--max_val_samples", "64"])

    subprocess.run(command, cwd=REMOTE_ROOT, env=env, check=True)
    model_volume.commit()
    hf_cache_volume.commit()
    return {
        "model": BASELINE_NAME,
        "model_dir": f"{output_dir}/final",
        "epochs": 1 if smoke else num_epochs,
        "smoke": smoke,
    }


@app.function(
    image=runtime_image,
    cpu=4,
    memory=8192,
    timeout=5 * 60,
    scaledown_window=5 * 60,
    volumes={MODEL_VOLUME_ROOT: model_volume},
)
@modal.asgi_app()
def api():
    """Serve the promoted baseline without downloading weights locally."""

    sys.path.insert(0, REMOTE_ROOT)
    os.environ["CLIMB_MODEL_DIR"] = f"{BASELINE_DIR}/final"
    os.environ[
        "CLIMB_VALID_HOLDS_CSV"
    ] = f"{REMOTE_ROOT}/data_reordered/moonboard_train_quality_filtered.csv"
    from backend.api.server import create_app

    return create_app()
