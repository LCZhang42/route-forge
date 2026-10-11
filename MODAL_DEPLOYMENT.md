# Modal training and deployment

DistilGPT-2 weights are downloaded, fine-tuned, stored, and served on Modal. No
model weights are required in this repository or on the local machine.

## Authentication

Authenticate with Modal's browser flow:

```powershell
python -m modal token new
```

The resulting credential is stored in the user's Modal profile outside this
repository. Do not paste token values into source code, `.env` files, chat, or
frontend configuration.

The DistilGPT-2 base model is public, so the current training job needs no
Hugging Face secret. If a future private model needs one, create a named Modal
Secret and reference only its name in `modal_app.py`.

## Smoke training

This verifies the image, GPU, data, and checkpoint path using 128 training rows:

```powershell
python -m modal run modal_app.py::train_baseline --smoke
```

## Baseline training

```powershell
python -m modal run modal_app.py::train_baseline --num-epochs 3
```

Training uses a T4 GPU. Model output is persisted under
`/models/distilgpt2-baseline/final` in the `climb-path-models` Modal Volume.
The Hugging Face cache is stored separately in `climb-path-hf-cache`.

## Development API

```powershell
python -m modal serve modal_app.py
```

## Persistent deployment

```powershell
python -m modal deploy modal_app.py
```

Modal prints the API URL after serving or deployment. Configure the frontend
with that public URL as `VITE_API_URL`. This value is not a secret.

Current Phase 0 deployment:

```text
https://lczhang42--climb-path-api.modal.run
```

The endpoint is intentionally unauthenticated for browser access during Phase
0. Add authentication and rate limiting before exposing it as a public product.

## Local guarantees

- Importing the API or running tests never downloads model weights.
- The inference loader uses `local_files_only=True`; on Modal, “local” means the
  mounted Modal Volume.
- `checkpoints/` is ignored and the old project-local artifacts were removed.
- The local CLI helper calls the deployed API through `CLIMB_API_URL`.
