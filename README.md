# Route Forge

Route Forge is an AI-assisted MoonBoard 2016 problem generator. The React app
lets a setter choose a grade, one or two start holds, one or two finish holds,
and a movement style. It then compares a conservative training-data match with
up to two model-generated alternatives on an interactive board.

The frontend runs locally, while training and inference run on Modal. Model
weights are never downloaded to this repository or executed on the local
computer.

> Generated routes are suggestions, not certified safe or climbable problems.
> Check every route on the physical wall and use human judgment.

## Run the tool locally

### Requirements

- Node.js 18 or newer
- npm
- An internet connection to reach the deployed Modal API

Python, PyTorch, a GPU, and a Modal login are **not** required to use the web
app.

### Start the web app

From the repository root:

```powershell
cd frontend
npm ci
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). The frontend uses the
deployed Phase 3 API by default:

```text
https://lczhang42--climb-path-phase3-api.modal.run
```

The Modal container scales to zero after five idle minutes. The first request
after an idle period can therefore take around a minute while the GPU container
starts and loads the model; later requests are much faster.

To use another API deployment, set `VITE_API_URL` before starting Vite. This URL
is visible in the browser and must never contain a token or other secret.

PowerShell:

```powershell
$env:VITE_API_URL = "https://your-api.example.com"
npm run dev
```

Bash/zsh:

```bash
VITE_API_URL="https://your-api.example.com" npm run dev
```

### Verify the frontend

```powershell
cd frontend
npm test -- --run
npm run lint
npm run build
```

The production build is written to `frontend/dist/`.

## How to use Route Forge

1. Choose an optional **Quick setup**, or select a target grade manually. The
   three presets are examples only; all supported grades remain available.
2. Choose **Safer**, **Balanced**, or **Dynamic** movement. Grade controls the
   intended difficulty, while movement style controls how adventurous the
   allowed reach can be.
3. Select one or two green start holds and one or two orange finish holds. The
   UI only exposes known 2016 starting positions and valid top-row finishes.
4. Open **Generation settings** if you want a repeatable seed or a different
   sampling temperature, then select **Forge routes**.
5. Compare the conservative route with the model alternatives. The cards show
   included holds, longest beta move, novelty, and movement warnings.
6. On the movement canvas, select **Edit route** and drag any colored ring to a
   different hold. Select **Restore generated route** to discard manual edits.
   Keyboard users can press Enter on a ring and then Enter on its destination.
7. Use **Copy shareable setup** to copy the grade, endpoints, seed, and movement
   mode in the page URL. Manual canvas edits are temporary and are not included
   in that URL.

## Training and generation overview

The versioned MoonBoard 2016 data contract contains 6,597 deduplicated training
routes across nine grades from 6B+ through 7C+. Each route is represented with
atomic tokens for its grade, start holds, middle holds, and finish holds.

Training fine-tunes DistilGPT-2 on a Modal T4 GPU using completion-only loss,
fixed seeds, validation checkpoints, and a frozen evaluation set. Checkpoints
are stored in the `climb-path-models` Modal Volume and the Hugging Face cache is
stored in `climb-path-hf-cache`; neither is committed to Git.

The selected structured checkpoint did not beat nearest-route retrieval on
middle-hold F1 (`0.1171` versus `0.1462`). For that reason, the deployed Phase 3
service is retrieval-first rather than model-only:

- candidate 1 is a validated conservative match from the training data;
- later candidates are diverse DistilGPT-2 alternatives;
- invalid, duplicate, or severely unreachable candidates are rejected;
- survivors are ranked using grade-aware route length, move distance, upward
  progression, model score, and novelty.

F1 measures overlap with a held-out human route. It is useful for reproducible
model comparison, but it does not prove that a route is physically feasible.
See [`PHASE2_RESULTS.md`](PHASE2_RESULTS.md) and
[`PHASE3_RESULTS.md`](PHASE3_RESULTS.md) for the measured results.

## Run training on Modal

Training is optional and is separate from running the local web app.

### 1. Install and authenticate the Modal CLI

Use Python 3.11 or 3.12:

```powershell
python -m pip install "modal==1.6.1"
python -m modal token new
```

Modal stores its credential in the user's Modal profile outside this
repository. Do not paste Modal tokens into source code, `.env` files, frontend
configuration, or Git.

### 2. Run a small smoke training job

```powershell
python -X utf8 -m modal run modal_app.py::train_phase2 `
  --run-name structured-smoke `
  --num-epochs 1 `
  --max-train-samples 256 `
  --max-validation-samples 128
```

### 3. Run the measured structured training configuration

```powershell
python -X utf8 -m modal run modal_app.py::train_phase2 `
  --run-name structured-v1 `
  --num-epochs 3 `
  --learning-rate 0.0001
```

This trains and evaluates the checkpoint, then stores it at
`/models/phase2/structured-v1/final` in the Modal Volume.

### 4. Re-run evaluation without retraining

```powershell
python -X utf8 -m modal run modal_app.py::evaluate_phase3 `
  --model-version structured-v1 `
  --evaluation-split frozen
```

### 5. Deploy the API

```powershell
python -m modal deploy modal_app.py
```

Deployment and training commands do not automatically promote a checkpoint.
Keep the retrieval fallback until a new model passes the frozen evaluation and
physical-validation gates. Full operational details are in
[`MODAL_DEPLOYMENT.md`](MODAL_DEPLOYMENT.md).

## Project structure

```text
climb-path/
├── frontend/              React/Vite route studio
├── backend/
│   ├── api/               Phase 0 and Phase 3 FastAPI services
│   ├── data_processing/   Dataset preparation and validation
│   ├── evaluation/        Frozen evaluation and Phase 3 scoring
│   ├── models/            Tokenization and dataset helpers
│   └── training/          Baseline and structured fine-tuning
├── data_contract/v1/      Versioned splits, board holds, and manifests
├── tests/                 Local and opt-in deployed API tests
├── modal_app.py           Modal training, evaluation, and API definitions
└── *.md                   Design, results, and deployment documentation
```

## Documentation

| Guide | Purpose |
|---|---|
| [`DATA_CONTRACT.md`](DATA_CONTRACT.md) | Dataset schema, splits, hashes, and frozen benchmark |
| [`PHASE2_RESULTS.md`](PHASE2_RESULTS.md) | Structured fine-tuning results and promotion decision |
| [`PHASE3_RESULTS.md`](PHASE3_RESULTS.md) | Constrained generation, F1 interpretation, and API results |
| [`PHASE4_RESULTS.md`](PHASE4_RESULTS.md) | React UI, board calibration, and interaction design |
| [`MODAL_DEPLOYMENT.md`](MODAL_DEPLOYMENT.md) | Modal authentication, jobs, volumes, and deployment |
| [`PROJECT_REVIEW_AND_IMPLEMENTATION_PLAN.md`](PROJECT_REVIEW_AND_IMPLEMENTATION_PLAN.md) | Original review and phased implementation plan |

## License

MIT
