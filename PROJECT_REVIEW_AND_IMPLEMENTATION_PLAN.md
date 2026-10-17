# Climb Path Generator: Project Review and Implementation Plan

**Reviewed:** 11 October 2026  
**Scope:** data pipeline, model training and evaluation, Modal execution, inference API, and web-app UX

## Executive summary

The project has a useful dataset, a clear product idea, a working visualization concept, and two modeling experiments: an older custom transformer and a newer DistilGPT-2 fine-tune. The shortest route to a credible demo is **not** to replace everything. It is to make the DistilGPT-2 path reproducible, represent MoonBoard routes more faithfully, add deterministic validation and constraint-aware ranking, then connect one inference implementation to the React UI.

The recommended product architecture is:

```text
Versioned dataset
      |
      v
Modal GPU training --> versioned model + metrics in Modal Volume
                                      |
                                      v
React web app --> FastAPI on Modal --> constrained generation + ranking
                                      |
                                      v
                         1-3 route candidates + diagnostics
```

The first milestone should be a reliable baseline, not a larger model. With only 6,610 training routes and 141 observed hold positions, data correctness, constrained decoding, and evaluation will matter more than raw parameter count.

## What is already good

- The project has separate training, validation, and test CSVs and useful route metadata such as repeats, quality score, grade, and designated start/end holds.
- Endpoint conditioning is the right product direction: the user can choose a grade and endpoints, while the model fills in the route.
- A quality-filtering pipeline and physical-constraint experiments already exist.
- The React client, FastAPI server, Gradio prototype, and MoonBoard image establish a good proof-of-concept UI direction.
- `.env` is already ignored by Git, and model/runs directories are partially excluded.

## Current findings and risks

### 1. The runnable application paths are inconsistent

- `backend/api/server.py` and `web_interface.py` import `backend/training/generate_paths.py`, but that file is absent.
- Those interfaces expect the old custom-transformer checkpoints under `checkpoints/climb_path_cpu/`, but the current documentation describes DistilGPT-2 under `checkpoints/distilgpt2_climb/final/`.
- Neither expected final model is present. The committed DistilGPT-2 directory contains configuration/log artifacts, not a complete loadable model.
- The API accepts only grade and sampling settings, while the current DistilGPT-2 generator is designed around grade + start + end conditioning.

**Recommendation:** choose DistilGPT-2 as the first supported baseline, create one inference module used by CLI, API, and tests, and retire or clearly archive the older custom-model application path.

### 2. Validation currently uses the test file

`backend/training/finetune_huggingface.py` defaults `--val_csv` to `moonboard_test_quality.csv`, even though `moonboard_val_quality.csv` exists. This turns the test set into a model-selection set.

**Recommendation:** use train for optimization, validation for checkpoint/parameter selection, and test exactly once for the final report.

### 3. Split and grade inconsistencies weaken evaluation

Observed during this review:

| Split | Rows | Grades | Notes |
|---|---:|---:|---|
| Train | 6,610 | 9 | 20 duplicate route strings within the split |
| Validation | 828 | 11 | Contains grades `6B` and `8A`, absent from training |
| Test | 829 | 11 | Contains grades `6B` and `8A`, absent from training |

There are 3 exact route strings shared between train and validation and 4 shared between train and test. The train distribution is also highly imbalanced: `6B+` has 3,251 examples while `7C+` has 51.

**Recommendation:** rebuild all splits in one deterministic operation after normalizing routes. Group by a canonical route fingerprint so duplicates cannot cross splits. Either restrict validation/test to trained grades or label unseen grades as a separate out-of-distribution evaluation. Report macro-per-grade metrics so `6B+` cannot dominate the score.

### 4. The endpoint representation loses important information

Of the 6,610 training rows, 3,161 have more than one designated start hold. The current fine-tuning code treats only the first item in the reordered `full_path` as the start and the last as the end. That is not equivalent to the dataset's `start_holds` and `end_holds` fields and cannot represent two-hand starts or multiple finish holds.

**Recommendation:** preserve starts, intermediates, and ends as separate sets/sequences. A compact serialization is preferable to natural-language punctuation:

```text
<GRADE_7A> <START> A4 B4 <END> G17 <PATH> D7 F9 C12 G14 <EOS>
```

Add each grade, board hold, and section marker as an atomic token. This shortens sequences, eliminates coordinate-format parsing errors, and makes constrained decoding straightforward.

### 5. The current language-model objective is inefficient

- The corpus is padded to 256 tokens even though routes are short; dynamic batch padding would reduce wasted compute.
- Padding reuses GPT-2's EOS token. With the standard language-model data collator, this can mask genuine EOS targets and weaken stopping behavior.
- Loss is applied to the entire prompt as well as the generated middle section. For endpoint-conditioned generation, prompt tokens should normally have label `-100` so training focuses on the completion.
- `compute_metrics` returns an empty dictionary, there is no seed, and runs do not record a data fingerprint or Git revision.
- `eval()` is used to parse CSV cells; `ast.literal_eval()` or, preferably, JSON should be used.

**Recommendation:** add a dedicated PAD token, dynamic padding, completion-only labels, fixed seeds, resume-from-checkpoint, exact dependency versions, and a run manifest containing code revision, data fingerprint, arguments, metrics, and model ID.

### 6. Current evaluation measures loss, not route quality

Validation loss cannot answer whether a route is valid, novel, climbable, or appropriate for its requested grade.

**Recommended automatic metrics:**

- exact parse success;
- requested grade and endpoint match;
- percentage of generated coordinates that correspond to real holds;
- duplicate-hold rate;
- hold-count fit by grade;
- reachability violations and maximum move distance;
- exact training-set memorization rate;
- candidate uniqueness and pairwise diversity;
- grade accuracy from a separately trained lightweight grade classifier;
- latency and generation success rate.

Create a frozen evaluation manifest of grade/start/end prompts. Use fixed seeds and compare every model against simple baselines: nearest known route, frequency/Markov route, and the current model.

For product acceptance, add a small blind human review by experienced climbers or setters. Ask for climbability, flow, grade fit, and novelty on a 1-5 scale, and record rejection reasons.

### 7. The web UI is a good sketch but not yet a trustworthy route editor

- The board overlay divides the entire image into an 11 x 18 grid. The image includes labels and margins, so circles will not align precisely with the physical hold centers.
- The canvas sizes itself only on image load, not on later viewport changes, and does not account for high-DPI displays.
- Errors use `alert()`, there is no empty/skeleton/retry state, and unsupported grades are hard-coded in the client.
- The UI cannot select start/end holds, compare candidates, inspect warnings, or reproduce a route from a seed.
- Color is the main distinction between start/middle/end, which is weak for accessibility.

**Recommendation:** use a calibrated board coordinate map (or SVG overlay) and make the board the primary interactive control. Fetch supported grades from the API, let users click one or two starts and one or two finishes, and return three ranked candidates. Use shapes, labels, and sequence numbers in addition to color.

## Recommended modeling approach

### Baseline: structured DistilGPT-2 fine-tune

Keep DistilGPT-2 for the first measured baseline, but use atomic domain tokens and completion-only loss. Fine-tune the full model initially; the model is small enough and the dataset is small. Compare against frozen-backbone or LoRA only if cost or overfitting justifies it.

Generation should produce several candidates and pass them through hard validation:

1. Preserve requested start and end holds exactly.
2. Permit only the 141 observed MoonBoard 2016 holds.
3. Prevent duplicate holds unless a deliberate match is represented.
4. Enforce grade-conditioned hold-count limits.
5. Reject routes with severe reachability violations.
6. Rank survivors using model log probability, reachability, grade fit, novelty, and smooth upward progression.

Do not silently modify an invalid route into a valid-looking one. Return explicit diagnostics or regenerate; otherwise model quality cannot be measured honestly.

### Next experiment: hold-token transformer

Because this is a small discrete board, compare DistilGPT-2 with a small decoder-only transformer trained from scratch over roughly 160-180 domain tokens (141 holds plus grades and control tokens). It may train faster, produce fewer invalid sequences, and be easier to constrain. Promote it only if it wins on the frozen evaluation suite and human review.

### Longer-term model

The available data describes selected holds, not limb-by-limb movements. A true four-limb state model would require limb/action labels or reliable reconstructed supervision. Until that data exists, treat “reachability” as a candidate filter and score, not as ground-truth biomechanics. A graph-based route model is a sensible later experiment after the baseline is measurable.

## Modal training and serving plan

### Secret and account handling

Do **not** put Modal credentials, Hugging Face tokens, proxy tokens, or account details in Python, shell scripts, Markdown, `.env.example`, frontend variables, command history, or Git.

Preferred setup:

1. Authenticate locally with Modal's interactive `modal setup` flow. Its local profile stays outside this repository.
2. Create named secrets in the Modal dashboard or CLI only when a workload needs them, for example a `huggingface-secret` containing `HF_TOKEN` for a private model or upload target.
3. Reference only the secret name in code with `modal.Secret.from_name(...)`.
4. Use separate Modal `dev` and `prod` environments and separate secrets where appropriate.
5. Never send a Modal token to the React browser. `VITE_API_URL` is public configuration, not a secret. A private Modal endpoint must be called through a trusted backend-for-frontend, or use user authentication that issues short-lived access.
6. Redact environment variables and request headers from logs. Rotate any credential immediately if it is pasted into chat, committed, or printed.

No raw account secret is needed for implementation. Once the machine is authenticated, commands can use the local Modal profile without exposing its contents.

### Modal resources

- Pin a known-supported Python version (prefer 3.11 or 3.12) and exact versions of PyTorch, Transformers, Datasets, Accelerate, Pandas, and Modal in the image.
- Start the 82M-parameter baseline on a `T4`; use `L4` if profiling shows a worthwhile speedup. Do not start with an A100 for this dataset.
- Use separate persistent Volumes for Hugging Face cache and project outputs/checkpoints, for example `climb-path-hf-cache` and `climb-path-models`.
- Write checkpoints and the final model under a unique run ID. Explicitly commit the Volume after important checkpoints and at successful completion.
- Set a finite timeout, retries for infrastructure interruption, and resume from the latest valid checkpoint.
- Store the cleaned dataset either as a versioned build input for this small corpus or in a read-only dataset Volume. Record its SHA-256 fingerprint in the run manifest.

### Proposed Modal entry points

Create `modal_app.py` with three responsibilities:

- `train(config)`: GPU function that builds/loads the dataset, fine-tunes, evaluates validation prompts, and commits artifacts.
- `evaluate(run_id)`: CPU or low-cost GPU function that evaluates a saved model on the frozen test manifest and emits `metrics.json` plus sample routes.
- `api()`: ASGI/FastAPI app that loads one promoted model once per container and exposes health, metadata, and generation endpoints.

For local iteration use `modal run` for jobs and `modal serve` for the API; use `modal deploy` only after evaluation gates pass. Modal Volumes require commit/reload awareness when writers and long-lived inference containers need to see a new model.

Useful current documentation:

- [Modal Secrets](https://modal.com/docs/guide/secrets)
- [Modal Volumes and checkpointing](https://modal.com/docs/guide/volumes)
- [Modal GPU selection](https://modal.com/docs/guide/gpu)
- [Modal ASGI/Web Functions](https://modal.com/docs/guide/webhooks)
- [Modal environments](https://modal.com/docs/guide/environments)

## API design

Use one typed API contract shared conceptually by tests and frontend:

```json
POST /api/v1/routes/generate
{
  "grade": "7A",
  "start_holds": ["A4", "B4"],
  "end_holds": ["G17"],
  "candidate_count": 3,
  "temperature": 0.8,
  "seed": 12345
}
```

The response should include model/data versions and per-candidate diagnostics:

```json
{
  "request_id": "...",
  "model_version": "...",
  "data_version": "...",
  "candidates": [
    {
      "holds": ["A4", "B4", "D7", "F9", "G17"],
      "score": 0.86,
      "warnings": [],
      "seed": 12345
    }
  ]
}
```

Also provide:

- `GET /api/v1/health/live` without loading the model;
- `GET /api/v1/health/ready` confirming model readiness;
- `GET /api/v1/model` for supported grades, board version, model version, and constraints.

Validate coordinates and min/max relationships server-side. Configure CORS with exact deployed origins rather than `*`, add request timeouts and a conservative candidate limit, and avoid returning raw exception text to clients.

## Web-app UX plan

### Primary flow

1. User chooses a supported grade.
2. User clicks the board to select start hold(s), then finish hold(s).
3. User selects a “balanced / safer / creative” preset; raw temperature remains under Advanced.
4. The app generates three candidates and keeps the board visible while loading.
5. User compares candidates, warnings, hold sequence, and estimated characteristics.
6. User can copy a route, regenerate with the same endpoints, or share a URL containing non-secret route state and seed.

### Layout

- Desktop: sticky controls/candidate rail on the left and a large board on the right.
- Mobile: board first, bottom-sheet controls, swipeable candidate cards.
- Keep advanced controls collapsed. Replace the 14-button grade wall with a compact segmented selector or dropdown grouped by band.
- Provide a clear first-run empty state on the board rather than an unlabelled blank image.

### Board implementation

- Prefer an SVG overlay or calibrated normalized center coordinates for each physical hold. Do not derive centers from full-image grid cells.
- Render at device pixel ratio for sharp overlays and recompute on container resize.
- Make every valid hold keyboard-focusable or provide an equivalent accessible list.
- Use distinct marker shapes and text: `S1/S2`, numbered middle moves, and `T1/T2`, plus color.
- Support path toggling and candidate comparison without redrawing the base bitmap each time.

### Feedback and trust

- Replace browser alerts with inline errors that explain whether the API is unavailable, the model is warming up, or no valid route was found.
- Show a model warm-up state and preserve the user's selections during retry.
- Display validation warnings such as “large move” or “unusual hold count.”
- State that generated routes should be checked by a competent climber/setter before use.
- Collect optional thumbs-up/down plus structured reasons; do not automatically train on feedback until it has been reviewed and versioned.

## Proposed repository shape

```text
backend/
  api/
    app.py                  # FastAPI routes and schemas
  climb_path/
    schema.py               # canonical route/hold models
    serialization.py        # special-token representation
    constraints.py          # hard validation and diagnostics
    inference.py            # one model loading/generation path
  data_processing/
    build_dataset.py        # deterministic normalization and split
  training/
    train.py                # framework-neutral entry point
    evaluate.py             # frozen automatic evaluation
frontend/
  src/
    features/board/
    features/generation/
    api/
tests/
  unit/
  integration/
  fixtures/
modal_app.py
pyproject.toml
```

The Gradio app can remain as an internal model-debug tool, but it should call the same inference module as FastAPI. It should not be a third independent model implementation.

## Phased implementation plan

### Phase 0: make the baseline honest and runnable (1-2 days)

- Select DistilGPT-2 as the active baseline and remove stale imports/checkpoint assumptions from runtime paths.
- Add a single canonical `Route` schema and inference module.
- Fix the failing tokenizer test (currently expects vocabulary size 45; implementation reports 46) and convert it to normal pytest tests.
- Use the real validation split during training; reserve test for final reporting.
- Add smoke tests for imports, one preprocessing batch, one generation call, and API schema validation.
- Pin dependencies and Python version; add a license file if MIT is intended.

**Exit gate:** fresh checkout can run tests; CLI/API fail clearly when no model is installed; one tiny overfit run can generate a parseable route.

**Status (2026-10-11): complete.** The canonical schema, shared inference
module, validation split, pinned Modal runtime, tests, and explicit missing-model
behavior are in place. A smoke run and a three-epoch T4 baseline both completed
on Modal; the full run reached validation loss 0.4184 and is deployed at
`https://lczhang42--climb-path-api.modal.run`. The local checkpoint and matching
Hugging Face cache were removed. Model-quality work such as completion-only
loss, atomic hold tokens, split-leakage measurement, and benchmark comparison
remains in the later phases below.

### Phase 1: rebuild the data contract (2-3 days)

- Normalize holds to a single `A1`-style ID and preserve multiple starts/ends.
- Deduplicate before splitting and group canonical duplicates.
- Define supported grades explicitly; create a separate OOD set if desired.
- Produce versioned train/validation/test JSONL or Parquet plus a manifest with counts, distributions, and hashes.
- Build the frozen prompt evaluation set and nearest-route baseline.

**Exit gate:** zero route-fingerprint overlap across splits; all coordinates are known board holds; dataset manifest is reproducible from one command.

**Status (2026-10-11): complete.** The deterministic v1 contract contains
6,597 train, 825 validation, 825 test, and 8 OOD routes using 141 canonical
`A1`-style hold IDs. Twelve duplicate rows are removed before splitting, six
conflicting-grade groups are recorded and resolved deterministically, and the
manifest verifies zero cross-split fingerprint overlap. A frozen 128-prompt
evaluation set and nearest-route retrieval baseline are included; the latter
scores 0.1462 mean middle-hold F1.

### Phase 2: train and compare on Modal (2-4 days)

- Implement atomic domain tokens, dynamic padding, prompt masking, dedicated PAD/EOS, seeds, and checkpoint resume.
- Add Modal image, Volumes, train/evaluate entry points, and named-secret references.
- Run a tiny smoke job, then a full T4 baseline.
- Sweep only a small set of high-value parameters: learning rate, epochs/early stopping, grade sampling, and decoding temperature/top-p.
- Compare structured DistilGPT-2 against the nearest-route baseline and, if time allows, the small hold-token transformer.

**Exit gate:** a run is reproducible from its manifest; model passes validity gates and improves on baselines without excessive memorization.

**Status (2026-10-11): implemented and evaluated; promotion gate not passed.**
Atomic hold/grade tokens, completion-only loss, dynamic padding, deterministic
grade sampling, checkpointing, manifests, and Modal train/evaluate entry points
are complete. The selected three-epoch T4 run is 100% valid, parseable,
duplicate-free, and endpoint-preserving with 99.22% novel routes. Its frozen
middle-hold F1 is 0.1171, below nearest-route retrieval at 0.1462, so it has not
replaced the Phase 0 deployment. Full run evidence and the Phase 3 decision are
recorded in `PHASE2_RESULTS.md`.

### Phase 3: constrained inference API (2-3 days)

- Generate multiple candidates, validate, rank, and return diagnostics.
- Add deterministic seeds, request IDs, version metadata, structured logs, warm model loading, and readiness checks.
- Deploy a development Modal ASGI endpoint with exact CORS configuration and rate/candidate limits.
- Add an integration test against a local fake model and a protected end-to-end smoke test for the deployed dev endpoint.

**Exit gate:** 95%+ of normal requests return at least one valid candidate within the agreed latency budget; no raw secret or internal traceback reaches the client.

**Status (11 October 2026): complete for development.** The frozen suite reached
100% request success and 100% returned-candidate validity, with 167.7 ms mean
warm service latency. A separate Modal endpoint is deployed with constrained
multi-candidate generation, deterministic seeds, diagnostics, readiness checks,
rate/candidate limits, structured logs, and sanitized errors. Retrieval remains
the conservative first candidate because the learned ranker did not beat it;
model candidates are exposed as alternatives. See `PHASE3_RESULTS.md`.

### Phase 4: polished React web app (3-5 days)

- Calibrate the board overlay and implement responsive, accessible endpoint selection.
- Add candidate cards, loading/warm-up/empty/error states, presets, warnings, route details, and shareable non-secret state.
- Fetch capabilities from `/api/v1/model` instead of duplicating grades in JavaScript.
- Add component tests, API-contract tests, responsive visual checks, and basic keyboard/screen-reader QA.

**Exit gate:** a new user can select endpoints and compare valid routes on desktop and mobile without setup knowledge; overlays line up with physical holds.

**Status (17 October 2026): complete for local development.** The React route
studio now uses live model capabilities, calibrated and accessible endpoint
selection, candidate comparison, route diagnostics, cold-start/error states,
presets, and shareable non-secret URL state. Tests, lint, production build, live
Modal generation, and 1440 px/360 px visual checks pass. Public frontend hosting
and its exact production CORS origin remain a Phase 5 deployment task. See
`PHASE4_RESULTS.md`.

### Phase 5: production hardening (1-2 days)

- Separate Modal dev/prod environments and promotion by immutable model version.
- Add usage limits, latency/error dashboards, cost alerts, and a rollback procedure.
- Document model/data limitations, licensing, privacy, secret rotation, and deployment steps.
- Run the frozen test suite and human A/B review before each promotion.

## Suggested first release acceptance criteria

- No exact canonical route overlap across train/validation/test.
- 100% requested endpoint preservation.
- 100% generated holds exist on the selected MoonBoard layout.
- Less than 1% duplicate-hold violations before filtering; 0% in returned candidates.
- At least 95% request success across the frozen prompt suite after bounded regeneration.
- Exact training-route memorization is measured and below an agreed threshold.
- Median warm inference and cold-start latency are recorded, not guessed.
- UI works at 360 px mobile width and with keyboard-only input.
- No secrets in Git history, application bundles, logs, model artifacts, or documentation.

## Immediate task order

1. Repair the runtime split-brain and tests.
2. Rebuild/deduplicate the dataset with correct multi-start/multi-end semantics.
3. Add evaluation before spending on a full training run.
4. Implement and smoke-test the Modal training job.
5. Train the structured baseline and promote one immutable model version.
6. Deploy one inference API and connect the redesigned React client.

This sequence reduces the risk of paying for a training run whose data contract, target representation, or evaluation is invalid.
