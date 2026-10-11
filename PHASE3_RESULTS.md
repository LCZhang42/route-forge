# Phase 3: constrained inference API results

**Completed:** 11 October 2026

**Runtime:** Modal T4; model weights remain in the `climb-path-models` Volume

## Outcome

Phase 3 is deployed as a separate development API:

```text
https://lczhang42--climb-path-phase3-api.modal.run
```

The existing Phase 0 endpoint remains unchanged. The Phase 3 service generates
up to 12 samples, rejects invalid candidates, ranks distinct survivors, and
returns up to three routes with diagnostics. Candidate 1 is the validated
nearest-route baseline when it passes the constraints; later candidates are
novel model alternatives. This policy remains in place because the learned
ranker has not yet beaten retrieval on the frozen suite.

## What F1 means here

Middle-hold F1 compares the **set of predicted middle holds** with the set in a
held-out human route. It balances:

- precision: how many predicted holds occur in the reference route;
- recall: how many reference holds were predicted.

An F1 of `1.0` is an exact set match and `0.0` means no overlap. It ignores hold
order, body size, handedness, hold orientation, wall angle, and whether a human
can actually perform each move. It is therefore a useful reproducible similarity
metric, not a climbability certificate.

Phase 3 supplements F1 with board membership, endpoint preservation, duplicate
checks, grade-conditioned route length, movement-distance limits, upward
progression warnings, and training-route novelty. These are data-derived
heuristics. A valid response should mean “internally consistent and plausible
enough to show,” not “guaranteed safe or climbable.” Human review remains
necessary.

## Frozen evaluation

The final policy was evaluated once on the 128 frozen prompts after ranking was
selected using the validation split.

| Metric | Result |
|---|---:|
| Successful requests | 128 / 128 (100%) |
| Returned candidates | 3.0 mean |
| Valid returned candidates | 100% |
| Novel returned candidates | 86.20% |
| Candidate 1 middle-hold F1 | **0.1459** |
| Nearest-route F1 | **0.1462** |
| Best-of-three oracle F1 | **0.2357** |
| Retrieval used in returned set | 89.84% of requests |
| Mean warm service latency | 167.7 ms |
| Failures | 0 |

The 0.0003 top-1 gap occurs when the nearest retrieved route fails the Phase 3
physical constraints and is omitted in favor of a valid model route. The
best-of-three score is not an automatic production score: it uses the reference
answer to choose the best candidate after the fact. It does show that candidate
diversity is useful and motivates candidate comparison in the Phase 4 UI.

The live deployed smoke test passed. Its first request after scale-to-zero took
74.6 seconds including GPU cold start and model loading. Modal is configured to
scale the function to zero after five idle minutes; persistent Volume storage
remains even while no inference container is running.

Relevant Modal runs:

- [Validation ranking calibration](https://modal.com/apps/lczhang42/main/ap-8LJ8l7NgilsxyDo2cXk1jZ)
- [Final frozen evaluation](https://modal.com/apps/lczhang42/main/ap-gpekPwNhfErIj0mg3qZ0z5)
- [Current deployment](https://modal.com/apps/lczhang42/main/deployed/climb-path)

## API behavior

- `GET /api/v1/health/live` reports process liveness.
- `GET /api/v1/health/ready` reports model readiness.
- `GET /api/v1/model` returns model/data versions, supported grades, board holds,
  endpoint limits, and the active selection policy.
- `POST /api/v1/routes/generate` accepts one or two start holds, one or two end
  holds, a grade, an optional deterministic seed, and at most three candidates.
- Responses include a request ID, route source, score, full path, movement
  diagnostics, warnings, novelty, and fallback status.
- Requests have a 45-second in-service timeout, a per-container rate limit of 30
  requests per minute per client, exact CORS origins, and sanitized errors.

No Modal token, account secret, model checkpoint, or Hugging Face cache is stored
in this repository. Authentication remains in the user's Modal profile.

## Phase 4 handoff

The web app should make the source of each candidate visible, describe candidate
1 as the conservative recommendation, and show model candidates as alternatives.
Warnings must remain visible rather than being converted into a generic quality
score. The UI should also explain cold starts and must not imply that a generated
route is guaranteed safe.
