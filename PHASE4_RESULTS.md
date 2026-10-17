# Phase 4: React route studio

**Completed:** 17 October 2026

## Outcome

The placeholder frontend has been replaced with a responsive React route studio
that uses the deployed Phase 3 API. It is implemented and verified locally but
is not yet published to a public frontend host. The API remains at:

```text
https://lczhang42--climb-path-phase3-api.modal.run
```

## User experience

- Fetches supported grades, model/data versions, limits, and the physical
  140-handhold runtime catalog from `GET /api/v1/model`; grades are not
  duplicated in frontend code.
- Provides one- or two-hold start and finish selection using calibrated board
  controls or native keyboard-accessible selects.
- Includes three route presets and reproducible seed/variety controls.
- Explains Modal warm-up and gives dedicated connecting, long-wait, retry,
  empty, generating, success, and sanitized error states.
- Compares one conservative retrieval route with up to two model alternatives.
- Shows route provenance, grouped route holds, longest move, novelty, and all backend
  warnings without claiming that a route is guaranteed climbable.
- Names the exact endpoints and grid distance of each unusually long move, and
  explains that the warning marks the longest 5% of recorded moves for the
  selected grade.
- Lets users switch the active candidate and see start, intermediate, and finish
  holds as transparent rings. The wall deliberately omits connecting lines and
  sequence numbers so it identifies the problem without prescribing beta.
- Maps the legacy screenshot/LED row values to the physical hold one row above,
  removes the extra white marker halo, and limits endpoint selection to the 24
  verified 2016 start positions and nine valid top-row finish positions.
- Excludes legacy internal `A5` (displayed as physical `A6`) at both the API and
  UI boundary. It appeared in only one malformed three-start source route and
  does not correspond to an installed 2016 handhold; its tokenizer entry is
  retained only so the existing checkpoint remains loadable.
- Stores only grade, endpoint labels, and seed in the URL for shareable,
  non-secret setup state.

## Board calibration

The previous canvas divided the complete photograph into an 11 × 18 grid. That
placed markers in the image margins rather than over physical holds. Phase 4
calibrates the usable MoonBoard grid to the photographed column and row centers:

| Edge | Image position |
|---|---:|
| Column A | 14.3% from left |
| Column K | 91.3% from left |
| Row 18 | 8.9% from top |
| Row 1 | 94.4% from top |

Multi-start and multi-finish routes are drawn as branches, so simultaneous
start or finish holds are not incorrectly rendered as climbing moves.

## Verification

- `npm test`: six component, state, and API-contract tests pass.
- `npm run lint`: zero warnings or errors.
- `npm run build`: production Vite build succeeds.
- Live browser QA generated three real Phase 3 candidates and verified the
  selected route against the board photograph.
- Responsive QA passed at 1440 px and 360 px with no horizontal overflow.
- The 360 px layout preserves setup → comparison → board reading order.
- Endpoint selection has visible focus styles, semantic labels, radio/pressed
  states, native select fallbacks, and reduced-motion support.
- `npm audit --omit=dev`: zero production dependency vulnerabilities.

The older Node 18-compatible Vite/Vitest development toolchain reports five
development-only advisories. npm's automated fix requires breaking upgrades to
Vite 8 and Vitest 5, so those upgrades are deferred to Phase 5 with a Node
runtime upgrade rather than forced into this release.

## Deployment handoff

Before public frontend deployment:

1. choose the frontend host and URL;
2. add that exact origin to `CLIMB_ALLOWED_ORIGINS` for the Modal API;
3. build with `VITE_API_URL` pointing to the Phase 3 URL;
4. run the live browser smoke test from the deployed origin;
5. keep the local-host origins only for development.
