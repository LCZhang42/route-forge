# Route Forge frontend

Responsive React interface for selecting MoonBoard 2016 endpoints, generating
Phase 3 routes, comparing candidates, and inspecting movement diagnostics.

## Run locally

```powershell
npm install
npm run dev
```

Open `http://localhost:3000`. The default API is the deployed Phase 3 Modal
endpoint. Override it without placing secrets in the frontend:

```powershell
$env:VITE_API_URL = "https://example-api.invalid"
npm run dev
```

`VITE_API_URL` is a public browser-visible URL, not a credential.

## Checks

```powershell
npm test
npm run lint
npm run build
npm audit --omit=dev
```

The production build is written to `dist/`. Model weights are never downloaded
to the frontend or local project.

## API contract

- `GET /api/v1/model` supplies grades, board holds, model/data versions, and
  endpoint limits.
- `POST /api/v1/routes/generate` returns up to three validated candidates with
  route source, hold sequences, scores, and diagnostics.

The deployed API uses exact CORS origins. Add the eventual production frontend
origin to `CLIMB_ALLOWED_ORIGINS` before publishing the site.

See [`../PHASE4_RESULTS.md`](../PHASE4_RESULTS.md) for the UI design, board
calibration, test results, and deployment handoff.
