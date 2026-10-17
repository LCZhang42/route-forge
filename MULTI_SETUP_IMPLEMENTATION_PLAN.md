# Multi-setup MoonBoard research and implementation plan

**Prepared:** 17 October 2026  
**Status:** proposal; no external route dataset has been imported

## Recommendation

Expand Route Forge beyond the 2016 setup, but do it in two stages:

1. Support multiple **known official layouts** using a layout ID, board angle,
   versioned layout manifest, and a hard available-hold mask.
2. Add arbitrary/custom hold subsets later using a set or graph encoder.

For known MoonBoards, do not serialize all 130–198 available positions into
every text prompt. The list is constant for a layout, wastes context, and does
not tell the model which holds define the desired climb. Instead, treat the
available positions as a hard legality mask and keep grade plus start/finish
conditioning. A second “free setting” mode can let the model choose starts and
finishes itself.

## Official setups

Moon Climbing currently lists six unique setups:

| Setup | Board size | Supported angle(s) | Initial product priority |
|---|---|---|---|
| MoonBoard 2016 | Standard, 11 × 18 grid | 25° and 40° | Already supported |
| MoonBoard Masters 2017 | Standard, 11 × 18 grid | 25° and 40° | Later |
| MoonBoard Masters 2019 | Standard, 11 × 18 grid | 25° and 40° | Recommended next |
| Mini MoonBoard 2020 | Mini, 11 × 12 grid | 40° only | Data-source gap |
| MoonBoard 2024 | Standard, 11 × 18 grid | 25° and 40° | After 2019 |
| Mini MoonBoard 2025 | Mini, 11 × 12 grid | 40° only | After standard layouts |

Sources: [official setup overview](https://moonclimbing.com/moonboard),
[official MoonBoard FAQ](https://moonclimbing.com/using-your-moonboard), and
[official build guide](https://moonclimbing.com/build-your-moonboard).

The angle is a first-class condition, not display metadata. The same selected
holds can climb and grade differently at 25° and 40°. Mini boards also need a
different geometry manifest rather than a cropped standard-board image.

## Route-data findings and constraints

### Sources that exist

- The official MoonBoard app is the sanctioned complete problem source and
  stores problem and logbook data remotely. Moon Climbing does not document a
  public bulk training-data export. Its 2026 app migration also retires the old
  app/API after a transition period, so an undocumented endpoint is not a
  dependable production dependency. See the
  [official app migration notice](https://moonclimbing.com/News/post/moonboard-app-update-important-information).
- [Boardhang](https://boardhang.app/about) reports a curated community mirror
  of roughly 12,000 problems across 2016, Masters 2017, Masters 2019, 2024, and
  Mini 2025. It does not currently cover Mini 2020. Its application is MIT
  licensed, but its catalog is mirrored from Boardsesh and is explicitly
  unofficial.
- Older open-source API wrappers such as
  [spookykat/MoonBoard](https://github.com/spookykat/MoonBoard) require a
  MoonBoard account and target undocumented endpoints. They are useful for
  understanding historical schemas, not as a stable or automatically
  authorized ingestion method.
- Published research confirms that 2016, 2017, and 2019 datasets have been used
  together. It also finds materially worse performance on unseen layouts,
  showing why the model must know the setup identity. See
  [Board-to-Board](https://arxiv.org/abs/2311.12419).

### Rights gate

An open-source client license does not necessarily license the route catalog,
problem names, setter identities, or user-contributed grades for redistribution
or model training. Before importing anything beyond the current dataset:

1. Record the dataset owner, original source, exact license, permitted uses,
   attribution terms, and update method.
2. Ask Moon Climbing or the community dataset maintainer for written permission
   when the dataset license is absent or ambiguous.
3. Keep credentials and session tokens out of the repository and Modal images.
4. Do not scrape or depend on undocumented authenticated endpoints until the
   rights and acceptable-use questions are resolved.
5. Store raw source snapshots separately from the normalized contract and
   record checksums and provenance in every manifest.

The safest first acquisition options are an explicitly licensed export, a
maintainer-approved Boardhang/Boardsesh snapshot, or data exported by the user
under terms that permit this project’s use.

## Data contract v2

Split board geometry from routes.

```json
{
  "layout_id": "moonboard-2019",
  "display_name": "MoonBoard Masters 2019",
  "columns": 11,
  "rows": 18,
  "supported_angles": [25, 40],
  "holds": [
    {
      "position": "A5",
      "hold_id": "school-a-17",
      "hold_set": "school-a",
      "orientation_degrees": 30,
      "roles": ["start", "middle"]
    }
  ]
}
```

```json
{
  "route_id": "...",
  "layout_id": "moonboard-2019",
  "angle_degrees": 40,
  "grade": "7A",
  "start_holds": ["A5"],
  "middle_holds": ["C8", "F11"],
  "end_holds": ["G18"],
  "benchmark": false,
  "repeats": 42,
  "source": {"dataset": "...", "source_id": "..."}
}
```

Important invariants:

- Every route position must exist in its exact layout manifest.
- Start and finish roles must be valid for that setup.
- `layout_id + angle_degrees + route roles` defines the canonical fingerprint.
- Duplicate physical routes stay in one split, including copies at different
  angles where necessary to avoid leakage.
- Coordinate labels and physical hold identity are different concepts. `A5` on
  the 2016 and 2019 layouts is not automatically the same hold or difficulty.

## Generation design

### Product modes

**Guided mode — recommended default**

Inputs: setup, angle, target grade, one or two starts, one or two finishes.
The model generates only intermediate route holds. This preserves the current
useful workflow while the layout mask prevents impossible positions.

**Free-setting mode**

Inputs: setup, angle, target grade, and optional pinned/blocked holds. The model
generates start, middle, and finish roles. This is useful when the user wants a
complete suggestion rather than a path between chosen anchors.

**Custom/spray-wall mode — later**

Inputs include an arbitrary set of available holds and their geometry/features.
This requires a set/graph encoder; it should not be represented as a very long
DistilGPT-2 text prefix.

### First multi-layout model

Keep the current structured decoder for the first experiment and add atomic
conditions:

```text
<LAYOUT_2019> <ANGLE_40> <GRADE_7A>
<START> <H_A5> <END> <H_G18> <MIDDLE> ... <EOS>
```

At each decoding step, mask logits to the positions installed in the selected
layout and exclude already-used or role-invalid holds. Balance batches by both
layout and grade. Compare:

1. nearest-route retrieval within the selected layout;
2. one separate structured model per layout;
3. one shared model with layout and angle tokens;
4. a shared base with small layout-specific adapters if the shared model loses
   too much per-layout quality.

The shared model is preferable only if it matches the per-layout baselines. It
reduces Modal cold-start memory and can learn common spatial movement, but the
research evidence shows that layout-blind pooling harms generalization.

### Model for arbitrary available-hold sets

For a future custom-board feature, use a small encoder-decoder model:

- Encode every available hold as a feature vector: normalized `(x, y)`, hold
  identity/type, size, orientation, role eligibility, and wall angle.
- Process this unordered collection with a Set Transformer or graph neural
  network.
- Let a route decoder cross-attend to those encoded holds.
- Apply the availability mask again at output so invalid coordinates are
  impossible even if the model makes a bad prediction.

This architecture directly supports “generate from these available holds.” It
also separates geometry from a specific year, unlike a vocabulary made only of
setup-specific coordinate tokens.

## Frontend design

Add a setup selector before grade and endpoint controls:

1. Setup: 2016 / Masters 2017 / Masters 2019 / 2024 / Mini 2020 / Mini 2025.
2. Angle: only the angles supported by the selected setup.
3. Load the matching versioned board image and calibrated layout manifest.
4. Reset or remap endpoints when the setup changes; never retain a coordinate
   that is missing or role-invalid on the new layout.
5. Store `layout` and `angle` in the shareable URL.
6. Keep colored rings without sequence numbers or connecting lines.

The API should expose `GET /api/v2/layouts`,
`GET /api/v2/layouts/{layout_id}`, and a generation request containing
`layout_id` and `angle_degrees`.

## Evaluation

Report every result per layout and angle as well as overall:

- schema validity and available-hold compliance;
- requested endpoint preservation;
- route success rate after bounded sampling;
- duplicate/memorization rate;
- retrieval similarity and hold-set F1 as descriptive metrics only;
- reachability/geometry diagnostics;
- blinded human ratings for climbability, flow, grade fit, and novelty.

Reserve one layout as a transfer experiment, but do not ship zero-shot support
unless human review is strong. A route valid on the grid can still be poor
because hold shape and orientation differ.

## Implementation sequence

### Phase A — provenance and layout registry

- Request/confirm permission for one multi-layout data source.
- Create versioned manifests for all six official layouts from authoritative
  setup data.
- Refactor the 2016 UI and API to use a layout manifest without changing model
  behavior.

**Exit gate:** the existing 2016 app behaves identically through the new schema,
and every manifest position is visually audited against its setup.

### Phase B — add Masters 2019 at 40°

Masters 2019 is the recommended first expansion because it uses the same
standard board geometry, has known research/community data, and provides a
meaningful layout change without adding Mini-board geometry at the same time.

- Ingest and validate a licensed 2019 snapshot.
- Build per-layout splits and retrieval baseline.
- Add the setup switcher and 2019 board presentation.
- Ship retrieval-only 2019 support before claiming model generation quality.

**Exit gate:** no invalid coordinates, no cross-split route leakage, and the
2019 retrieval experience passes visual and human review.

### Phase C — model comparison

- Train separate 2016/2019 structured baselines on Modal.
- Train the shared layout-and-angle-conditioned model.
- Promote the shared model only if neither setup regresses materially.
- Otherwise retain one shared service with per-layout adapters or models.

### Phase D — remaining full and Mini layouts

- Add Masters 2017 and 2024 one at a time.
- Add Mini 2025 after the 11 × 12 rendering path is verified.
- Add Mini 2020 only after an explicitly permitted dataset is available.

### Phase E — custom available-hold generation

- Define hold-shape/orientation features and a custom-layout editor/importer.
- Train the set/graph-conditioned generator.
- Evaluate on held-out layouts and user-created subsets.

## Immediate next step

Do not start another paid training run yet. First obtain an explicitly usable
2019 dataset and create a machine-checked 2019 layout manifest. Then implement
the layout registry while keeping the current 2016 model and API stable. This
turns multi-board support into a controlled data-contract change rather than a
large simultaneous rewrite.
