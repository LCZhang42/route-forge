# Phase 1 data contract

The versioned dataset under `data_contract/v1/` is the canonical input for
future training and evaluation. Holds use MoonBoard labels such as `A1` and
`K17`; coordinate arrays are no longer mixed into the persisted contract.

## Build

```powershell
python -m backend.data_processing.build_phase1_dataset
python -m backend.evaluation.nearest_route_baseline
```

Both commands are deterministic. The builder combines the legacy quality
splits, normalizes route roles, groups identical physical routes before
splitting, and resolves conflicting rows by community repeats, quality score,
and stable problem ID. The original split is retained only as source metadata.

## Route schema

Each JSONL record contains:

- `route_id`: stable ID derived from the physical route fingerprint;
- `fingerprint`: SHA-256 over sorted start holds, ordered middle holds, and
  sorted end holds; grade is intentionally excluded;
- `grade`: the selected supported grade label;
- `start_holds`, `middle_holds`, and `end_holds`: canonical hold IDs;
- `metadata`: source IDs/files, all observed grades, repeat total, and selected
  quality score.

Start and end holds remain arrays because real data contains multi-start and
multi-end problems. Code must not infer those roles from the first and last
hold in a flattened path.

## Splits and invariants

Seed 42 produces:

| Split | Routes |
|---|---:|
| Train | 6,597 |
| Validation | 825 |
| Test | 825 |
| OOD | 8 |

The supported distribution is `6B+` through `7C+`. The scarce `6B` and `8A`
records are isolated in `ood.jsonl`. The manifest verifies:

- 141 known MoonBoard 2016 holds;
- zero route-fingerprint overlap across all splits;
- 12 duplicate rows removed before splitting;
- multiple starts and ends preserved;
- SHA-256 checksums for every core artifact.

`frozen_evaluation.jsonl` contains 128 deterministic test prompts and their
references. It must not be regenerated selectively after model results are
known.

## Non-neural baseline

The nearest-route baseline retrieves from train using grade distance, endpoint
set distance, and stable route ID. Its Phase 1 frozen-set results are:

| Metric | Result |
|---|---:|
| Same-grade retrieval | 1.0000 |
| Mean endpoint distance | 1.3906 |
| Mean middle-hold F1 | 0.1462 |
| Exact middle match | 0.0078 |

Learned-model comparisons should use the same 128 prompts and report at least
valid-hold rate, endpoint preservation, duplicate rate, middle-hold F1, exact
match, novelty, and latency.
