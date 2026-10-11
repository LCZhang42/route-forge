# Phase 2: structured DistilGPT-2 results

**Run date:** 11 October 2026
**Execution:** Modal T4; model weights and Hugging Face cache remain in Modal Volumes

## Outcome

The Phase 2 training and evaluation pipeline is implemented and reproducible,
but the model-quality promotion gate did **not** pass. The best structured
DistilGPT-2 checkpoint produces valid routes reliably, yet it does not beat the
Phase 1 nearest-route retrieval baseline on the frozen 128-prompt suite.

For that reason, the Phase 0 production API has not been replaced. Phase 3
should treat the structured model as a candidate generator and add multi-candidate
decoding, physical validation, and ranking before another promotion decision.

## Implemented training contract

- Atomic tokens for 141 board holds, nine grades, route sections, PAD, and EOS.
- Correct preservation of multiple start and end holds.
- Completion-only loss: prompt and padding labels are masked with `-100`.
- Dynamic batch padding and a dedicated PAD token.
- Fixed model/data seeds, finite-loss guard, checkpoints, best-validation-loss
  restore, and resume support.
- Run manifests containing data hashes, runtime versions, hyperparameters, and
  training metrics.
- Optional deterministic inverse-frequency grade sampling without increasing
  the number of samples per epoch.
- Frozen-set evaluation with only known, unused hold tokens available during
  decoding. EOS is enabled after a train-derived grade-median route length.

## Selected checkpoint

| Item | Value |
|---|---:|
| Modal Volume path | `/models/phase2/structured-v1/final` |
| Base model | `distilbert/distilgpt2` |
| Epochs | 3 |
| Learning rate | `1e-4` |
| Batch size | 16 |
| Training rows | 6,597 |
| Validation rows | 825 |
| Training loss | 3.5004 |
| Validation loss | 2.7296 |

Frozen evaluation after the final decoding constraint:

| Metric | Structured model | Nearest-route baseline | Delta |
|---|---:|---:|---:|
| Mean middle-hold F1 | **0.1171** | **0.1462** | -0.0291 |
| Exact middle match | 0.0000 | 0.0078 | -0.0078 |
| Valid holds | 100% | n/a | n/a |
| Parseable routes | 100% | n/a | n/a |
| Duplicate-free routes | 100% | n/a | n/a |
| Endpoint preservation | 100% | n/a | n/a |
| Novel routes | 99.22% | n/a | n/a |
| Mean generated middle holds | 4.484 | n/a | n/a |
| Mean warm GPU latency | 35.2 ms | n/a | n/a |

The latency number measures generation inside an already-running T4 container;
it does not include Modal cold start or network time.

## Experiments

| Run | Configuration | Frozen mean F1 | Notes |
|---|---|---:|---|
| `structured-sweep-lr5e5` | 1 epoch, LR `5e-5` | 0.0511 | Early LR sweep |
| `structured-sweep-lr1e4` | 1 epoch, LR `1e-4` | 0.0713 | Selected learning rate |
| `structured-v1` | 3 epochs, LR `1e-4` | 0.1123 | Greedy unique-hold decoding |
| `structured-v1` | same checkpoint + length prior | **0.1171** | Best measured result |
| `structured-balanced-v1` | 3 epochs, equal expected grade mass | 0.1132 | Did not improve the aggregate score |

The one-epoch sweep predates the train-derived length constraint, so its scores
are useful for selecting the learning rate but are not a pure decoder comparison.

Relevant Modal runs:

- [Three-epoch selected checkpoint](https://modal.com/apps/lczhang42/main/ap-qoMvu27RIhlgMdOzhAb320)
- [Selected-checkpoint constrained re-evaluation](https://modal.com/apps/lczhang42/main/ap-IBJZ9BqC3JqHiSSc6YQNwh)
- [Grade-balanced training](https://modal.com/apps/lczhang42/main/ap-b6LW3W6EE0f50yOrtGm5eR)
- [Grade-balanced constrained re-evaluation](https://modal.com/apps/lczhang42/main/ap-cHGoXxdqYj5ob3cswJhsek)

## Interpretation and next decision

The model learned useful low-grade hold patterns, but quality declines sharply
for scarcer high grades. Equal-grade sampling changed the score only marginally,
which suggests that imbalance is not the only limitation. Enforcing a realistic
minimum length corrected early EOS behavior but also did not close the retrieval
gap.

The next useful experiment is not another blind epoch sweep. Phase 3 should:

1. generate several deterministic candidates per request;
2. reject invalid, repeated, and severely unreachable moves;
3. rank survivors by model log probability, reachability, upward progression,
   grade-conditioned length fit, and novelty;
4. compare the ranked result and retrieval fallback on the same frozen prompts;
5. preserve the nearest-route baseline as a fallback until the learned path wins.

No checkpoint is stored in this repository or required on the local machine.
