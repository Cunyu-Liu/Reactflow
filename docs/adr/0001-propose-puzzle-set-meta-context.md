# ADR-0001: Propose puzzle-set meta-context as the next distinct point capability

## Status

Proposed, implementation-only, no training or score authority. V14 remains the
sole active experiment, and exact V14M3 PASS supersedes this proposal by routing
only to V14M4.

## Context

V11–V13 show that a focal-construct neural residual can improve feature41 but
adds almost no stable increment over its matched parent or nested null. Exact
mutant encoding, larger frozen foundations, structure/contact propagation,
additional shrinkage capacity and residual mixtures have already produced
negative or sub-threshold complete evidence.

Each OpenKnot puzzle nevertheless supplies eight outcome-blind WT constructs
from different design methods for one shared target. Existing rescue models
encode these constructs independently. If V14 identifies a point/transfer
bottleneck, the remaining untested information source is cross-construct
context inside the unseen puzzle.

Functional requirements:

- consume only WT-derived hidden states and observed masks;
- operate on exactly the registered eight-construct puzzle set;
- remain permutation equivariant over construct order;
- produce a focal-construct context usable by a feature41-anchored point head;
- handle the real zero-observed `P20_Eterna` construct without fabricating a
  reactivity target;
- exclude method ID, puzzle ID and every held target-side field.

Non-functional/scientific requirements:

- candidate and null have identical parameters, initialization, inputs,
  optimizer and downstream calibration;
- only cross-construct attention connectivity differs;
- no V14 artifact, authority, process or score is modified;
- implementation tests establish mechanics, not scientific performance.

## Decision

Implement a proposed `PuzzleSetMetaContextPointModel` with a shared
outcome-blind WT encoder and `PuzzleSetMetaContext` adapter:

```text
8 WT sequence/reactivity contexts
        │ shared outcome-blind WT encoder
        ▼
8 WT construct hidden sequences
        │ observed-mean pooling; all-position fallback only for P20_Eterna
        ▼
8 construct tokens + observed fractions
        │
        ├── candidate: full permutation-equivariant set attention
        └── matched null: identical attention with block-diagonal mask
        │
        ▼
focal mixed token + existing focal source/receiver/mutation/feature41 features
        │
        ▼
zero-initialized residual head → feature41 + residual
```

The block-diagonal null uses every trainable layer on the focal token but cannot
receive information from non-focal constructs. This identifies the
cross-construct capability rather than model size. Both arms replay feature41
exactly at initialization.

The current implementation materializes `6,071,729` parameters in each arm:
`4,767,280` in the shared-form WT encoder and `1,304,449` in the set mixer and
point adapter. These are implementation observations, not an active contract;
an eligible future amendment must freeze the final count before real-data
training.

The proposed training unit is one whole puzzle, not one pooled mutant table.
Within a puzzle, loss is averaged position within mutant, then mutant within
each of the eight method cells, then equally across those eight cells. Every
outer-train puzzle is visited once per epoch in a deterministic shuffled order.
Candidate and null reset the same Torch random stream before fitting, so model
connectivity—not cell order or dropout randomness—is the intended difference.

The proposed held path assembles all eight WT contexts once, encodes and mixes
them once per arm, then emits a prediction for every registered mutant and full
construct position. The prediction schema contains only biological keys,
registration status, fold/seed, feature41, candidate point and null point;
mutant target, target error, qualified mask, loss and score are scorer-only.

An implementation-only fold runner now exists, but real outer-train outcome
access remains fail-closed. It requires a future active task ID
`reactflow_delta_puzzle_set_meta_context`, the exact training token
`PUZZLE_SET_META_CONTEXT_REAL_DATA_TRAINING_ONLY`, held and partial scores closed,
and external outcomes locked. V14 authority cannot satisfy this predicate.

The implementation-only merger requires the future amendment to supply the
exact fold, seed, epoch and parameter-count universe. It rejects missing,
duplicate or unexpected fold-seed pairs, target-bearing predictions, changed
connectivity, incomplete histories, row misalignment, repeated biological keys
within a seed and absent checkpoints. It emits only a complete unscored merge;
no target scorer or Gate is active at this stage.

## Consequences

### Positive

- Tests the only registered outcome-blind input relationship absent from all
  previous rescue models.
- Supplies an exact parameter-matched attribution null.
- Directly targets unseen-puzzle transfer rather than adding focal capacity.
- Cleanly supports a future full model without changing the frozen evaluator.

### Negative

- The eight designs may be redundant or encode design-method composition rather
  than transferable puzzle biology.
- Set attention increases per-puzzle orchestration and memory, although the set
  length is fixed at eight and is negligible relative to position attention.
- Repeated development use means even a positive result remains post-hoc until
  independently confirmed.

### Neutral

- The adapter is not activated unless the mechanical post-V14 decision tree
  selects a point/transfer branch.
- Exact parameter counts, training epochs and top-journal Gate are frozen only
  in a future amendment after eligibility is established.

## Alternatives considered

**Another larger focal encoder or WT masking schedule**

Rejected: V14 already tests focal capacity plus task-matched WT pretraining, and
its contract terminates same-family iteration on failure.

**Method-ID or puzzle-ID conditioning**

Rejected: these identifiers can become shortcuts and do not identify a
transferable biological capability.

**More structure/contact or exact-mutant features**

Rejected: complete historical evidence and V13's matched-null result already
make these low-information repeats.

**Another Gaussian residual head**

Rejected as the default: point representation is upstream. A residual-only
model remains eligible solely under the separately frozen distribution-only
branch.

## Failure modes and handling

- If non-focal input gradients reach the block-diagonal null, the matched null
  is invalid and no real-data probe may run.
- If construct permutation changes the corresponding candidate output, method
  order has leaked into the model and the design is invalid.
- If zero-observed P20 cannot be represented without a fake target, exclude the
  adapter rather than inventing data.
- If a future complete candidate-minus-null effect misses its attribution Gate,
  terminate the puzzle-set family without connectivity, width or epoch search.

## References

- `docs/plans/2026-08-27-post-v14-model-contingency.md`
- `autoresearch/orchestrator-260827-v14-wt-profile/research.md`
- `docs/prospective_v2/model_rescue_v13_decision_ledger.yaml`
