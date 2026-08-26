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

Implement a parent-preserving `PuzzleSetMetaContextPointModel`. The frozen V13
seed-0 candidate from the same outer fold supplies the point anchor; the frozen
encoder subset of the V14 seed-0 candidate from that fold supplies the
outcome-blind representation. Training is staged: masked-WT initialization
updates `PuzzleSetMetaContext` plus a temporary decoder; point fitting updates
`PuzzleSetMetaContext` plus its incremental head. The imported encoder and V13
parent remain frozen:

```text
frozen V13 outer-fold point ───────────────────────────────────────┐
8 WT sequence/reactivity contexts                                 │
        │ frozen outer-fold V14 outcome-blind WT encoder           │
        ▼
8 × 177 WT construct hidden states
        │ append observed flag; shared projection at each position
        ▼
177 aligned sets of eight construct states
        │
        ├── candidate: full cross-construct attention at each position
        └── matched null: identical self-only attention at each position
        │ outer-train masked-WT initialization of this set operator
        │ identical 40% masks/budgets; temporary decoder; no mutant outcomes
        │
        ▼
mixed source + mixed receiver + focal/mutation/feature41/parent features
        │
        ▼
zero-initialized incremental head → frozen V13 point + increment
```

The block-diagonal null uses every trainable layer at every focal position but
cannot receive information from non-focal constructs. Cross-construct attention
is position aligned: full-sequence position is the attention batch axis, so it
cannot mix different coordinates. This identifies the
cross-construct capability rather than model size. Both arms replay the frozen
V13 parent point exactly at initialization. V13 is fixed before V14 scoring
because it has the strongest known combined point profile; using its prediction
as an immutable parent does not reopen the terminated exact-mutant mechanism
claim.

The final pre-run representation also projects four missingness-aware aligned
WT statistics. Candidate statistics are leave-one-construct mean, spread,
support fraction and focal deviation across the other seven constructs. The
matched null uses the identical projection and parameter count but supplies
only focal value, zero spread, focal observed flag and zero deviation. This
turns the positive WT-only alignment diagnostic into a sample-efficient input
bias while preserving a strict no-cross-construct null.

The set operator is not left randomly initialized before the scarce
mutation-effect objective. Within each outer fold, both arms receive a fixed
200-epoch masked-WT stage over the nineteen outer-train puzzles. One eligible
focal construct at a time has 40% of its observed WT positions hidden using the
frozen V14 deterministic mask schedule. A temporary, exactly matched
769-parameter decoder reconstructs those values with L1 loss. Candidate may
draw on the other seven aligned WT profiles; null remains self-only. The V14
encoder and zero-initialized incremental point head are bitwise frozen, and the
pretraining API accepts only `{puzzle, contexts}`. Candidate and null use the
same mask, order, optimizer, epoch budget, decoder initialization and dropout
stream. The decoder is frozen and unused after this stage. Thus the stage
initializes the new cross-construct capability without reopening V14 focal
pretraining or exposing a mutant outcome.

The current implementation materializes `6,171,697` parameters in each arm:
`4,767,280` frozen V14 encoder parameters and `1,404,417` trainable
position-aware mixer and incremental-head parameters. The head receives the
aligned mixed states at both source and receiver plus the explicit parent-point
scalar. The V13 parent is evaluated outside the new module and is never
optimized by P1. The temporary 769-parameter reconstruction decoder is not part
of the final 6,171,697-parameter prediction model. These are implementation
observations, not an active contract; an eligible future amendment must freeze
the final count before real-data training.

The proposed training unit is one whole puzzle, not one pooled mutant table.
Pretraining visits every outer-train puzzle once per epoch, giving 3,800 WT-only
puzzle updates per arm at 200 epochs. Point loss is then averaged position
within mutant, mutant within each qualified method cell, and equally across
available cells. All eight WT constructs remain in the set context. The
registered zero-outcome P20_Eterna construct contributes outcome-blind WT
context but no fabricated supervised cell. Every outer-train puzzle is visited
once per point epoch in a deterministic shuffled order, giving 760 point
updates per arm at 40 epochs. Candidate and null reset the same Torch random
stream before each stage, so connectivity—not cell order, mask or dropout
randomness—is the intended difference.

The proposed held path first computes the same-fold V13 parent point, then
assembles all eight WT contexts once, encodes them once, and mixes each aligned
eight-construct position once per arm before emitting a prediction for every
registered mutant and full construct position.
The V14 encoder stays in evaluation mode throughout point fitting. After point
fitting, both complete incremental models are frozen.
Each arm then receives an exactly initialized copy of the V10
median-preserving asymmetric residual family, trained only on its outer-train
residuals with the same feature41 basis, frozen V8 direct features, optimizer,
epoch count and puzzle-balanced method-cell CRPS objective. Calibration cannot
move either point median. The prediction schema contains only biological keys, registration
status, fold/seed, feature41, frozen parent, candidate/null points and their distribution
weights, locations, scales and expected absolute delta; mutant target, target
error, qualified mask, loss and score are scorer-only.

An implementation-only fold runner now exists, but real outer-train outcome
access remains fail-closed. It requires a future active task ID
`reactflow_delta_puzzle_set_meta_context`, the exact training token
`PUZZLE_SET_META_CONTEXT_REAL_DATA_TRAINING_ONLY`, held and partial scores closed,
and external outcomes locked. V14 authority cannot satisfy this predicate.

The implementation-only merger requires the future amendment to supply the
exact fold, seed, pretraining/point/calibration epoch and parameter-count
universe. It rejects missing, duplicate or unexpected fold-seed pairs,
target-bearing predictions, changed connectivity, held-puzzle or mutant-outcome
pretraining access, unequal candidate/null pretraining budgets, wrong mask
rate, missing decoder evidence, missing same-fold seed-0 V13/V14 parent
checkpoints, initial or post-pretraining parent replay error above `1e-7`,
changed trainable counts, incomplete histories, row misalignment, repeated
biological keys within a seed, invalid mixture weights/scales, a shifted point
median and absent point, decoder or residual checkpoints. It emits only a
complete unscored merge. Implementation-only P1M3 and P1M4 scoring paths now
exist, but neither authority predicate can be satisfied by V14. P1M3 requires
training to be closed and the exact future
`PUZZLE_SET_COMPLETE_MERGE_SCORE_ONCE_ONLY` token. If that screen passes
exactly, the P1M4 path independently runs all 20 folds for seeds 0--4, assembles
candidate and matched-null ten-component equal-seed mixtures without scoring,
then requires training to be closed and the exact
`PUZZLE_SET_FORMAL_COMPLETE_SCORE_ONCE_ONLY` token. Partial scores and external
outcomes remain locked throughout. The target join therefore remains
impossible until a future amendment ratifies the model, complete universe and
unchanged top-journal Gate.

## Consequences

### Positive

- Tests the only registered outcome-blind input relationship absent from all
  previous rescue models.
- Supplies an exact parameter-matched attribution null.
- Directly targets unseen-puzzle transfer rather than adding focal capacity.
- Preserves the strongest known point level and reduces the trainable problem
  from 6.17M parameters to the 1.40M parameters that implement the new ability.
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
- The implementation fixes the realized parameter counts and proposes the
  unchanged top-journal margins; only a future amendment can ratify them and
  open real training.

## Alternatives considered

**Another larger focal encoder or focal WT masking schedule**

Rejected: V14 already tests focal capacity plus task-matched focal WT
pretraining, and its contract terminates same-family iteration on failure. The
selected outer-train masked-WT stage is narrower and different: it initializes
only the new cross-construct set operator against a self-only matched null while
the imported V14 encoder remains frozen.

**Train the full puzzle-set model from scratch**

Rejected: it would spend most of the nineteen-puzzle training signal relearning
the focal predictor and could hide a real cross-construct increment behind
variance or parent degradation.

**Fine-tune the imported V14 encoder end to end**

Rejected: it can overwrite the representation that P1 is meant to augment and
would entangle focal transfer with the cross-construct attribution contrast.

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
- If a mixed state at position `j` depends on a non-focal input at a different
  position `k`, the registered coordinate alignment is not being implemented.
- If zero-observed P20 cannot be represented without a fake target, exclude the
  adapter rather than inventing data.
- If either arm fails to replay its same-fold V13 parent at `1e-7` before the
  first optimizer step or after masked-WT pretraining, or if any frozen V14
  encoder or point-head parameter changes, no scientific run may proceed.
- If a pretraining batch contains the held puzzle, mutant cells or target-side
  fields; if candidate/null masks, eligible-construct counts or optimizer steps
  differ; or if the temporary decoder remains trainable downstream, no
  scientific run may proceed.
- If a non-focal WT-profile counterfactual changes the trained matched null, or
  does not change the trained candidate, the proposed capability is absent and
  no real-data probe may proceed. The same fail-closed decision applies if the
  cross-construct projection and attention do not receive finite nonzero
  gradients after the zero-initialized output layer has been bootstrapped.
- The WT-only registered-position audit is positive in all 20 puzzles for both
  pairwise and leave-one-construct consensus alignment increments, while mean
  pairwise sequence identity is only 0.511. The input-level termination branch
  is therefore closed; this is architecture support, not mutation-effect
  performance evidence.
- If a future complete candidate-minus-null effect misses its attribution Gate,
  terminate the puzzle-set family without connectivity, width or epoch search.

## References

- `docs/plans/2026-08-27-post-v14-model-contingency.md`
- `docs/plans/2026-08-27-puzzle-set-position-alignment-design.md`
- `autoresearch/orchestrator-260827-v14-wt-profile/research.md`
- `docs/prospective_v2/model_rescue_v13_decision_ledger.yaml`
