# Parent-preserving puzzle-set meta-context design

**Status:** `PRE_SCORE_IMPLEMENTATION_DESIGN_ONLY_NO_TRAINING_AUTHORITY`

**Evidence boundary:** this decision was written while V14M3 remained a
score-blind, incomplete twenty-fold run. No V14 loss, fold score, per-puzzle
effect, Gate direction, or external outcome was used.

## 1. Problem being fixed

The first puzzle-set implementation trained a 6.07-million-parameter encoder,
set mixer, and point head from scratch on only nineteen outer-train puzzles.
Although that implementation was a valid matched candidate/null probe, it
unnecessarily asked a new model to relearn the useful focal-construct point
predictor before it could test the genuinely new capability: conditioning one
construct on the other seven WT constructs in the same unseen puzzle.

This creates two avoidable risks. First, a cross-construct signal can be hidden
by variance or catastrophic loss of an already strong point predictor. Second,
a candidate-minus-null effect can be difficult to interpret if both arms also
learn a new focal encoder and focal head from scratch.

## 2. Alternatives considered

### A. Keep the from-scratch 6.07M implementation

Rejected. It is parameter-matched, but it provides no performance floor and
spends most of its statistical capacity relearning a capability that V11–V13
already established.

### B. Import V14 and fine-tune the complete model end to end

Rejected. It would preserve a useful initialization but could immediately
overwrite it. It also entangles WT-profile pretraining, focal point adaptation,
and cross-construct transfer, weakening both sample efficiency and attribution.

### C. Freeze a qualified parent and train only a paired-head cross-construct
increment

Selected. The parent point is the completed V13 seed-0 candidate from the same
outer fold. V13 is selected before V14 scoring because it has the strongest
known combined point profile: 9.8818% signed-delta improvement and 5.2556%
point-absolute improvement versus feature41. Reusing its frozen prediction does
not reopen the terminated exact-mutant mechanism claim; it treats V13 only as a
development baseline that the new capability must preserve and exceed.

The outcome-blind context encoder is initialized from the V14 seed-0 candidate
checkpoint from the same outer fold and then frozen. This choice is architectural,
not outcome-selected: P1 uses the checkpoint regardless of the eventual V14
score. The V14 encoder is exactly the 256-wide, six-block representation already
implemented by `OutcomeBlindWTEncoder`, and every source checkpoint is trained
using only that outer fold's nineteen training puzzles.

## 3. Frozen data flow

For outer fold \(f\):

1. Load `v13_candidate_point_fold{f}_seed0.pt` and compute a frozen parent point
   for every outer-train cell and every registered held-puzzle key.
2. Load the encoder subset of `v14_candidate_point_fold{f}_seed0.pt` into both
   P1 arms. Freeze it and keep it in evaluation mode throughout P1.
3. Encode all eight outcome-blind WT construct contexts in their common
   full-sequence coordinate frame. The registered zero-outcome P20_Eterna
   construct remains present at all aligned positions but contributes no
   fabricated target.
4. Initialize only the new set operator using the nineteen outer-train puzzle
   contexts. Deterministically mask 40% of one focal WT construct's observed
   positions, reconstruct them with a temporary 769-parameter decoder, and
   average L1 across eligible constructs and puzzles. Candidate may use the
   other seven aligned WT profiles; the null uses the same seven profiles at a
   fixed 17-position circular derangement. Both use identical masks, puzzle
   order, initialization, optimizer and 200-epoch budget. The frozen encoder
   and shared point head cannot change, and no mutant cell or outcome
   is accepted by the pretraining interface.
5. Freeze and remove the reconstruction decoder from downstream prediction.
   Require both arms still to replay the V13 parent within `1e-7`.
6. Operator `POSITION_ALIGNED_CROSS_CONSTRUCT_CONSENSUS_V5` uses the focal state
   as the one-token query and never includes it in K/V. Candidate K/V consists
   of seven correctly aligned non-focal individual states plus one summary
   without a dedicated learned token; the matched null uses the same
   eight-token support after shifting all seven
   non-focal streams by 17 positions. The attention output does not add the
   focal query as a residual. For both arms, a reference first zeros all seven
   non-focal hidden/reactivity/observed streams before learned projection while
   reusing the real focal query exactly. Actual and raw-zero reference traverse
   the same complete projection, attention, FFN and output-normalization path
   with one shared dropout draw; the returned state is `F(q,V)-F(q,V0)` and RNG
   advances as one `F` call. Inputs, parameters, initialization, optimizer,
   legal attention support, order and epochs are otherwise identical;
   attention-weight dropout is zero in both arms.
7. One shared point head receives `base=[focal V14 source/receiver, signed
   distance, mutation identity, feature41, frozen V13 parent]` and the
   zero-preserving nonfocal-dependent source/receiver states. The residual is
   `h(base,cross)-h(base,0)` under one shared dropout draw, so base-only and
   focal-only shortcuts cancel exactly.
8. The prediction is

   \[
   \widehat\Delta = \widehat\Delta_{V13,f,seed0} +
   g_{\mathrm{set}}(x).
   \]

At initialization \(g_{\mathrm{set}}(x)=0\), so both arms replay the V13
parent point exactly on every registered key. More strongly, zero cross context
always yields zero residual after training because both shared-head calls use
the same base and dropout realization. During WT pretraining only the set mixer
and temporary decoder are trainable. Point epoch 0 updates only the shared head
at learning rate `1e-3` while keeping context bitwise frozen; epochs 1–39 update
the head at `1e-3` and set mixer at `3e-4`. Adam has zero weight decay and
gradient clipping `5.0`. The parent point and frozen encoder cannot receive
gradient in either stage.

The V5 representation itself has the stronger invariant that raw non-focal
input `V=0` yields exact elementwise zero for every focal query and arbitrary
learned biases. Subtracting `F(q,V0)` removes projection biases, summary
constants and query-only attention/FFN/norm terms before the point-head contrast
is applied. A nonzero representation proves causal dependence on non-focal
inputs, not predictive usefulness.

P1 experiment seeds vary the new mixer/head initialization and training order;
the parent remains the predeclared V13/V14 seed-0 outer-fold checkpoint pair for
all P1 seeds. This makes the parent a fixed foundation rather than allowing a
post-hoc parent or seed selection.

The stage-specific trainable counts are explicit. Masked-WT initialization
updates 857,600 set-operator parameters plus the 769-parameter temporary
decoder, for 858,369 trainable parameters. Joint point fitting updates the
857,600 set parameters plus the 546,817-parameter shared paired head, giving the
frozen final model's 1,404,417 trainable parameters; the first warmup epoch
updates only those 546,817 head parameters. Residual calibration then freezes
the entire point model and trains only the predeclared residual head.

The point objective remains the original method-balanced signed-delta L1, and
the frozen-point residual objective remains the original puzzle-balanced CRPS.
CRPS gradients cannot return to the point model. MGDA, absolute-delta auxiliary
losses, learned objective weights and other multi-objective changes are reserved
for a separate amendment rather than confounded with the cross-context test.

After point fitting, a fixed outer-train-only capability-retention diagnostic
uses the same final frozen reconstruction decoder and deterministic mask epoch
`200` to evaluate initial, post-pretraining and post-point context snapshots.
It reports per-puzzle L1 and an equal-puzzle mean without reading the held puzzle
or any mutant outcome. The candidate must have both
`pretraining_established=true` and `retention_positive=true` to qualify; the
matched null is reported identically but its retention booleans are descriptive
only. The diagnostic cannot choose a checkpoint, learning rate, model or Gate.

## 4. Error and provenance rules

- A P1 fold must reject a parent checkpoint from another outer fold or any seed
  other than zero.
- The V13 and V14 source records must identify the same held puzzle as the P1
  fold and certify held-puzzle exclusion from their training path.
- The encoder import must cover exactly input projection, input norm, six
  context blocks, and output norm. Missing or shape-mismatched tensors are an
  implementation error.
- Prediction artifacts record both parent checkpoint paths and the maximum
  absolute initial and post-pretraining replay differences, but contain no
  target, score, loss, target error, or qualified target mask.
- Pretraining batches contain exactly `{puzzle, contexts}` for the outer-train
  puzzle IDs. The runner mechanically rejects a held puzzle, duplicate puzzle,
  mutant cell or target-bearing batch.
- Fold artifacts record the fixed mask rate, pretraining epoch/history/step
  count, eligible-construct counts, candidate/null decoder checkpoints and the
  bitwise-frozen encoder/point result. They also record the one-epoch head-only
  warmup, 39 joint epochs, separate head/context learning rates and update
  counts, plus the outer-train-only retention reports. The temporary decoders
  are frozen before point fitting.
- The active V14 authority cannot run P1. A future narrow amendment must freeze
  the P1 fold/seed/epoch/Gate universe before real training.

## 5. Verification required before any amendment

1. V14 encoder import produces exactly the same hidden state as
   `V14PointModel.encode(context, None)` in evaluation mode.
2. Candidate and null have identical state and parameter counts after import.
3. Both arms exactly replay an arbitrary frozen parent point before training.
4. Masked-WT pretraining uses identical masks and budgets for both arms, changes
   the set operator, leaves the encoder and point head bitwise unchanged, and
   preserves parent replay at `1e-7`.
5. The real P20 case may have seven eligible reconstruction targets while all
   eight constructs remain present as context; candidate and null must report
   the same eligibility set.
6. The temporary decoder is frozen downstream and never enters point or
   calibration prediction.
7. Point epoch 0 changes only the shared head at learning rate `1e-3`; the
   context path is bitwise unchanged. Epochs 1–39 update the head at `1e-3` and
   set mixer at `3e-4`; the encoder and parent arrays remain unchanged.
8. Every focal query excludes itself from K/V, attends to exactly seven
   non-focal individual tokens plus one non-focal summary, and is not added back
   as an attention residual. Candidate and null Q/K/V gradients are all finite
   and nonzero. Perturbing a non-focal WT value at the registered coordinate
   changes the candidate but not the null when the null's shifted-coordinate
   input is held fixed; this establishes dependence only.
9. The V5 raw-zero reference zeros all seven non-focal
   hidden/reactivity/observed streams before projection, reuses the focal query,
   traverses the full shared cross block, replays one dropout draw and advances
   RNG once. Raw-zero actual input cancels exactly despite arbitrary focal query
   and nonzero learned biases.
10. The paired shared head reuses one dropout draw, and
   `h(base,cross)-h(base,0)` is exactly zero whenever cross is zero in both train
   and evaluation modes after arbitrary head updates.
11. Construct-order permutation preserves the corresponding focal output.
12. Seven-cell P20 training retains all eight context constructs without
   inventing supervision.
13. The fixed epoch-200 retention diagnostic reads only outer-train
   `{puzzle, contexts}`, preserves model/decoder state, gradients and modes, and
   requires candidate `pretraining_established` and `retention_positive`; null
   retention is report-only.
14. The fold runner rejects mismatched parent fold/seed provenance and emits one
   complete target-free prediction universe.
15. The scorer remains closed until folds 0–19 have one complete prediction-only
   merge, training has been closed, and a future authority issues the exact
   score-once token.

If reference zeros are introduced after projection, the focal query differs
between paths, any layer is skipped, stochastic draws differ, RNG advances
twice, or raw-zero actual input does not cancel exactly, the operator is not V5
and real-data eligibility remains closed. The downstream point-head subtraction
cannot compensate for that upstream failure.

The final position-aware operator, raw-zero cancellation and coordinate audit
are specified in `docs/plans/2026-08-27-puzzle-set-position-alignment-design.md`
and `docs/plans/2026-08-27-puzzle-set-cross-only-retention-design.md`.

## 6. Scientific interpretation

This design isolates a new causal input path without making the architecture
broader for its own sake. It preserves the strongest known point level, reduces
trainable capacity to the new capability, and keeps the matched-null contrast
identifiable. Raw-zero cancellation or a nonzero counterfactual response is not
performance evidence. A future complete PASS would support the claim
that cross-construct WT context adds predictive information beyond a strong
focal parent. A failure would terminate puzzle-set meta-context rather than be
explained away as loss of the parent model during retraining.
