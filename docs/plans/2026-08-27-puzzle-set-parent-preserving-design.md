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

### C. Freeze a qualified parent and train only a zero-initialized
cross-construct increment

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
   P1 arms. Freeze it and keep it in evaluation mode during point training.
3. Encode all eight outcome-blind WT construct contexts in their common
   full-sequence coordinate frame. The registered zero-outcome P20_Eterna
   construct remains present at all aligned positions but contributes no
   fabricated target.
4. At every position, candidate uses full permutation-equivariant attention
   across the eight constructs. The matched null uses block-diagonal self-only
   attention with identical inputs, parameters, initialization, optimizer,
   order, and epochs.
5. A zero-initialized incremental head receives focal V14 source/receiver
   features, signed distance, mutation identity, feature41, the frozen V13
   parent point, and the position-aligned mixed states at the mutation source
   and receiver.
6. The prediction is

   \[
   \widehat\Delta = \widehat\Delta_{V13,f,seed0} +
   g_{\mathrm{set}}(x).
   \]

At initialization \(g_{\mathrm{set}}(x)=0\), so both arms replay the V13
parent point exactly on every registered key. Only the set mixer and incremental
head are trainable. The parent point and frozen encoder cannot receive gradient.

P1 experiment seeds vary the new mixer/head initialization and training order;
the parent remains the predeclared V13/V14 seed-0 outer-fold checkpoint pair for
all P1 seeds. This makes the parent a fixed foundation rather than allowing a
post-hoc parent or seed selection.

## 4. Error and provenance rules

- A P1 fold must reject a parent checkpoint from another outer fold or any seed
  other than zero.
- The V13 and V14 source records must identify the same held puzzle as the P1
  fold and certify held-puzzle exclusion from their training path.
- The encoder import must cover exactly input projection, input norm, six
  context blocks, and output norm. Missing or shape-mismatched tensors are an
  implementation error.
- Prediction artifacts record both parent checkpoint paths and the maximum
  absolute initial replay difference, but contain no target, score, loss, target
  error, or qualified target mask.
- The active V14 authority cannot run P1. A future narrow amendment must freeze
  the P1 fold/seed/epoch/Gate universe before real training.

## 5. Verification required before any amendment

1. V14 encoder import produces exactly the same hidden state as
   `V14PointModel.encode(context, None)` in evaluation mode.
2. Candidate and null have identical state and parameter counts after import.
3. Both arms exactly replay an arbitrary frozen parent point before training.
4. Backpropagation changes only the set mixer and incremental head; the encoder
   and parent arrays remain unchanged.
5. Non-focal construct gradients are nonzero for the candidate and exactly zero
   for the block-diagonal null.
6. Construct-order permutation preserves the corresponding focal output.
7. Seven-cell P20 training retains all eight context constructs without
   inventing supervision.
8. The fold runner rejects mismatched parent fold/seed provenance and emits one
   complete target-free prediction universe.
9. The scorer remains closed until folds 0–19 have one complete prediction-only
   merge, training has been closed, and a future authority issues the exact
   score-once token.

The final position-aware operator and its coordinate audit are specified in
`docs/plans/2026-08-27-puzzle-set-position-alignment-design.md`.

## 6. Scientific interpretation

This design raises the probability of a real performance improvement without
making the architecture broader for its own sake. It preserves the strongest
known point level, reduces trainable capacity to the new capability, and keeps
the matched-null contrast identifiable. A future PASS would support the claim
that cross-construct WT context adds predictive information beyond a strong
focal parent. A failure would terminate puzzle-set meta-context rather than be
explained away as loss of the parent model during retraining.
