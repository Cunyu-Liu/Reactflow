# ReactFlow-Delta Model Rescue v3 autoresearch

## Goal

Produce a frozen candidate that passes the original R2M4 five-seed, 20-fold dual-primary
Gate against the same B1 comparator. Model Rescue v2 remains terminal and immutable.

## Baseline

- Candidate: Model Rescue v2 MeanAligned + zero-mean calibrated residual.
- Seed-0 signed-delta relative gain: 0.883309% (16/20 puzzles positive).
- Seed-0 CRPS gain: +0.00547832 (20/20 puzzles positive).
- Qualification: calibration-only; Mean Gate failed the frozen 1% threshold.

## Pinned success predicate

The work is complete only when the frozen five-seed 20-fold qualification JSON reports
`R2M4_POST_HOC_DEVELOPMENT_PASS` and every original R2M4 guardrail is true. A seed-0
screen improvement is only a working signal, never completion.

## Iteration policy

1. One focused hypothesis per iteration.
2. Keep only if seed-0 signed-delta relative gain improves and the CRPS/calibration,
   target-invariance, coverage and failure guards remain valid.
3. Discard regressions; do not lower Gates or change the comparator.
4. After selecting one candidate, freeze it before the five-seed formal run.
5. All evidence remains consumed-development unless a separate external amendment exists.

## Cycle 0 hypothesis

The contracted training hierarchy averages positions within mutant before method-cell
aggregation, but evaluator_v2 currently pools all positions within a method. Quantify
the mismatch on the complete v2 OOF ledger. If the contracted estimator materially
changes the result, build evaluator_v3 and realign both training and scoring before
testing a model change. If not, prioritize a train-only residual mean adapter.

## Cycle 0 result — discarded as a performance explanation

- The implementation mismatch is real, but it is numerically inert on the complete
  seed-0 OOF universe: the contracted mutant-balanced and implemented position-pooled
  estimands both give 0.883309% signed-delta relative gain and +0.00547832 CRPS gain.
- The recomputed implemented estimator matches the frozen qualifier to
  `1.83e-11`; this is not an artifact parsing discrepancy.
- Therefore an evaluator-only change cannot rescue the model and is not retained as
  the next performance intervention.
- The complete OOF error atlas localizes the limiting behavior:
  - large-effect positions (`|delta| > 0.20`) regress by 2.5066% signed-delta MAE;
  - near-zero positions improve by 34.7801% and moderate positions by 2.3233%;
  - `gRNAde-no3d`, `gRNAde`, and `Starting sequence` regress, while the other methods
    improve by roughly 2.25% to 3.87%;
  - mutation substitution identity is not the dominant failure axis.
- Interpretation: pure L1 MeanAligned primarily learns a shrinkage/conditional-median
  solution. It is useful in the dense near-zero regime but gives back B1's tail skill.
  Zero-mean residual calibration cannot repair a biased point mean.

## Cycle 1 hypothesis

A method-conditioned convex expert mean can preserve B1 in the methods/tails where
MeanAligned regresses and use MeanAligned where it wins. Before any new training,
run a complete-OOF diagnostic with alpha learned only from the other 19 puzzle folds:

`mu_blend = (1 - alpha_method) * mu_B1 + alpha_method * mu_MeanAligned`.

Shift the already-fitted zero-mean residual component locations by
`mu_blend - mu_MeanAligned`, leaving scales and weights unchanged. Test both a global
alpha and method-specific alphas. This probe is only a go/no-go estimate because its
base OOF experts were not trained as a nested meta-learning procedure; it cannot be
used as Gate evidence. Keep this direction only if the leave-one-puzzle probe exceeds
1% signed-delta relative gain and preserves positive CRPS gain. A kept implementation
must fit the gate inside each outer-training fold and be rerun end to end.

## Cycle 1 result — global blend kept; method-conditioned result ineligible

- Global leave-one-puzzle alpha probe: 1.129861% signed-delta relative gain,
  17/20 positive puzzles, +0.00652286 CRPS gain and 20/20 CRPS-positive puzzles.
- Method-conditioned probe: 1.724694% signed-delta relative gain, 19/20 positive
  puzzles, +0.00759215 CRPS gain and 20/20 CRPS-positive puzzles.
- The method alphas are stable and mechanistically coherent: all available folds select
  alpha=0 for `Starting sequence`, `gRNAde`, and `gRNAde-no3d`; Rosetta/Shujun are near
  alpha=1. This confirms expert complementarity rather than a one-puzzle accident.
- The frozen endpoint-v7 legal inputs do not include design-method provenance. The
  method-conditioned variant is therefore excluded from candidate eligibility: using
  it would change the prediction task and make comparison with B1 unfair.
- The global blend uses no new predictor input and is retained as the current working
  incumbent. The probe remains meta-cross-fold and is not Gate evidence; alpha must be
  made fold-legal and the candidate rerun end to end.

## Cycle 2 hypothesis

The tail failure can be gated using legal, outcome-blind expert outputs rather than an
illegal method label. Probe a low-capacity blend in which alpha depends only on frozen
B1/Mean predictions, especially `abs(B1 signed-delta)` and expert disagreement. Fit all
thresholds/weights on the other 19 puzzle OOF cells and apply them to the held puzzle.
Keep only if it materially exceeds the 1.129861% global-blend incumbent while CRPS
stays positive in at least 12/20 puzzles. Any kept version must later fit the gate on
outer-train outcomes only; this probe itself remains non-qualifying.
