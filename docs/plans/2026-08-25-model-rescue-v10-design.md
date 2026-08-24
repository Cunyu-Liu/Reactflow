# ReactFlow-Delta Model Rescue v10 design

Status: `DESIGN_FROZEN_AFTER_POST_V9_DIAGNOSTIC; NOT_EXECUTION_AUTHORITY`

## 1. Decision and alternatives

V9 fixed the main failure of the earlier joint mean/calibration models: it
preserved the V8 L1 point, improved signed-delta MAE by 8.36%, recovered
distribution-derived absolute magnitude by 9.81%, and improved CRPS by 4.23%.
It failed only the pre-frozen 5% CRPS threshold. The complete post-Gate
diagnostic found a replicated positive residual mean-minus-median gap and
positive quantile asymmetry in 20/20 held puzzles. Therefore the remaining
error is not well described as symmetric zero-mean noise around the L1 point.

Three approaches were considered:

1. **Increase only symmetric residual capacity.** This is the necessary null:
   it tests whether V9 was limited by raw, poorly conditioned inputs and a
   3,011-parameter head. It cannot express the observed skew, so it is not the
   final hypothesis.
2. **Median-preserving asymmetric Gaussian mixture.** Recommended. It keeps the
   frozen V8 output as the 0.5 quantile, adds trained direct representations and
   train-only input standardization, and permits asymmetric component
   locations under an exact mixture-median constraint. A parameter-matched
   symmetric model identifies the incremental effect of asymmetry.
3. **Unconstrained mixture, normalizing flow, or larger backbone.** Rejected for
   this iteration. These could move the point functional, confound capacity
   with skew, and make a positive result scientifically uninterpretable.

## 2. Frozen architecture and data flow

The V8 MeanAligned encoder and direct point head remain frozen. For every
outer fold, the calibration input is exactly 244 features: feature41, the
frozen point and its absolute value, and 201 detached features from the path
actually trained by V8 (`source hidden 96 + receiver hidden 96 + signed
distance 1 + mutation one-hot 8`). Randomly initialized unused V8 branches are
forbidden. All 244 inputs are standardized using outer-train-only mean and
scale and the same transform is applied to held rows.

The identification ladder is fixed:

- historical `V9-SmallSymmetric`, raw 43 inputs, hidden 64;
- `V10-CapacitySymmetricNull`, 244 standardized inputs, hidden 256, three
  outputs, 63,491 parameters, both locations equal the frozen point;
- `V10-MedianAsymmetric`, identical inputs and budget, four outputs, 63,748
  parameters; the additional 257 parameters only allocate component CDF mass.

The asymmetric head uses two Gaussian components. Mixture weight is bounded to
`[0.1,0.9]`. With fixed numerical interior `eps=1e-4`, one output selects a
valid component CDF value `a`; the other is determined analytically as
`b=(0.5-w*a)/(1-w)`. Residual locations are
`-sigma_1*Phi^-1(a)` and `-sigma_2*Phi^-1(b)`, so the mixture CDF at zero is
exactly 0.5. Initialization `a=b=0.5` exactly recovers the symmetric null.

Both corrected feature41 and MeanAligned point predictors receive symmetric
and asymmetric heads with the same input permissions, initialization of common
parameters, optimizer, 40 epochs and method-balanced CRPS. This prevents an
unfair uncertainty-family advantage in the task-level comparison.

## 3. Scientific comparisons and Gate

The final candidate is fixed before training as MeanAligned-MedianAsymmetric;
the symmetric model cannot be promoted after outcomes. Required comparisons:

- asymmetric versus capacity-symmetric at the same MeanAligned point isolates
  asymmetric locations;
- capacity-symmetric versus historical V9 isolates conditioning/capacity;
- MeanAligned-asymmetric versus feature41-asymmetric is the fair task-level
  comparison;
- the frozen V8 point replay proves signed-delta predictions did not move.

The seed0 20-fold screen must satisfy all of the following: task-level signed
MAE gain at least 5%, distribution absolute-delta MAE gain at least 5%, and
CRPS gain at least 5% versus the equi-calibrated feature41-asymmetric
comparator; puzzle-level CI lower above zero; at least 16/20 positive puzzles
for signed and CRPS and at least 16/20 for magnitude; candidate CRPS gain at
least 1% versus historical V9 with CI lower above zero; asymmetric CRPS gain at
least 1% versus its parameter-matched symmetric null with CI lower above zero
and at least 14/20 positive puzzles; all three headline effects remain positive
under leave-one-puzzle-out; no puzzle contributes more than 20%; coverage is
100%, failure is zero, and 68%/95% coverage error is not more than two
percentage points worse than the fair feature41 comparator.

Failure of the asymmetric-versus-symmetric contrast closes the asymmetry claim
even if the larger head improves over V9. Failure of any headline Gate stops
V10 before multi-seed confirmation. Only a complete PASS opens fixed residual
seeds 0-4; the V8 point remains frozen in every seed. No external outcome is
accessed and no V10 result can establish SOTA or publication readiness without
a separate external amendment.

## 4. Implementation and verification boundary

Implementation is limited to the two fixed residual heads, train-only input
standardization, extraction of trained-path V8 features, prediction-only fold
artifacts, complete merge, scorer and mechanical qualifier. Tests must prove:
exact V8 point replay; identical common initialization; exact symmetric nested
null at allocation zero; mixture CDF at the frozen point equals 0.5; finite
gradients at the CDF bounds; train-only standardization; no target in the
prediction path; complete registered output; fair feature41 and MeanAligned
head families; and rejection of duplicate or incomplete folds.

Training is score-blind until all 20 folds exist. Candidate, width, features,
component count, epochs, optimizer and Gate cannot change after any fold is
run. GPU0-7 may be used where memory permits without preempting unrelated jobs.
