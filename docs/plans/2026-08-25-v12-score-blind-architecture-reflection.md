# ReactFlow-Delta score-blind architecture reflection before the V12 verdict

## Status and decision boundary

This document was written while V12M3 remained score-blind and before any V12
fold score was available.  It is an architecture reflection, not a contract,
candidate selection, Gate change, or training authorization.  The binding V12
amendment and the post-V12 diagnostic routing remain controlling.

- **Focal question:** if V12 does not pass, which missing capability is actually
  supported by the already-terminal V1--V11 evidence, and which additional
  evidence must exist before more model capacity is justified?
- **Decision owner:** project owner; execution is delegated to the model-rescue
  agent under the active amendment.
- **In scope:** point residual representation, legal shrinkage functions, and a
  frozen-point residual distribution.
- **Out of scope:** method or puzzle identifiers, new external outcomes,
  threshold relaxation, best-fold/seed selection, another rank search, larger
  encoder by default, or treating an oracle as model performance.
- **Current decision:** no new model is authorized.  V12 must first produce its
  complete twenty-puzzle verdict and, on FAIL, the pre-frozen D1--D4 diagnostic.

## Located evidence and actual architecture

### Located evidence

1. The terminal V11 diagnostic artifact is
   `/mnt/cunyuliu/reactflow_delta_model_rescue_v11/v11m3_screen_seed0/v11m3_postscreen_diagnostics.json`.
   It reports a stable association between the anchored neural residual and the
   held target residual (mean association `0.2218677`, positive in `20/20`
   puzzles).  The corresponding unanchored association is only `0.0724605` and
   is not directionally stable.  This is evidence for a real feature41-anchored
   residual signal, not evidence that every residual application is useful.
2. V11 improves signed-delta MAE over feature41 by `9.8041%` (`20/20` positive)
   and the anchored model improves over its parameter-matched unanchored null by
   `12.3351%` (`19/20` positive).  The feature41 anchor is therefore a necessary
   part of the current point estimator.
3. The V11 residual is harmful at edit distance zero (`-1.4280%`) and neutral
   when `abs(feature41)<0.05` (`-0.2730%`), but it improves signed MAE by
   `10.9658%` at distances 6--20 and `11.3720%` beyond 20.  Its gain is
   `8.51%--14.03%` in all three non-negligible feature41-magnitude bins.  The
   error is localized rather than a uniform lack of capacity.
4. Only `2/20` V11 folds show at least 1% final-window improvement and the
   median final-window decrease is `0.2225%`; simply training longer is not
   supported.  Only `3/20` folds show a train-to-held gain drop of at least five
   percentage points; uniform global shrinkage as an overfit remedy is also not
   supported.
5. The terminal distribution diagnostic shows stable 90% and 95% undercoverage
   and a stable lower/upper tail miss imbalance for the anchored distribution.
   It also shows that predicted scale tracks absolute residual error strongly
   (mean association `0.4856`, positive in `20/20`).  There is useful
   heteroscedastic information, but its tail allocation remains imperfect.

### Actual V11 capability

`scripts/reactflow_delta/model_rescue_v11.py::V11PointModel` encodes WT
sequence, reactivity, precision, observed mask, normalized position and region
with four width-192 relative-attention blocks.  For every mutation and receiver
position, it concatenates:

- source hidden state;
- receiver hidden state;
- normalized signed sequence distance;
- ref/alt one-hot identity;
- feature41 point.

A two-hidden-layer width-256 MLP predicts a residual, and the final point is
`feature41 + residual`.  This is already a substantial nonlinear model.  Its
main structural limitation is not raw parameter count: source and receiver
interact only after concatenation in a generic MLP.  It has no explicit
source-conditioned receiver modulation or relation/message representation.

`scripts/reactflow_delta/run_model_rescue_v11.py::fit_point_model` optimizes the
method-cell-balanced L1 objective with Adam.  The point model is frozen before
`scripts/reactflow_delta/model_rescue_v10.py::MedianAsymmetricResidual` is fit.
Thus point and probability optimization are already separated; a future
probability-only change must not retrain or move the point.

## Independent candidate hypotheses

The following ideas were generated before checking external modeling papers.
They are mutually conditional proposals, not a candidate menu to be searched.

### Idea A -- non-product monotone shrinkage

- **Idea:** replace the V12 product of two sigmoid factors with one four-parameter
  monotone interaction surface,
  `sigmoid(b + softplus(wd)x + softplus(wm)y + softplus(wi)xy)`, where
  `x=log1p(distance)` and `y=log1p(abs(feature41)/0.05)`.
- **Assumption:** the product form is too restrictive because it suppresses the
  residual whenever either coordinate is small, while the true reliability may
  be governed by an interaction or by either coordinate independently.
- **Prediction:** a fixed 4x4 held oracle materially improves both signed and
  point-absolute loss over a global oracle, and the learned V12 gate is poorly
  aligned with that oracle surface.
- **Falsifier:** global-to-4x4 headroom is below 1%, is not stable across at
  least 14/20 puzzles, or V12 is already within 1% of the 4x4 oracle.
- **Required contrast:** one prospectively frozen candidate versus the exact
  four-parameter V12 product gate; same V11 residual, inner-OOF data, optimizer,
  steps, folds, seeds and residual calibration.  It must not be accompanied by
  another gate family.
- **Authorization condition:** only `COORDINATE_LIMITED` from the complete D2/D3
  diagnostic.  A V12 near miss alone is insufficient.

### Idea B -- explicit source-to-receiver conditional interaction

- **Idea:** preserve the feature41 anchor and V11 WT encoder, but replace only
  the concat-only residual head with a source/mutation-conditioned affine
  modulation of receiver features, followed by a small relation head.  Capacity
  may increase here, but the encoder does not grow by default.
- **Assumption:** the residual represents a directed perturbation from an edit
  source to every receiver.  A generic concat MLP can approximate interactions,
  but an explicit conditional modulation may learn them more data-efficiently
  and make direction-specific effects identifiable.
- **Prediction:** after the best legally identifiable shrinkage is applied, a
  stable residual ceiling remains that cannot be explained by gate coordinates;
  within fixed regimes, the target residual remains associated with the V11
  residual, yet the train-only gate cannot transfer it to held puzzles.
- **Falsifier:** the 4x4 oracle itself cannot reach the point thresholds,
  residual association disappears after localization, or the observed failure
  is fully explained by the shrinkage surface.
- **Required contrast:** a parameter-matched concat residual null and a
  modulation residual candidate sharing encoder, feature41 anchor, loss,
  optimization, calibration and evaluation.  The additional parameters must
  be allocated between the two heads before training; a larger candidate versus
  the unchanged V11 head is not an interpretable contrast.
- **Authorization condition:** not authorized by V12 FAIL alone.  It requires a
  separate prospective amendment showing residual signal remains but neither
  gate capacity nor simple transfer diagnostics explain the shortfall.

### Idea C -- frozen-point tail distribution repair

- **Idea:** keep the final V12 point bitwise fixed and change only the residual
  distribution so that asymmetric tail mass/scale can be represented without
  changing the predictive median.
- **Assumption:** V11's stable upper-tail undercoverage and tail-miss imbalance
  persist after the V12 point is corrected; the current two-component
  median-constrained family lacks tail flexibility rather than scale signal.
- **Prediction:** every V12 point Gate passes, whereas CRPS or
  distribution-absolute Gate is the only failed headline family; D4 shows a
  stable tail effect with CI excluding zero and at least 14/20 puzzles in the
  same direction.
- **Falsifier:** any point Gate fails, the tail effect is unstable, or the
  candidate's scale/weight already tracks error without a systematic residual
  distribution defect.
- **Required contrast:** same frozen point and same calibration inputs, with a
  parameter-matched residual null.  CRPS is optimized directly; point gradients
  remain impossible.  No simultaneous point-model change is allowed.
- **Authorization condition:** only `DISTRIBUTION_ONLY_SUPPORTED` from D4.

## Adversarial review

### Against Idea A

A held 4x4 oracle is fitted in-sample and can exaggerate useful gate capacity.
Even a stable oracle ceiling does not prove an inner-OOF interaction gate will
transfer.  The candidate therefore remains deliberately four-parameter and
must beat the exact V12 parent, not only feature41.

### Against Idea B

V11's width-256 nonlinear head can already approximate multiplicative
interactions, and earlier low-rank/operator experiments did not identify a
stable low-rank increment.  Calling explicit modulation a new capability may
only relabel additional capacity.  This idea is the highest-risk route and must
use a genuinely parameter-matched null; otherwise it cannot support a method
claim even if its metric improves.

### Against Idea C

Improving interval coverage does not necessarily improve CRPS, and a richer
mixture can overfit the heavily reused 20-puzzle development set.  Stable
undercoverage in V11 is not enough: the effect must survive around the final V12
point, and only complete puzzle-level results can justify the route.

## External modeling check after independent generation

- Deep lattice networks demonstrate that flexible functions can retain formal
  monotonicity through constrained calibrators and lattices.  This supports the
  feasibility of Idea A but does not show that a more flexible gate will transfer
  on OpenKnot: [You et al., NeurIPS 2017](https://proceedings.neurips.cc/paper/2017/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf).
- FiLM implements conditioning through feature-wise affine transformations.
  This supplies a concrete mechanism for source/mutation-conditioned receiver
  modulation in Idea B, but its evidence comes from visual reasoning, not RNA
  mutation response: [Perez et al., 2017](https://arxiv.org/abs/1709.07871).
- Message-passing neural networks make sender, receiver and relation features
  explicit in the message function.  This supports the relational inductive
  bias behind Idea B, not the claim that RNA sequence distance is a valid graph:
  [Gilmer et al., ICML 2017](https://www.cs.toronto.edu/~gdahl/papers/gilmer17a.pdf).
- CRPS is strictly proper for ordinary real-valued distributional regression;
  improving a coverage diagnostic alone is not a substitute for improving the
  complete predictive distribution.  This supports keeping direct CRPS and the
  fixed point in Idea C: [Pic et al., 2023](https://arxiv.org/abs/2205.04360).

The literature check supplies implementation precedents only.  No task-matched
paper was located here that validates any of these three hypotheses for
full-construct 2A3 mutation response.

## Frozen judgment before the V12 outcome

1. Do not enlarge the V11 encoder or extend training merely because the current
   model is below the desired headline threshold; the terminal convergence and
   localization evidence contradict that diagnosis.
2. V12 remains the only active model experiment.  If it passes, run only the
   already-frozen V12M4.
3. On V12 FAIL, use D1--D4 first.  Advance at most one idea whose explicit
   authorization condition is met.  Do not compare Ideas A--C in one screen.
4. If no condition is met, terminate the V11/V12 residual family and return to
   the benchmark/measurement paper rather than manufacturing another rescue.
5. Any later candidate must improve both point and probabilistic evidence under
   the existing top-journal thresholds; parameter growth is allowed only when
   it adds an identified capability and is accompanied by a parameter-matched
   null.
