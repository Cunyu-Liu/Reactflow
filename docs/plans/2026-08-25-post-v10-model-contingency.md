# Post-V10 model contingency — pre-score decision register

**Status:** `PRE_SCORE_CONTINGENCY_ONLY`

**Frozen before:** any V10M3 score or Gate direction is read

**Authority effect:** none; this document does not authorize V10M3, V10M4,
V11 training, external outcome access, or candidate selection.

## 1. Focal question and scope

**Question.** If the fixed V10 median-constrained residual model does not pass
the top-journal development screen, which failed scientific capability—not
which attractive module—should determine the next model-level experiment?

**Purpose.** Prevent outcome-driven architecture shopping while preserving a
credible route to improve point and probabilistic performance before
submission.

**Decision owner.** Project owner; execution delegated to Codex under the
existing instruction to continue model improvement without repeated approval.

**In scope.** Point representation, residual distribution family, calibration,
capacity-matched nulls, optimization/functional alignment, and heterogeneity
that changes a model decision.

**Out of scope.** Lowering V10 Gates, post-hoc seed selection, method-ID
shortcuts, random untrained branches, new external outcomes, changing split_v4,
or presenting development-consumed evidence as external/SOTA confirmation.

## 2. Located evidence and unresolved assumptions

### Located evidence

- `LOCATED_EVIDENCE`: corrected feature41 signed-delta MAE is approximately
  0.19116.
- `LOCATED_EVIDENCE`: frozen V8 MeanAligned improves signed-delta MAE by 8.36%
  on 20/20 puzzles but its point-derived absolute-delta MAE is worse, showing
  that the conditional median is not a conditional magnitude estimator.
- `LOCATED_EVIDENCE`: V9 zero-mean residual calibration improves
  distribution-derived absolute-delta MAE by 9.81% and CRPS by 4.23%, but misses
  the pre-frozen 5% top-journal CRPS margin.
- `LOCATED_EVIDENCE`: post-V9 held residual mean-minus-median and normalized
  quantile asymmetry are positive with puzzle-level CIs excluding zero and
  20/20 direction consistency.
- `LOCATED_EVIDENCE`: V10 therefore tests a matched-capacity symmetric null and
  an exactly median-preserving asymmetric-location model. It does not retrain
  the V8 point.

### Assumptions that V10 will test

- `ASSUMPTION A1`: residual asymmetry supplies incremental CRPS beyond matched
  scale-mixture capacity.
- `ASSUMPTION A2`: V8 point quality plus a stronger residual family is enough to
  exceed a fair feature41 distribution by at least 5% CRPS.
- `ASSUMPTION A3`: gains are broad across puzzles rather than driven by one
  construct family.
- `ASSUMPTION A4`: calibration can remain acceptable without moving the frozen
  median.

## 3. Independently generated model-level directions

These are proposals, not findings or authorized candidates.

### I1 — Matched symmetric capacity route

- `IDEA`: retain the 244-dimensional trained representation and hidden-256
  residual head but remove asymmetric locations.
- `PREDICTION`: matched symmetric V10 should improve historical V9 even if the
  asymmetric increment is absent.
- `FALSIFIER`: capacity-symmetric does not improve V9 CRPS with a positive
  puzzle-level CI.
- `ALTERNATIVE_EXPLANATION`: any improvement may come from more parameters or
  direct features, not a publishable residual mechanism.

### I2 — New mean representation under a fixed median objective

- `IDEA`: if distribution modeling works but task-level CRPS remains limited,
  improve the conditional-median representation while keeping mean-first,
  calibration-second separation. A future amendment may compare one larger
  point backbone against a parameter-matched old-backbone null; the residual
  family must be frozen across both.
- `PREDICTION`: signed-delta MAE and fair task CRPS improve together while the
  residual head's incremental effect remains stable.
- `FALSIFIER`: a stronger point representation fails to improve signed-delta
  MAE, or CRPS gains disappear under a shared residual family.
- `ALTERNATIVE_EXPLANATION`: extra capacity may memorize the 20 development
  puzzles; no publication claim is allowed without new independent evidence.

### I3 — Magnitude/sign functional factorization

- `IDEA`: only if absolute-delta remains the failed axis, model mutation-effect
  magnitude and sign as separate train-only functionals, then recombine into a
  coherent predictive distribution. The signed median remains the primary
  point estimator; magnitude cannot be substituted for it.
- `PREDICTION`: distribution-derived absolute-delta MAE improves without
  reducing signed-delta MAE or CRPS.
- `FALSIFIER`: magnitude improves only by moving the signed point, or the
  recombined distribution is not proper/calibrated.
- `ALTERNATIVE_EXPLANATION`: apparent magnitude gains can arise from predicting
  the marginal `|Delta|` distribution and ignoring mutation-specific signal.

### I4 — Train-only residual recalibration

- `IDEA`: only if all performance margins pass and coverage is the sole failed
  axis, fit one pre-specified train-only monotone scale recalibrator after
  freezing locations and mixture weights.
- `PREDICTION`: 68%/95% coverage error returns within the frozen guardrail while
  CRPS and point metrics are unchanged or improved.
- `FALSIFIER`: calibration repair costs the task CRPS margin or requires
  puzzle-specific/method-specific thresholds.
- `ALTERNATIVE_EXPLANATION`: coverage mismatch may reflect distribution-family
  misspecification rather than a global scale error.

### I5 — Stop residual rescue and return to benchmark route

- `IDEA`: if neither matched capacity nor asymmetric locations provide stable
  CRPS improvement, stop adding residual modules. Preserve V8 as the point
  model and V9/V10 as negative model-rescue evidence.
- `PREDICTION`: no further residual-only amendment has sufficient expected
  information gain.
- `REVISIT_TRIGGER`: a new independent dataset, a demonstrably stronger point
  representation, or a new mechanistic feature with a matched null.

## 4. Pre-score decision tree

The first matching branch controls. No branch changes the V10 verdict.

1. **V10 exact PASS.** Open only the frozen seeds0–4 formal confirmation. Do not
   create V11.
2. **Integrity, signed-delta, or key-universe failure.** Treat as an audit
   failure, not a model result. Close scoring, diagnose identity/estimand, and
   do not select a new architecture.
3. **Asymmetric-vs-symmetric Gate fails and capacity-symmetric-vs-V9 also lacks
   a positive CI.** Reject A1 and residual-capacity rescue; select I5.
4. **Asymmetric-vs-symmetric Gate fails but capacity-symmetric-vs-V9 has a
   positive CI and at least 14/20 positive puzzles.** Asymmetry mechanism is
   rejected. I1 may be proposed in a new amendment only as a performance model,
   never promoted inside V10 and never claimed as asymmetric-mechanism evidence.
5. **Asymmetry increment passes, signed and absolute margins pass, but fair
   task CRPS remains below 5%.** Residual asymmetry is useful but A2 fails;
   prioritize I2. The next experiment must change the point representation and
   hold the residual family, evaluator, split, and budget fixed.
6. **Signed margin fails while distribution axes pass.** Prioritize I2; do not
   add calibration capacity. The primary falsifier is point learnability.
7. **Absolute-delta alone fails while signed and CRPS pass.** Consider I3 only
   after verifying that the failure is not caused by a scorer or expected-value
   implementation error.
8. **Coverage guardrail alone fails while every performance/influence Gate
   passes.** Consider I4. No architecture search is allowed.
9. **Positive-puzzle, leave-one-out, or 20% influence Gate fails.** Treat the
   effect as heterogeneous. Run only pre-specified method/region/distance
   diagnostics; no new model is authorized until one falsifiable source of
   heterogeneity is identified.

If multiple scientific margins fail, choose the earliest upstream failure:
point representation → task distribution → asymmetry increment → calibration.
Do not combine several rescue modules in one amendment.

## 5. Evaluation criteria for any post-V10 amendment

Any proposed V11 must satisfy all of the following before training:

- one primary capability and one exact nested or parameter-matched null;
- the same corrected target identity, split_v4, puzzle-level independent unit,
  method-balanced estimand, coverage universe, and fair input permissions;
- a single candidate family, fixed epoch/seed/fold universe, and complete-before-
  score rule;
- separate point and distribution gradients unless the new hypothesis is
  explicitly about joint training and includes a frozen-mean null;
- a practical margin, puzzle-level CI, direction count, LOO and influence Gate;
- a negative result that terminates that hypothesis rather than spawning an
  unbounded module search;
- external/SOTA/publication claims remain closed until a separate independent
  validation authority exists.

## 6. Adversarial review

- `RISK`: the same 20 puzzles have been repeatedly consumed. `MITIGATION`: all
  V11 evidence remains post-hoc development; a new amendment cannot restore
  prospective status.
- `RISK`: selecting a branch from V10 outcomes is model selection. `MITIGATION`:
  this branch mapping is frozen before score, and any selected branch requires
  a new explicit amendment and new confirmation evidence.
- `RISK`: more parameters can masquerade as mechanism. `MITIGATION`: every
  architecture increment requires a parameter/representation-matched null.
- `RISK`: CRPS can improve through uncertainty while point quality worsens.
  `MITIGATION`: signed-delta and absolute-delta remain independent mandatory
  Gates; locations and scales are reported separately.
- `RISK`: a strong development number may still be unpublishable. `MITIGATION`:
  publication readiness and SOTA remain explicitly unestablished without
  task-matched external confirmation.

## 7. Decision log

- `DECISION`: V10 remains the only active model experiment.
- `DECISION`: this register does not authorize any post-V10 training.
- `DECISION`: next-direction eligibility is controlled by the first matching
  branch above, not by the visually largest improvement.
- `DECISION`: no outcome, partial score, Gate direction, or external outcome was
  read while creating this register.
- `UNRESOLVED`: whether any future point representation has enough independent
  signal to justify another development-consumed amendment.
