# ReactFlow-Delta post-V9 model contingency (not active authority)

Status: `PRE_OUTCOME_ARCHITECTURE_NOTE_ONLY`

This note was written while V9M2 remained score-blind. It does not amend the
V9 model, training budget, candidates, thresholds, or Gate, and it does not
authorize any run. V9 must first finish its frozen 20-fold prediction universe
and receive the mechanical V9M3 verdict.

## Located evidence

- `CONFIRMED_FACT`: V8 trains `MeanAlignedModel` with method-balanced L1. Under
  the usual conditional-risk interpretation, the population target of L1 is a
  conditional median, not necessarily a conditional expectation.
- `CONFIRMED_FACT`: V8 improved signed-delta MAE versus corrected feature41 by
  8.36% with 20/20 positive puzzles, but its absolute point prediction failed
  the frozen guardrail. Its method-balanced `abs(mu)` was 0.04284 versus a true
  `abs(delta)` mean of 0.18332, with underprediction in 20/20 puzzles.
- `CONFIRMED_FACT`: V9 freezes this L1-trained point predictor, calls it a
  signed mean, and places both Gaussian mixture locations exactly at that
  value. Consequently the V9 residual family is symmetric about the frozen
  L1 point and has zero conditional mean by construction.
- `CONFIRMED_FACT`: the V9 residual MLP receives raw feature41 values. The
  corrected ridge artifact already contains outer-train-only `mean_x` and
  `scale_x`, but V9 does not apply them to the residual-head input. Most
  channels are bounded near one, while global energy channels range to about
  12, so this is a conditioning limitation rather than proof of catastrophic
  saturation.

## Primary unresolved mechanism

`HYPOTHESIS`: V8's strong signed-MAE result is a useful conditional-median
signal. Treating that median as a conditional expectation and restricting the
residual distribution to a symmetric scale mixture may be too rigid for the
skewed, sparse mutation-response distribution. V9 can still succeed by
inflating conditional scales, but it cannot express residual skew or a
distribution mean that differs from the L1-optimal point median.

`FALSIFIER`: if V9 passes signed, distribution-derived absolute-delta, CRPS,
coverage, influence, and leave-one-puzzle-out Gates simultaneously, this
restriction is not a current performance bottleneck and no successor should
be opened.

## Pre-outcome decision map

The following mapping is fixed before V9 scores are read. It is a planning
rule, not execution authority.

1. `V9M3_TOP_JOURNAL_SCREEN_PASS`: proceed only to the already frozen five-seed
   V9 confirmation. Do not open a new architecture search.
2. Signed Gate fails: audit replay/scoring identity first. Because the signed
   point predictions are frozen V8/TIC2A replays, this is not a residual-model
   failure that more capacity can fix.
3. Signed passes, CRPS passes, absolute-delta fails: test one explicit
   method-balanced absolute-magnitude head while leaving the V9 signed point
   and probability distribution unchanged. This isolates the output
   functional mismatch; it is a multi-output model, not evidence that the
   probability distribution recovered magnitude.
4. Signed passes, absolute-delta passes, CRPS fails: test one
   median-preserving asymmetric residual distribution. The frozen signed point
   remains the 0.5 quantile; calibration is allowed to model skew but cannot
   move that point prediction.
5. Signed passes while both absolute-delta and CRPS fail: use the same
   median-preserving asymmetric family, with outer-train standardization and
   frozen V8 source/receiver representations added to the residual input. Do
   not first search mixture counts, hidden sizes, backbones, or loss weights.

## Highest-information successor if required

Candidate name: `MedianAligned-AsymmetricGaussianMixture`.

`IDEA`: reinterpret the frozen V8 output as the conditional median
`m_0.5(x)`. Retain the existing two-Gaussian mixture and exact closed-form
mixture CRPS, but allow the component locations to differ while constraining
the mixture CDF at the frozen point to equal 0.5 exactly:

```text
Delta = m_0.5(x) + epsilon,
P(epsilon <= 0 | x) = 0.5.
```

For mixture weight `w`, component scales `s1,s2`, and component CDF values at
zero `a,b`, enforce

```text
w*a + (1-w)*b = 0.5.
```

Use a bounded `w`, map one network output into the analytically valid interval
for `a`, set `b=(0.5-w*a)/(1-w)`, and obtain residual locations through
`l1=-s1*Phi_inverse(a)` and `l2=-s2*Phi_inverse(b)`. Initializing `a=b=0.5`
recovers the exact V9 nested null with both residual locations equal to zero.

Minimum implementation:

- reuse the frozen, target-corrected V8 encoder and L1 point checkpoint;
- input outer-train-standardized feature41 plus detached V8 direct features;
- use one fixed two-layer residual head, width 256;
- retain two Gaussian components, exact Gaussian-mixture CRPS, and the existing
  expected-absolute-delta calculation;
- keep the signed point prediction equal to the frozen V8 conditional median;
- train a same-input symmetric-location nested null and the
  median-constrained asymmetric candidate with identical budget;
- train the corrected feature41 comparator with the same residual family and
  input permissions;
- run one seed, 20-fold, complete-universe-before-score; do not inspect partial
  metrics or search mixture count, width, loss, epoch, or location constraint.

`PREDICTION`: allowing residual asymmetry should improve CRPS and
distribution-derived absolute magnitude relative to V9 while preserving the
already observed V8 signed-MAE gain exactly.

`ADVERSARIAL_ALTERNATIVE`: apparent gains could come from adding detached V8
representations or outer-train standardization rather than the median
constraint. The same-input symmetric-location nested null is therefore
required to identify the incremental value of asymmetric locations.

`RISK`: inverse-normal parameterization can produce unstable gradients if CDF
probabilities approach zero or one; the valid CDF interval and a fixed
numerical interior bound must be specified before outcome access. A null
result would mean that residual asymmetry is not the missing capability and
should close residual-only rescue rather than trigger an unbounded family
search.

## Evidence status and missing perspectives

- The located evidence is internal development evidence; it is not external
  replication or SOTA evidence.
- The successor is an `IDEA`, not a result. Literature novelty is
  `NOT_CHECKED` in this note.
- No new external outcome is used.
- This is a single-agent, AI-assisted architecture analysis. Independent RNA
  biophysics, probabilistic-forecasting, and statistical review remain absent
  and would be required before a confirmatory amendment.
