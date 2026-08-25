# Post-V12 model contingency — frozen before V12 score access

## Status and boundary

This document does not authorize V13, another gate, a new residual family, or any
additional training.  It freezes the diagnostic and routing logic while V12M3
is still score-blind so that a near miss cannot be converted into a post-hoc
model family, threshold, or claim.

- `LOCATED_EVIDENCE`: V11 improved signed-delta MAE by 9.8041% versus corrected
  feature41, but its edit-site effect was negative and its gain was near zero
  when `abs(feature41) < 0.05`.
- `LOCATED_EVIDENCE`: V11 gains were about 11% at distances above five and about
  8.5–14% when `abs(feature41) >= 0.05`.
- `LOCATED_EVIDENCE`: V11 did not show a stable unfinished-schedule or global
  train-to-held overfit signature, and its task CRPS missed the 5% top-journal
  threshold.
- `HYPOTHESIS`: V12 can remove the localized V11 over-correction with one
  train-only four-parameter monotone shrinkage gate while keeping the useful
  nonlocal residual and the fixed V10 median-constrained distribution family.
- `PROHIBITED`: no partial-fold diagnostic, no partial score, no external
  outcome, no threshold relaxation, no best puzzle/seed subset, no method-ID
  shortcut, and no new candidate training under V12 authority.

All outcome-bearing diagnostics below may run only after folds0–19 have been
merged into the complete prediction-only V12M3 universe, the single scorer has
run, and the mechanical V12M3 verdict is frozen.  The 20 held puzzles remain the
only independent units.  Every diagnostic loss is position → mutant → method →
puzzle balanced; positions, SNVs, methods, bins, or Gaussian components never
increase independent N.

## D1 — Learned gate geometry without held outcomes

Before joining targets, report for every fold:

- the four frozen gate parameters;
- gate, distance-factor, and magnitude-factor weighted quantiles;
- the fraction of registered rows with `g < 0.1`, `g < 0.25`, `g > 0.75`, and
  `g > 0.9`;
- the gate surface on the predeclared grid
  `distance={0,1,5,10,20,40}` ×
  `abs(feature41)={0,0.01,0.05,0.10,0.20,0.40}`;
- across-fold dispersion of each parameter and grid value.

`INTERPRETATION`: saturation or cross-fold dispersion is descriptive.  It
cannot itself authorize more parameters because the gate may correctly differ
across outer-train sets.

## D2 — Held-puzzle shrinkage ceilings

For diagnosis only, define the parent residual

\[
r_{11}=\mu_{11}-f_{41},\qquad
\mu(g)=f_{41}+g r_{11},\quad 0\le g\le1.
\]

Use the fixed grid `g={0.00,0.01,...,1.00}` and report four explicitly
in-sample oracle ceilings on each held puzzle:

1. one global scalar `g`;
2. one scalar in each frozen distance bin `0`, `1–5`, `6–20`, `>20`;
3. one scalar in each frozen magnitude bin `[0,.05)`, `[.05,.10)`,
   `[.10,.20)`, `[.20,inf)`;
4. one scalar in each fixed 4×4 distance–magnitude cell.

Each oracle minimizes the same method-balanced signed-delta L1.  Ties select
the smallest `g`, fixed before outcomes are read.  Report its signed-delta MAE,
point absolute-delta MAE, selected gate values, and gain over feature41, V11,
and V12.

`LIMITATION`: these are in-sample upper bounds fitted on the held puzzle.  They
are not valid candidate estimates, cannot generate a PASS, and cannot be
reported as model performance.  Their only purpose is to distinguish missing
residual signal from insufficient shrinkage coordinates or capacity.

## D3 — Transfer failure versus gate-family failure

At puzzle level compare the learned V12 gate with D2:

- V12 versus global-oracle signed and point-absolute losses;
- global oracle versus distance-only, magnitude-only, and 4×4 oracle losses;
- method-balanced correlation between V12 gate values and the 4×4 oracle gate;
- the signed residual association between `target-feature41` and
  `V11-feature41` within every fixed distance and magnitude bin.

Predeclared interpretation:

- `RESIDUAL_SIGNAL_ABSENT`: even the 4×4 oracle cannot make the corresponding
  point metric reach its frozen V12 top-journal threshold.
- `TRANSFER_LIMITED`: a global oracle improves V12 by at least 2% of the
  feature41 baseline in at least 14/20 puzzles with puzzle-level CI lower > 0,
  while the 4×4 oracle adds less than 1%.  More gate capacity is contraindicated;
  the bottleneck is outer-train-to-held transfer.
- `COORDINATE_LIMITED`: the 4×4 oracle improves the global oracle by at least
  1% of the feature41 baseline in at least 14/20 puzzles with CI lower > 0.
  Only then may a later amendment consider a low-capacity non-product gate.
- `MONOTONE_PRODUCT_ADEQUATE`: V12 is within 1% of the 4×4 oracle on both point
  metrics.  More gate capacity is not supported even if a headline threshold is
  narrowly missed.

Because D2 is an in-sample ceiling, `COORDINATE_LIMITED` supports only a later
cross-fitted test; it does not establish that a flexible gate will generalize.

## D4 — Point versus residual-distribution bottleneck

For feature41, V11 parent, terminal V10, and V12 distributions report:

- method-balanced CRPS and distribution-derived absolute-delta error;
- probability-integral-transform mean and variance;
- 50%, 68%, 80%, 90%, and 95% central coverage error;
- lower- and upper-tail miss rates;
- the same values in the fixed distance and magnitude bins;
- association of learned scale/mixture weight with absolute residual error.

`DISTRIBUTION_ONLY_SUPPORTED` requires all V12 point Gates to pass while CRPS
or distribution absolute-delta is the only failed headline family, plus one
predeclared tail/asymmetry effect whose 20-puzzle CI excludes zero and whose
direction agrees in at least 14/20 puzzles.  Otherwise a new distribution
family is not the identified bottleneck.

## Frozen routing after the complete V12 verdict

1. **V12M3 exact PASS:** open only the already-frozen V12M4 seeds0–4 formal
   confirmation.  Do not run D2 to select another gate.
2. **V12 point Gates PASS; only probability/magnitude Gates FAIL:** V12 remains
   terminal FAIL.  Run D4; a later residual-only amendment is eligible only
   under `DISTRIBUTION_ONLY_SUPPORTED`, with the V12 point frozen.
3. **V12 beats V11 but misses feature41 point thresholds:** run D2/D3.  A later
   gate amendment is eligible only under `COORDINATE_LIMITED`; a near miss alone
   is insufficient.
4. **V12 fails to beat V11:** run D2/D3.  If `TRANSFER_LIMITED`, stop increasing
   gate capacity.  If `RESIDUAL_SIGNAL_ABSENT`, terminate the V11-residual
   architecture family.
5. **V12 is within 1% of the 4×4 oracle but misses a point Gate:** terminate the
   shrinkage-gate route; the parent residual, not the gate capacity, limits the
   attainable result.
6. **V12M3 PASS but V12M4 FAILS:** the signal is seed-unstable.  Preserve the
   complete negative formal result; do not select seeds, puzzles, or gates.
7. **Any later amendment:** name exactly one diagnostic-supported bottleneck,
   one candidate, one task/parameter-matched null, fixed folds/seeds/epochs, and
   top-journal Gates before training.  Use a new worktree and artifact universe,
   preserve V1–V12 verdicts, and keep external outcomes locked.

No route in this document authorizes a model automatically.  If no listed
condition is met, the default decision is to stop this architecture family and
return the paper to the benchmark/measurement route rather than continue an
open-ended rescue search.
