# Post-V14 P2 Monotone Quantile Residual Design

**Status:** `FOCUSED_PRE_SCORE_DESIGN_FROZEN_INACTIVE`

**Authority effect:** none. This document does not issue source-projection,
training, prediction, scoring, qualification, formal-confirmation, or external
outcome authority, and it does not change
`configs/reactflow_delta/active_contract.yaml`.

**Evidence boundary:** the numerical choices below are new focused pre-score
design judgments. They are not historical V14 or earlier-contract facts and
must never be reported as such.

## 1. Selected capability and non-goals

If and only if the canonical first-matching post-V14 router selects branch 6 and
the once-only branch-6 diagnostic returns exact
`POST_V14_BRANCH6_TAIL_DIAGNOSTIC_PASS`, a later focused authority may activate
P2. P2 asks whether a monotone conditional quantile residual distribution, with
its median fixed exactly to the V14 point prediction, repairs the remaining
distribution failure beyond a parameter-matched replay of the V10
median-asymmetric residual family.

The selected implementation is a dedicated fixed-grid quantile vertical slice.
Two alternatives are closed:

- Do not learn a monotone warp of the V10 Gaussian-mixture CDF. It would blur
  the distinction between nonparametric conditional tail shape and another
  scale/location-mixture transformation.
- Do not generalize the P1 calibration pipeline behind a family switch. P1 and
  P2 are mutually exclusive routes, so that abstraction would add coupling for
  no second live use case.

P2 does not change or retrain the V14 point model, revisit router order, relax a
V14 Gate, read a new external outcome, or restore historical qualification. It
does not run until a future amendment, ledger, validator, exact source binding,
and phase-specific active authority all exist.

## 2. Unique entry and fail-closed binding

Every future activation prerequisite is mandatory:

1. A complete, valid V14 terminal handoff exists.
2. The canonical first-matching router selects branch ID `6` with classification
   `DISTRIBUTION_ONLY_FAILURE`.
3. Every signed and point-absolute V14 Gate passed, while CRPS and/or
   distribution-derived absolute error was the sole failed family.
4. The canonical diagnostic schema is
   `reactflow_delta.post_v14_branch6_tail_diagnostic.v1`, its primary statistic
   is `LOWER_MINUS_UPPER_TAIL_MISS90`, and its exact status is
   `POST_V14_BRANCH6_TAIL_DIAGNOSTIC_PASS`.
5. Its puzzle-level 95% interval is wholly on one side of zero and at least 14
   of 20 puzzles agree in direction.
6. Its exact next action is `OPEN_FOCUSED_P2_AMENDMENT_AUTHORITY`.
7. A later amendment binds every canonical source/output path and a separate
   active-pointer commit issues exactly one phase token.

Missing, inconsistent, stale, or differently routed parent state must fail
before output-directory creation or source/model/data reads. Diagnostic PASS is
route eligibility only, not a P2 result and not training authority.

Actual V14 terminal values, diagnostic artifacts, realized absolute paths, and
phase tokens are `PENDING_TERMINAL_BINDING` in inactive preparation. They must
not be guessed.

## 3. Data, estimand, and claim ceiling

P2 retains OpenKnot M2 v4.5.2, split-v4 20-fold leave-one-puzzle-out, and exact
identity `EXACT_PUZZLE_METHOD_MUTATION`. The prediction universe is every
registered mutant full-construct position.

Training and scoring retain this hierarchy:

```text
position -> equal mutant mean -> equal method cell -> equal puzzle mean
```

The 20 held puzzles are the independent inference units. Primary incremental
estimands are paired equal-puzzle differences between the candidate and its
newly trained parameter-matched V10 replay for CRPS and MAE of the
distribution-derived expected absolute delta. Relative gain is:

```text
(mean_puzzle_comparator_error - mean_puzzle_candidate_error)
/ mean_puzzle_comparator_error
```

From P2M3 screen onward, the 13 quantile nodes and their fixed weights define
the candidate predictive distribution itself: a 13-atom distribution with atom
values `q_i` and masses `w_i`. Candidate scientific CRPS is the exact CRPS of
that finite distribution. The matched V10 replay remains its two-Gaussian
predictive distribution and uses exact Gaussian-mixture CRPS. Thus both arms
are compared under the same proper scoring rule—CRPS—evaluated exactly for each
arm's declared predictive distribution. Their training objectives need not be
the same.

Every canonical V14 feature41 and terminal-comparator distribution Gate is
repeated unchanged. Signed and point-absolute results are replay invariants, not
P2-optimized outcomes.

Even exact formal PASS is capped at `POST_HOC_DEVELOPMENT_FORMAL_PASS`. It does
not establish external replication, SOTA, mechanism, practical utility,
publication readiness, or restored historical qualification.

## 4. Frozen target-free input

Candidate and comparator receive the same 244-dimensional input in this order:

```text
feature41 basis                              41
frozen V14 point                             1
absolute frozen V14 point                    1
frozen trained V8 direct features          201
                                           ---
                                           244
```

The V8 direct-feature definition remains:

```text
source hidden96 + receiver hidden96 + signed distance1 + mutation one-hot8
```

The same-fold trained V8 MeanAligned direct path and same-fold frozen TIC2A
feature41 basis/caches are required. Random/unused V8 branches, method ID,
puzzle ID, dataset ID, held target, held error, held qualified-target mask,
external outcome, and score-derived fields are forbidden inputs.

Reuse the V10 `calibration_input` order. Fit one
`TrainOnlyStandardizer` per outer fold using outer-train rows only. It computes
all 244 means and population standard deviations and replaces any scale below
`1e-6` with `1.0`. Held rows only transform under the frozen statistics.

## 5. Frozen V14 point anchor

Bind the same-fold seed-0 V14 candidate checkpoint and prediction:

```text
v14_candidate_point_fold{outer_fold}_seed0.pt
v14_predictions_fold{outer_fold}_seed0.npz
```

For outer-train rows, load the checkpoint in evaluation mode, make every
parameter non-trainable, clear gradients, snapshot the full state, and compute
points under `no_grad`. After distribution fitting, every tensor must remain
bitwise equal and every point gradient must remain absent.

For held rows, read `candidate_point` directly by biological key from the bound
V14 prediction artifact. GPU recomputation is not the held-point authority. The
key universe, fold, held puzzle, and seed must match the complete V14 merge.

The raw point is passed separately to each distribution head. Point channels in
the 244-dimensional input are context only. The candidate assigns the detached
float64 raw point directly to tau `0.50`; it never reconstructs the median from
standardized inputs.

## 6. Quantile grid, predictive atoms, and training surrogate

The candidate predicts exactly:

```text
taus = [
  0.025, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
  0.60,  0.70, 0.80, 0.90, 0.95, 0.975,
]

weights = [
  0.0375, 0.0375, 0.075,
  0.1,    0.1,    0.1,   0.1, 0.1, 0.1, 0.1,
  0.075,  0.0375, 0.0375,
]
```

These are fixed full-interval midpoint/Voronoi weights. They sum to `1.0`, with
mass `0.45` below the median node, `0.10` at it, and `0.45` above it. Exact
`0.05/0.95` nodes retain the branch-6 90% tail boundary; `0.025/0.975` add the
minimum extra tail resolution. Beginning with P2M3 scientific screen, these
nodes and weights are not merely a numerical integration grid: they define the
candidate predictive distribution exactly as
`F_candidate = sum_i w_i delta(q_i)`.

For residual target `y`, quantile `q_i`, and `u_i = y - q_i`:

```text
rho_tau(u) = u * (tau - 1[u < 0])

candidate_training_surrogate
  = 2 * sum_i weights[i] * rho_taus[i](y - q_i)

candidate_expected_absolute_delta
  = sum_i weights[i] * abs(q_i)

candidate_scientific_crps(y)
  = sum_i weights[i] * abs(y - q_i)
    - 0.5 * sum_i sum_j weights[i] * weights[j] * abs(q_i - q_j)
```

The `2 x` weighted pinball expression is the candidate training surrogate only.
It must not be written into a screen/formal score artifact as scientific CRPS,
used to adjudicate a CRPS Gate, or compared directly with Gaussian-mixture
CRPS. Screen candidate CRPS uses the exact 13-atom expression above. Matched
V10 scientific CRPS uses its existing exact Gaussian-mixture expression. The
weighted absolute value is exact for the declared candidate atom distribution.
Learned atom masses, grid search, interpolation, extrapolation, and
result-dependent tail refinement are prohibited.

## 7. Candidate architecture and exact parameter count

The candidate is exactly:

```text
Linear(244, 248) -> ReLU -> Linear(248, 12)
```

Its parameter count is:

```text
(244 + 1) * 248 + (248 + 1) * 12
= 60,760 + 2,988
= 63,748
```

The 12 outputs define strictly positive adjacent gaps:

```text
gap_j = 1e-4 + softplus(raw_j), j = 0..11

q_6 = detached_float64_frozen_v14_point
q_i = q_6 - sum(gap_j for j in i..5),    i = 0..5
q_i = q_6 + sum(gap_j for j in 6..i-1),  i = 7..12
```

Monotonicity and the median are structural, not penalties or post-hoc repairs.
Learned layers remain float32; gap construction, cumulative quantiles,
quadrature, and persisted distributions use float64. There is no dropout, batch
normalization, skip projection, learned global scale, extra tail parameter, or
auxiliary loss.

Candidate `output_layer.weight` initializes entirely to zero. Its 12 biases are
set exactly to:

```text
inverse_softplus(target_adjacent_gap_j - 1e-4)
```

Before inverse softplus, the validator requires every
`target_adjacent_gap_j > 1e-4`. Target gaps come from the float64 inverse CDF of
the P2-specific input-independent V10 initialization defined below, evaluated
at the 13 frozen taus. Fixed float64 bisection computes those quantiles. Do not
store hand-rounded gaps or add trainable initialization parameters.

The sole executable initial-grid replay criterion is frozen as:

```text
INITIAL_GRID_REPLAY_ATOL_1E_6_RTOL_0

np.allclose(
    candidate_initial_grid_float32,
    registered_comparator_initial_grid_float64,
    atol=1.0e-6,
    rtol=0.0,
)
```

This tolerance covers only the fixed float64 inverse-CDF bisection followed by
the candidate's float32 bias and forward round-trip. It does not apply to the
frozen V14 median point, any point-prediction replay, the scientific CRPS, or
any other score. Point replay remains `atol=1e-7, rtol=0`, and focused core
tests must additionally retain exact array equality for the assigned candidate
median.

## 8. Exact parameter-matched V10 replay

The comparator is a newly trained existing `MedianAsymmetricResidual`:

```text
Linear(244, 256) -> ReLU -> Linear(256, 4)

(244 + 1) * 256 + (256 + 1) * 4
= 62,720 + 1,028
= 63,748
```

It keeps the V10 two-Gaussian median-preserving construction and exact
Gaussian-mixture CRPS objective. It receives the same frozen V14 point, input
rows/channels, standardizer, outer-train universe, seed, epochs, puzzle order,
Adam optimizer, and gradient clipping as the candidate. Historical V10
predictions are not the matched replay; the comparator is trained again around
the V14 point in every authorized P2 fold and seed. Model/family selection is
forbidden.

P2 freezes a specific input-independent comparator initialization. Construct
the existing `MedianAsymmetricResidual`, then zero its complete
`output_layer.weight` tensor. Keep or set its existing four output biases
exactly as:

```text
mixture-weight logit = 0
narrow-scale raw     = inverse_softplus(0.08)
wide-gap raw         = inverse_softplus(0.20)
allocation raw       = 0
```

Because the full comparator output weight is zero, its initial mixture is the
same for every 244-dimensional input. Candidate output weights are also all
zero, and its biases are computed from this comparator mixture's 13 inverse-CDF
values. Therefore, for every input, the candidate initial quantile grid matches
the registered comparator float64 inverse-CDF grid at all 13 taus within
`INITIAL_GRID_REPLAY_ATOL_1E_6_RTOL_0`. This is registered-tolerance grid replay
only: the 13-atom candidate and continuous two-Gaussian comparator are different
complete predictive distributions and must never be described as identical.

## 9. Training and P2M0-P2M5 lifecycle

Real training and GPU validation must call the existing CUDA fail-fast helper.
CPU fallback is forbidden. No free-memory or GPU-memory Gate is added.

### P2M0 — inactive preparation

Design, amendment, ledger, validator, implementation, and focused synthetic
tests only. No artifact access, active-pointer change, training, score, or
external outcome.

### P2M1 — source projection only

Bind 20 same-fold source records and canonical output paths. Validate exact
terminal/router/diagnostic state before source read or output creation. Training,
prediction, scoring, and external access remain closed.

### P2M2 — engineering smoke only

- folds `[0, 1]`, seed `[0]`, 3 epochs for both models;
- prediction-only artifacts and engineering qualification;
- scientific scorer prohibited.

### P2M3 — twenty-fold screen

- folds `[0..19]`, seed `[0]`, 40 epochs;
- complete prediction-only universe and merge before one scorer/qualifier;
- partial-fold and partial-score inspection prohibited.

### P2M4 — fixed five-seed formal confirmation

- only exact P2M3 PASS opens it;
- folds `[0..19]`, seeds `[0,1,2,3,4]`, 40 epochs;
- seed0 retrains; screen predictions are not reused;
- all 100 fold-seed runs complete before assembly and scoring;
- no failed-seed removal, seed subset, or best-seed selection.

### P2M5 — terminal

Formal PASS or scientific FAIL closes the family. Incomplete universe, invalid
provenance, nonfinite output, integrity failure, or unauthorized access is
`INDETERMINATE`, never scientific PASS/FAIL.

Both models use Adam, learning rate `1e-3`, zero weight decay, gradient clipping
`5.0`, no early stopping, and no best-epoch selection. Puzzle order uses
`seed * 100003 + epoch`. Candidate training minimizes the fixed `2 x` weighted
pinball surrogate; matched V10 training minimizes exact Gaussian-mixture CRPS.
Scientific comparison later uses exact CRPS for each declared predictive
distribution, not either arm's training-loss representation.

## 10. Prediction and merge boundary

Prediction-only fold files minimally contain:

```text
schema_version, keys, biological_scoring_key, outer_fold, held_puzzle, seed,
registered_status, v14_candidate_point, taus, quadrature_weights,
candidate_quantiles, candidate_expected_absolute_delta,
v10_replay_weights, v10_replay_locations, v10_replay_scales,
v10_replay_expected_absolute_delta
```

For P2M3 scoring, `candidate_quantiles` are the 13 atom values and
`quadrature_weights` are their predictive masses. The scorer must not reinterpret
them as samples, interpolate them into another distribution, or substitute the
training pinball surrogate for exact atom-distribution CRPS.

They must not contain held target/error/mask, score, per-puzzle effect, Gate, or
external outcome. Fold result metadata may contain paths, standardizers,
parameter counts, histories, and invariants but may not move histories into the
scientific merge.

Merge requires the exact authorized fold/seed universe, canonical files, unique
biological keys, same-fold provenance, both 63,748 counts, identical taus and
weights, full coverage, point replay, strict monotonicity, finite arrays, and
prediction-only schemas. It rejects duplicate/missing/unexpected keys, wrong
paths, point drift, target fields, partial-score markers, and external-access
markers. It never joins targets or computes a metric.

Material artifacts are atomically finalized and never overwritten. Future large
artifacts belong under bound `/mnt/cunyuliu` paths; source and commits stay under
`/home`. This inactive design creates and reads neither.

## 11. P2M3 screen Gates

Integrity requires coverage `1.0`, failure rate `0.0`, unexpected keys `0`, a
complete 20-fold universe, unchanged point state, absent point gradients,
finite strictly increasing quantiles, exact `q_0.50 == V14 point`, retained
point replay `atol=1e-7, rtol=0`, no held-target prediction input, no partial
score, and no external outcome.

At activation, copy every canonical V14 feature41/terminal CRPS,
distribution-absolute, coverage, and calibration Gate verbatim and repeat them
without relaxation. All signed/point-absolute V14 Gates replay unchanged.

For every screen row, candidate scientific CRPS is computed exactly as:

```text
sum_i w_i |y - q_i| - 0.5 sum_i sum_j w_i w_j |q_i - q_j|
```

Matched V10 scientific CRPS remains the exact Gaussian-mixture CRPS. These are
two exact evaluations of the same proper-scoring-rule estimand for the two
declared distributions. Weighted pinball is absent from scientific score and
Gate fields.

The two new capability Gates are:

```text
candidate vs matched V10 replay CRPS relative gain >= 0.015
candidate vs matched V10 replay distribution-absolute MAE gain >= 0.01
```

Each comparison additionally requires:

- paired 95% t-interval lower bound greater than zero over 20 puzzles, with
  `t_0.975,df19 = 2.093024054408263`;
- at least 14 of 20 positive puzzle differences;
- all headline leave-one-puzzle-out effects positive; and
- maximum single-puzzle effect fraction at most `0.20`.

Exact V14 coverage/calibration guardrails repeat; no P2-specific relaxation is
allowed. A valid complete score failing a scientific Gate is FAIL. An integrity
defect is INDETERMINATE.

## 12. P2M4 formal assembly and Gates

Formal candidate assembly is an equal-seed 65-atom distribution:

```text
atom_value(seed, i) = q_seed,i
atom_weight(seed, i) = weights[i] / 5
```

Do not average quantile curves. The 65 weights sum to `1.0`. Since each seed has
mass `0.45/0.10/0.45` below/at/above the common point, the mixture retains the
V14 median. Compute mixture CRPS as weighted finite-distribution CRPS and
expected absolute delta as the weighted absolute atom mean. Assemble the five
V10 Gaussian replays as their equal distribution mixture.

The 65-atom candidate uses the same exact finite-distribution formula as the
13-atom screen candidate, with the 65 values and masses substituted. This
repeats the same CRPS estimand; it does not switch back to integrated pinball or
average per-seed CRPS values. The equal five-seed V10 mixture is likewise scored
by its exact mixture CRPS.

The formal mixture repeats every screen integrity/scientific Gate. Additionally,
at least 4 of 5 seeds must have positive candidate-versus-V10 increments for
CRPS and at least 4 of 5 for distribution-absolute error. All seeds/runs are
reported and none may be removed after scoring.

## 13. Failure policy and frozen/pending boundary

Authority validation precedes output creation and artifact reads. CUDA
validation precedes model/data construction. CUDA absence or CPU fallback stops
the run and preserves the error; it never retries on CPU. No hashes, migration
framework, generic model registry, compatibility wrapper, interpolation layer,
alternative grid, hyperparameter search, or automated contingency selection is
added. Smoke, proxy, or training loss is never a scientific result.

Frozen here are the unique branch-6 PASS entry, inactive status, 244 input,
frozen V14 point, exact taus/weights, weighted-pinball training surrogate,
exact predictive-distribution scientific CRPS, exact candidate/comparator
architectures and counts, `INITIAL_GRID_REPLAY_ATOL_1E_6_RTOL_0`, P2M0-M5
schedule, Adam/epochs/seeds, new matched replay Gates, 65-atom formal mixture,
4-of-5 rules, and claim ceiling.

Still pending—and prohibited from invention—are actual terminal/router/
diagnostic values and paths, realized source/output paths, future authority
tokens, and numeric V14 feature41/terminal/coverage/calibration Gate values.
Those V14 values must be copied verbatim from the canonical terminal contract at
activation, never transcribed from memory.
