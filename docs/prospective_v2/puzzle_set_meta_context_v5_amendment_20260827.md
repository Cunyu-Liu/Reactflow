# ReactFlow-Delta Puzzle-Set Meta-Context V5 Amendment

**Status:** `DRAFT_FROZEN_INACTIVE_V14_SOLE_ACTIVE`

**Proposed task:** `reactflow_delta_puzzle_set_meta_context`

**Operator:** `POSITION_ALIGNED_CROSS_CONSTRUCT_CONSENSUS_V5`

**Evidence ceiling:** `POST_HOC_DEVELOPMENT_PASS`

## 1. Contract status and authority boundary

This document freezes a prospective, narrowly scoped Puzzle-Set V5 experiment.
It is not active authority. Model Rescue V14 remains the sole active model
experiment, and this draft does not modify, interrupt, score, supersede or
reinterpret any V14 run, artifact, protocol, current Gate or future terminal
handoff. V1--V13 terminal verdicts remain immutable; V14 does not yet have a
terminal verdict.

Puzzle-Set V5 may become active only after all of the following are true:

1. V14 has reached a complete terminal handoff under its existing contract;
2. the first applicable branch of
   `docs/plans/2026-08-27-post-v14-model-contingency.md` mechanically selects
   P1 rather than V14M4, P2 or model-rescue termination;
3. the selected router branch and its probe state are bound mechanically:
   branches 3/4 require `NOT_APPLICABLE/NOT_APPLICABLE`, while branch 5
   requires `REQUIRED/EXACT_PASS` for the fixed complete outer-train-only
   linear route probe;
4. the realized V14 terminal source artifacts and every frozen input path,
   role, seed and realized parameter count have been bound in the already
   existing inactive machine-contract draft and a decision-ledger event;
5. a new focused commit activates P1, points `active_contract.yaml` at that
   machine contract and keeps training closed until contract validation and
   focused tests pass.

An exact V14M3 PASS routes only to V14M4 and cannot activate P1. A V14M4 formal
failure is seed-instability evidence and routes to model-rescue termination
unless a new independent dataset becomes available; it cannot be reused to
activate this amendment. Merely completing the V5 implementation, passing
tests or observing a near miss in another model does not confer authority.

The registered router is
`docs/plans/2026-08-27-post-v14-model-contingency.md`, and its first matching
branch controls. At draft time `selected_router_branch_id`, route-probe
requirement are pending the complete V14 terminal handoff, and route-probe
status is exactly `NOT_RUN`; this is not evidence that P1 will become eligible.

No P1 training, target join or scientific score is authorized while the status
is `DRAFT_FROZEN_INACTIVE_V14_SOLE_ACTIVE`.

### Frozen branch-5 route probe — specification only

This draft now freezes the exact probe required only if the first matching
post-V14 router branch is branch 5, `INDEPENDENT_CONSTRUCT_TRANSFER_LIMITED`.
It does not authorize that probe. Before a complete V14 terminal handoff its
status is `NOT_RUN`; no execution token, prediction authority, held-score
authority or external-outcome authority has been issued. Branches 3 and 4 use
`NOT_APPLICABLE/NOT_APPLICABLE` and must not manufacture a probe artifact.
The future runtime identity is frozen as project task
`reactflow_delta_post_v14_branch5_route_probe`. A terminal-only `B5RP0` first
binds the frozen checkpoint directories and manifest path with status
`POST_V14_BRANCH5_SAFE_SOURCE_MANIFEST_PENDING_PROJECTION`; it keeps training
and all score access closed. Only after all 40 checkpoint files pass strict
architecture inspection and the 20-row manifest exists may a focused commit
set the manifest status to `POST_V14_BRANCH5_SAFE_SOURCE_MANIFEST_PASS` and
open prediction phase `B5RP1` with
token `POST_V14_BRANCH5_LINEAR_CROSS_CONSTRUCT_ROUTE_PREDICTION_ONLY`, score
phase `B5RP2` with token
`POST_V14_BRANCH5_COMPLETE_MERGE_SCORE_ONCE_ONLY`, and qualifier/terminal phase
`B5RP3`. Every one of these authorities and both tokens are currently
`NOT_AUTHORIZED/NOT_ISSUED`; `active_contract.yaml` remains unchanged.
Any future route-probe active authority must additionally bind the exact parent
state `v14_status=TERMINAL_V14M3_TOP_JOURNAL_SCREEN_FAIL`,
`post_v14_first_matching_branch_id='5'` and
`post_v14_route_classification=INDEPENDENT_CONSTRUCT_TRANSFER_LIMITED`.
The V14M4 path is explicitly ineligible and must fail closed.

The future authority is also path-bound, not merely schema-bound. It fixes the
V13 checkpoint directory to
`/mnt/cunyuliu/reactflow_delta_model_rescue_v13/v13m3_screen_seed0`, the V14
checkpoint directory to
`/mnt/cunyuliu/reactflow_delta_model_rescue_v14/v14m3_screen_seed0`, the M2 CSV,
strict TIC2A registry and both feature caches to the absolute paths registered
in the machine contract, and the prediction directory to
`/mnt/cunyuliu/reactflow_delta_post_v14_branch5_route_probe/b5rp1_seed0`.
The only eligible complete merge, complete score and qualification are the
three registered files in that directory. Projector, runner, scorer and
qualifier must reject a CLI path that differs from its active-authority
binding. This prevents a compatible but differently trained checkpoint or a
second scored universe from being substituted after the route is selected.
In particular, B5RP0 must not predeclare the not-yet-created manifest as
`PASS`; `PENDING_PROJECTION → PASS` requires the manifest artifact and a
separate authority commit. Its executable projector derives the active pointer
only from `<repo-root>/configs/reactflow_delta/active_contract.yaml`; an
operator-supplied alternate YAML cannot authorize projection.

The probe may consume parent checkpoints only through a terminal safe source
manifest with schema
`reactflow_delta.post_v14_branch5_safe_source_manifest.v1`. That manifest is
not generated or accepted now. If branch 5 is later selected, it must contain
exactly folds 0--19 with canonical `P01`--`P20`, seed 0, V13M3 candidate and
V14M3 candidate checkpoint paths, and score-closed/external-locked provenance.
It must not contain targets, masks, training histories, losses, scores,
per-puzzle effects or Gate values. The route runner may not parse the wide V13
or V14 fold-result artifacts as a substitute.

For each outer fold, the probe freezes the same-fold V13 candidate seed-0 point
as an immutable additive anchor and the same-fold V14 candidate seed-0 encoder
as an immutable outcome-blind representation source. Neither is trained by the
probe. The regression target on the nineteen outer-train puzzles is

\[
r_{ui}=\Delta_{ui}-\widehat\Delta^{V13}_{ui}.
\]

For each focal construct and coordinate, exactly the other seven constructs
form one raw 260-dimensional non-focal summary:

- the arithmetic mean of their seven zero-preserving V14 content contrasts,
  width 256;
- `pooled_safe_wt`, the sum of finite observed WT reactivities with unobserved
  entries replaced by zero, divided by seven, width 1;
- the finite-observed WT-reactivity mean, width 1;
- the finite-observed WT-reactivity population standard deviation, width 1;
- the finite-observed support fraction, count divided by seven, width 1.

Zero support gives zero pooled value, mean and standard deviation. Concatenating
the mutation-source and receiver summaries produces the only 520-dimensional
ridge input. The probe contains no focal hidden/reactivity/mask feature, V13
point as a ridge feature, feature41, signed distance, mutation identity,
method ID, puzzle ID or dataset ID. V13 enters only as the frozen additive
anchor.

Each V14 content contrast is computed in frozen eval mode as
`encode(real WT context) - encode(coordinate-only reference)`. The reference
preserves the same `position` and `region` tensors and zeros `sequence`,
`reactivity`, `precision` and `observed`. The fixed shift is applied only after
this subtraction. Therefore arbitrary positions and regions with zero
biological/measurement content must give an exactly zero contrast and
identical zero aligned/shift-17 features. This prevents a pass caused only by
raw V14 absolute-position or region encoding.

The aligned arm uses the registered shared full-construct source and receiver
coordinates. The matched arm circularly shifts every non-focal stream by the
fixed offset 17 before computing all five summary components. The two arms are
standardized and fit separately using only their outer-train rows. Each feature
uses its arm-specific weighted outer-train mean and population standard
deviation. A standard deviation below `1e-8` is assigned scale `1.0`; a truly
constant feature must therefore standardize exactly to zero. Each arm fits weighted
ridge with `alpha=1` and an unpenalized intercept. Sample weights implement the
frozen hierarchy: equal outer-train puzzle×method cells, equal mutants within a
cell and equal qualified positions within a mutant, normalized to mean one over
the fitted rows. No alpha, standardization, feature, intercept, fold or model
selection is allowed.

The two predictions are

\[
\widehat\Delta_{aligned}=\widehat\Delta_{V13}+\widehat r_{aligned},\qquad
\widehat\Delta_{shift17}=\widehat\Delta_{V13}+\widehat r_{shift17}.
\]

All 20 LOPO folds must first produce prediction-only artifacts. Every fold
records the exact same-fold V13 checkpoint and V14 encoder checkpoint from the
terminal safe manifest at
`/mnt/cunyuliu/reactflow_delta_post_v14_branch5_route_probe/source_binding/post_v14_branch5_safe_source_manifest.json`,
their roles, outer fold and seed 0. The future B5RP1 active authority must bind
that exact path and status `POST_V14_BRANCH5_SAFE_SOURCE_MANIFEST_PASS`, and the
runner must reject any different CLI manifest path. Every fold additionally
records the strict same-fold TIC2A feature41 model projection and the global M2 CSV,
TIC2A target-free registry and two feature caches. The five global source paths
must be identical across all 20 folds; every path and role is part of fold
provenance and the merger revalidates the TIC2A registry/model projection
without opening its prediction artifacts. Each fold also records the fitted
standardization and ridge-model artifact. Prediction artifacts
must contain V13, aligned and shift-17 point predictions but no held target,
target error, qualified target mask, score or per-puzzle effect. Only a complete,
unique, provenance-qualified 20-fold merge may be scored exactly once under the
frozen method-balanced position→mutant→method→puzzle evaluator.

The terminal-safe source manifest becomes final only after all forty
checkpoints pass and a same-directory atomic rename completes. Prediction,
ridge and fold JSON files become final only through same-directory
atomic rename after their complete write. The merger itself remains under
B5RP1 authority: its input directory and output path must equal the registered
`prediction_dir` and `complete_unscored_merge_path`; every fold's prediction
and ridge path must equal the fixed filename inside that directory, and the
merger rechecks the frozen V13/V14 checkpoint parent directories. The complete
merge also requires the manifest, M2 CSV, TIC2A registry, unconstrained cache
and constrained cache recorded by every fold to equal the five frozen
active-authority paths, rather than merely agreeing with one another. The complete
merge likewise becomes final only after complete validation and atomic rename.
The one complete score and qualification use the same complete-write then
atomic-rename rule.
This prevents a recovered half-file or a compatible alternate fold universe
from being mistaken for the registered experiment.

The registered schemas are:

- fold: `reactflow_delta.puzzle_set_branch5_route_probe_fold.v1`;
- prediction: `reactflow_delta.puzzle_set_branch5_route_probe_prediction.v1`;
- ridge model: `reactflow_delta.puzzle_set_branch5_route_probe_ridge.v1`;
- merged: `reactflow_delta.puzzle_set_branch5_route_probe_merged.v1`;
- score: `reactflow_delta.puzzle_set_branch5_route_probe_score.v1`;
- qualification:
  `reactflow_delta.puzzle_set_branch5_route_probe_qualification.v1`.

The aligned prediction must pass four separate comparisons: signed-delta MAE
and point absolute-delta MAE, each versus the frozen V13 parent and versus the
shift-17 ridge. Every comparison must have relative gain at least 1%, a
two-sided 95% paired t-CI lower bound strictly above zero across the 20 held
puzzles, and at least 14/20 puzzles with strictly positive comparator-minus-
aligned error. The t-CI uses the sample standard deviation with `ddof=1`,
19 degrees of freedom and `t(0.975,19)=2.093024054408263`.
Relative gain is the puzzle-macro mean comparator-minus-aligned error divided
by the puzzle-macro mean comparator error, not a mean of per-puzzle ratios.
Because the 20 LOPO training sets overlap, this paired t-CI is a strict
development-routing Gate, not independent confirmatory inference. Any paper-
level generalization claim still requires a new independent puzzle/study or a
separately frozen retraining-aware uncertainty analysis.

The exact three-state decision is:

- `PASS`: the complete universe and provenance are valid and all four
  comparison Gates pass;
- `FAIL`: a complete valid score exists and at least one of the four Gates
  fails;
- `INDETERMINATE`: the universe, source provenance, target identity,
  aggregation, finite-score, coverage or prediction-integrity qualification is
  incomplete or invalid.

Only exact `PASS` can make branch 5 eligible for a later focused P1 activation
commit. `FAIL` and `INDETERMINATE` both route to P3/stop-model-rescue and cannot
be repaired by changing alpha, shift, features, threshold or Gate. The present
draft remains `NOT_RUN` and cannot itself create any of these terminal states.

## 2. Scientific question, hypothesis and falsifier

### Scientific question

Does correctly aligned, outcome-blind WT context from the other seven designs
in the same unseen puzzle add transferable mutation-effect information beyond
a strong frozen focal-construct parent and an otherwise identical
position-deranged control?

### Hypothesis

The eight registered WT constructs in one OpenKnot puzzle jointly describe a
puzzle-specific structural and measurement regime that an independently
encoded focal construct cannot fully recover. A zero-preserving aligned
cross-construct operator should therefore improve point and probabilistic
prediction over both historical comparators and a task- and
parameter-matched wrong-coordinate null.

### Falsifier

The family is falsified if any one of the following occurs:

- the candidate fails the train-only capability-retention Gate before scoring;
- the complete P1M3 candidate fails any frozen top-journal performance Gate;
- the candidate-minus-matched-null increment misses its 1--1.5% metric-specific
  attribution margin, has a non-positive puzzle-level CI lower bound, has fewer
  than 14/20 positive puzzles or fails the frozen stability checks;
- the raw-zero representation, focal exclusion, paired-dropout,
  parent-replay, target-isolation or candidate/null-matching invariants fail;
- an exact P1M3 PASS does not survive the fixed five-seed P1M4 confirmation.

Failure terminates the Puzzle-Set meta-context family. It does not authorize a
different shift, wider mixer, more epochs, another summary token, loss-weight
search, a relaxed Gate or a same-family follow-up amendment.

## 3. Frozen candidate and matched null

### Candidate

`puzzle_set_meta_context_v5_aligned` uses the registered shared full-construct
coordinate. For every focal query and position, its K/V support contains:

- seven individual tokens derived from the seven non-focal constructs; and
- one parameter-free non-focal summary token derived only from those seven
  projected non-focal states and their missingness-aware WT mean, spread and
  support statistics.

The focal construct supplies the one-token query and is absent from K/V. No
focal-query value residual is added after attention. There is no method-ID,
puzzle-ID or dataset-ID embedding.

### Matched null

`puzzle_set_meta_context_v5_shift17_null` has the same inputs, parameters,
initialization, focal query, eight-token legal attention support, optimizer,
training order, masks, dropout stream, losses, epochs and residual calibration.
Its only identifying difference is that all seven non-focal hidden,
reactivity and observed streams are circularly shifted by exactly 17 positions
before both the individual tokens and summary token are constructed.

Attention-weight dropout is zero in both arms. FFN/output residual dropout is
matched at 0.1. The null is an attribution control for correct coordinate
alignment, not a weaker model.

### Zero-preserving cross state

Both arms create a raw-zero reference before learned non-focal projection by
setting all seven non-focal hidden, reactivity and observed streams to zero.
The actual and reference paths reuse the focal query element by element and
traverse the same individual/summary projection, Q/K/V attention, FFN and
output-normalization block with one replayed dropout draw:

\[
c(q,V)=F(q,V)-F(q,V_0).
\]

The random-number state advances as one ordinary block evaluation. Therefore
projection biases, summary constants, query-only terms and downstream block
biases cancel. For arbitrary focal queries and learned parameters,
`V=0` must produce exact elementwise `c=0`. A nonzero cross state establishes
dependence on non-focal input, not predictive utility.

### Parent-preserving point residual

For both arms, the immutable same-fold V13 candidate seed-0 prediction is the
point anchor, and the immutable same-fold V14 candidate seed-0 encoder is the
outcome-blind representation source. These parents are selected before any P1
score and remain fixed for every P1 seed.

One shared point head produces

\[
g(b,c)=h([b,c])-h([b,0]),\qquad
\widehat\Delta=\widehat\Delta_{V13}+g(b,c),
\]

where `b` contains frozen focal V14 features, signed distance, mutation
identity, feature41 and the frozen V13 point. The two head evaluations reuse
one dropout draw and advance the random stream once. Thus zero cross evidence
must exactly replay the V13 parent even after arbitrary head updates.

### Frozen input-source registry

Activation must bind the realized path and exact role of every source for each
outer fold 0--19 during the score-closed P1M1 source-projection step and before
P1M2 can open. The frozen registry is:

- same-fold V13 candidate seed-0 point checkpoint: immutable point anchor,
  2,064,737 parameters, never trained by P1;
- same-fold V14 candidate seed-0 checkpoint: only its 4,767,280-parameter
  outcome-blind encoder subset is imported into the P1 point module and kept
  frozen;
- same-fold V8 seed-0 MeanAligned checkpoint: frozen source of the trained
  201-dimensional direct-feature block used by calibration, 109,581 upstream
  parameters, exact filename
  `v8_corrected_mean_fold{outer_fold}_seed0.pt`, constructed directly from the
  frozen V8 directory rather than parsed from the wide V8 fold result, and not
  the P1 point anchor;
- TIC2A corrected outer-fold `v6_feature41` weighted ridge and its
  41-dimensional feature basis from
  `tic2a_corrected_models_fold{outer_fold}.json`, together with the frozen
  unconstrained and constrained feature caches used to construct that basis;
- V10 is not a frozen training or prediction input, and the P1 scorer does not
  open any V10 fold result. After a complete prediction merge and the exact
  score token, the scorer opens the sole historical bundle
  `/mnt/cunyuliu/reactflow_delta_model_rescue_v13/v13m3_screen_seed0/v13m3_complete_score.json`.
  That exact 20-fold V13 artifact carries V12/V11/V10 comparator metrics only
  through its already-frozen transitive lineage. No learned V10 checkpoint or
  wide V10 result is a P1 point, feature or direct score input.

The realized TIC2A ridge parameter count, both cache paths, all fold-specific
source paths and the complete upstream parameter footprint are deliberately
`PENDING_ACTIVATION_BINDING`. They must be measured and recorded at activation;
this draft does not invent them.

Every fold artifact must use
`reactflow_delta.puzzle_set_meta_context_fold.proposed.v10` and carry the exact
`frozen_input_sources` records `v13_point_checkpoint`,
`v14_encoder_checkpoint`, `v8_meanaligned_checkpoint`,
`tic2a_feature41_model_artifact`, `tic2a_merged_registry`,
`unconstrained_feature_cache` and `constrained_feature_cache`. Each record
contains exactly `path`, `role`, `used_in_candidate_prediction`, `outer_fold`
and `seed`. The first four learned sources are fold-scoped; the TIC2A registry
and two caches are global. Roles and candidate-input booleans must exactly
match the machine contract. V10 is absent from these records; its distribution
absolute comparator enters only as a validated transitive field of the sole
V13M3 historical score bundle.

The terminal binding artifact is fixed at
`/mnt/cunyuliu/reactflow_delta_puzzle_set_meta_context/source_binding/puzzle_set_source_manifest.json`
with schema `reactflow_delta.puzzle_set_meta_context_source_manifest.v1`, status
`PUZZLE_SET_SOURCE_MANIFEST_BOUND`, and binding status
`REALIZED_PATHS_ROLES_AND_COUNTS_BOUND`. It contains exactly folds 0--19 and
exactly the seven training/prediction source records above, including realized
parameter counts and `trainable_in_p1=false`. The initial P1M1 authority binds
this exact absolute path as `source_manifest_path` with
`REALIZED_PATHS_ROLES_AND_COUNTS_PENDING`, keeps training and every outcome read
closed, and permits only the canonical source projector. After the projector
atomically writes and validates all twenty rows, a separate focused authority
commit changes only `source_binding_status` to
`REALIZED_PATHS_ROLES_AND_COUNTS_BOUND`. No P1M2 training authority may be
issued before that transition, and the runner rejects any CLI path not
identical to the bound authority path.

The future active authority also freezes the exact output universe for every
phase. P1M2 is confined to `p1m2_real_smoke`, P1M3 to
`p1m3_screen_seed0`, and P1M4 to `p1m4_formal_seeds0_4`, including their
single merge, score, qualification and (for P1M4) assembly paths. Runner,
merger, assembler, scorer and qualifier each compare their real CLI paths to
those flat active-authority fields before opening or writing scientific
artifacts. Before constructing `M2Universe` or reading training data, the
runner additionally requires its resolved M2 CSV path to equal the active
`m2_csv_path`; each fold then proves that its actual universe was constructed
from that same authorized path. A same-content alternate CSV is not an
equivalent training source. P1M2 engineering qualification remains score-blind
but is bound in the same way. Every production P1M2/P1M3/P1M4 entry point,
including merge, assembly, score and qualification, must reject a pending,
missing, non-absolute or invalid source manifest before opening a phase
artifact; successful runner validation alone cannot be reused as authority.

Every complete merge must use
`reactflow_delta.puzzle_set_meta_context_merged.proposed.v9`, preserve all fold
records, set `complete_frozen_input_provenance_all_runs=true`, and record the
expected candidate-specific trainable count of 1,468,165 for every run.
The fold result is the completion sentinel and is atomically renamed only after
its prediction and checkpoint artifacts exist. Complete merge, formal assembly,
score and qualification JSON files likewise become visible only through an
atomic final rename after their entire validated content is written. This is
required because interrupted remote runs are an observed execution mode, not a
hypothetical adversarial case.
Production merge also derives the exact folds, seeds, three epoch counts and
point-module parameter counts from P1M2/P1M3/P1M4. Caller CLI values cannot
redefine a canonical phase universe or occupy its merge path with a partial
run. For every fold-seed, the prediction artifact and six trained checkpoint
paths must be the frozen canonical filenames inside that phase's active
`prediction_dir`, and the fold's seven `frozen_input_sources` records must
match the currently bound source-manifest row. Structurally compatible stale
or alternate artifacts are not eligible for a production merge.

### Frozen capacity

Each P1 point module contains:

- 6,171,697 point-module parameters (not the full model pipeline);
- 4,767,280 frozen V14 encoder parameters;
- 857,600 trainable set-operator parameters;
- 546,817 trainable shared-head parameters;
- 1,404,417 trainable point-module parameters.

The temporary 769-parameter WT reconstruction decoder is stage-specific and is
not part of the 6,171,697-parameter point module. Each arm adds a 63,748-
parameter trainable residual head, so the point-plus-residual modules contain
6,235,445 parameters and candidate-specific trainable point-plus-distribution
capacity is 1,468,165. These counts exclude the upstream V13 point, V8
MeanAligned model, TIC2A ridge and feature caches. Candidate and null must have
identical realized counts. Any count change requires a new prospective
amendment and cannot be accepted as V5.

## 4. Frozen data, estimand and objectives

- Dataset: real OpenKnot M2 v4.5.2, 20 puzzles, eight registered WT constructs
  per puzzle, `DEVELOPMENT_CONSUMED`.
- Split: fixed `split_v4` outer leave-one-puzzle-out; the held puzzle is the
  independent evaluation unit.
- Prediction target: every registered mutant and every full-construct
  position, with held target, target error and qualified target mask excluded
  from the prediction artifact and model path.
- Primary aggregation: position within mutant, mutant within qualified method
  cell, equal method-cell weighting within puzzle, then equal weighting over 20
  held puzzles.
- Point objective: exact method-balanced signed-delta L1.
- Calibration objective: frozen-point, puzzle-balanced method-cell CRPS using
  the unchanged V10 median-preserving asymmetric residual family for each arm.
- Calibration cannot change either point prediction.
- No external outcome is available to P1. Existing or new external outcomes
  cannot select a model, parent, shift, epoch, checkpoint, residual family,
  threshold or Gate.

The registered `P20_Eterna` construct with zero WT-observed positions remains
available as outcome-blind context and receives full predictions, but no
supervised target is fabricated.

## 5. Frozen optimization protocol

For each outer fold and arm:

1. **Masked-WT initialization:** 200 epochs over the nineteen outer-train
   puzzles, one eligible focal construct at a time, deterministic 40% masking
   of its observed WT positions, masked-position L1, AdamW at `3e-4`, weight
   decay `0.01`, clipping `5.0`. Only the set operator and temporary decoder
   update. This yields exactly 3,800 puzzle updates per arm.
2. **Point warmup:** point epoch 0 updates only the shared point head at
   `1e-3`; the context path is bitwise frozen.
3. **Joint point fitting:** point epochs 1--39 continue the same Adam, keep the
   head at `1e-3`, update the context path at `3e-4`, use zero weight decay and
   clipping `5.0`. Across all 40 point epochs, the head receives 760 updates,
   the context receives 741 updates and every available supervised cell is
   exposed exactly 40 times.
4. **Residual calibration:** freeze the complete point model and train the
   unchanged V10 residual family for 40 epochs with the puzzle-balanced
   method-cell CRPS objective.

There is no early stopping; epoch, checkpoint, shift, seed, loss,
parameter-count, model-family or calibration selection is forbidden. The
candidate and null reset and replay the same prescribed random streams for
each matched stage.

## 6. Train-only capability-retention Gate

Before any held score can be read, each P1M3/P1M4 fold-seed run is evaluated on
only its nineteen outer-train `{puzzle, contexts}` records. One fixed mask epoch
200, disjoint from formal pretraining mask epochs 0--199, and the same final
frozen decoder evaluate the initial, post-pretraining and post-point context
snapshots using puzzle-balanced masked-position L1.

For the candidate, every fold-seed run must satisfy:

\[
L_{postpretraining}<L_{initial}
\]

and

\[
R=\frac{L_{initial}-L_{postpoint}}
        {L_{initial}-L_{postpretraining}}>0.
\]

These are recorded as `pretraining_established=true` and
`retention_positive=true`. The diagnostic must also record that no checkpoint,
model or threshold selection occurred, no mutant outcome was used and the held
puzzle was not accessed. Candidate failure produces
`PUZZLE_SET_TRAIN_ONLY_RETENTION_GATE_FAIL`, freezes the complete negative
merge and leaves held scoring closed.

The null receives the same continuous diagnostic, but its two capability
booleans are report-only and cannot block or select the candidate. The P1M2
three-epoch engineering smoke records its actual training mask range and does
not turn retention into scientific evidence.

## 7. Phase sequence and authority transitions

### P1M0 — inactive draft and future contract activation

Current state: `DRAFT_FROZEN_INACTIVE_V14_SOLE_ACTIVE`.

After a qualifying complete V14 terminal handoff, promote the existing inactive
machine-contract draft by binding the selected router branch, branch-specific
probe state and the exact source-path universe. Activate only
`P1M1/SOURCE_MANIFEST_PROJECTION_ONLY` in one focused commit with the manifest
binding still pending; training, held score, partial score and external outcome
access remain closed. If V14 routing does not select P1, this draft is archived
without execution.

### P1M1 — implementation and invariants

Run the canonical source projector from the activated clean checkout, validate
the exact twenty-fold/seven-source manifest and bind the realized source counts
and upstream footprint in a separate focused authority commit. Then rerun the
complete V5 focused test suite.
Require exact candidate/null counts, parent provenance and replay, focal
exclusion, 7+1 K/V support, shift-17 matching, raw-zero cross cancellation,
paired point-head cancellation, finite nonzero Q/K/V gradients, permutation
equivariance, target isolation, optimizer schedule and retention-diagnostic
invariants. Also require fold schema `proposed.v10`, merged schema `proposed.v9`
and complete frozen-input provenance. Only exact mechanical PASS may open
real-data smoke.

### P1M2 — real-data engineering smoke

Run only folds 0/1, seed 0 and 3 masked-WT + 3 point + 3 calibration epochs.
The only eligible training token is
`PUZZLE_SET_P1M2_REAL_DATA_ENGINEERING_SMOKE_ONLY`.
Produce prediction-only artifacts and a complete unscored engineering merge.
Do not compute a scientific score or use smoke direction to change the model.
Only exact `P1M2_ENGINEERING_SMOKE_PASS` may open P1M3.

### P1M3 — seed-0 complete screen

Run seed 0, folds 0--19 and the frozen `200+40+40` schedule in persistent
sessions under only
`PUZZLE_SET_P1M3_TWENTY_FOLD_PREDICTION_ONLY_SCREEN_ONLY`. Before all twenty
complete fold artifacts exist, inspect only process health, log
timestamps/sizes, expected filenames and non-metric engineering errors. Do not
inspect loss or partial score directions.

After all workers exit, merge once. The candidate train-only retention Gate
must pass for all twenty folds. Then close training in a focused authority
commit, set the exact held-score token to
`PUZZLE_SET_COMPLETE_MERGE_SCORE_ONCE_ONLY`, keep partial and external outcomes
closed, score once and qualify once.

Only exact `PUZZLE_SET_M3_TOP_JOURNAL_SCREEN_PASS` may open P1M4.

### P1M4 — fixed five-seed formal confirmation

Run a new independent 20-fold by seeds 0--4 universe; seed 0 is rerun rather
than copied from P1M3. Every seed independently trains both V5 arms with the
same `200+40+40` schedule under only
`PUZZLE_SET_P1M4_FIXED_FIVE_SEED_FORMAL_ONLY`. No seed, fold, epoch or
checkpoint may be selected or omitted.

After 100 complete fold-seed artifacts and candidate retention PASS for all
runs, create the complete unscored merge and the equal-seed prediction-only
assembly. The formal point is the arithmetic mean of the five seed points. Each
arm's probability distribution is a ten-component Gaussian mixture: each seed
contributes its two components and exactly one fifth of total probability mass.
The formal scorer must reconstruct the assembly from the merged source
artifacts before target access.

Then close formal training in a focused commit, set the exact score token to
`PUZZLE_SET_FORMAL_COMPLETE_SCORE_ONCE_ONLY`, keep partial and external outcomes
closed, score once and qualify once.

### P1M5 — artifact freeze and handoff

Freeze code, contracts, parent provenance, checkpoints, prediction-only
artifacts, merges, score and qualification. Close all training and outcome
access. Exact formal PASS returns a strong post-hoc development baseline to the
benchmark manuscript route; any failure returns the strongest already valid
baseline and terminates Puzzle-Set meta-context.

## 8. Top-journal qualification Gates

Every Gate is conjunctive. Relative gain means comparator error minus candidate
error, divided by comparator error, using the frozen 20-puzzle estimator.

| Candidate metric | vs feature41 | vs terminal historical comparator | vs frozen parent | vs matched shift-17 null |
|---|---:|---:|---:|---:|
| signed-delta MAE | at least 12% | at least 2% vs terminal V12 | at least 2% vs V13 parent | at least 1.5% |
| point absolute-delta MAE | at least 7% | at least 2% vs terminal V11 | at least 2% vs V13 parent | at least 1% |
| CRPS | at least 5% | at least 2% vs terminal V12 | not a registered comparison | at least 1.5% |
| distribution absolute-delta MAE | at least 15% | at least 2% vs terminal V10 | not a registered comparison | at least 1% |

For signed-delta and point absolute-delta, feature41 comparisons must be
positive on at least 16/20 puzzles; the terminal historical, V13-parent and
matched-null comparisons must each be positive on at least 14/20 puzzles. For
CRPS and distribution absolute-delta, feature41 comparisons must be positive
on at least 16/20 puzzles and the terminal historical and matched-null
comparisons each on at least 14/20.

Every registered paired comparison must have a puzzle-level 95% CI lower bound
strictly above zero, remain positive under every leave-one-puzzle-out analysis,
and receive no more than 20% of total effect from any one puzzle. Registered
prediction coverage must equal 100%, failure rate must equal zero and
unexpected prediction keys must equal zero. At both 68% and 95% nominal
coverage, the candidate's absolute coverage error may exceed feature41's by at
most 0.01.

P1M4 repeats the complete P1M3 Gate on the predeclared equal-seed mixture and
adds all of the following:

- exact P1M3 PASS is a prerequisite;
- all individual seeds have complete prediction integrity;
- `formal_assembly_reconstructed_exactly_from_same_100_run_merged_sources=true`
  before target access;
- candidate signed-delta mean gain versus feature41 is positive in at least
  four of five individual seeds;
- candidate CRPS mean gain versus feature41 is positive in at least four of
  five individual seeds.

No near miss, nominal p-value, engineering PASS, retention PASS or single-metric
PASS may be promoted to overall scientific PASS.

## 9. Claim and publication boundary

An exact P1M4 PASS may support only this bounded statement:

> On the repeatedly consumed 20-puzzle OpenKnot development benchmark,
> correctly aligned, zero-preserving cross-construct WT context added stable
> predictive information beyond a frozen focal parent and a position-deranged,
> parameter-matched null under the registered full-construct estimator.

It remains `POST_HOC_DEVELOPMENT_PASS`. It does not establish SOTA, external
replication, biological mechanism, practical or clinical utility, independent
generalization or publication readiness. Any such claim requires a separately
sealed external-confirmation amendment with genuinely independent
publication/study/batch units.

## 10. Non-negotiable boundaries

- V1--V13 terminal verdicts remain immutable.
- V14 is the sole active model experiment until its complete terminal handoff.
- Do not alter V14's frozen protocol, current Gates or eventual terminal
  handoff while preparing or activating this draft.
- Do not access a new external outcome.
- Do not use held mutant target, error or qualified target mask in prediction,
  checkpointing, training, calibration, retention or model selection.
- Do not add method-ID, puzzle-ID or dataset-ID shortcuts.
- Do not change the candidate, matched null, shift, 7+1 support, raw-zero
  construction, parameter count, optimizer, epoch schedule, losses, residual
  family, seeds or Gates.
- Do not activate P1 and P2 together.
- Do not score incomplete P1M3/P1M4 universes or inspect partial directions.
- Do not overwrite complete artifacts or drop failed folds/seeds.
- GPU0--7 may be used when memory is sufficient and safe co-location is
  possible, but unrelated processes may not be preempted, signalled or changed.
- Persistent sessions and 900-second low-frequency health monitoring are
  required during scientific runs.
- Any P1 failure terminates this model family and returns the project to its
  strongest qualified benchmark/measurement route.

## 11. Frozen evidence sources

- `docs/adr/0001-propose-puzzle-set-meta-context.md`
- `docs/plans/2026-08-27-post-v14-model-contingency.md`
- `docs/plans/2026-08-27-puzzle-set-position-alignment-design.md`
- `docs/plans/2026-08-27-puzzle-set-parent-preserving-design.md`
- `docs/plans/2026-08-27-puzzle-set-cross-only-retention-design.md`
- `scripts/reactflow_delta/puzzle_set_meta_context.py`
- `scripts/reactflow_delta/puzzle_set_meta_context_pretraining.py`
- `scripts/reactflow_delta/puzzle_set_meta_context_retention.py`
- `scripts/reactflow_delta/puzzle_set_safe_sources.py`
- `scripts/reactflow_delta/preflight_puzzle_set_meta_context_sources.py`
- `scripts/reactflow_delta/project_puzzle_set_meta_context_sources.py`
- `scripts/reactflow_delta/puzzle_set_score_chain.py`
- `scripts/reactflow_delta/run_puzzle_set_meta_context_probe.py`
- `scripts/reactflow_delta/merge_puzzle_set_meta_context_probe.py`
- `scripts/reactflow_delta/qualify_puzzle_set_meta_context_smoke.py`
- `scripts/reactflow_delta/score_puzzle_set_meta_context.py`
- `scripts/reactflow_delta/qualify_puzzle_set_meta_context.py`
- `scripts/reactflow_delta/assemble_puzzle_set_meta_context_formal.py`
- `scripts/reactflow_delta/score_puzzle_set_meta_context_formal.py`
- `scripts/reactflow_delta/qualify_puzzle_set_meta_context_formal.py`
- `scripts/reactflow_delta/run_puzzle_set_meta_context_smoke_controller.sh`
- `scripts/reactflow_delta/run_puzzle_set_meta_context_screen_controller.sh`
- `scripts/reactflow_delta/run_puzzle_set_meta_context_score_once.sh`
- `scripts/reactflow_delta/run_puzzle_set_meta_context_formal_controller.sh`
- `scripts/reactflow_delta/run_puzzle_set_meta_context_formal_score_once.sh`
- `scripts/reactflow_delta/validate_puzzle_set_meta_context_v5_contract.py`
