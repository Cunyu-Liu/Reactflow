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
requirement and route-probe status are all pending/`NOT_EVALUATED`; this is not
evidence that P1 will become eligible.

No P1 training, target join or scientific score is authorized while the status
is `DRAFT_FROZEN_INACTIVE_V14_SOLE_ACTIVE`.

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
outer fold 0--19 before P1M1. The frozen registry is:

- same-fold V13 candidate seed-0 point checkpoint: immutable point anchor,
  2,064,737 parameters, never trained by P1;
- same-fold V14 candidate seed-0 checkpoint: only its 4,767,280-parameter
  outcome-blind encoder subset is imported into the P1 point module and kept
  frozen;
- same-fold V8 seed-0 MeanAligned checkpoint: frozen source of the trained
  201-dimensional direct-feature block used by calibration, 109,581 upstream
  parameters, exact filename
  `v8_corrected_mean_fold{outer_fold}_seed0.pt`, not the P1 point anchor;
- TIC2A corrected outer-fold `v6_feature41` weighted ridge and its
  41-dimensional feature basis from
  `tic2a_corrected_models_fold{outer_fold}.json`, together with the frozen
  unconstrained and constrained feature caches used to construct that basis;
- the same-fold terminal V10 row only as historical-comparator and residual-
  family provenance under
  `v10_fold_result_fold{outer_fold}_seed0.json`. No learned V10 checkpoint is a
  P1 point or feature input.

The realized TIC2A ridge parameter count, both cache paths, all fold-specific
source paths and the complete upstream parameter footprint are deliberately
`PENDING_ACTIVATION_BINDING`. They must be measured and recorded at activation;
this draft does not invent them.

Every fold artifact must use
`reactflow_delta.puzzle_set_meta_context_fold.proposed.v10` and carry the exact
`frozen_input_sources` records `v13_point_checkpoint`,
`v14_encoder_checkpoint`, `v8_meanaligned_checkpoint`,
`tic2a_feature41_model_artifact`, `tic2a_merged_registry`,
`unconstrained_feature_cache`, `constrained_feature_cache` and
`v10_fold_comparator`. Each record contains exactly `path`, `role`,
`used_in_candidate_prediction`, `outer_fold` and `seed`. The first four learned
sources and V10 comparator are fold-scoped; the TIC2A registry and two caches
are global. Roles and candidate-input booleans must exactly match the machine
contract, in particular V10 is
`COMPARATOR_PROVENANCE_ONLY_NOT_CANDIDATE_INPUT`.

Every complete merge must use
`reactflow_delta.puzzle_set_meta_context_merged.proposed.v9`, preserve all fold
records, set `complete_frozen_input_provenance_all_runs=true`, and record the
expected candidate-specific trainable count of 1,468,165 for every run.

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
probe state, exact frozen-input paths/roles/counts and the full upstream
footprint. Then rerun contract validation and focused tests and activate P1 in
one new focused commit. Training remains closed during that commit. If V14
routing does not select P1, this draft is archived without execution.

### P1M1 — implementation and invariants

Rerun the complete V5 focused test suite from the activated clean checkout.
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
- `scripts/reactflow_delta/run_puzzle_set_meta_context_probe.py`
- `scripts/reactflow_delta/merge_puzzle_set_meta_context_probe.py`
- `scripts/reactflow_delta/qualify_puzzle_set_meta_context_smoke.py`
- `scripts/reactflow_delta/score_puzzle_set_meta_context.py`
- `scripts/reactflow_delta/qualify_puzzle_set_meta_context.py`
- `scripts/reactflow_delta/assemble_puzzle_set_meta_context_formal.py`
- `scripts/reactflow_delta/score_puzzle_set_meta_context_formal.py`
- `scripts/reactflow_delta/qualify_puzzle_set_meta_context_formal.py`
- `scripts/reactflow_delta/validate_puzzle_set_meta_context_v5_contract.py`
