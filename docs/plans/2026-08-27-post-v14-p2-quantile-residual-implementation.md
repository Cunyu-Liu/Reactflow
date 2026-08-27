# Post-V14 P2 Quantile Residual Implementation Plan

> **Execution policy:** Implement by functional batch under the global lightweight development and verification strategy. Do not apply task-by-task review or full-validation gates unless the batch risk warrants them.

**Goal:** Build an inactive, fail-closed P2 quantile-residual pipeline that can later run only after exact post-V14 branch-6 diagnostic PASS authority.

**Architecture:** A dedicated 244-input monotone fixed-grid quantile head preserves the frozen V14 point as its exact median and is compared with a newly trained, exactly parameter-matched V10 median-asymmetric replay. Source projection, prediction-only execution, complete merge, once-only scoring, and mechanical qualification remain separate authority phases; inactive implementation never activates itself.

**Tech Stack:** Python 3, PyTorch, NumPy, SciPy, PyYAML, pytest, Bash controllers, and existing ReactFlow-Delta split/source/scoring utilities.

---

## Global execution constraints

- Use @Code for implementation/focused verification and @git-essentials only
  for scoped commit/push work.
- Implement in a fresh worktree based on the then-approved staging commit, not
  in active V14 or this design-prep branch.
- Preparation may add inactive files/tests. Only a later explicit authority
  commit may change the canonical active pointer or issue a runnable token.
- Batches 1-2 perform no `/mnt` or V14 artifact access. Batches 3-4 use only
  synthetic/mocked fixtures until exact terminal binding is authorized.
- Inactive implementation runs no training or GPU validation. A later real
  smoke must require CUDA and stop on CPU fallback. Do not add a memory Gate.
- Code/commits stay under `/home`; future large artifacts use only bound
  `/mnt/cunyuliu` paths.
- Use focused tests. Run a broader suite only if a focused failure identifies a
  real cross-module risk.

### Batch 1: Freeze inactive amendment, human contract, ledger, and validator

**Files:**

- Create: `configs/reactflow_delta/post_v14_p2_quantile_residual_amendment.yaml`
- Create: `docs/prospective_v2/post_v14_p2_quantile_residual_amendment_20260827.md`
- Create: `docs/prospective_v2/post_v14_p2_quantile_residual_decision_ledger.yaml`
- Create: `scripts/reactflow_delta/validate_post_v14_p2_quantile_residual_contract.py`
- Test: `tests/reactflow_delta/test_validate_post_v14_p2_quantile_residual_contract.py`

**Risk:** Medium — no runtime authority changes, but ambiguity could later bind
the wrong parent or alter a frozen scientific decision.

**Implementation:** Transcribe
`docs/plans/2026-08-27-post-v14-p2-quantile-residual-design.md` into one machine
amendment, one readable contract, and one ledger. Label all numerical decisions
as new focused pre-score choices. Set `DRAFT_FROZEN_INACTIVE`, with activation,
source projection, training, prediction, smoke, screen, scoring, qualification,
formal confirmation, partial-score access, and external access all false.

Require exact branch `6`, classification `DISTRIBUTION_ONLY_FAILURE`, diagnostic
schema `reactflow_delta.post_v14_branch6_tail_diagnostic.v1`, diagnostic PASS,
primary statistic `LOWER_MINUS_UPPER_TAIL_MISS90`, and next action
`OPEN_FOCUSED_P2_AMENDMENT_AUTHORITY`. Keep actual terminal paths and copied V14
Gate values `PENDING_TERMINAL_BINDING`; validator must reject an active/runnable
interpretation while they are pending.

Freeze the input order/widths, taus, weights, objectives, candidate/comparator
parameter formulas, training phases, artifact schemas, matched-replay Gates,
formal mixture, integrity rules, forbidden actions, and claim ceiling. Load the
canonical repo amendment/active pointer rather than accepting a CLI authority
override. Validate both top-level and nested inactive booleans, exact arrays and
sums, parameter counts, unique median tau, phase state, parent tokens, pending
bindings, and absence of a generic training token.

Focused tests prove the committed inactive contract passes and that any wrong
parent field, reopened authority/access, altered number/array/formula/schedule/
Gate, realized source inserted into the draft, or generic token fails. They also
prove validation reads no artifact and creates no output. Do not create or
modify `configs/reactflow_delta/active_contract.yaml`.

**Minimum verification:**

```bash
pytest -q tests/reactflow_delta/test_validate_post_v14_p2_quantile_residual_contract.py
git diff --check
```

**Independent review:** No — keep this coupled contract/validator/test batch
with one owner; review at the mathematical and final scientific milestones.

### Batch 2: Implement quantile core and matched V10 replay

**Files:**

- Create: `scripts/reactflow_delta/post_v14_p2_quantile_residual.py`
- Test: `tests/reactflow_delta/test_post_v14_p2_quantile_residual.py`

**Risk:** High — median drift, wrong quadrature, non-monotone output, or a false
parameter match invalidates attribution despite a clean run.

**Implementation:** Define constants for the 244 input channels, 13 taus and
weights, 12 gaps, hidden width 248, gap floor `1e-4`, and expected 63,748
parameters. Reuse `TrainOnlyStandardizer`, `calibration_input`, and
`MedianAsymmetricResidual` from
`scripts/reactflow_delta/model_rescue_v10.py`; do not add a generic family
wrapper.

Implement `MonotoneQuantileResidual` as
`Linear(244,248) -> ReLU -> Linear(248,12)`. Keep layers float32; construct
float64 positive gaps; assign the detached float64 point directly at tau `0.5`;
and cumulatively subtract/add gaps. Implement fixed weighted pinball CRPS,
weighted expected absolute delta, parameter count, deterministic
V10-grid-matched initialization, outer-train standardization, and the frozen
position-to-mutant-to-method-cell-to-puzzle loss.

One paired fitting entry point accepts authorized outer-train rows and a frozen
point; creates candidate/comparator from the same seed; gives them identical
rows, standardizer, puzzle order, epochs, Adam `1e-3`, zero weight decay, and
clip `5.0`; checks point snapshots/gradients; performs no early stopping,
selection, held scoring, or artifact I/O; and returns prediction primitives plus
histories without a verdict.

Initialize candidate output weights to zero. Use bounded float64 bisection to
obtain the existing V10 initial mixture's fixed-grid quantiles and set the 12
biases to inverse-softplus adjacent gaps. Test the bound/tolerance rather than
adding a runtime recovery loop.

Focused tests cover exact arrays and `0.45/0.10/0.45` mass; hand-computable loss
and weighted absolute values; strict monotonicity; exact median and absent point
gradient; both 63,748 counts; identical rows/statistics/seed/epochs/order;
initial-grid equality; finite/alignment rejection; and point immutability after
a minimal synthetic optimization step.

**Minimum verification:**

```bash
pytest -q tests/reactflow_delta/test_post_v14_p2_quantile_residual.py
git diff --check
```

**Independent review:** Yes — independently verify quantile algebra,
quadrature, parameter counts, initialization, V10 fairness, and the point
gradient boundary before runtime work.

### Batch 3: Implement source projection, runtime/controller, and merger

**Files:**

- Create: `scripts/reactflow_delta/project_post_v14_p2_quantile_sources.py`
- Create: `scripts/reactflow_delta/run_post_v14_p2_quantile_residual.py`
- Create: `scripts/reactflow_delta/run_post_v14_p2_quantile_residual_controller.sh`
- Create: `scripts/reactflow_delta/merge_post_v14_p2_quantile_residual.py`
- Test: `tests/reactflow_delta/test_project_post_v14_p2_quantile_sources.py`
- Test: `tests/reactflow_delta/test_run_post_v14_p2_quantile_residual.py`
- Test: `tests/reactflow_delta/test_run_post_v14_p2_quantile_residual_controller.py`
- Test: `tests/reactflow_delta/test_merge_post_v14_p2_quantile_residual.py`

**Risk:** High — wrong fold/seed sources, held-target access, incomplete universe,
or CPU fallback would invalidate the experiment.

**Implementation:** The projection command loads canonical
`configs/reactflow_delta/active_contract.yaml` before output creation or source
read. It accepts only exact P2M1 projection authority and binds 20 same-fold
records for V14 candidate checkpoint/prediction, V8 MeanAligned checkpoint,
TIC2A feature41 model/registry/caches, and M2 identity source. Manifest records
contain paths, roles, fold, held puzzle, seed, parameter counts, and trainability
but no target, error, loss, score, or history.

The runner validates exact phase authority, bound CLI paths, phase universe,
no-overwrite state, and CUDA before any reader/directory/model creation. It
builds outer-train calibration rows, uses the V14 checkpoint for frozen
outer-train points, reads held `candidate_point` directly by biological key,
and generates candidate quantiles plus V10 replay mixtures. Held-target/scorer
code is absent from its import/data path.

The phase-aware controller dispatches only:

```text
P2M2 folds0/1 seed0 3 epochs
P2M3 folds0-19 seed0 40 epochs
P2M4 folds0-19 seeds0-4 40 epochs
```

It skips only complete independently valid fold artifacts, propagates child
failure, never launches a scorer, never polls metrics, and has no memory Gate.

The merger revalidates canonical authority, manifest, exact fold/seed universe,
filenames, schemas, keys, same-fold provenance, parameter counts, taus/weights,
point replay, monotonicity, finite values, coverage, and forbidden fields. It
atomically writes one prediction-only merge, refuses overwrite, never joins
targets, and never reads scores.

Focused synthetic/mock tests prove that wrong authority/parent/source/fold/
seed/epoch/path/key/point/count/order, a target field, absent CUDA, or incomplete
universe fails before the relevant reader/output side effect. Valid synthetic
phase universes merge without target or score access.

**Minimum verification:**

```bash
pytest -q \
  tests/reactflow_delta/test_project_post_v14_p2_quantile_sources.py \
  tests/reactflow_delta/test_run_post_v14_p2_quantile_residual.py \
  tests/reactflow_delta/test_run_post_v14_p2_quantile_residual_controller.py \
  tests/reactflow_delta/test_merge_post_v14_p2_quantile_residual.py
git diff --check
```

**Independent review:** No at this boundary — keep coupled authority/runtime
files with one owner; any active-pointer proposal is outside this batch and must
be reviewed separately.

### Batch 4: Implement scorer, qualifier, and formal assembly

**Files:**

- Create: `scripts/reactflow_delta/score_post_v14_p2_quantile_residual.py`
- Create: `scripts/reactflow_delta/qualify_post_v14_p2_quantile_residual.py`
- Create: `scripts/reactflow_delta/assemble_post_v14_p2_quantile_formal.py`
- Test: `tests/reactflow_delta/test_score_post_v14_p2_quantile_residual.py`
- Test: `tests/reactflow_delta/test_qualify_post_v14_p2_quantile_residual.py`
- Test: `tests/reactflow_delta/test_assemble_post_v14_p2_quantile_formal.py`

**Risk:** High — aggregation, comparator, CI, mixture, or status errors could
create a false scientific PASS.

**Implementation:** The screen scorer accepts only exact P2M3 score-once
authority and a complete validated merge. Validate authority/bound paths before
target read or output creation, join the exact M2 target universe once, compute
the frozen hierarchical puzzle estimands, and repeat every canonical V14
feature41/terminal CRPS, distribution-absolute, coverage, and calibration Gate
unchanged. Signed/point-absolute values replay from the frozen V14 point.

Add both paired capability Gates:

```text
CRPS relative gain vs matched V10 >= 0.015
distribution-absolute relative gain vs matched V10 >= 0.01
paired 95% CI lower > 0 for both
positive puzzles >= 14/20 for both
all headline LOO effects positive
maximum single-puzzle effect fraction <= 0.20
```

The qualifier is mechanical: all integrity, replayed V14, new capability, and
guardrail conditions yield PASS; a complete valid scientific Gate miss yields
FAIL; incomplete/invalid/nonfinite/unauthorized/point-drift state yields
INDETERMINATE. Smoke, loss, or proxy evidence cannot become PASS.

Formal assembly accepts exact P2M4 authority and 100 complete prediction-only
runs. It constructs 65 candidate atoms per row with `weight[tau] / 5`, never
averages quantile curves, and equally mixes the five V10 replay distributions.
Formal scoring repeats screen Gates, reports every seed, and requires at least
4/5 positive individual-seed matched-V10 increments for CRPS and separately
4/5 for distribution-absolute error. Failed seeds cannot be removed.

All outputs are atomic/no-overwrite, bind exact sources/tokens, and cap PASS at
`POST_HOC_DEVELOPMENT_FORMAL_PASS`. Tests use hand-computable fixtures for
pinball CRPS, finite-mixture CRPS, weighted absolute error, hierarchy, paired
CI, puzzle counts, LOO, influence, boundaries, 65 weights, 4/5 logic,
PASS/FAIL/INDETERMINATE, score-once, and rejection before target read.

**Minimum verification:**

```bash
pytest -q \
  tests/reactflow_delta/test_score_post_v14_p2_quantile_residual.py \
  tests/reactflow_delta/test_qualify_post_v14_p2_quantile_residual.py \
  tests/reactflow_delta/test_assemble_post_v14_p2_quantile_formal.py
git diff --check
```

**Independent review:** Yes — independently recompute counts, quadrature,
65-atom mixture, paired estimands, Gate inequalities, and terminal mapping
before any activation proposal.

## Final inactive milestone

After all batches, run only the focused P2 suite and directly reused V10 core/
contract tests, followed by `git diff --check`. The final reviewer confirms
inactive status, no `active_contract.yaml` change, no real artifact reads in
tests, exact parameter match, no CPU fallback or memory Gate, no partial score,
and the claim ceiling.

Passing this milestone means only that inactive preparation is ready for a
future terminal-binding decision. It is not P2 scientific PASS and does not
authorize P2M1.

## Execution handoff

Recommended grouping: keep Batch 1 with one contract owner; give Batch 2 to one
mathematical-model owner and review it independently; keep Batches 3-4 with one
pipeline owner, followed by the final independent scientific review. Do not
split coupled model/scorer algebra across agents, and do not open an authority
task before exact V14 router/diagnostic terminal state is available.
