# ReactFlow-Delta Model Rescue v2 Implementation Plan

> **Execution policy:** Implement by functional batch under the global lightweight development and verification strategy. Do not apply task-by-task review or full-validation gates unless the batch risk warrants them.

**Goal:** Build and evaluate a mean-first, calibration-second residual distribution model without changing the B1 backbone or the immutable v1 failure result.

**Architecture:** Train the B1 K_rank=0 encoder and direct mean head from scratch using method-balanced signed-delta L1. Freeze that complete model, then fit either a global zero-mean Gaussian residual scale or a conditional two-Gaussian zero-mean scale mixture using exact CRPS; calibration can never alter the point mean.

**Tech Stack:** Python, PyTorch, NumPy/SciPy, pytest, existing OpenKnot M2 universe/split_v4/evaluator_v2.

---

### Batch 1: Contract and authority

**Files:** v2 human/machine contract, decision ledger, active pointer, contract validator.

**Risk:** High — a wrong pointer could reopen prohibited v1 phases or external access.

**Implementation:** Bind the amendment to parent head `00c0cf3`, leave v1 immutable, open only R2M1, and keep training/external outcomes closed until implementation invariants pass.

**Minimum verification:** YAML parse, path/phase/claim consistency test, git diff confirming no v1 contract modifications.

**Independent review:** Yes — milestone authority transition controls every later scientific action.

### Batch 2: Mean and residual model

**Files:** `model_rescue_v2.py`, `run_model_rescue_v2.py`, focused model/runner tests.

**Risk:** High — gradient leakage from calibration into mean would invalidate the central hypothesis.

**Implementation:** Add explicit mean forward, cell-balanced L1, mean freeze, global Gaussian calibrator, conditional zero-mean scale mixture, exact differentiable mixture CRPS, and prediction-only artifacts.

**Minimum verification:** CRPS parity, single-component reduction, gradient isolation, state immutability, point-mean identity, target-invariance, full key coverage, and method-balance fixtures.

**Independent review:** Yes — the mean/calibration boundary is the core scientific intervention.

### Batch 3: Mechanical qualification

**Files:** `qualify_model_rescue_v2.py` and focused qualifier tests.

**Risk:** High — an incorrect Gate could turn calibration-only gains into a false method PASS.

**Implementation:** Implement the layered R2M3 Mean/Calibration Gate and the fixed R2M4 five-seed Gate exactly as frozen, including puzzle CIs, relative effects, LOO, influence and coverage guardrails.

**Minimum verification:** handcrafted pass/fail fixtures for each individual condition and rejection of incomplete/duplicate folds or seeds.

**Independent review:** Yes — milestone scientific verdict generation.

### Batch 4: Real-data smoke and screen

**Files:** result artifacts and phase handoffs only.

**Risk:** High — real outcomes are development-consumed and must not trigger post-hoc configuration changes.

**Implementation:** Run P01/P02 engineering smoke, then seed-0 20-fold screen on GPU0–5 using disjoint fold shards. Do not use smoke or partial-fold scores for selection.

**Minimum verification:** complete artifacts, 100% registered coverage, zero failure/unexpected keys, target-invariance, shared mean checkpoint, qualifier replay.

**Independent review:** Yes — R2M3 determines whether formal computation is authorized.

**Recovery note (2026-08-22):** The original R2M3 process stopped after verified folds 0 and 1.
Record the execution substate as `R2M3_INTERRUPTED_RECOVERABLE` while leaving the scientific
Gate `IN_PROGRESS`. Preserve folds 0/1, run only missing folds 2--19 in the persistent tmux
session `reactflow_delta_r2m3_recovery_20260822`, and save a persistent log. Monitor at most
once every four hours unless the process exits. Monitoring is health-only: do not inspect
partial CRPS, signed-delta MAE, puzzle effects, or Gate direction. Merge all 20 folds and run
the frozen qualifier only after every fold artifact exists. R2M4 remains closed unless both
the Mean Gate and Calibration Gate pass.

### Batch 5: Formal confirmation and handoff

**Files:** `run_model_rescue_v2_formal.py`, formal artifacts, decision ledger, active
contract and manuscript handoff.

**Risk:** High — final evidence qualification and return to the main contract.

**Implementation:** Only after R2M3 PASS, run fixed B1/candidate seeds0–4 without nested selection, evaluate the unique mixtures, freeze artifacts, close training and return to main M6. On failure, skip further model search.

**Minimum verification:** complete 20×5 universe for both models, headline-mixture replay, Gate validation, clean-checkout contract/table replay.

**Independent review:** Yes — final delivery and claim boundary.

**Terminal execution result (2026-08-22):** R2M3 completed all 20 seed-0 folds. The
Calibration Gate passed, but the Mean Gate failed because the relative signed-delta MAE
gain was 0.883%, below the frozen 1% threshold. R2M4 was therefore not authorized or run.
R2M5 closed training and returned the project to main-contract M6 with
`CALIBRATION_BASELINE_ONLY / BENCHMARK_ROUTE_LOCKED`. No third rescue is permitted.
