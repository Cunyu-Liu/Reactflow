# Model Rescue v12 Monotone Residual Shrinkage Implementation Plan

> **Execution policy:** Implement by functional batch under the global lightweight development and verification strategy. Do not apply task-by-task review or full-validation gates unless the batch risk warrants them.

**Goal:** Build and prospectively test one continuous monotone, outcome-blind gate that shrinks the frozen V11 neural residual only where legal inner-OOF evidence supports shrinkage.

**Architecture:** Reuse the exact V11 anchored point as the parent and compute `feature41 + gate × (V11 − feature41)`. Fit the four-parameter product-logistic gate on puzzle-grouped inner-OOF predictions, freeze the gated point, and fit the unchanged V10 median-constrained asymmetric residual family.

**Tech Stack:** Python, PyTorch, NumPy, SciPy, YAML, split_v4, evaluator_v2, tmux, pytest.

---

### Batch 1: Contract validator and monotone gate core

**Files:**
- Create: `scripts/reactflow_delta/model_rescue_v12.py`
- Create: `scripts/reactflow_delta/validate_model_rescue_v12_contract.py`
- Create: `tests/reactflow_delta/test_model_rescue_v12.py`
- Modify: `configs/reactflow_delta/active_contract.yaml`
- Modify: `docs/prospective_v2/model_rescue_v12_decision_ledger.yaml`

**Risk:** High — a gate that is not monotone, uses target-derived inputs, or changes the V11 parent would invalidate the only scientific contrast.

**Implementation:** Implement the four-parameter product-logistic gate, exact `feature41 + g × residual` composition, fixed-one parent null, method-balanced inner-OOF L1, gate serialization, contract consistency checks, and fail-closed training authority. Do not implement alternative gates or thresholds.

**Minimum verification:** Focused tests prove gate range and monotonicity, fixed-one null exact replay, no method/target inputs, method-balanced weighting, deterministic initialization, and contract/active/ledger agreement.

**Independent review:** Yes — the nested scientific contrast and leakage boundary are high-risk.

### Batch 2: Inner crossfit runner and prediction-only artifacts

**Files:**
- Create: `scripts/reactflow_delta/run_model_rescue_v12.py`
- Create: `scripts/reactflow_delta/qualify_model_rescue_v12_smoke.py`
- Create: `scripts/reactflow_delta/run_model_rescue_v12_smoke_controller.sh`
- Modify: `tests/reactflow_delta/test_model_rescue_v12.py`

**Risk:** High — an inner-held puzzle entering point training or an outer-held target entering gate fit would create direct leakage.

**Implementation:** Reuse split_v4 inner groups, train exact V11 inner models, generate one OOF prediction for every outer-train puzzle, fit the gate once, load authoritative V11 outer predictions/checkpoints, fit the unchanged residual family around the gated point, and write target-free full-key artifacts plus the inner-crossfit ledger.

**Minimum verification:** Tensor fixtures plus folds0/1 real-data smoke prove inner disjointness/completeness, target invariance, parent replay, point freeze, median invariance, finite optimization, full registered output, duplicate/missing rejection, and no scientific score generation.

**Independent review:** Yes — leakage and parent replay can directly reverse the scientific verdict.

### Batch 3: Complete merge, scorer, qualifier, and controllers

**Files:**
- Create: `scripts/reactflow_delta/merge_model_rescue_v12.py`
- Create: `scripts/reactflow_delta/score_model_rescue_v12.py`
- Create: `scripts/reactflow_delta/qualify_model_rescue_v12.py`
- Create: `scripts/reactflow_delta/run_model_rescue_v12_screen_controller.sh`
- Modify: `tests/reactflow_delta/test_model_rescue_v12.py`

**Risk:** High — partial scoring, pooled aggregation, or a permissive qualifier would turn an exploratory gate into false top-journal evidence.

**Implementation:** Require folds0–19 exactly once, merge prediction-only files, join targets only after complete merge, calculate puzzle-level method-balanced signed MAE, point absolute MAE, CRPS, distribution absolute MAE, coverage and integrity, then mechanically apply every frozen screen Gate.

**Minimum verification:** Tests reject missing/duplicate folds, target fields in predictions, wrong key universes, pooled-mutant aggregation, any failed headline Gate, and score access before a committed authority transition.

**Independent review:** Yes — this batch produces the scientific verdict.

### Batch 4: Formal five-seed assembly, qualification, and handoff

**Files:**
- Create: `scripts/reactflow_delta/assemble_model_rescue_v12_formal.py`
- Create: `scripts/reactflow_delta/score_model_rescue_v12_formal.py`
- Create: `scripts/reactflow_delta/qualify_model_rescue_v12_formal.py`
- Create: `scripts/reactflow_delta/run_model_rescue_v12_formal_controller.sh`
- Modify: `tests/reactflow_delta/test_model_rescue_v12.py`
- Modify: `docs/prospective_v2/model_rescue_v12_decision_ledger.yaml`
- Modify: `configs/reactflow_delta/active_contract.yaml`

**Risk:** High — seed selection or unequal mixture weights would inflate the final effect.

**Implementation:** Only after exact screen PASS, run seeds0–4 without selection, give every seed equal mass, apply the screen Gates and 4/5 seed direction Gate, freeze artifacts, close training, and hand the qualified result back to M6.

**Minimum verification:** Focused formal tests prove complete 100 fold×seed universe, equal seed weights, no failed-seed deletion, single scorer/qualifier semantics, and terminal claim boundaries.

**Independent review:** Yes — formal confirmation is the last internal evidence gate before any sealed external amendment.
