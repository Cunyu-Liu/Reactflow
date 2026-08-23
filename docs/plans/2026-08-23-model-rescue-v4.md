# Model Rescue v4 Implementation Plan

> **Execution policy:** Implement by functional batch under the global lightweight development and verification strategy. Do not apply task-by-task review or full-validation gates unless the batch risk warrants them.

**Goal:** Build and prospectively qualify one mutation-conditioned 1D/2D RNA response transformer that can clear a strict dual-metric top-journal development gate without modifying v1–v3 conclusions.

**Architecture:** Use frozen paired WT/exact-mutant RNA-FM embeddings, a 512-wide WT sequence tower, a 128-wide pair tower, and a mutation-conditioned receiver tower. Train the signed-delta mean first with method-balanced L1, freeze it, then fit a strictly zero-mean residual distribution by exact CRPS.

**Tech Stack:** Python 3.10, PyTorch 2.5, NumPy, pandas, SciPy, PyYAML, pytest, OpenKnot M2 v4.5.2, official `ml4bio/RNA-FM`.

---

### Batch 1: Contract, authority, provenance, and phase graph

**Files:**
- Create: `docs/prospective_v2/model_rescue_v4_amendment_20260823.md`
- Create: `configs/reactflow_delta/model_rescue_v4_amendment.yaml`
- Create: `docs/prospective_v2/model_rescue_v4_decision_ledger.yaml`
- Create: `tests/reactflow_delta/test_model_rescue_v4_contract.py`
- Modify: `configs/reactflow_delta/active_contract.yaml`

**Risk:** High — an authority error could overwrite v1/v2 terminal evidence or allow v4 to interfere with the in-progress v3 run.

**Implementation:** Record parent HEAD `a560a179d620f5be083c8fce617e3cc0a7016908`, immutable v1/v2 results, concurrent v3 diagnostic status, GPU resource partition, foundation provenance, unique candidate, fixed controls, phase transitions, top-journal development and external gates, prohibited outcome access, and failure handoff. Start fail-closed at `V4M1_IMPLEMENTATION_ONLY` with real-data training disabled.

**Minimum verification:** Parse all YAML and run `pytest -q tests/reactflow_delta/test_model_rescue_v4_contract.py`. Confirm the original v3 worktree is unchanged by comparing its HEAD and status with the pre-v4 snapshot.

**Independent review:** Yes — the resource-partitioned authority and immutable-parent semantics are a scientific governance boundary.

### Batch 2: Foundation cache and model invariants

**Files:**
- Create: `scripts/reactflow_delta/precompute_model_rescue_v4_rnafm.py`
- Create: `scripts/reactflow_delta/model_rescue_v4.py`
- Create: `tests/reactflow_delta/test_model_rescue_v4_foundation.py`
- Create: `tests/reactflow_delta/test_model_rescue_v4.py`

**Risk:** High — an outcome-bearing cache, incorrect mutation coordinate, non-frozen foundation, or calibration gradient into the mean would invalidate the method.

**Implementation:** Build an outcome-blind sequence registry using only `id`, `puzzle`, `method`, and `sequence`; load the pinned RNA-FM checkpoint; export WT and exact-mutant nucleotide embeddings plus a provenance manifest. Implement pair-biased attention, axial pair updates, WT tower, mutation receiver tower, mean head, capacity-matched null, scratch and foundation-only controls, zero-mean residual calibrator, and mechanical parameter accounting.

**Minimum verification:** Tensor-fixture tests for exact-mutant sequence construction, full/design coordinate separation, target-invariant forward output, output shapes, finite forward/backward, pair-source row/column use, frozen RNA-FM gradients, calibration mean invariance, mixture locations, parameter range, and null/main parameter ratio.

**Independent review:** Yes — the 2D tensor/data flow and gradient boundaries are the central novelty and highest implementation risk.

### Batch 3: Training runner, prediction ledger, merge, and qualifier

**Files:**
- Create: `scripts/reactflow_delta/run_model_rescue_v4.py`
- Create: `scripts/reactflow_delta/merge_model_rescue_v4.py`
- Create: `scripts/reactflow_delta/qualify_model_rescue_v4.py`
- Create: `tests/reactflow_delta/test_run_model_rescue_v4.py`
- Create: `tests/reactflow_delta/test_merge_model_rescue_v4.py`
- Create: `tests/reactflow_delta/test_qualify_model_rescue_v4.py`

**Risk:** High — pooled-mutant training, partial-fold inspection, held-target leakage, or a hand-edited verdict would make the result scientifically uninterpretable.

**Implementation:** Train each puzzle-method cell with exact hierarchical L1, chunk mutants without changing cell weight, run all fixed controls, serialize prediction-only full-construct ledgers, score by independent evaluator join, reject duplicate/missing folds, and mechanically apply seed-0 and formal gates. The runner accepts only the frozen phases, folds, seeds, epochs, model IDs, authorized GPU0–7 mapping, and artifact roots.

**Minimum verification:** Focused tests for cell-weight invariance under mutant duplication, all-position output, held target/error/mask invariance, duplicate/missing fold rejection, no partial qualification, exact gate boundary behavior, and replay-stable verdict generation.

**Independent review:** Yes — this batch controls estimand, leakage, and statistical qualification.

### Batch 4: Remote environment qualification and real-data engineering smoke

**Files:**
- Modify after evidence: `configs/reactflow_delta/active_contract.yaml`
- Append: `docs/prospective_v2/model_rescue_v4_decision_ledger.yaml`
- Create artifact: `/mnt/cunyuliu/reactflow_delta_model_rescue_v4/v4m2_real_smoke/`

**Risk:** Medium — checkpoint/dependency incompatibility or pair-memory pressure can prevent execution but must not change the scientific candidate.

**Implementation:** Install or create a pinned environment for RNA-FM without changing the v3 environment; generate the outcome-blind embedding cache; run P01/P02, seed 0, at most three mean and calibration epochs on an authorized GPU0–7 with sufficient available memory; verify finite gradients, parameter counts, prediction universe, foundation freeze, target invariance, and memory headroom. Do not read or use smoke score direction.

**Minimum verification:** Run the v4 focused suite in the remote environment and the engineering qualifier. Record `ENGINEERING_SMOKE_ONLY`; open V4M3 only on exact engineering PASS.

**Independent review:** No additional review beyond the engineering qualifier; score direction is deliberately unavailable.

### Batch 5: Complete seed-0 screen and formal confirmation

**Files:**
- Create artifacts under `/mnt/cunyuliu/reactflow_delta_model_rescue_v4/v4m3_screen_seed0/`
- Create formal artifacts under `/mnt/cunyuliu/reactflow_delta_model_rescue_v4/v4m4_formal_seeds0_4/`
- Modify only after complete qualification: `configs/reactflow_delta/active_contract.yaml`
- Append only after complete qualification: `docs/prospective_v2/model_rescue_v4_decision_ledger.yaml`

**Risk:** High — partial outcome access or post-hoc model changes would invalidate the top-journal gate.

**Implementation:** Use persistent GPU0–7 sessions with non-overlapping folds and sufficient available memory. Complete all 20 seed-0 folds for all five fixed families before merging and scoring. If and only if the exact V4M3 gate passes, run seeds 0–4 for the pre-frozen model universe. Never delete failed seeds, modify thresholds, or inspect partial metric directions.

**Minimum verification:** File-name-only low-frequency monitoring; complete-universe merge; mechanical qualifier; clean replay of headline tables from the sole merged artifacts.

**Independent review:** Yes — review only the complete merged evidence and qualifier, never partial fold scores.

### Batch 6: Handoff or independent external amendment

**Files:**
- Create: `docs/prospective_v2/model_rescue_v4_handoff_202608xx.yaml`
- Modify: `docs/prospective_v2/claim_evidence_map.yaml`
- Modify: `configs/reactflow_delta/active_contract.yaml`

**Risk:** High — development-only evidence can be overstated as SOTA or publication evidence.

**Implementation:** On internal failure, freeze the negative result and return to the benchmark route. On internal pass, label it only `HIGH_EFFECT_POST_HOC_DEVELOPMENT_PASS`, close training, and draft a separate sealed external amendment. Only a later external gate can generate `TOP_JOURNAL_EVIDENCE_CHAIN_PASS`.

**Minimum verification:** Claim-evidence test ensures engineering, development, external, SOTA, mechanism, and publication statuses cannot be promoted automatically.

**Independent review:** Yes — final claim scope and manuscript language require evidence-level review.
