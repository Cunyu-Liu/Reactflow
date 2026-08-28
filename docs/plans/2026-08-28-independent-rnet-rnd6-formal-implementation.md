# Independent RNet2 RND6 Formal Implementation Plan

> **Execution policy:** Implement by functional batch under the global lightweight development and verification strategy. Do not apply task-by-task review or full-validation gates unless the batch risk warrants them.

**Goal:** Build the complete inactive RND6 seeds0–4 formal prediction, assembly, score-once, qualifier-once, and terminal-control path without reading RND1 progress or any scientific outcome.

**Architecture:** Extend the existing RND2/RND3 runner and merger for an exact 100 `(fold, seed)` target-free universe, then add an RNet-specific equal-seed assembler and formal scorer/qualifier. Treat RND6 as an umbrella and use explicit mutually exclusive machine phases `RND6P`, `RND6S`, `RND6Q`, and terminal `RND6T` so rights never overlap and existing outputs cannot be rerun.

**Tech Stack:** Python 3.10, PyTorch/CUDA, NumPy/SciPy, YAML, Bash dynamic GPU queue, pytest, Git.

---

### Batch 1: Freeze the formal scientific and authority contract

**Files:**
- Modify: `configs/reactflow_delta/independent_rnet_distill_contract.yaml`
- Modify: `configs/reactflow_delta/active_contract.yaml`
- Modify: `docs/prospective_v2/independent_rnet_distill_decision_ledger.yaml`
- Modify: `scripts/reactflow_delta/validate_independent_rnet_distill_contract.py`
- Modify: `tests/reactflow_delta/test_independent_rnet_distill_contract.py`
- Modify: `autoresearch/orchestrator-260828-independent-rnet-distill/research.md`
- Create: `docs/plans/2026-08-28-independent-rnet-rnd6-formal-design.md`
- Create: `docs/plans/2026-08-28-independent-rnet-rnd6-formal-implementation.md`

**Risk:** High — authority drift could open training and held-score access simultaneously or authorize formal work after a non-PASS screen.

**Implementation:** Freeze folds0–19, seeds0–4, 40+40 epochs, equal weight 0.2, unchanged RND5 mixture Gates, four-metric 4-of-5 matched-null seed stability, exact canonical paths/statuses, and explicit `RND6P/S/Q/T` phases. On the inactive prep branch only, add the frozen inactive-chain map and formal access fields while preserving the exact current RND1 phase, token, runnable set, permissions, and next action. Validator changes must continue accepting current RND1 authority and exact future RND6P/S/Q/T fixtures; no prep field is allowed to activate formal work.

**Minimum verification:** Run the focused contract suite, including current RND1 PASS with an inactive formal map, valid fixtures for RND6P/S/Q/T, and parameterized rejection of premature activation, predecessor, token, scope, permission, path, action, output-existence, score-access, and terminal-state drift.

**Independent review:** Yes — this defines all later scientific and mutation rights before results are read.

### Batch 2: Produce and assemble the exact 100-fold-seed prediction universe

**Files:**
- Modify: `scripts/reactflow_delta/run_independent_rnet_distill_downstream.py`
- Modify: `scripts/reactflow_delta/run_model_rescue_v11.py`
- Modify: `scripts/reactflow_delta/merge_independent_rnet_distill.py`
- Create: `scripts/reactflow_delta/run_independent_rnet_distill_formal_controller.sh`
- Create: `scripts/reactflow_delta/assemble_independent_rnet_distill_formal.py`
- Modify: `tests/reactflow_delta/test_independent_rnet_distill_downstream.py`
- Modify: `tests/reactflow_delta/test_merge_independent_rnet_distill.py`
- Modify: `tests/reactflow_delta/test_independent_rnet_distill_controller.py`
- Create: `tests/reactflow_delta/test_assemble_independent_rnet_distill_formal.py`

**Risk:** High — a missing/mixed fold, favorable-seed selection, comparator drift, or partial canonical publish would invalidate the formal result.

**Implementation:** Add RND6 exact experiment/schedule/path binding; allow only seeds0–4; make the shared held-prediction helper's explicitly required authoritative Feature41 replay seed-agnostic while leaving old callers unchanged; force that comparator for every formal seed; retain result-last fold publication; use a dedicated formal dynamic-GPU controller; merge exactly 100 unique pairs with one Git commit; and atomically publish 20 equal-seed assembled predictions while preserving fixed Feature41, V8, and historical-V10 fields.

**Minimum verification:** Focused runner, controller, merger, and assembler tests covering seed/fold/path drift, CUDA evidence fields, exact 100-pair universe, mixed commit rejection, fixed-comparator equality, candidate/null mean and ten-component mixture math, no best-seed flag, target-free schemas, no overwrite, and failure-before-merge behavior. Run `bash -n` and `py_compile` for changed entrypoints.

**Independent review:** Yes — the target-free formal universe and aggregation are a scientific risk node.

### Batch 3: Add complete score-once and qualifier-once

**Files:**
- Create: `scripts/reactflow_delta/score_independent_rnet_distill_formal.py`
- Create: `scripts/reactflow_delta/qualify_independent_rnet_distill_formal.py`
- Create: `scripts/reactflow_delta/run_independent_rnet_distill_formal_score_once.sh`
- Create: `scripts/reactflow_delta/run_independent_rnet_distill_formal_qualifier_once.sh`
- Create: `tests/reactflow_delta/test_score_independent_rnet_distill_formal.py`
- Create: `tests/reactflow_delta/test_qualify_independent_rnet_distill_formal.py`

**Risk:** High — target access ordering, comparator mismatches, Gate drift, or reruns could turn a development result into an invalid selected result.

**Implementation:** Validate all target-free merge/assembly inputs before reading M2 or historical score, score one equal mixture plus all five complete individual seeds, atomically publish one score, apply the unchanged RND5 Gate set plus the pre-frozen 4-of-5 matched-null stability rule, and emit exact PASS/FAIL/INDETERMINATE semantics with exits 0/1/2. Both wrappers bind only canonical paths and reject any existing corresponding or downstream output.

**Minimum verification:** Focused scorer/qualifier tests for complete universes, target-read ordering, canonical paths, preserved comparators, no-selection metadata, unchanged Gates, all four stability metrics, no overwrite, and legal exit semantics; plus shell syntax and Python compilation.

**Independent review:** Yes — this is the held-outcome access and final scientific verdict boundary.

### Batch 4: Milestone integration and GitHub handoff

**Files:**
- Modify only files from Batches 1–3 if integration defects are found.
- Update: `autoresearch/orchestrator-260828-independent-rnet-distill/research.md`

**Risk:** Medium — independently correct modules may disagree on a basename, status, schema, or authority token.

**Implementation:** Commit and push each focused batch as soon as it passes its scoped checks. Then run one combined focused suite across contract, downstream, merger, controller, assembler, scorer, and qualifier; compile changed Python modules; syntax-check changed shell scripts; verify clean diff and exact linear ancestry from `d07f766`; and push any integration-only correction as its own focused commit. Fast-forward the remote `/home` RND6 prep worktree after each pushed batch and keep it non-authoritative. Do not merge into the active RND1 worktree until RND1 terminal conditions are exact.

**Minimum verification:** One combined focused pytest invocation, changed-entrypoint compilation/shell syntax, `git diff --check`, clean worktree, upstream `0/0`, and remote branch HEAD equality.

**Independent review:** Yes — final review verifies the end-to-end inactive path without running it or reading outcomes.
