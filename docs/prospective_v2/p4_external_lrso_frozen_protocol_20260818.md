---
# P4 External Final-LRSO Frozen Protocol (FROZEN_BEFORE_OUTCOME_ACCESS)
schema_version: reactflow_delta.p4_external_lrso_frozen_protocol.v1
frozen_at: 2026-08-18 (Asia/Shanghai)
status: FROZEN_BEFORE_OUTCOME_ACCESS
---

## 1. Identity / authority

- audit basis: `提示词/ReactFlow_Delta_strict_scientific_engineering_audit_20260817.md`
  §9.1 decision tree, §9.2 solution 3 ("外部候选必须是最终 LRSO，而不是 reg_direct"),
  §13 P0-4/P0-6.
- This protocol SUPERSEDES the ridge external candidate
  (`run_p4_external_v1.py`, candidate = reg_direct/ridge) for the external
  evidence role. The ridge result is not re-used for confirmatory claims.
- execution: `run_p4_external_lrso_v1.py`; single locked outcome access.

## 2. Scientific question (frozen)

Does the **final LRSO** (rank=2, five-seed Gaussian mixture, the deployment
candidate established by P2-v3 selected-rank vs rank0, D_p=+0.0018) replicate a
low-rank incremental signal on qualified, development-disconnected external M2
2A3-MaP components, at the highest independent (study/batch) cluster level?

Because external effective independence is K_joint=2 < K_required_planned=9
(audit P0-4/P0-6), this run is **EXPLORATORY replication evidence only**; no
confirmatory PASS is authorized.

## 3. Frozen statistical settings (pre-outcome-access)

- `delta_stat = 0` (statistical null); effect direction D > 0 = LRSO better.
- `K_required_planned = 9` (audit P0-6 power calc, unchanged).
- `K_preaccess = 24` components (frozen direct_external pool), `K_joint = 2`
  study clusters (SL5 PNAS 2024; Ribonanza 2024).
- statistical units: component (transparency) and **study cluster** (primary
  exploratory). Cluster-macro D = mean of component D within cluster; primary
  exploratory CI = two-sided 95% t-CI over the K=2 cluster means (df=1).
- LOSO: leave-one-cluster-out mean D (K=2 => each LOO set is the other cluster).
- evaluator: Gaussian CRPS energy form; baseline ZeroResponse at fixed scale
  0.3 (identical to P2/P3 evaluator). LRSO predictive CRPS scored on the
  five-seed equal-weight Gaussian mixture with the model's per-position
  (learned) scale.
- estimator per component: component-macro CRPS over the shared-region ∩
  observed-window positions; `D_component = L_zero - L_lrso`.
- verdict: `DEVELOPMENT_REPLICATION_EXPLORATORY`; `confirmatory_eligibility =
  NOT_ESTABLISHED`; `practical_importance = NOT_ESTABLISHED`.

## 4. Final LRSO (frozen, outcome-blind to external)

- rank = 2 (P2-v3 per-fold inner-selected rank; 18/20 folds chose rank 2).
- cfg = {lr=1e-3, wd=0, likelihood=student_t} (HP-selected on development,
  `--no-inner-select` frozen defaults from P3-v3).
- seeds {0..4}, equal-weight deployment mixture (contract 9.1).
- epoch count: dev-level puzzle-grouped inner 4-fold early stopping
  (max 200, patience 20), then final training on **ALL** development OK7a_M2
  records for that epoch count. No external outcome influences any choice.

## 5. External component graph (frozen, unchanged from audit P0-4)

Role `direct_external` (development-disconnected biology, 2A3-MaP, zero
sequence identity overlap with OK7a_M2): M2SL5=3, M3SARS=3, 15KLIB=18; 3237
single-SNV mutants. Loaded from the frozen outcome-blind manifest
`p4_external_components.json` (shared-region masks from sequence identity
only).

## 6. SEQPOS-CORRECT alignment (audit finding P4-M1; hard requirement)

The external RDAT reactivity arrays cover a window
[seqpos[0]-1, seqpos[0]-1+n_seqpos) of the full construct sequence, NOT index
0. For all three datasets seqpos starts at X27 => offset 26. This run:

1. builds the LRSO WT context over the observed window only (sequence sliced
   from seqpos offset; WT-observed mask from reactivity presence; missing
   mean-filled NOT 0; region = design_region because external RDAT carries no
   per-position region annotation);
2. maps shared-region positions and edit positions into the window coordinate
   system (offset cancels in distances);
3. scores only positions in `shared_region ∩ [off, off+W)` with non-missing WT
   and mutant reactivity (attrition rule 3).

The legacy ridge run (`run_p4_external_v1.py`) indexed reactivity by raw
sequence index (misalignment ~+26), corrupting candidate features and scoring
positions; its `P4_EXTERNAL_STATISTICAL_PASS` is therefore not usable as
replication evidence (see `external_qualification_v1.md` update).

## 7. Frozen attrition rules (applied after open, no back-editing)

1. WT anchor has an observed 2A3 reactivity profile (non-empty window).
2. >= 20 single-SNV mutants matched (rule 3 realized per mutant).
3. a mutant is scored only if >= 20 window positions with non-missing WT and
   mutant reactivity; edit position must fall inside the observed window.
No component is added/removed/re-thresholded after open; `K_eff_realized` is
reported by these fixed rules.

## 8. Locked access control

- locked_outcome_access_count = 1 for this LRSO evaluation; external RDAT
  reactivity is opened only AFTER final-model training and frozen-graph load.
- one execution only; the consumed external outcomes are already marked
  consumed/development-replication (audit P0-6) and cannot be renamed a new
  confirmatory set.
