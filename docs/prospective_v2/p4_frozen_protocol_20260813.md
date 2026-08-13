---
# P4 Locked External Confirmatory Protocol (FROZEN)
schema_version: reactflow_delta.p4_frozen_protocol.v1
frozen_at: 2026-08-13T23:40:00+08:00
status: FROZEN_BEFORE_OUTCOME_ACCESS
---

## 1. Identity / authorization

- contract: ReactFlowDelta_prospective_full_spectrum_scientific_contract_v2_20260813.md
  (sha256 c1ea88c1532530576e726064501b66b52cad3fa3acaaf7379ea37cf73bd3d5c8)
- branch: codex/reactflow-delta-prospective-v2-20260813
- authority: PROSPECTIVE_V2_EPOCH21_ACTIVE
- owner directive (2026-08-13): "先保证 P0-3 所有 gate pass，自己去找 P4 的数据集，然后执行 P5"
  -> treated as Phase4 + Phase5 authorization to run the frozen protocol once.
- Phase4 token: OWNER_AUTHORIZATION_AUTHORIZE_REACTFLOW_DELTA_PROSPECTIVE_V2_P4_P5
  (granted by owner instruction; locked outcome access counter starts at 0)

## 2. Scientific question (frozen)

Does the prospective direct-learnability signal established on the development
benchmark (P2: D_p^P2 = L_T* - L_Direct*, 20-puzzle CI lower +0.0079 > 0)
replicate on **qualified, development-disconnected** external M2 2A3-MaP
components?

Because P3 concluded NO_INCREMENTAL_LRSO_SKILL (contract 17.2), the adopted
deployment model is the **direct model** (reg_direct). P4 therefore tests the
**external confirmation of the direct-learnability signal**: on each external
component, does the frozen direct model beat the trivial baseline
(ZeroResponse / WT-anchor prediction)?

## 3. Frozen statistical settings (pre-outcome-access)

- `delta_stat = 0` (statistical null; P4_EXTERNAL_STATISTICAL_PASS needs
  component-macro 95% CI lower > 0).
- `delta_practical = NOT_ESTABLISHED` (no independent repeatability/noise/
  decision-utility evidence established before this open). => even a PASS
  carries `PRACTICAL_IMPORTANCE_NOT_ESTABLISHED`; no utility/material claim.
- `delta_power` (effect to power against) = 0.0127 (the DEVELOPMENT per-puzzle
  D_p mean from P2 — independent of locked external outcomes).
- component SD for power: sd = 0.0153 (development sd 0.0102 inflated by 1.5x
  conservative factor for smaller external component sizes).
- alpha (one-sided component-macro t-CI) = 0.025 per family test.
- FWER: Holm-Bonferroni over the pre-registered secondary comparator family:
  {zero (primary), train_median (sensitivity)}. Primary PASS uses zero; median
  must not overturn direction.
- power >= 0.80.
- `K_required_planned = ceil(((1.645+0.842)^2 * sd^2) / delta_power^2)`
  = ceil((6.185 * 0.0153^2) / 0.0127^2) = ceil(6.185*1.452) = ceil(8.98) = **9**.
- `K_preaccess = 24` components (direct_external pool: M2SL5=3, M3SARS=3,
  15KLIB=18) with 3237 single-SNV mutants, all outcome-blind qualified.
- `K_preaccess >= K_required_planned` (24 >= 9) -> underpowered preaccess is
  NOT triggered; opening is permitted.
- evaluator: same Gaussian CRPS (energy form) at fixed scale 0.3 as P2/P3
  (`crps_gaussian`), shared-region scoring domain.
- estimator: component = one WT anchor + its single-SNV mutant library.
  Per component: mean per-mutant full-construct CRPS for direct and for
  baseline; `D_component = L_baseline - L_direct` (positive = direct better).
  Component-macro mean and 95% t-CI (df = K_eff-1).
- multiplicity family for direct comparators: only the adopted direct model is
  the primary comparator; no post-hoc comparator switching.

## 4. Candidate / B*_external (frozen)

- Candidate = **reg_direct** (input-matched regularized direct model:
  signed distance x exact ref->alt x WT edit/readout state; ridge lambda=1e-2;
  the Direct* selected by inner-OOF in P2 and the adopted model per P3/17.2).
- `B*_external` = refit the SAME reg_direct procedure on ALL development-only
  OK7a_M2 data (20 puzzles) once, with the frozen ridge lambda; config identity
  recorded. No outcome-driven re-selection. This is the frozen comparator
  "trivial baseline" = WT-anchor prediction (ZeroResponse) and train median
  (sensitivity).
- Point baselines use only outer-train residuals (frozen scale 0.3) — no
  held-external outcome used to set scale.

## 5. External component graph (frozen)

Role `direct_external` (development-disconnected biology, 2A3-MaP same
chemistry family as OK7a_M2, zero sequence identity overlap with dev set):

| dataset | biology | WT anchors (components) | single-SNV mutants |
|---|---|---|---|
| M2SL5_2A3_0000 | betacoronavirus SL5 (SARS-CoV-2/MERS/HKU5) | 3 | 584 |
| M3SARS_2A3_0000 | coronavirus frameshift elements | 3 | 147 |
| 15KLIB_2A3_0000 | diverse (TTR, SAM riboswitch, SARS windows, HDV) | 18 | 2506 |
| **total** | | **24** | **3237** |

Adjacent (OpenKnot program lineage; SENSITIVITY ONLY, not primary):
M2PK50/M2PK90 Pilot, OK1LIB/OK2TRN — reported as sensitivity, excluded from
the primary external estimand.

Task identity: full-construct mutant 2A3 reactivity response given WT profile +
exact SNV. External constructs carry unique 3' barcodes, so the scoring domain
is the **shared region** (positions where WT and mutant sequences are identical
plus the edit position itself), frozen below — analogous to
`full_construct ∩ target_qualified_positions`.

## 6. Frozen attrition rules (applied after open, no back-editing)

A component is evaluable iff:
1. WT anchor has an observed 2A3 reactivity profile (non-empty shared region).
2. >= 20 single-SNV mutants matched to it.
3. each mutant has >= 20 shared-region positions with non-missing WT and
   mutant reactivity.
4. no sequence identity with any OK7a_M2 development sequence.
No component is added, removed, or re-thresholded after open; `K_eff_realized`
is computed by these fixed rules and reported.

## 7. Frozen P5 mechanism contrasts (for P5, first-outcome-access-frozen)

- direct-vs-baseline effect decomposition by region where available;
- signed continuous distance curve (edit-readout |dist| bands);
- real vs WT-anchor (zero) baseline;
- negative control: shuffled/permuted direct features must not produce skill;
- FWER + leave-dominant-component-out sensitivity.
NOTE: external datasets have limited region annotations; where a frozen
contrast cannot be evaluated, it is reported EXPLORATORY_POST_HOC or
MECHANISM_NOT_ESTABLISHED rather than over-claimed.

## 8. Locked access control

- locked_outcome_access_count must be 0 before this open; after the single
  locked evaluation it is set to 1 and never reset.
- no preview/caching/back-tuning on external outcomes.
- one execution only; outcomes consumed cannot be renamed as a new
  confirmatory set.
