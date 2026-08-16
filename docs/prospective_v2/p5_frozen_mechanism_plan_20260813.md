---
# P5 Frozen Mechanism Plan (FROZEN at first P5 execution)
schema_version: reactflow_delta.p5_frozen_mechanism_plan.v1
frozen_at: 2026-08-14T03:00:00+08:00
status: FROZEN_BEFORE_P5_ANALYSIS
supersedes_analysis_of: p4_external_result.json (locked outcome, read-only)
---

## 1. Identity / authority

- contract: ReactFlowDelta_prospective_full_spectrum_scientific_contract_v2_20260813.md
  (sha256 c1ea88c1532530576e726064501b66b52cad3fa3acaaf7379ea37cf73bd3d5c8), section 12.7 P5.
- P5 contrast family was pre-frozen in p4_frozen_protocol_20260813.md section 7
  ("Frozen P5 mechanism contrasts (for P5, first-outcome-access-frozen)"):
  direct-vs-baseline decomposition, signed continuous distance curve, real vs
  WT-anchor baseline, negative control (shuffled/permuted features), FWER +
  leave-dominant-component-out sensitivity. This document fixes the exact
  operationalization (band edges, permutation seed, multiplicity family, PASS
  criteria) at first P5 execution, on the SAME locked external components.
- owner directive (2026-08-13): "先保证 P0-3 所有 gate pass，自己去找 P4 的数据集，然后执行 P5"
  -> Phase5 authorization (mirrors p4_frozen_protocol_20260813.md section 1).
- P4 precedent: p4_external_result.json -> P4_EXTERNAL_STATISTICAL_PASS.

## 2. Inputs (all frozen / locked)

- frozen outcome-blind component graph: /mnt/cunyuliu/prospective_v2_p4_20260813/
  p4_external_components.json (24 components, 3237 single-SNV mutants).
- locked P4 outcome (read-only, carried forward, NOT recomputed):
  p4_external_result.json (component-macro CI, Holm-Bonferroni FWER,
  leave-dominant-out sensitivity).
- candidate model: B*_external = reg_direct refit once on ALL development
  OK7a_M2 (identical procedure to P4; single coef shared by P4 and P5).
- profiles: same Ribonanza M2-style 2A3 rdat (M2SL5, M3SARS, 15KLIB).

## 3. Frozen statistical settings

- evaluator: Gaussian CRPS fixed scale 0.3, shared-region scoring domain
  (identical to P4); per-position D = CRPS_zero - CRPS_direct (positive =
  direct better), component-macro per stratum.
- alpha (one-sided t-CI) = 0.025 per family test.
- distance bands over |dist| = |readout position - edit position|:
  - B_edit: |dist| = 0
  - B_near: 1-3
  - B_mid: 4-10
  - B_far: 11-25
  - B_vfar: >= 26
- multiplicity family A (distance curve): one-sided component-macro t-tests on
  D_vs_zero within each of the 5 bands; Holm-Bonferroni at 0.025 family alpha.
- multiplicity family B (negative control): single pre-registered permutation
  (seed 20260813) that shuffles each mutant's feature vectors across its shared
  positions; the permuted direct model must show NO skill.
- region stratum: biology/dataset (M2SL5 betacoronavirus SL5, M3SARS
  coronavirus FSE, 15KLIB diverse); direction-level replication across datasets
  (only the pooled K_required=9 design is formally powered).

## 4. Frozen PASS criteria -> MECHANISM_EVIDENCE_PASS

1. P4 already P4_EXTERNAL_STATISTICAL_PASS (carried forward).
2. Primary mechanism contrast (signed distance curve): the edit-site band
   (B_edit) has D_vs_zero > 0 with Holm-Bonferroni-adjusted significance across
   family A (adjusted p < 0.025), AND distance heterogeneity holds: the
   edit-site advantage exceeds the very-far advantage,
   D_vs_zero(B_edit) - D_vs_zero(B_vfar), with one-sided 95% CI lower > 0.
   This replicates the development finding that direct skill is concentrated at
   / near the signed edit site.
3. Negative control: permuted-feature direct model shows no skill --
   component-macro D_vs_zero(permuted) one-sided 95% CI upper <= 0.
4. Region replication: component-macro D_vs_zero > 0 in at least 2 of the 3
   datasets (direction), confirming the effect is not a single-biology artifact.
5. Failure cases reported (components with D_vs_zero < 0) but do not overturn
   the primary contrast (consistent with P4 leave-dominant-out sensitivity).

If any criterion 1-4 fails -> MECHANISM_NOT_ESTABLISHED (delete mechanism
claim; keep qualified prediction / resource route).

## 5. Frozen outputs

- /mnt/cunyuliu/prospective_v2_p4_20260813/p5_mechanism_result.json
  - per-band component-macro D_vs_zero + CI + Holm p-values
  - distance-heterogeneity contrast
  - negative-control D_vs_zero(permuted) + CI
  - per-dataset region stratum D_vs_zero
  - failure-case list
  - claim-evidence map (claim -> evidence -> verdict)
  - verdict: MECHANISM_EVIDENCE_PASS | MECHANISM_NOT_ESTABLISHED
