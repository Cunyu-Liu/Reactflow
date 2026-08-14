---
# P5b Frozen Mechanism Protocol (FROZEN before outcome access on NEW independent set)
schema_version: reactflow_delta.p5b_frozen_mechanism_plan.v1
frozen_at: 2026-08-14T22:30:00+08:00
status: FROZEN_BEFORE_OUTCOME_ACCESS
supersedes_analysis_of: NONE (new set; no prior outcome access)
---

## 1. Identity / authority / rationale

- contract: ReactFlowDelta_prospective_full_spectrum_scientific_contract_v2_20260813.md
  (sha256 c1ea88c1532530576e726064501b66b52cad3fa3acaaf7379ea37cf73bd3d5c8), section 12.7 P5.
- P5 (first set) verdict = MECHANISM_NOT_ESTABLISHED: the pre-frozen "edit-site
  concentration" claim did NOT replicate (distance heterogeneity edit-vfar CI lower
  -0.0199 < 0); skill is spatially UNIFORM on the 24-component locked set.
- Contract section 16.1: a mechanism claim that was not confirmed on one external
  set can only be established on NEW independent components with the contrast
  frozen BEFORE outcome access on that new set. This document is that freeze.
- candidate model unchanged: reg_direct (B*_external, same coef as P4/P5, refit
  once on ALL development OK7a_M2). No outcome-driven re-selection.
- authorization: owner directive 2026-08-13 (P4+P5) and current owner instruction
  to pass the P5 gate; new locked external access requires explicit owner
  confirmation (see section 8).

## 2. New independent component set (outcome-blind, NEVER outcome-accessed)

- datasets: M2RFOK_2A3_0000 (rfam-OK), M2RFPK_2A3_0000/0001/0002 (rfam-PK)
  = DasLab BigLib2 OneMil2 M2 sub-libraries (2A3-MaP, RNAFramework v2.8.4).
- p5b_external_components.json (frozen graph, built outcome-blind from sequence
  identity only): 694 components / 106,904 single-SNV mutants.
- development-disconnected: zero sequence overlap with OK7a_M2 dev (WT+all).
- disjoint from consumed P4/P5 set: zero WT-name overlap with the 24 components.
- K_preaccess = 694 >= K_required_planned = 9  => power sufficient.

## 3. Mechanism claim to confirm (frozen, honest finding from set 1)

The reproducible phenomenon on the first locked set was NOT edit-site
concentration but **spatial extension**: direct-vs-zero full-spectrum skill was
significant (Holm-pass) at EVERY distance band, including very-far (|dist|>=26,
CI lower +0.0149). The mechanism claim to confirm on the NEW independent set is:

  "RFD-Direct full-construct skill extends to REMOTE readout positions
   (very-far band) and is not confined to the edit-site neighborhood."

This is the constructive replacement for the deleted "edit-site concentration"
claim and is exactly the direction the first set supported.

## 4. Frozen statistical settings

- evaluator: Gaussian CRPS fixed scale 0.3, shared-region scoring domain
  (identical to P4/P5). per-position D = CRPS_zero - CRPS_direct (positive =
  direct better). component-macro per stratum.
- alpha (one-sided t-CI) = 0.025 per family test.
- distance bands over |dist| = |readout - edit|:
  B_edit=0, B_near=1-3, B_mid=4-10, B_far=11-25, B_vfar>=26 (identical to P5).
- multiplicity family A (distance curve): one-sided component-macro t-tests on
  D_vs_zero within each band; Holm-Bonferroni at 0.025 family alpha.
- multiplicity family B (negative control): pre-registered permutation
  (seed 20260813) shuffling each mutant's feature vectors across its shared
  positions; permuted direct must show NO skill.
- region stratum: M2RFOK vs M2RFPK_0000/0001/0002 (direction-level replication).

## 5. Frozen PASS criteria -> MECHANISM_EVIDENCE_PASS

1. P4 carried: P4_EXTERNAL_STATISTICAL_PASS (already established, carried).
2. Primary mechanism claim (spatial extension): component-macro
   D_vs_zero(B_vfar) > 0 with one-sided 95% CI lower > 0 AND Holm-Bonferroni-
   adjusted significance in family A (adjusted p < 0.025). (Remote skill is
   real; not an edit-site-only artifact.)
3. Full-band consistency: D_vs_zero(B_edit) also Holm-pass (skill present at
   the edit site too), i.e., construct-wide coverage not remote-only noise.
4. Negative control: permuted-feature direct model shows no skill --
   component-macro D_vs_zero(permuted) one-sided 95% CI upper <= 0.
5. Region replication: component-macro D_vs_zero(B_edit) > 0 in >= 2 of the 4
   dataset groups (direction-level).
6. Failure cases reported but do not overturn primary (leave-dominant-out
   sensitivity: after removing the dominant component, the very-far CI lower
   must remain > 0).

If any criterion 2-5 fails -> MECHANISM_NOT_ESTABLISHED (delete claim; keep
qualified prediction/transportability narrative).

## 6. Frozen outputs

- /mnt/cunyuliu/prospective_v2_p4_20260813/p5b_mechanism_result.json
  - per-band component-macro D_vs_zero + CI + Holm p-values
  - very-far spatial-extension contrast (primary)
  - negative-control D_vs_zero(permuted) + CI
  - per-dataset region stratum D_vs_zero
  - failure-case list + leave-dominant-out
  - claim-evidence map
  - verdict: MECHANISM_EVIDENCE_PASS | MECHANISM_NOT_ESTABLISHED

## 7. Frozen attrition / access control

- same attrition rules as P4 section 6 (>=20 mutants/component, >=20 shared
  non-missing positions/mutant, reactivity-array length bounding).
- locked_outcome_access_count: P4/P5 = 1 (consumed set). This P5b run would be
  access count 2 on a NEW set; count is never reset.
- one execution only; outcomes of the new set, once opened, cannot be renamed
  as yet another confirmatory set.

## 8. Authorization requirement (blocking)

Per contract section 15.2 the Phase4 token is necessary (not sufficient) to
access locked external outcomes. Opening the NEW set (M2RFOK/M2RFPK) is a NEW
locked outcome access distinct from the already-consumed 24-component set.
Execution MUST NOT start until the owner confirms authorization for this
specific new locked external access.
