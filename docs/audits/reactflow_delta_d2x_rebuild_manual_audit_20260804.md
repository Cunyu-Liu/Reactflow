# D2-X Publication-Level Split Rebuild — Manual Audit (epoch 8)

**Audit ID:** reactflow_delta_d2x_rebuild_manual_audit_20260804
**Phase:** D2-X (publication-level split/exposure rebuild)
**Authority epoch:** 8
**Run ID:** `d2x_split_publication_v1`
**Supersedes run ID:** `d2x_split_20260804_v1` (epoch 6, study-level)
**Reviewer role:** CODEX_PRIMARY_IMPLEMENTATION_AGENT
**Reviewer external identity:** NOT_EXTERNALLY_VERIFIED
**Date:** 2026-08-04

## 1. Rationale for rebuild

The epoch-6 D2-X split grouped studies only by RMDB entry identity (study-level),
not by publication. Re-audit revealed publication-level leakage:

- `CIDGMP` was assigned to **validation** (190 primary pairs).
- `TRP4P6` was assigned to **train** (358 primary pairs).
- Both resolve to the **same publication**: NAR 2014, pmid 25303992 (SHAPE-Seq 2.0).
- The epoch-6 exposure audit's `publication_level` field was empty (`{}`), i.e. no
  publication-level grouping or leakage audit was performed.

This violates contract §10 (zero cross-split publication-level leakage) and
contaminates Tier B+ independence criterion #1 (>=3 independent study/publication
units). The epoch-6 split/exposure closure is therefore voided (#1) and rebuilt
at publication level.

## 2. Publication crosswalk (deterministic, outcome-blind)

| Study | Publication | Split |
|-------|-------------|-------|
| 16SFWJ | pmid_25183835 (RNA 2014) | test |
| CIDGMP | pmid_25303992 (NAR 2014) | validation |
| TRP4P6 | pmid_25303992 (NAR 2014) | validation |
| ADD140 | pmid_29446752 (eLife 2018) | train |
| ETBSTR | pmid_24469816 (PNAS 2014) | train |
| PSL2IAV | pmid_35982307 (Nat Med 2022) | train |
| RNAPZ18 | pub_RNAPuzzle18_daslab | train |
| RNAPZ5 | pmid_25883046 (RNA 2015) | train |
| TBWND | pmid_26566145 (PLoS CB 2015) | train |

Distinct publications: 7. Independent publication units: >=3 (7). No single
publication spans multiple splits.

## 3. Audit scope and results

| # | Check | Result |
|---|-------|--------|
| 1 | split schema v1 + outcome_blind | PASS |
| 2 | all studies assigned to train/validation/test | PASS |
| 3 | non-empty splits | PASS |
| 4 | pair counts reconcile | PASS |
| 5 | exposure schema v1 | PASS |
| 6 | overlap_zero (design-lineage/exact-seq/near-dup) | PASS |
| 7 | no near-dup leakage | PASS |
| 8 | **no publication leakage** (no single pub spans splits) | PASS |
| 9 | **distinct publications >= 3** | PASS (7) |
| 10 | tier schema v1 | PASS |
| 11 | tier_b_plus_data_candidate present | PASS (candidate) |
| 12 | **independent publications >= 3** | PASS (7) |
| 13 | test_is_unconsumed | PASS |
| 14 | test seal SEALED | PASS |
| 15 | ledger append-only, no sample-level access | PASS |
| 16 | blind certificate validity | PASS |
| 17 | data card schema | PASS |

## 4. Tier B+ candidate checklist (changers deferred to PH0-X)

- studies_ge_3: PASS (9 studies)
- independent_publications_ge_3: PASS (7)
- parents_ge_10: PASS (42 parents)
- pairs_ge_1000: PASS (4472 primary pairs)
- test_study_ge_100_pairs: PASS (test=408)
- test_is_unconsumed: PASS
- overlap_zero: PASS
- single_parent_lt_40pct: PASS
- probe_domain_separable: PASS
- changer_counts (training/val/test/controls): UNKNOWN_NOT_ASSERTED → PH0-X
- noise_bound_ge_80pct: UNKNOWN_NOT_ASSERTED → PH0-X

## 5. Disposition

**PASS.** The publication-level D2-X split/exposure rebuild is accepted. Full
Tier B+ requires PH0-X identifiability; full Tier A+ requires B0-X. Test remains
sealed. No sample-level label, no pair identity, no prediction, no normalization,
no model output was used in this audit.