# PH0-X Identifiability and Reliability — Manual Audit (epoch 9)

**Audit ID:** reactflow_delta_ph0x_manual_audit_20260804
**Phase:** PH0-X (identifiability/reliability)
**Authority epoch:** 9
**Run ID:** `ph0x_identifiability_20260804_v1`
**Reviewer role:** CODEX_PRIMARY_IMPLEMENTATION_AGENT
**Reviewer external identity:** NOT_EXTERNALLY_VERIFIED
**Date:** 2026-08-04

## 1. Scope

Verifies that the D2-X publication-level split (epoch 8, 4472 primary exact
delta pairs) supports a reproducible changer signal under controls/replicates,
per contract §20.7 and §13.3. This audit is **data-qualification only**; it
writes TIER_B_PLUS as PASS but does NOT claim full Tier A+ (which requires B0-X
frozen cross-parent/cross-study learnability) and does not unseal the test split.

## 2. Inputs (frozen, D2-X closure)

- D1-X canonical records: `d1x_canonical_records.jsonl`
- D2-X publication-level split manifest: `d2x_split_publication_20260804T1600+0800`
- Test seal + test access ledger (unchanged, sealed)

## 3. PH0-X outputs

| Output | Result |
|--------|--------|
| Noise manifest (`ph0x_noise_manifest.json`) | PASS |
| Caller manifest (`ph0x_caller.json`) | PASS |
| Permutation report (`ph0x_permutation.json`) | PASS |
| Blind test certificate (`ph0x_blind_certificate.json`) | PASS |

## 4. Audit checks and results

| # | Check | Result |
|---|-------|--------|
| 1 | noise schema v2 | PASS |
| 2 | matched noise coverage >= 80% | PASS (100%) |
| 3 | noise coverage == 100% | PASS |
| 4 | noise source hierarchy, no fabricated value | PASS (per_position/replicate_block/study_probe_median/study_median) |
| 5 | ICC pooled median >= 0.5 | PASS (0.84) |
| 6 | Tier B+ cond 7 controls/replicates >= 100 obs | PASS (4472 obs, 11 blocks) |
| 7 | caller schema v1 | PASS |
| 8 | caller frozen on train+validation only | PASS |
| 9 | training changers >= 100 | PASS (1306) |
| 10 | validation changers >= 20 | PASS (49) |
| 11 | test changers >= 20 | PASS (193) |
| 12 | permutation schema v1 | PASS |
| 13 | real signal > group-aware permutation null | PASS (real 465 vs null median 197, max 225; p=0.0099) |
| 14 | p-value <= 0.05 | PASS (0.0099) |
| 15 | **no single study driven** (leave-one-study-out) | PASS (all 9 LOSO pass) |
| 16 | blind certificate schema v1 | PASS |
| 17 | blind certificate aggregate-only | PASS |
| 18 | test split remains sealed | PASS (seal SHA unchanged) |
| 19 | test changers >= 20 in certificate | PASS (193) |

## 5. Key evidence

### 5.1 Matched noise (contract §9.1)

- 100% of 4472 primary pairs carry a matched noise estimate.
- Sources: per-position upstream error (3241), replicate-block (602),
  study/probe median (190), study median (439). All propagate genuine measured
  noise; no outcome-derived value.
- Pooled ICC(1,1) median = 0.84 across 11 WT-anchor replicate blocks.

### 5.2 Frozen replicate-aware caller (contract §9.2)

- Statistic: local sliding-window max-cluster (window=15) of
  control-standardized delta on the eligible mask.
- Null: training replicate-block WT-WT disagreement (resampled, N=2000).
- FDR: Benjamini-Hochberg within study, q=0.05.
- Tier changer counts: train=1306, validation=49, test=193.

### 5.3 Group-aware permutation (contract §13.3)

- Exchangeability block = study (preserves parent/shared-WT/mask structure).
- Statistic: # pairs where mutation position lies inside the winning local
  max-cluster above the replicate-null 95th percentile.
- Real statistic = 465; group-aware null median = 197, max = 225; p = 0.0099.
- Leave-one-study-out: every study removal keeps the signal significant →
  **no single study drives the result**.

### 5.4 Blind test certificate (contract §20.7)

- Aggregate-only: test pair count 408, aggregate changer count 193.
- No pair identity, position label, profile, prediction, or per-pair statistic
  emitted. Test seal SHA unchanged (test remains sealed).

## 6. Conclusion

**PASS.** All PH0-X identifiability/reliability gates are met. The exact-mutation
delta signal is reproducible above matched measurement noise under a frozen
train-only caller and is not driven by any single study. **TIER_B_PLUS is written
PASS (data qualification).** Full TIER_A_PLUS remains pending B0-X frozen
cross-parent/cross-study learnability. No test sample was unsealed; no model was
trained; no outcome percentile noise was used.