# P2 Learnability re-audit — endpoint_v4 non-degenerate macro (Route A, authority epoch 16)

Run: `p2_v3_learnability_20260808b` · caller `caller_v3` · predictions recomputed under `evaluate_v4` endpoint_v4 semantics (macro over NON-DEGENERATE publications) · no retraining · source held-out predictions fixed.

## 1. Verdict

- **Verdict:** `STOP` (fail-closed)
- **Estimand status (primary publication-macro AUPRC):** `IDENTIFIABLE` — now IDENTIFIABLE under endpoint_v4 (both the caller_v3 calibration artifact and the evaluate_v2 degeneracy artifact are resolved; the relaxed macro yields a numeric value).
- **Permutation status:** `DEGENERATE_NO_POWER` (see §4).
- **Best model:** `p2_mlp` (by mean macro delta over trivial) · delta +0.0298 over trivial `0.6517`
- **Best paired publication-block bootstrap delta CI:** `[-0.0833, 0.5000]` — lower bound **< 0**, includes 0.

## 2. Non-degenerate macro AUPRC (endpoint_v4) by model

| model | macro (5-seed mean) | delta over trivial | per-pub direction (seed 0) |
|---|---|---|---|
| trivial (constant/prevalence) | 0.6517 | — | — |
| deepsets | 0.6585 | +0.0068 | 4/8 |
| gbm | 0.6409 | -0.0108 | 5/8 |
| logistic | 0.6787 | +0.0270 | 6/8 |
| p2_mlp | 0.6815 | +0.0298 | 6/8 |

## 3. Excluded (constant-label) publications

- **Excluded from macro (documented, not silent):** `pmid_25883046, pmid_35982307`
- **Non-degenerate (mixed-label) publications used:** 8 of 10 eligible.

### Per-publication AP (seed 0)

| publication | logistic | gbm | p2_mlp | deepsets |
|---|---|---|---|---|
| UNKNOWN_PUBLICATION:CL1LIG | 0.9039 | 0.8781 | 0.9181 | 0.9208 |
| UNKNOWN_PUBLICATION:HC16M2R | 0.0643 | 0.0601 | 0.0547 | 0.0419 |
| UNKNOWN_PUBLICATION:RNASEP | 0.5817 | 0.4676 | 0.6567 | 0.5022 |
| pmid_24469816 | 0.3427 | 0.4811 | 0.3208 | 0.4003 |
| pmid_25183835 | 0.8874 | 0.7526 | 0.8304 | 0.8090 |
| pmid_25303992 | 0.7984 | 0.7717 | 0.8515 | 0.8311 |
| pmid_25883046 | DEGENERATE | DEGENERATE | DEGENERATE | DEGENERATE |
| pmid_29446752 | 0.8707 | 0.7681 | 0.8433 | 0.8475 |
| pmid_35982307 | DEGENERATE | DEGENERATE | DEGENERATE | DEGENERATE |
| pub_RNAPuzzle18_daslab | 0.9804 | 0.9781 | 0.9756 | 0.9631 |

## 4. Why the permutation is non-informative (DEGENERATE_NO_POWER)

The publication-block permutation (evaluate_v2/`evaluate_v4` exchangeable-null) permutes score-blocks **within equal-size classes**. Here all 8 non-degenerate publications have **unique sizes** (36, 62, 64, 68-excluded, 71, 128, 220, 408, 2366, …), so no two publications share a size and every permutation returns the exact same block assignment. The null is a **point mass at the observed macro** (e.g. logistic seed0: null min=mean=max=0.6787, all 1000 equal), giving p=1.0 with **zero power**. This does NOT by itself refute learnability — it means the permutation test is inapplicable here. The decision therefore rests on the paired publication-block bootstrap delta CI, which is a valid publication-level resampling interval.

## 5. Dialectical interpretation (no gate-lowering, no fabricated number)

- **Progress:** both the calibration artifact (caller_v3, epoch 15) and the degeneracy artifact (endpoint_v4, epoch 16) are resolved; the primary estimand is now numeric and IDENTIFIABLE. 8/10 publications have mixed labels and several show strong per-pub AP (e.g. pub_RNAPuzzle18_daslab ~0.96–0.98).
- **Honest negative:** the best learned model (p2_mlp) beats the trivial baseline by only +0.030 macro AUPRC, and its paired publication-block bootstrap delta CI lower bound is **negative** (CI includes 0). At publication-level resampling the incremental cross-publication learnability on the **primary binary-changer estimand is NOT statistically established.** The permutation cannot rescue this because it has no power under unique block sizes.
- **Contract consequence (§13.2 R6 / §13.4 / termination condition):** *“publication-level P2 learnability不能胜permutation与简单baseline”* is met in the fail-closed sense. Phase 3 architecture iteration remains **BLOCKED**; per the contract the route should pivot to resource/measurement/negative (or a secondary/conditional magnitude estimand), NOT open Phase 3 by adding hidden layers.
- **Route A was not used to force a GO.** The relaxed macro made the estimand identifiable, but the verdict is still STOP on the evidence (CI includes 0). This is the contract's fail-closed behavior; no gate was lowered and no degenerate publication was silently dropped.

No AUPRC was fabricated; every number above is recomputed from the frozen held-out predictions. This document is evidence for the authority-epoch-16 endpoint_v4 re-adjudication only.
