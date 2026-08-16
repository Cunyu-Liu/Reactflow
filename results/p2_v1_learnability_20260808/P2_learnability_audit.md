# P2 Learnability Audit — run_id `p2_v1_learnability_20260808`

**Verdict: STOP_METHOD_ROUTE — primary binary changer estimand is UNIDENTIFIABLE**
under the frozen `caller_v2.py` on the frozen `d1x_v2` data. This is a **contract-vs-data
conflict**, not a learned-model result.

## 1. Setup (nested leave-one-publication-out)
- 19 pool studies / **18 distinct publications**; nested LOOCV with publication as outer unit.
- 6385 usable WT↔exact-single-mutant pairs in pool; SL5 test family excluded.
- Models: `trivial, logistic, gbm, p2_mlp, deepsets`; seeds `[0,1,2,3,4]`.
- All neural training on GPU 0 (A100-40GB, CUDA confirmed, ~29 GB free).
- 18 folds processed: **10 had eligible held-out pairs**, **8 were all-NO_CALL (skipped)**.
  Folds with held-out data: CL1LIG, HC16M2R, RNASEP, pmid_24469816, pmid_25183835,
  pmid_25303992, pmid_25883046, pmid_29446752, pmid_35982307, pub_RNAPuzzle18_daslab.

## 2. Why the primary table/learning-curve cannot be produced
The caller emits an effectively **constant** label: **3 changers / 3178 non / 3204 NO_CALL**.
In LOOCV the 3 changers sit one-per-publication in exactly 3 held-out folds; the other 7+
held-out folds have **0 changers** → per-publication AP is DEGENERATE, and publication-macro
AUPRC is unidentifiable. Endpoint `degenerate_policies.constant_label` → return UNIDENTIFIABLE,
**no fabricated numbers**. Running the full experiment again would only reproduce this.

## 3. Root cause (data-scale / error-calibration)
- **Cross-study reactivity scale heterogeneity**: reactivity medians span 0.0005–4.08
  (~4000×), maxima to **47,222** (TRP4P6), 2413 (ETBSTR), 1814 (HC16M2R).
- **Miscalibrated errors**: reported per-position errors (median 0.034, some negative, max 644)
  are a median **2.5×** / mean **8.4×** (up to **642×**) smaller than the empirical
  cross-replicate SD.
- **Inflated null**: WT-WT replicate |z| up to **1521** (p99 44.3) pushes the caller's pooled
  spatial-block null to ~44 → significance threshold astronomically high → only 3 changers.
- **Position-level signal exists**: 22.7% of 458k eligible positions have |Δreact|>0.3.
  The degeneracy is a caller/null calibration artifact — **NOT** evidence that mutations do
  not change reactivity.

## 4. Recommended path forward (requires approval, not a silent change)
1. Per-study reactivity normalization to a common scale + error recalibration using empirical
   replicate scatter as noise → re-derive labels. This requires **caller_v3 / endpoint_v3**
   (new authority epoch), per the endpoint's `PRIMARY_ENDPOINT_NEVER_SILENT_CHANGE` policy.
2. The conditional/secondary regression estimand (|Δr| magnitude) has real signal but is
   gated by the endpoint to require an identifiable primary + reliable caller — not currently
   enabled.

## 5. Engineering fix applied
`run_p2_v1.py` `predict_torch`/`predict_deepsets` now `.squeeze(-1)` (previously returned
`(n,1)` → `TypeError: float() ... not 'list'` at evaluation). Compiles clean. Applies to any
future re-run.

## 6. Artifacts
- `results/p2_v1_learnability_20260808/` → `main.log`, `explore.log`, `P2_learnability_verdict.json` (this audit).
- Cache: `/mnt/cunyuliu/.../p2_cache/p2_cache.pkl` (101 MB, frozen).
- Diagnostics: `scripts/reactflow_delta/diag_p2_error_scale.py`, `diag_p2_study_scale.py`, `diag_p2_calib.py` (CPU-only).
