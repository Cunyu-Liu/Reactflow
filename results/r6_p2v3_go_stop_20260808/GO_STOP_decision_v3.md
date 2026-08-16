# GO/STOP Terminal Decision — P2_v3 re-gate (epoch 15, caller_v3/endpoint_v3)

**Adapter:** `r6_p2v3_go_stop.py`
**Run:** `results/p2_v3_learnability_20260808b`
**Overall decision: `STOP`**
**Route: `STOP_METHOD_ROUTE`**

Independent disk-evidence re-adjudication of `P2_LEARNABILITY_GO` on the authority-epoch-15
`caller_v3` / `endpoint_v3` labels (per-study empirical-scatter noise recalibration). The frozen
R5/R6 script `r6_go_stop_adjudicate.py` is NOT modified; this is the new epoch-15 re-adjudication.

## P2_LEARNABILITY_GO — FAIL (UNIDENTIFIABLE)

| Check | Status | Value |
|---|---|---|
| (a) estimand identifiable | FAIL | publication-macro AUPRC = `UNIDENTIFIABLE` for ALL models (incl. trivial) |
| (b) >=5 seeds, fixed budget | PASS | 5 seeds (0..4), fixed budget |
| (c) best model beats trivial w/ CI>0 & p<0.05 | FAIL | no numeric metric; best_model = null |
| (d) direction consistent across publications | FAIL | 2/10 eligible pubs degenerate (pmid_25883046, pmid_35982307) |

### Why the estimand is still UNIDENTIFIABLE

`evaluate_primary` → `publication_macro_auprc` returns `UNIDENTIFIABLE` (fail-closed) when ANY
publication has a constant label set. In the pooled held-out evaluation, 2 of the 10 publications
with eligible held-out pairs (`pmid_25883046`, `pmid_35982307`) are `DEGENERATE` (constant labels)
across every model and seed. Per `endpoint_v3.degenerate_policies.constant_label`, the whole metric
is `UNIDENTIFIABLE` rather than a number. This is the contract-mandated fail-closed behavior, not a
bug and not a silent exclusion.

### Per-publication AP (seed 0) — 8/10 are numeric, 2 degenerate

| method | pmid_24469816 | pmid_25183835 | pmid_25303992 | pmid_29446752 | pub_RNAPuzzle18 | CL1LIG | HC16M2R | RNASEP | pmid_25883046 | pmid_35982307 |
|---|---|---|---|---|---|---|---|---|---|---|
| trivial | 0.500 | 0.730 | 0.906 | 0.757 | 0.972 | 0.883 | 0.048 | 0.417 | DEGENERATE | DEGENERATE |
| logistic | 0.343 | 0.887 | 0.798 | 0.871 | 0.980 | 0.904 | 0.064 | 0.582 | DEGENERATE | DEGENERATE |
| gbm | 0.481 | 0.753 | 0.772 | 0.768 | 0.978 | 0.878 | 0.060 | 0.468 | DEGENERATE | DEGENERATE |
| p2_mlp | 0.321 | 0.830 | 0.851 | 0.843 | 0.976 | 0.918 | 0.055 | 0.657 | DEGENERATE | DEGENERATE |
| deepsets | 0.400 | 0.809 | 0.831 | 0.848 | 0.963 | 0.921 | 0.042 | 0.502 | DEGENERATE | DEGENERATE |

## Verdict

- `P2_LEARNABILITY_GO` = **FAIL** (estimand UNIDENTIFIABLE at the publication-macro level).
- Per §13.4 any non-PASS gate stops the route. Phase 3 model-architecture iteration remains **BLOCKED**.
- The caller_v3/endpoint_v3 calibration fix improved the label structure vs. caller_v2
  (3 changers → thousands of changers, most per-publication APs now numeric) **but did not**
  establish an identifiable publication-macro AUPRC, because two publications retain constant
  binary labels and the frozen degenerate policy forces the whole macro to UNIDENTIFIABLE.

## Recommendation

Proceed on the **model-conditional synthetic / theory / descriptive-real-cases** route
(`ONLY_MODEL_CONDITIONAL_THEORY_SOFTWARE_AND_DESCRIPTIVE_REAL_CASES`). Real-data claims that require
an identifiable publication-macro AUPRC remain gated. A further endpoint amendment (e.g. restricting
the macro to publications with non-constant labels, i.e. a defined subset estimand) would itself be a
new endpoint version under a new authority epoch and would require explicit user authorization — it is
NOT a silent in-place change and is not performed here.