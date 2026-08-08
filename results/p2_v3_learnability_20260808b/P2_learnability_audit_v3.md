# P2 Learnability re-audit — caller_v3 (endpoint_v3, authority epoch 15)

Run: `p2_v3_learnability_20260808b` · caller `caller_v3` · evaluator/split `evaluate_v2` / `split_v2` (unchanged, frozen) · GPU `NVIDIA A100-PCIE-40GB MIG 1g.5gb` (cuda_available=true, no silent CPU fallback) · source hashes all match manifest.

## 1. Verdict

- **Verdict:** `STOP` (fail-closed)
- **Estimand status (primary publication-macro AUPRC):** `UNIDENTIFIABLE`
- **Eligible held-out publications:** 10 of 18
- **Root cause of UNIDENTIFIABLE:** the frozen `evaluate_v2.publication_macro_auprc` returns UNIDENTIFIABLE (degenerate_policies.constant_label, fail-closed, no silent exclusion) whenever ANY held-out publication has a constant label set. Here 2 of 10 eligible publications (pmid_25883046, pmid_35982307) have all held-out pairs labeled changers.

## 2. Calibration fix validated (progress vs R5/R6 STOP)

caller_v3 empirical-scatter recalibration recovered abundant real changers. The R5 near-constant label (3 changers / 6359) is resolved:

| publication | n held-out | #changer | #nonchanger | constant? |
|---|---|---|---|---|
| UNKNOWN_PUBLICATION:CL1LIG | 128 | 113 | 15 | no |
| UNKNOWN_PUBLICATION:HC16M2R | 62 | 3 | 59 | no |
| UNKNOWN_PUBLICATION:RNASEP | 36 | 15 | 21 | no |
| pmid_24469816 | 220 | 110 | 110 | no |
| pmid_25183835 | 408 | 298 | 110 | no |
| pmid_25303992 | 64 | 58 | 6 | no |
| pmid_25883046 | 68 | 68 | 0 | YES |
| pmid_29446752 | 2366 | 1792 | 574 | no |
| pmid_35982307 | 68 | 68 | 0 | YES |
| pub_RNAPuzzle18_daslab | 71 | 69 | 2 | no |

8 of 10 eligible publications have mixed labels and therefore produce a **numeric** per-publication AP. 2 publications are internally constant (all-changer) and are the sole trigger of the macro UNIDENTIFIABLE (per the frozen fail-closed rule).

## 3. Per-publication AP (seed 0)

| publication | logistic | gbm | p2_mlp | deepsets |
|---|---|---|---|---|
| UNKNOWN_PUBLICATION:CL1LIG | 0.9039 | 0.8781 | 0.9181 | 0.9208 |
| UNKNOWN_PUBLICATION:HC16M2R | 0.0643 | 0.0601 | 0.0547 | 0.0419 |
| UNKNOWN_PUBLICATION:RNASEP | 0.5817 | 0.4676 | 0.6567 | 0.5022 |
| pmid_24469816 | 0.3427 | 0.4811 | 0.3208 | 0.4003 |
| pmid_25183835 | 0.8874 | 0.7526 | 0.8304 | 0.809 |
| pmid_25303992 | 0.7984 | 0.7717 | 0.8515 | 0.8311 |
| pmid_25883046 | DEGENERATE | DEGENERATE | DEGENERATE | DEGENERATE |
| pmid_29446752 | 0.8707 | 0.7681 | 0.8433 | 0.8475 |
| pmid_35982307 | DEGENERATE | DEGENERATE | DEGENERATE | DEGENERATE |
| pub_RNAPuzzle18_daslab | 0.9804 | 0.9781 | 0.9756 | 0.9631 |

Several publications show strong per-pub AP (CL1LIG ~0.88–0.92, pmid_25183835 ~0.75–0.83, pmid_25303992 ~0.77–0.85, pmid_29446752 ~0.77–0.87, pub_RNAPuzzle18_daslab ~0.96–0.98). HC16M2R is low (~0.04–0.06) for all models. pmid_25883046 and pmid_35982307 are DEGENERATE (constant labels).

## 4. Primary metric: publication-macro AUPRC

All models × all 5 seeds return `UNIDENTIFIABLE` for the primary publication-macro AUPRC because of the 2 constant-label publications (frozen fail-closed rule; degenerate_policies.constant_label / pair_any_all_positive). No numeric macro AUPRC, no confirmatory CI, and no permutation p<0.05 can be reported on the primary estimand. Per contract §13.4 the P2 gate on the PRIMARY estimand is therefore **not established**.

## 5. Contract interpretation & discussion point (dialectical)

The frozen degenerate policy “any constant publication ⇒ whole macro UNIDENTIFIABLE, no silent exclusion” is scientifically conservative (prevents an all-positive publication from inflating a macro AP). However it also means a single all-changer publication blocks any numeric learnability number even when 8/10 publications show real, strong per-pub signal. Two defensible routes exist; both require explicit authorization (new authority epoch + amendment), because they change either the evaluator degeneracy policy or the endpoint metric semantics:

1. **Relax to macro-over-non-degenerate-publications** (document explicit exclusions): compute macro AUPRC over the 8 mixed-label publications and report CI/permutation on that subset, with the 2 constant-label publications excluded and documented. This would likely produce a numeric primary metric and enable a proper GO/STOP on the recalibrated labels.
2. **Accept fail-closed STOP on the primary estimand** and treat the strong 8-pub per-pub signal as motivation for a secondary/conditional estimand (|delta_r| magnitude regression), which is currently gated by endpoint requiring an identifiable primary + reliable caller.

No number was fabricated, and no degenerate publication was silently dropped. This document is evidence for the authority-epoch-15 re-adjudication only; it does not modify the frozen R6 script, endpoint, or any prior verdict.
