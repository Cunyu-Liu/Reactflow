# GO/STOP Terminal Decision — P2 position-granularity re-gate (epoch 17, endpoint_v5, Route C)

**Adapter:** `scripts/reactflow_delta/run_p2_position_v5.py`
**Run:** `results/p2_position_v5_learnability_20260808`
**Overall decision: `STOP`**
**Route: `STOP_METHOD_ROUTE`**

Independent disk-evidence re-adjudication of `P2_LEARNABILITY_GO` on the authority-epoch-17
`endpoint_v5` position-granularity estimand (eligible-position unit, Z_CUT=2.0 pre-registered).
This is the user-authorized Route C pivot (amendment
`reactflow_delta_v4_epoch17_endpoint_v5_position_granularity_20260808.yaml`): after the pair-level
endpoint_v4 STOP (wide bootstrap CI / low power), the evaluation unit is reduced from pair to
eligible position so each publication contributes thousands of position units.

## Model × seed criterion checks (all 5 seeds, all 3 learned models)

Pre-registered GO requires ALL of: `delta>0`, paired publication-block bootstrap `CI lower>0`
(alpha=0.05, n_boot=1000), publication-block `permutation p<0.05` (n_perm=1000), `n_non_degenerate>=5`.

| model | metric | trivial | delta | CI lower | CI upper | perm p | ndeg | GO |
|---|---|---|---|---|---|---|---|---|
| logistic s0 | 0.3645 | 0.3222 | +0.0423 | +0.0055 | 0.0955 | 1.0 | 10 | NO |
| gbm s0 | 0.3669 | 0.3222 | +0.0447 | -0.0000 | 0.1120 | 1.0 | 10 | NO |
| p2_mlp s0 | 0.3612 | 0.3222 | +0.0390 | +0.0010 | 0.0994 | 1.0 | 10 | NO |

Across ALL models and ALL 5 seeds: `delta>0` = True, `n_non_degenerate>=5` = True,
`CI lower>0` = True for logistic (5/5) and p2_mlp (3/5), but **`permutation p<0.05` = False for
every model and every seed (perm p = 1.0)**. No seed passes all four criteria ⇒ `all_seeds_go=False`
for logistic, gbm, and p2_mlp ⇒ **verdict STOP**.

## P2_LEARNABILITY_GO — FAIL (confirmatory permutation not significant)

| Check | Status | Value |
|---|---|---|
| (a) estimand identifiable | PASS | publication-macro position-AUPRC numeric, 10 non-degenerate publications |
| (b) >=5 seeds, fixed budget | PASS | 5 seeds (0..4) |
| (c) delta > 0 | PASS | best gbm +0.0447, logistic/MLP +0.04 |
| (d) paired bootstrap delta CI lower > 0 | PARTIAL | logistic 5/5, p2_mlp 3/5, gbm CI lower ~0 |
| (e) publication-block permutation p < 0.05 | **FAIL** | p=1.0 for ALL models × ALL seeds |
| (f) n_non_degenerate >= 5 | PASS | 10 |

### Why the permutation fails decisively

The paired bootstrap CI (cluster-bootstrap on per-publication delta) is tight at position granularity
and often excludes 0, which endpoint_v4 lacked. However the publication-block **permutation** —
the confirmatory test of whether the observed macro-AUPRC exceeds chance when scores are permuted
among equal-size publications — gives **p=1.0** for every model and every seed. The positive delta is
therefore NOT distinguishable from the within-publication structure reachable under the permutation
null. Incremental **cross-publication** learnability of the mutation-induced reactivity response at the
position level is NOT established.

## Verdict

- `P2_LEARNABILITY_GO` = **FAIL** (confirmatory permutation p=1.0 across all models/seeds).
- Per §13.4 any non-PASS gate stops the route. Phase 3 model-architecture iteration remains **BLOCKED**.
- The epoch-17 position-granularity pivot repaired the bootstrap power (CI lower > 0 for
  logistic/MLP) but did NOT establish cross-publication learnability: the permutation test remains
  decisively non-significant.

## Recommendation

Proceed on the **model-conditional synthetic / theory / descriptive-real-cases** route
(`ONLY_MODEL_CONDITIONAL_THEORY_SOFTWARE_AND_DESCRIPTIVE_REAL_CASES`). Real-data claims that require an
identifiable, permutation-significant, cross-publication learnability remain gated. A further endpoint
amendment (e.g. a within-publication comparison against a different null, or a raw per-replicate
counts kill test) would be a new endpoint version under a new authority epoch and require explicit
user authorization — it is NOT a silent in-place change and is not performed here.