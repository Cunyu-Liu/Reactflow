# Statistical Design v1 — power / null-space / minimum-attainable p

authority_scope: benchmark_v3
schema_version: "reactflow_delta.statistical_design.v1"
generated_by: "scripts/reactflow_delta/verify_physical_test_isolation_v1.py"
status: PRE-MODEL (precomputed before any model results exist)

## 1. Exchangeable unit
The highest-level exchangeable unit is the **publication** (resolved PMID,
else DOI, else `UNRESOLVED_PUBLICATION:<study>`).  Resampling and permutation
are performed only at the publication level.  Seeds are optimization repeats
and do NOT increase publication N.

## 2. Null space
For a paired sign-flip (permutation) test across `N` exchangeable publications,
the null space is the set of sign assignments over the `N` publication-level
paired effects:

- number of unique null assignments = `2^N` (up to global sign: `2^(N-1)`
  distinct orderings).
- The unique null space is enumerated up front; if the same assignment maps to
  two different outcomes the statistic is **degenerate / UNIDENTIFIABLE**.

## 3. Minimum attainable p-value
The smallest exact p-value achievable with `N` exchangeable publications
(two-sided sign test) is:

    p_min(N) = 2 / 2^N = 2^(1-N)

| N | p_min(N) | unique null assignments |
|---|----------|-------------------------|
| 5 | 0.0625   | 32                      |
| 6 | 0.03125  | 64                      |
| 7 | 0.015625 | 128                     |
| 8 | 0.0078125| 256                     |

## 4. Required publication N for target alpha
- alpha = 0.05 (two-sided): require `2^(1-N) <= 0.05`  =>  **N >= 6**.
- alpha = 0.01 (two-sided): require `2^(1-N) <= 0.01`  =>  **N >= 8**.

## 5. Power / confirmatory CI
- A confirmatory confidence interval is only reported when there are at least
  **3 independent** (publication-disjoint) publications; otherwise the effect
  is **UNIDENTIFIABLE**.
- Power is not estimated from confirmatory outcomes.  The precomputed null
  space and p_min are fixed before any model result is produced.

## 6. Isolation
- Development builders never open/deserialize the confirmatory test outcome
  store (enforced by `verify_physical_test_isolation_v1.py`).
- Any open/hash/stat/evaluation event touching the test store is appended to
  the append-only ledger `data_registry/reactflow_delta/test_outcome_access_ledger_v1.jsonl`.
- Mixed-cache / load-then-filter fixtures are forbidden (they could leak test
  outcomes into model / hyperparameter selection).