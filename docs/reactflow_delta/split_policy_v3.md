# Split Policy v3 — publication-disjoint + homology sensitivity

authority_scope: benchmark_v3
schema_version: "reactflow_delta.split_policy_v3.v1"
generated_by: "scripts/reactflow_delta/sequence_lineage_overlap_v1.py"
status: PRE-MODEL (homology thresholds fixed before any model results exist)

## 1. Publication-disjoint primary split
- The highest-level exchangeable unit is the **publication** (resolved PMID;
  falls back to DOI; otherwise UNRESOLVED_PUBLICATION:<study>).
- Development roles:
  - `16SFWJ` = DEVELOPMENT_CONSUMED / INVALID_FOR_CONFIRMATORY_USE
  - existing Phase-3 pool = DEVELOPMENT_USED
  - SL5CV2 / SL5HKU / SL5MER = merged into a single publication domain
    `pmid_38427602` (publication N = 1, NOT sufficient for confirmatory).
- New confirmatory candidates MUST come from unexposed, provenance-confirmed
  publications (never from an exposed/development publication domain).

## 2. Stricter publication + homology sensitivity split
- Homology thresholds are PRE-SPECIFIED and fixed (never tuned after seeing
  model results): 70/80/90 (identity% AND coverage%).
- Sensitivity is reported at every threshold jointly; we do NOT pick the most
  favorable threshold afterward.
- A pair is homology-flagged if its sequence falls in a connected component
  that spans more than one distinct publication at the given threshold.

## 3. Conservative leakage guard
- Confirmatory test pairs must be publication-disjoint AND not homology-flagged
  at the strictest prespecified threshold (identity>=90 AND coverage>=90)
  relative to any development pair.

## 4. Homology sensitivity snapshot (computed over eligible exact pairs)
- identity/coverage 70/70: 317/7961 flagged (sensitivity 0.039819)
- identity/coverage 80/80: 317/7961 flagged (sensitivity 0.039819)
- identity/coverage 90/90: 317/7961 flagged (sensitivity 0.039819)

## 5. Exact-sequence / parent / lineage counts
- exact-sequence duplicate sequences: 39
- exact-sequence duplicate pairs: 7961
- shared WT parent groups: 0
- shared WT parent pairs: 0
- existing lineage count: 39
- existing parent count: 75
- existing family count: 1

