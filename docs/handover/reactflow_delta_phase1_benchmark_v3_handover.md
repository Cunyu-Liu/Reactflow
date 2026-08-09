# ReactFlow-Delta Phase 1 (benchmark_v3) Handover

- date: 2026-08-09
- authority epoch: 20 (PHASE1_BENCHMARK_V3_ONLY)
- gate verdict: PHASE1_BENCHMARK_V3_PASS_WITH_NON_BLOCKING_NOTICE
- Phase-2 prereq PASS: authority_semantic_closure, asset_disposition, pair_publication_resolution, sequence_lineage_leakage_audit, split_independence, physical_test_isolation, caller_stability, endpoint_mask_alignment, keyed_prediction_schema, evaluator_fixtures, license_release_status
- Phase-2 prereq NOT-PASS (non-blocking notice): test_statistical_sufficiency

## What was built

- 1024-asset controlled disposition (unique asset_id, no silent drop).
- Pair publication registry (7961 eligible pairs; resolved/unresolved citation status; same-PMID merging including SL5->pmid_38427602).
- Sequence/lineage/homology leakage audit (70/70, 80/80, 90/90 sensitivity).
- Split v3 (publication-disjoint) + statistical design.
- Physical test isolation + append-only access ledger.
- Endpoint v6 (three-tier task) + CallerV4 (STRICT primary / TRANSDUCTIVE sensitivity).
- Keyed prediction schema v2 + evaluate_v6 (publication-anchored, no position zip).

## Scientific gate

- All benchmark-construction gates PASS.
- `test_statistical_sufficiency` is NOT_ESTABLISHED: no untouched, provenance-confirmed confirmatory publication set exists yet. This blocks Phase 4 (locked test), NOT Phase 2 (development learnability).
- No model trained; no confirmatory outcome opened; fail-closed.

## Next

- Await `AUTHORIZE_PHASE2_LEARNABILITY` to run strong simple/generic baselines for cross-real-publication learnability.
- Method modeling is NOT authorized until learnability is established.

Status: STOPPED_AT_OWNER_REVIEW

