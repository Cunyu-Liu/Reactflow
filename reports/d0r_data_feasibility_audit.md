# D0-R Functional-Anchor Data Feasibility Audit Report

**Generated:** 2026-07-30T19:06:30.003672+08:00
**Audit Method:** `functional_anchor_124nt_window_offset_31`
**Triage Decision:** `qualified_to_propose_v3_1_non_learning_d1_cleanup_only`
**Stage Permissions:** d1_allowed=`False`, training_allowed=`False`

## Summary

- Audit method: 124 nt functional window at offset 31 within the 206 nt full anchor
- M2SL5 candidate relations: 744 (372 2A3 + 372 DMS)
- COVSL5/SL5CV2 candidate relations: 0
- Grand total candidate relations: 744
- Expected (handoff): 384 (192/probe)
- Matches expected: `False`

## Anchor Verification

- Full anchor (SL5CV2_NOM_0002): 206 nt, matches expected=`True`
- Functional anchor (COVSL5_NOM_0002): 124 nt, matches expected=`True`
- Functional anchor in full anchor: valid=`True`, offset=31, occurrences=[31]

## Candidate Counts: Actual vs Expected

- Actual: **744** M2SL5 candidates (372/probe x 2 probes = 124 positions x 3 mutations = 372/probe)
- Expected: 384 (192/probe, likely 64 nt sub-region x 3 = 192/probe)
- Species among candidates: SARS_CoV_2=744
- Positions covered (aggregate): 124 (range [0, 123]), mutations per position (aggregate): [6]
- Per probe, all 124 positions covered with exactly 3 mutations: `True`

**Discrepancy explanation:** Expected 384 (192/probe x 2 probes), likely assuming a 64 nt sub-region (64 positions x 3 mutations = 192). Actual is 744 (372/probe x 2 probes) = full per-probe saturation of the 124 nt COVSL5 functional anchor (124 positions x 3 mutations = 372/probe; aggregate 744 = [6] per position across 2 probes). Species distribution among candidates: SARS_CoV_2=744. Per probe, all 124 positions covered with exactly 3 mutations: `True`. All 744 have functional Hamming == 1 AND matching pos/ref/alt. The handoff explicitly states expected numbers must NOT be forced; actuals are reported honestly.

## Exclusion Breakdown (per M2SL5 file)

### M2SL5_2A3_0000

- functional_hamming_not_1: 810
- multiple_name_mutation_encodings: 1314
- no_name_mutation_encoding: 2
- candidate_single_functional_anchor: 372

### M2SL5_DMS_0000

- functional_hamming_not_1: 810
- multiple_name_mutation_encodings: 1314
- no_name_mutation_encoding: 2
- candidate_single_functional_anchor: 372

## COVSL5/SL5CV2 Files

- Total candidates: 0
- Source files: COVSL5_DMS_0001, COVSL5_DMS_0002, COVSL5_NOM_0001, COVSL5_NOM_0002, SL5CV2_2A3_0001, SL5CV2_DMS_0001, SL5CV2_NOM_0001, SL5CV2_NOM_0002
- SL5CV2 files lack the WT anchor sequence (wt_anchor_found=false); COVSL5 files have the WT anchor but their mutation annotations use a different format (e.g. `G159C`) that does not match the name-encoded `<pos><ref>-<alt>` scheme.

## Test Results

- Command: `/home/cunyuliu/miniconda3/envs/editflow311/bin/python -m pytest tests/reactflow_delta --color=no --tb=short -p no:cacheprovider`
- Python interpreter: `/home/cunyuliu/miniconda3/envs/editflow311/bin/python`
- Python version: Python 3.11.15
- Exit code: 0
- Passed: 93 | Failed: 0 | Errors: 0 | Skipped: 0
- All passed: `True`
- Summary: 93 passed in 0.41s

## Stage Permissions

- **d1_allowed:** `False` — D0-R is an audit-only feasibility stage. D1 execution has NOT started and is NOT authorized. The triage decision records that the candidate count QUALIFIES for proposing EPRO v3.1 and a non-learning (cleanup-only) D1 permission; it does not itself authorize D1. Until v3.1 is published, no D1 work is permitted.
- **training_allowed:** `False` — Learned training requires D2 Tier B approval. D0-R never authorizes training.
- **triage_decision:** `qualified_to_propose_v3_1_non_learning_d1_cleanup_only`

744 functional-anchor candidate single-mutant relations found (functional Hamming == 1, name-encoded single mutation with matching pos/ref/alt). This QUALIFIES for proposing EPRO v3.1 and a non-learning (cleanup-only) D1 permission; it does NOT authorize D1. D1 execution has not started, and learned training remains prohibited (d1_allowed=False, training_allowed=False).

## Key Findings

1. D0-R functional-anchor audit uses a 124 nt window at offset 31 within the 206 nt full anchor (SL5CV2_NOM_0002), stricter than the prior SEQPOS approach.
2. Functional anchor verified: 124 nt from COVSL5_NOM_0002 occurs exactly once at offset 31 in the 206 nt full anchor (valid=True).
3. WT anchor identified by exact 206 nt sequence match in both M2SL5 files.
4. M2SL5 produced 744 candidate single-mutant relations (372 2A3 + 372 DMS), vs 384 expected. Expected 384 (192/probe x 2 probes), likely assuming a 64 nt sub-region (64 positions x 3 mutations = 192). Actual is 744 (372/probe x 2 probes) = full per-probe saturation of the 124 nt COVSL5 functional anchor (124 positions x 3 mutations = 372/probe; aggregate 744 = [6] per position across 2 probes). Species distribution among candidates: SARS_CoV_2=744. Per probe, all 124 positions covered with exactly 3 mutations: `True`. All 744 have functional Hamming == 1 AND matching pos/ref/alt. The handoff explicitly states expected numbers must NOT be forced; actuals are reported honestly.
5. COVSL5/SL5CV2 files produced 0 candidates (their mutation annotations use a different format, e.g. G159C, not name-encoded <pos><ref>-<alt>; SL5CV2 files lack the WT anchor sequence).
6. Species distribution among the 744 candidates: SARS_CoV_2=744. Non-WT-species profiles are excluded (functional Hamming >> 1).
7. Previous SEQPOS-based result (m2sl5_candidate_relations.json) is preserved as historical evidence and NOT overwritten.
8. All candidate relations are candidate_only_pending_parent_lineage_and_functional_region_validation with true_pair=False. No pair, tier, or model claim is made.

## Scientific Boundary

D0-R is a fail-forward data feasibility audit. Candidate relations are unverified (candidate_only_pending_parent_lineage_and_functional_region_validation, true_pair=False). No pair, tier, or model claim is made. The original D0 acceptance (NO_GO) and the previous D0-R acceptance are both preserved as historical evidence.
