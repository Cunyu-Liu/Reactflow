# D0-R Functional-Anchor Data Feasibility Audit Report

**Generated:** 2026-07-30T18:19:19.309001+08:00
**Audit Method:** `functional_anchor_124nt_window_offset_31`
**Triage Decision:** `non_zero_candidate_pairs_authorize_d1`
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

**Discrepancy explanation:** Expected 384 (192/probe x 2 probes), likely assuming a 64 nt sub-region (64 positions x 3 mutations = 192). Actual is 744 (372/probe x 2 probes) = full saturation of the 124 nt COVSL5 functional anchor (124 positions x 3 mutations = 372). All 744 candidates are SARS_CoV_2 (the WT anchor species), every functional position 0-123 is covered with exactly 3 mutations, and all 744 have functional Hamming == 1 AND matching pos/ref/alt (0 name_sequence_mismatch exclusions). The handoff explicitly states expected numbers must NOT be forced; actuals are reported honestly.

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

- Command: `/home/cunyuliu/miniconda3/envs/editflow/bin/python -m pytest tests/reactflow_delta --color=no --tb=short -p no:cacheprovider`
- Exit code: 0
- Passed: 93 | Failed: 0 | Errors: 0 | Skipped: 0
- All passed: `True`
- Summary: 93 passed in 1.69s

## Stage Permissions

- **d1_allowed:** `False` — D0-R is an audit-only feasibility stage. D1 execution has NOT started. The triage decision RECOMMENDS authorizing D1 via EPRO v3.1; it does not itself permit D1 training to run.
- **training_allowed:** `False` — Learned training requires D2 Tier B approval. D0-R never authorizes training.
- **triage_decision:** `non_zero_candidate_pairs_authorize_d1`

744 functional-anchor candidate single-mutant relations found (functional Hamming == 1, name-encoded single mutation with matching pos/ref/alt). Recommend publishing EPRO v3.1 to authorize D1. D1 execution itself and learned training remain gated (d1_allowed=False, training_allowed=False).

## Key Findings

1. D0-R functional-anchor audit uses a 124 nt window at offset 31 within the 206 nt full anchor (SL5CV2_NOM_0002), stricter than the prior SEQPOS approach.
2. Functional anchor verified: 124 nt from COVSL5_NOM_0002 occurs exactly once at offset 31 in the 206 nt full anchor (valid=True).
3. WT anchor (profile 1, SARS_CoV_2) identified by exact 206 nt sequence match in both M2SL5 files.
4. M2SL5 produced 744 candidate single-mutant relations (372 2A3 + 372 DMS), vs 384 expected. Expected 384 (192/probe x 2 probes), likely assuming a 64 nt sub-region (64 positions x 3 mutations = 192). Actual is 744 (372/probe x 2 probes) = full saturation of the 124 nt COVSL5 functional anchor (124 positions x 3 mutations = 372). All 744 candidates are SARS_CoV_2 (the WT anchor species), every functional position 0-123 is covered with exactly 3 mutations, and all 744 have functional Hamming == 1 AND matching pos/ref/alt (0 name_sequence_mismatch exclusions). The handoff explicitly states expected numbers must NOT be forced; actuals are reported honestly.
5. COVSL5/SL5CV2 files produced 0 candidates (their mutation annotations use a different format, e.g. G159C, not name-encoded <pos><ref>-<alt>; SL5CV2 files lack the WT anchor sequence).
6. Species partitioning per probe (2499 profiles): SARS_CoV_2=798 (1 WT + 372 single-mut + 425 double-mut), MERS=888, BtCoV=813. MERS/BtCoV excluded (functional Hamming >> 1).
7. Previous SEQPOS-based result (m2sl5_candidate_relations.json, 744 candidates) is preserved as historical evidence and NOT overwritten.
8. All candidate relations are candidate_only_pending_parent_lineage_and_functional_region_validation with true_pair=False. No pair, tier, or model claim is made.

## Scientific Boundary

D0-R is a fail-forward data feasibility audit. Candidate relations are unverified (candidate_only_pending_parent_lineage_and_functional_region_validation, true_pair=False). No pair, tier, or model claim is made. The original D0 acceptance (NO_GO) and the previous D0-R acceptance are both preserved as historical evidence.
