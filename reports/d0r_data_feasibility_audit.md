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


---

# D0-R v2 Re-Audit (Tier A Non-Ribonanza Expansion)

> **Forward-only addendum.** All v1 content above is preserved unchanged as
> historical evidence. This section records a READ-ONLY re-audit that expands
> the candidate pool beyond the M2SL5 single-study set. It does NOT supersede,
> lower, or retract any prior gate; it only adds new evidence. D1 remains
> unauthorized.

**Date:** 2026-07-30  **Branch:** `codex/reactflow-delta-d0r`  **Script:** `scripts/reactflow_delta/d0r_reaudit_tierA.py`

## Scope

Re-audit the Tier A non-Ribonanza subset of the RMDB accession registry
(`data_registry/d0r_accession_registry.jsonl`) that was NOT already downloaded
in D0-R v1 (13 files). Selection filters: exclude Ribonanza/Kaggle library
entries (`RIBO_RE`); include entries whose text matches Tier A mutation-relevant
signals (`TIERA_PATS`: mutate-and-map, m2-seq, m2r-rescue, riboSNitch/SNP).

## Method

For each selected RDAT file, parse with `parse_rdat`, then identify a WT anchor
and classify each non-WT profile as a single-mutant candidate. Two modes:

- **Sequence-based** (`classify_profile_general`) — when per-profile sequences
  exist. Edits are partitioned by SEQPOS into functional vs flanking; a
  candidate requires exactly one name-encoded OR annotation-encoded mutation
  with functional_edit_count==1 and matching position/ref/alt (DNA->RNA T->U).
- **Annotation-only** (`classify_profile_annotation_only`) — when NO per-profile
  sequences exist (standard M2-seq / mutate-and-map-seq files, e.g. RNASEP,
  L21RNA). A candidate is exactly one `<ref><pos><alt>` annotation mutation
  (e.g. `G159C`) whose `ref` is verified against the header `SEQUENCE` at the
  encoded position (construct-local 1-indexed, with optional OFFSET adjustment).

### Annotation-only caveat (important)

For M2-seq files the encoded alt base is `X` (variable pool), so the
sequence-level edit is **NOT verifiable** without per-profile sequences. All
annotation-only candidates are therefore `true_pair=False` with lineage status
`candidate_only_pending_parent_lineage_and_functional_region_validation`. This
is an honest, weaker evidentiary bar than the sequence-based path; it is
recorded as such and must be upgraded in D1 before any pair claim.

## Result Summary

| Metric | Value |
|---|---|
| Tier A entries selected | 101 |
| Downloaded OK | 100 |
| Download failed | 1 (`TRP4P6_DMS_0007`) |
| Files parsed OK | 72 |
| Parse errors (honest) | 28 |
| **Total candidate single-mutant pairs** | **7,761** |
| Distinct studies (owner, doi) | 8 |
| Distinct parents (rmdb_id prefix) | 31 |
| Distinct owners | 6 |
| `re_tier_judgment` | **Tier A** |
| `re_d1_allowed` | **False** |

### Tier gate check (contract §8)

Tier A requires >=5 studies AND >=20 parents AND >=5,000 pairs. Observed
8 / 31 / 7,761 — **all three gates met** -> `Tier A`. (Tier B: >=3 / >=10 / >=1,000.)

### Per-owner candidate counts

| Owner | Candidates |
|---|---|
| Kalli Kappel | 4,528 |
| Rhiju Das | 2,001 |
| rui huang | 640 |
| Gun Woo Byeon | 394 |
| Ivan Zheludev | 143 |
| Clarence Cheng | 55 |

### Per-parent candidate counts (top files)

| Parent / file | Candidates |
|---|---|
| L21 (`L21RNA_DMS_0000`) | 433 |
| hc16-prod (`HC16PRO_DMS_0000`) | 417 |
| hc16 (`HC16APO_DMS_0000`) | 408 |
| scaRNA6 (`SCARNA6_DMS_0000`) | 335 |
| VCKT-apo / VCKT-gly | 301 + 301 |
| RNASEP (3 files) | 265 x 3 = 795 |
| RB1 (`RB1UTR_DMS_0000`) | 262 |
| 24-3 | 250 |
| FNKT / FNKT+gly | 237 + 237 |

(31 distinct parents total; full per-parent and per-file breakdowns in
`artifacts/reactflow_delta/d0r/d0r_reaudit_tierA_summary.json`.)

## Parse Errors (28, honest forward-only)

| Error type | Count | Files |
|---|---|---|
| `RDAT_VERSION != 0.34` (not accepted in D0) | 26 | `BSUGLY_DMS_0003..0014` (12, v0.4), `TRP4P6_DMS_0002..0014` excl. 0007 (12, v0.22/v0.24/VERSION-key), `CBAG4P_DMS_0003..0004` (2) |
| `invalid indexed annotation key` | 2 | `GLYCFN_KNK_0001`, `GLYCFN_KNK_0002` |

These are recorded, not silently dropped. The version-gated files can be
revisited in D1 if the parser is extended to accept later RDAT versions.

## 0-Candidate Files (24, parsed OK but no candidate)

| Group | Count | Reason |
|---|---|---|
| `HIV3PR_*` (DMS/NMD x8) | 8 | `annotation_ref_mismatch` for all mutants — annotation uses HIV genome numbering (offset), not construct-local; WT found but ref does not match header `SEQUENCE` at the encoded position. Solvable in D1 with the correct offset. |
| `SL5CV2/SL5HKU/SL5MER_*` (x12) | 12 | `no_wt_anchor` — no `mutation:WT` annotation and no per-profile sequences to identify a WT anchor. |
| `TODEX_1M7/DMS_0000` (x2) | 2 | `no_encoded_mutation` — different annotation format, no `<ref><pos><alt>` mutation token. |
| `RNASEP_RSQ_0000`, `TODS7_MUT_0001` | 2 | `no_wt_anchor`. |

## v1 vs v2 Comparison

| | D0-R v1 | D0-R v2 re-audit |
|---|---|---|
| Studies | 1 (M2SL5) | 8 |
| Parents | 1 (SL5 SARS-CoV-2) | 31 |
| Candidate pairs | 744 | 7,761 |
| Method | name-encoded `<pos><ref>-<alt>` + 124 nt functional anchor | general SEQPOS functional window + annotation-only fallback |
| Evidentiary bar | sequence-level edit verified (pos/ref/alt) | sequence-verified OR annotation-ref-verified (alt=X unverified for M2-seq) |

The v1 744 candidates (M2SL5, sequence-verified) are a strict subset
evidentiarily; v2 adds 7,017 annotation-only candidates at a weaker bar. Both
are preserved; neither is retracted.

## Stage Permissions (re-affirmed)

- **`re_d1_allowed`: `False`** — This re-audit QUALIFIES the data to propose
  EPRO v3.1 and a non-learning (cleanup-only) D1, but does NOT authorize D1.
  D1 is blocked until v3.1 is published.
- **`training_allowed`: `False`** — Learned training requires D2 Tier B
  approval; D0-R never authorizes training.
- **`re_triage_decision`:** `reaudit_qualified_to_propose_v3_1_non_learning_d1_cleanup_only`

## Artifacts

- Gitignored (large): `artifacts/reactflow_delta/d0r/d0r_reaudit_tierA_audit.json`,
  `..._relations.json` (7,761 relations), `..._summary.json`.
- Tracked (slim manifest with download records + sha256):
  `manifests/reactflow_delta/d0r/d0r_reaudit_tierA_manifest.json`.
- Raw RDAT files (100, ~49 MB): `/mnt/cunyuliu/reactflow_delta_raw/rmdb/rdat_tierA_20260730/`.

## Scientific Boundary (re-affirmed)

Read-only D0 re-audit. All candidate relations are unverified
(`candidate_only_pending_parent_lineage_and_functional_region_validation`,
`true_pair=False`). No pair, tier, or model claim is made. The Tier A judgment
is a data-availability gate, not a model-quality claim. D1 is not authorized
(v3.1 contract not published). Previous D0-R v1 results and the original D0
acceptance (NO_GO) are preserved as historical evidence.
