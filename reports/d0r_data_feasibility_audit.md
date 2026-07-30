# D0-R Data Feasibility Audit Report

**Generated:** 2026-07-30T17:46:11.948508+08:00
**Triage Decision:** `non_zero_candidate_pairs_authorize_d1`

## Summary

- RDAT files audited: 13
- Files with parse errors: 8
- Files with WT anchor: 2
- Total profiles: 5482
- Profiles with per-profile sequence: 4998
- Accession registry entries: 1024
- M2SL5 candidate relations: 744
- M2SL5 strict candidates (functional_edit_count=1): 744

## Triage Decision

744 strict candidate single-mutant relations found (functional_edit_count=1, name-encoded single mutation). Publish v3.1 authorizing D1. Learned training still requires D2 Tier B.

## Key Findings

1. D0 used filename-keyword recall; D0-R uses paper-accession mapping from 1024 RMDB _entries YAML front matter.
2. D0 only downloaded RMDB metadata; D0-R downloaded 13 RDAT payload files (~33MB) with sha256 checksums.
3. D0 RDAT parser only accepted global SEQUENCE header; D0-R parser supports per-profile SEQUENCE:N lines and sequence: annotation tokens (M2-seq style).
4. D0 parser rejected 5012 M2SL5 profiles as 'parent sequence masked/noncanonical'; D0-R recovers per-profile sequences from annotation tokens.
5. D0 pairing used mutation labels only; D0-R computes actual edit sets from per-profile sequence vs WT anchor, partitioned by SEQPOS window.
6. ETERNA_R78 (PMID 36192461, paper-explicit) uses RDAT_VERSION 0.33, inaccessible to fail-closed v0.34 parser. Recorded as parse error.
7. M2SL5 (Ribonanza pre-competition) produced 744 strict candidate single-mutant relations (functional_edit_count=1).
8. Ribonanza Kaggle data inaccessible (no credentials), but M2SL5 derivative accessible via RMDB. D0 404 reinterpreted as探测 method failure.
9. All candidate relations are candidate_only_unverified — lineage NOT independently confirmed.

## Ribonanza Audit

- Kaggle CLI installed: False
- Kaggle credentials present: False
- M2SL5 accessible via RMDB: True
- Conclusion: Ribonanza Kaggle data is NOT directly accessible in this environment (no kaggle CLI, no credentials). However, M2SL5 (Ribonanza pre-competition derivative) IS accessible via RMDB and has been downloaded and parsed. The D0 404 is reinterpreted as a探测 method failure, not data inaccessibility.

## D1 Authorization

**Authorized:** True

Conditions:
- All D1 training pairs must come from strict candidates (functional_edit_count=1)
- Learned training requires D2 Tier B approval
- Lineage must be verified before upgrading from candidate to confirmed pair
- Raw RDAT files remain read-only (checksum-verified, not modified)

## Scientific Boundary

D0-R is a fail-forward data feasibility audit. Candidate relations are unverified. No pair, tier, or model claim is made. The original D0 acceptance (NO_GO) is preserved as historical evidence.
