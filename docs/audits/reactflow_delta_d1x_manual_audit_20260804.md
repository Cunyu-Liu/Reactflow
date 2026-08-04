# D1-X Manual Audit — ReactFlow-Delta V4

- **Phase**: D1-X (Exact Canonicalization and Cleaning)
- **Reviewer role**: `CODEX_PRIMARY_IMPLEMENTATION_AGENT`
- **Reviewer external identity**: `null` (`NOT_EXTERNALLY_VERIFIED`)
- **Sample mode**: `FULL_D1X_CANONICALIZATION_AUDIT`
- **Audit date**: 2026-08-04
- **Evidence class**: `DATA_QUALIFICATION_ONLY` (no exact-pair eligibility, Tier,
  split, training, or scientific claim)

## 1. Scope of this audit

This audit confirms that the D1-X canonicalization is a complete, auditable,
fail-closed record of the D0-X candidate inventory transformed into canonical
records with closed-set roles, exact ref/alt, WT-mutant condition pairing, and
traceable parent lineage. It does NOT make any scientific claim. Scope items:

1. Schema conformance — every canonical record carries `reactflow_delta.data_record.v4.0`.
2. Required-field retention — every required field present on every record.
3. Reactivity layers — raw/upstream/train-frozen present; missing positions masked, never zero-filled.
4. Primary records — exact ref/alt, verified coordinate, WT anchor, and matched condition on 100%.
5. Non-primary records — every excluded record has a controlled reason.
6. Pair records — schema `d1x_pair.v1`, role PRIMARY_EXACT_DELTA, traceable pointers.
7. Count reconciliation — no silent drop; counts match the canonicalization summary.

## 2. Audit method

The canonicalization audit (`scripts/reactflow_delta/d1x_audit_canonical.py`) was
run over the full canonical records and primary-pairs JSONL produced by
`d1x_canonicalize.py` (run_id `d1x_canonicalization_20260804_v1`). Every check
had to PASS for the D1-X gate to be eligible.

## 3. Canonicalization outputs

- Canonical records: `d1x_canonical_records.jsonl`
- Primary pairs: `d1x_primary_pairs.jsonl`
- Summary: `d1x_canonicalization_summary.json`

## 4. Findings

The automated audit reported all checks PASS, including:
- schema_all_v4, fields_all_present, layers_all_present
- primary_ref_alt_all_present, primary_wt_anchor_all_present,
  primary_condition_all_matched
- non_primary_all_have_reason
- pair_schema_all_v1, pair_role_all_primary, pair_all_traceable
- record_count_matches_summary, pair_count_matches_summary

## 5. Disposition

- **PASS** (no failing scope item).
- The scientific boundary is unchanged: D1-X canonicalization is closed; no
  exact-pair eligibility, Tier, split, training, or scientific claim is made.
  D2-X is not started.

## 6. Data-role summary (from the canonicalization summary)

Recorded in `d1x_canonicalization_summary.json` under `data_role_counts` and
`exact_mutation_evidence_status_counts`. The canonicalization is outcome-blind:
role and exclusion are derived only from mutation token, coordinate, sequence,
condition, and lineage metadata — never from observed reactivity or Delta.