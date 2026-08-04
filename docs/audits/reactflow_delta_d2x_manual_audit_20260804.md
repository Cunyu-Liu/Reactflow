# D2-X Manual Audit — 2026-08-04

## Scope
Full D2-X split/exposure/data-tier-candidate freeze audit over the D1-X canonical
records and primary pairs. Data-qualification only; no scientific claim.

## Method
- Inspected the D1-X canonical records (`d1x_canonical_records.jsonl`) and primary
  pairs (`d1x_primary_pairs.jsonl`) for grouping atoms (study / parent / design_lineage).
- Verified the deterministic, outcome-blind split assignment (test=16SFWJ, validation=CIDGMP).
- Re-ran the split build and confirmed overlap audit (design-lineage, exact-sequence,
  near-duplicate split-aware) reports zero cross-split leakage.
- Verified Tier B+ data-candidate computation (changers UNKNOWN_NOT_ASSERTED).
- Verified test seal SEALED, test access ledger append-only with no sample access,
  and blind viability certificate aggregate-only.

## Findings
- None blocking. All 20 audit checks pass.

## Disposition
PASS

## Boundary
Full Tier B+ requires PH0-X identifiability; full Tier A+ requires B0-X. Test remains sealed.
