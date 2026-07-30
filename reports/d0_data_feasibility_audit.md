# ReactFlow-Delta D0 data feasibility audit

## Scope and scientific boundary

This is a public-data feasibility audit only. It reports parsed-artifact evidence; it does not report a trained model, a benchmark result, or biological validation.

## Audited counts

- Audited entries: 6
- Audited profile/construct records: 5175
- Filename-only mutational candidates: 15 M2-named; 11 M2R-named-unconfirmed.
- Explicit WT profiles: 1
- Confirmed single-mutant profiles: 0
- Single-site labels with unknown endpoint: 159
- Confirmed true WT--single-mutant pairs: 0
- Confirmed double/rescue profiles: 0
- Same-sequence replicate groups/profiles: 7/14
- Explicit no-edit profiles: 1
- Known studies/parents: 0/0

## Probe, condition, and observation metadata

- Probe profile counts: {"2A3": 2499, "DMS": 2499, "SHAPE": 14, "unknown_probe": 163}
- Condition strata: 5
- In-vitro/in-vivo metadata: {"unknown_metadata": 5175}
- Exclusion reasons: {"no explicit single-mutant endpoint": 163, "parent sequence is masked or noncanonical": 5012}

## Pair and tier decision

- RMDB fixture-scope candidate pairs: 0
- Ribonanza same-condition single-edit pairs: unknown (raw data was not acquired in this environment).
- Tier A: not_supported: zero confirmed true pairs in the six-fixture audit scope
- Tier B: not_assessable: Ribonanza raw table unavailable in current environment
- Tier C: audit_only: parsed public construct/profile observations are not primary intervention truth
- Highest currently supported status: below_Tier_B_audit_only
- Allow D1: False

## Largest uncertainties

1. Ribonanza raw data was not acquired in this environment, so same-batch/same-condition single-edit pair count is unknown rather than zero.
2. The six frozen RMDB fixtures are a narrow audited sample; their RDAT header parent sequences are masked/noncanonical or their mutation endpoints are unspecified.
3. Study IDs and independently established parent lineage are absent from all 5,175 audited construct records, preventing true-pair provenance.

## Stop rule

D1 and any learned training are blocked. No metric threshold is lowered; resolving the documented data/provenance gaps is required before a new gate decision.

## Version control

- Evidence-generating commit: `a79fd073dd1d28ab8b2a2554efd6e8050e52550d`
- Branch: `codex/reactflow-delta-r0`
- Push status: `verified_pushed` to `origin/codex/reactflow-delta-r0`
- The companion immutable D0 acceptance certificate records hashes for this report, the data summary, parser results, and the v3 contract.
