# D0-X Manual Audit — ReactFlow-Delta V4

- **Phase**: D0-X (Exact Mutation Recovery — candidate inventory)
- **Reviewer role**: `CODEX_PRIMARY_IMPLEMENTATION_AGENT`
- **Reviewer external identity**: `null` (`NOT_EXTERNALLY_VERIFIED`)
- **Sample mode**: `FULL_D0X_INVENTORY_AUDIT` (all 1,024 frozen source assets, not a sample)
- **Audit date**: 2026-08-03
- **Evidence class**: `DATA_QUALIFICATION_ONLY` (no exact-pair count, eligibility, Tier, split, training, or scientific claim)

## 1. Scope of this audit

The D0-X manual audit confirms that the candidate inventory is a complete,
auditable, fail-closed record of the frozen D0-X source universe. It does NOT
make any scientific claim. Scope items:

1. Per-accession disposition — 100% coverage.
2. Parser coverage — full, with every failed parse explicitly recorded.
3. Field retention — presence of the D0-X profile schema on every record.
4. Silent-drop — zero unreported loss.
5. Raw-dir sidecar `.headers` files — documented.
6. Requalification — all frozen assets verified by SHA-256 + byte count.

## 2. Per-accession disposition (100% coverage)

Every one of the 1,024 frozen RMDB assets has a disposition. No `NOT_SEARCHED`.

| Disposition | Count |
|---|---|
| `PARSED` | 414 |
| `PARSE_FAILED` | 610 |
| `MISSING_FILE` | 0 |
| `NOT_SEARCHED` | 0 |
| **Total** | **1,024** |

## 3. Parser coverage

- Frozen assets: 1,024
- Parsed files: 414 (coverage rate 0.404297)
- Parse-failed files: 610 (explicitly recorded with per-file error reason)
- Missing files: 0
- Not-searched: 0

The 610 parse failures are all genuine RDAT data irregularities, not parser
silent-loss. The strict parser preserves every REACTIVITY/SEQPOS value
(`_numeric_values` converts each token to float or `None` and never drops
tokens), so the `REACTIVITY length does not match SEQPOS` failure is a real
length mismatch in the file. Failure breakdown (all `D0XContractError`):

| Failure category | Count |
|---|---|
| REACTIVITY length does not match SEQPOS | 311 |
| missing REACTIVITY rows | 168 |
| annotation token lacks key/value separator | 63 |
| malformed SEQPOS token (negative / non-standard coordinates) | 56 |
| missing SEQPOS | 6 |
| REACTIVITY non-numeric value | 2 |
| invalid positive profile index | 2 |
| header STRUCTURE requires one non-empty value | 2 |
| **Total** | **610** |

## 4. Field retention

All 24 D0-X profile schema keys are present on 100% of the 4,386,310 profile
records (`present_rate = 1.0` for every key). Fields that are legitimately null
in D0-X (no value assigned yet) are fully present but null:

- `data_role` — 0% non-null (role assignment is a D1-X concern).
- `replicate_block_id` — 0% non-null (no replicate blocks resolved at D0-X).
- `ref_allele` / `alt_allele` — 0.19% / 0.10% non-null (only the 8,333 records
  carrying an exact ref/alt mutation token have non-null alleles).
- `provisional_data_role` — 0.21% non-null (provisional roles are assigned only
  to the 9,083 records carrying an exact / rescue / latent-alt role).
- `exclusion_reason` — 99.79% non-null (most candidate profiles are flagged for
  exclusion pending D1 canonicalization).

## 5. Silent-drop audit

Total silent-drop count: **0** (`silent_drop_ok = true`). No profile record was
silently dropped during parsing or inventory construction.

## 6. Raw-dir sidecar files

- Manifest main files present: 1,024 (matches manifest name count 1,024).
- Sidecar `.headers` files: 101 (these are RMDB-generated per-file header
  sidecars retained alongside the raw `.rdat` files; they are not parse
  sources and are excluded from the manifest).
- Non-header sidecar files: 0.

## 7. Requalification

All 1,024 frozen assets were requalified against their expected SHA-256 hash
and byte count: disposition `VERIFIED` for all 1,024 (verified_count = 1,024,
hash_mismatch_count = 0, missing_count = 0).

## 8. Profile summary (candidate inventory only)

- Total profile records: **4,386,310**
- Total seqpos count: 499,705,712
- Total missing-reactivity count: 83,989,399
- Source-group counts: data-eterna 107, data-general 127, data-puzzle 10,
  data-riboswitches 51, data-rna-structures 119.

Exact-mutation evidence status counts (candidate-level, PROVISIONAL only):

| Status | Count |
|---|---|
| EXACT_REF_ALT_TOKEN_REF_VERIFIED_PROFILE_SEQUENCE_UNAVAILABLE | 4,496 |
| LATENT_ALT_X_REF_CHECKED | 3,837 |
| MISSING_MUTATION_ANNOTATION | 4,376,132 |
| MULTIPLE_MUTATION_ANNOTATION_VALUES | 750 |
| INVALID_MUTATION_TOKEN | 537 |
| WT_CONTROL_CANDIDATE | 558 |

Provisional data-role counts:

| Role | Count |
|---|---|
| PRIMARY_EXACT_DELTA | 4,496 |
| AUXILIARY_LATENT_ALT | 3,837 |
| RESCUE_MULTI_EDIT | 750 |
| null (no provisional role) | 4,377,227 |

## 9. Findings and disposition

- No finding blocked the audit. All scope items are satisfied.
- Disposition: **PASS**.
- The scientific boundary is unchanged: D0-X candidate inventory is closed;
  no exact pair count, eligibility, Tier, split, training, or scientific claim
  is made. D1-X is not started.