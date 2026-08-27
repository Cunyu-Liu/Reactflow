# ReactFlow-Delta prospective-v2: PR Readiness Checklist (auto-check)

> **SUPERSEDED HISTORICAL CHECKLIST — NOT LIVE AND NOT SUBMISSION-READY.** Generated
> 2026-08-14 against `codex/reactflow-delta-prospective-v2-20260813 @ 13d34ac`.
> That ref is not current authority. V13M3 is terminal FAIL; V14 has no terminal
> verdict. Rows below are retained for audit, with current scientific qualification
> corrections applied where the old status is no longer valid.

## A. Scientific gates (contract section 12)

| Check | Status | Evidence |
|-------|--------|----------|
| A1 P0 authority/truth reconciliation | HISTORICAL_ONLY | epoch21 is superseded; current authority is V14M3 score-closed |
| A2 P1 data/measurement/evaluator | FAIL_CLOSED_OPEN (non-blocking for P2/P3) | evaluator PASS, held_response_invariance PASS, primary caller exclusion PASS |
| A3 P2 direct learnability | HISTORICAL_POST_HOC_DEVELOPMENT_EVIDENCE | 20-puzzle development universe has been repeatedly consumed; not prospective confirmation |
| A4 P3 LRSO incremental skill | HISTORICAL_POST_HOC_DEVELOPMENT_EVIDENCE | v3 is the valid historical implementation, but its development result is not external or publication qualification |
| A5 P4 external statistical | INVALIDATED_BY_SEQPOS_ALIGNMENT | legacy external features/scoring were misaligned; old PASS is not citable |
| A6 P4 calibration | NOT_CURRENTLY_QUALIFIED | derived from the same alignment-invalid external run |
| A7 P5 mechanism | MECHANISM_NOT_ESTABLISHED (reported honestly) | edit-vfar CI lower -0.0199; negative control passes |
| A8 P6 replay | INTERNAL_RETAINED_ARTIFACT_REPLAY_ONLY; EXTERNAL_DENIED | default P2/P3 replay only; artifact consistency cannot restore qualification |

## B. Engineering / automated checks

| Check | Status | Notes |
|-------|--------|-------|
| B1 CI workflow dependency completeness | FIXED (this PR) | added numpy/pandas/scipy/PyYAML/sklearn to [dev]; installs .[dev,torch] |
| B2 test_caller_v2 sys.modules pollution | FIXED (this PR) | unique module name caller_v2_under_test; was causing 3 false failures in full-suite order |
| B3 P4/P5/P6 core test suites | PASS (45 tests) | test_p4_external_v1, test_p5_mechanism_v1, test_p4_calibration_v1, test_run_replay_v1, test_generate_p6_tables_figures_v1, test_build_p6_cards_v1 |
| B4 Full reactflow_delta suite | RUNNING (verify green after B2 fix) | previously 3 caller_v4 failures were pollution, now resolved |
| B5 verify-symbolic | PASS | reactflow.cli verify-symbolic |
| B6 Coverage 90% (reactflow package) | to verify on CI | coverage source = reactflow package only |

## C. Data isolation / exposure

| Check | Status | Notes |
|-------|--------|-------|
| C1 Historical P2/P3 primary path external-free | PASS (engineering scope only) | no rdat/rmdb refs in run_p2_direct_v2.py / valid run_p3_lrso_v3.py path |
| C2 Development-external disconnected | PASS | 24 components, zero sequence overlap |
| C3 Locked outcome access controlled | PASS | P4/P5 single open per frozen protocol |

## D. Artifact completeness

| Check | Status |
|-------|--------|
| D1 P2 held rows (975,599) | PASS |
| D2 P3 LRSO result | PASS |
| D3 P4 components + result + calibration | PASS |
| D4 P5 mechanism result | PASS |
| D5 P6 tables/figures/cards/env | PASS |
| D6 manuscript/supplement no placeholders | FAIL / OPEN | author-contribution placeholders remain; package is not submission-ready |

## E. Gate verdicts (machine-readable)

- P0/P1: historical authority record only; consult the current active contract.
- P2/P3: historical post-hoc development evidence; not prospective/external proof.
- P4: legacy external score invalidated by seqpos alignment; no current calibration
  or transportability qualification.
- P5: `MECHANISM_NOT_ESTABLISHED`.
- P6: retained P2/P3 artifact replay only; external replay denied by current authority.
- V13M3: terminal top-journal screen FAIL; V13M4 closed.
- V14M3: terminal verdict pending; submission readiness not established.
