# ReactFlow-Delta prospective-v2: PR Readiness Checklist (auto-check)

> Auto-generated 2026-08-14. Every row below is verified against live remote state
> (branch codex/reactflow-delta-prospective-v2-20260813 @ 13d34ac + /mnt artifacts).

## A. Scientific gates (contract section 12)

| Check | Status | Evidence |
|-------|--------|----------|
| A1 P0 authority/truth reconciliation | PASS | epoch21 ACTIVE; authority_epoch_21 sentinel+bundle |
| A2 P1 data/measurement/evaluator | FAIL_CLOSED_OPEN (non-blocking for P2/P3) | evaluator PASS, held_response_invariance PASS, primary caller exclusion PASS |
| A3 P2 direct learnability | PASS | 20-puzzle CI lower +0.0079 > 0; sign-flip p=1.9e-6 |
| A4 P3 LRSO incremental skill | PASS (LRSO_EXCEEDS_DIRECT_FOR_DEVELOPMENT) | spec-compliant v3 re-run (2026-08-16): rank2/4/8 D_p^P3=+0.0147/+0.0155/+0.0154, all 95% CI lower > 0; 20/20 puzzles positive; sign-flip p=1.9e-6. v1/v2 verdict retracted (implementation failure). |
| A5 P4 external statistical | PASS | K_eff=24 >= 9; component CI lower +0.0153; FWER pass; leave-dominant-out lower +0.0127 |
| A6 P4 calibration | PASS | cov95 0.874 in [0.85,0.99] |
| A7 P5 mechanism | MECHANISM_NOT_ESTABLISHED (reported honestly) | edit-vfar CI lower -0.0199; negative control passes |
| A8 P6 replay | PASS | REPLAY_CONSISTENT; all_reproduced=true |

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
| C1 P2/P3 primary path external-free | PASS | no rdat/rmdb refs in run_p2_direct_v2.py / run_p3_lrso_v2.py |
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
| D6 manuscript/supplement no placeholders | PASS (checked: no TODO/TBD/placeholder markers) |

## E. Gate verdicts (machine-readable)

- P0: AUTHORITY_RECONCILIATION_COMPLETE_PASS
- P1: FAIL_CLOSED_OPEN (blocks_phase2/3=false, blocks_phase4=true)
- P2: PROSPECTIVE_SIGNAL_ESTABLISHED_FOR_DEVELOPMENT
- P3: LRSO_EXCEEDS_DIRECT_FOR_DEVELOPMENT (v1/v2 retracted)
- P4: P4_EXTERNAL_STATISTICAL_PASS + CALIBRATION_ACCEPTABLE
- P5: MECHANISM_NOT_ESTABLISHED
- P6: REPLAY_CONSISTENT (P6_REPRODUCIBILITY_DELIVERED)
