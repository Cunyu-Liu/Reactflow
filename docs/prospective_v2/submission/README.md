# ReactFlow-Delta prospective-v2: Submission Materials Package

> Auto-generated 2026-08-14 from locked artifacts (branch
> codex/reactflow-delta-prospective-v2-20260813 @ 13d34ac).
> P6_REPRODUCIBILITY_DELIVERED; release decision is owner-controlled.

## Contents

| File | Purpose |
|------|---------|
| pr_readiness_checklist.md | PR automated check list + all gate states (A-E) |
| supplementary_data_manifest.md | Data availability + artifacts + results + attrition |
| declaration_statements.md | Data/code availability, competing interests, claims, limitations |
| out/main_tables.md / .tex | Main manuscript tables |
| out/figures/fig1..fig4.png | Main figures |
| out/cards.md | Model/Data/Code cards |
| environment.yml | Conda environment spec |

## Gate summary

P0 PASS | P1 FAIL_CLOSED_OPEN | P2 PROSPECTIVE_SIGNAL | P3 LRSO_EXCEEDS_DIRECT_FOR_DEVELOPMENT (v1/v2 retracted) |
P4 EXTERNAL_STATISTICAL_PASS + CALIBRATION | P5 MECHANISM_NOT_ESTABLISHED | P6 REPLAY_CONSISTENT

## How to verify

Everything in this package is auto-generated from locked artifacts on branch
`codex/reactflow-delta-prospective-v2-20260813` and is reproducible:

1. **Gates** — `pr_readiness_checklist.md` rows A1-A8 and E are checked against the
   live remote state (authority epoch 21 ACTIVE; P2/P3/P4/P5/P6 verdicts above).
2. **Replay** — one-click reproducibility via
   `PYTHONPATH=.:src python scripts/reactflow_delta/run_replay_v1.py`
   with the locked artifacts (P2/P3 re-derived; P4/P5/P5b fresh re-run; P5_COMBINED
   report-level aggregation) yields `REPLAY_CONSISTENT`; see
   `REPRODUCE_p6_20260814.md` for the full command.
3. **Tests** — the full `reactflow_delta` suite passes on the remote GPU machine,
   including the P3 spec-compliance fixtures (`test_p3_lrso_v3.py`, 14 tests) and the
   P5 combined fail-closed fixtures (`test_p5_combined_meta_v1.py`, 11 tests).
4. **No placeholders** — manuscript/supplement drafts contain no TODO/TBD/placeholder
   markers.