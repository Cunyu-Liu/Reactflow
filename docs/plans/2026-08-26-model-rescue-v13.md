# ReactFlow-Delta Model Rescue v13 implementation plan

## V13M0 — freeze

Create and validate the isolated amendment, machine contract, decision ledger, architecture decision and active authority. Parent files remain unchanged. Commit before implementation authority is opened.

## V13M1 — implementation and invariants

Add:

- `scripts/reactflow_delta/model_rescue_v13.py` — shared batched encoder, exact-mutant builder, anchored point head and exact WT-replay null;
- `scripts/reactflow_delta/run_model_rescue_v13.py` — prediction-only fold runner reusing V11 feature41 and V10 calibration inputs;
- `scripts/reactflow_delta/merge_model_rescue_v13.py` — complete-fold merge;
- `scripts/reactflow_delta/score_model_rescue_v13.py` — complete-only scorer;
- `scripts/reactflow_delta/qualify_model_rescue_v13.py` — mechanical Gate;
- `scripts/reactflow_delta/run_model_rescue_v13_screen_controller.sh` — persistent, missing-fold-only controller;
- focused tests under `tests/reactflow_delta/`.

Required tests before real data:

1. candidate/null exact parameter and initial-state match;
2. candidate mutates exactly one corrected token; null mutates zero;
3. WT replay produces zero hidden delta in evaluation mode;
4. exact-mutant delta is generally nonzero and receiver-shaped;
5. candidate/null difference is limited to second-pass sequence input;
6. WT and second-pass encoder rows replay an identical dropout mask, so stochastic mask noise cannot enter the counterfactual hidden difference;
7. point target/error/mask invariance;
8. method-balanced L1 and missing-target semantics;
9. point freeze and median equality;
10. prediction artifact contains no target-bearing fields;
11. merger rejects missing/duplicate folds;
12. qualifier fails every single pre-frozen Gate independently.

## V13M2 — real-data smoke

Folds 0/1, seed 0, 3+3 epochs. Produce prediction-only artifacts and run the engineering qualifier. No scientific scoring. Engineering failures may be repaired without changing the scientific protocol.

## V13M3 — complete seed-0 screen

Run 20 folds in persistent sessions on available GPU0–7. Each controller only fills missing artifacts. Monitor at 900-second cadence without reading loss or partial scores. Merge only at 20/20, then close training authority, enable one complete-score read, run scorer once and qualifier once.

## V13M4 — fixed five-seed confirmation

Open only on exact V13M3 PASS. Run seeds 0–4 and 20 folds without selection, merge equal-seed mixtures and apply the pre-frozen formal Gate.

## V13M5 — freeze/handoff

Freeze code, contracts, checkpoints, prediction-only universe, complete score and qualification. Restore main M6 authority with either `POST_HOC_DEVELOPMENT_PASS` or terminal representation-route failure. External outcomes remain locked.
