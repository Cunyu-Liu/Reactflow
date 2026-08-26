# ReactFlow-Delta Model Rescue v14 Implementation Plan

## Outcome

Implement and prospectively test one route only: masked WT-profile self-supervision of the exact downstream context encoder, compared against an identical-architecture from-scratch null. Preserve every V1–V13 verdict and all post-V13 diagnostic closures.

## V14M0 — contract freeze

- Add the human amendment, machine contract, decision ledger, this plan and the autoresearch record.
- Point `active_contract.yaml` at the V14 branch/worktree with implementation and training closed.
- Mechanically validate parent status, candidate/null identity, exact architecture and parameter counts, data-access boundary and Gate values.
- Commit only these authority artifacts.

Acceptance: YAML parses; three focused contract tests and validator pass; held/partial/external outcome access are false.

## V14M1 — implementation and invariants

Files:

- `scripts/reactflow_delta/model_rescue_v14.py`
- `scripts/reactflow_delta/run_model_rescue_v14.py`
- `scripts/reactflow_delta/merge_model_rescue_v14.py`
- `scripts/reactflow_delta/score_model_rescue_v14.py`
- `scripts/reactflow_delta/qualify_model_rescue_v14.py`
- smoke/screen controller scripts
- `tests/reactflow_delta/test_model_rescue_v14.py`

Implementation tasks:

1. Build the frozen 11-channel, width-256, six-block encoder, decoder and feature41 residual head.
2. Prove exact candidate/null parameter equality and the registered parameter counts.
3. Build per-outer-fold WT-only pretraining records from all 152 registered train construct IDs; fail if the held construct enters the universe, and exclude only constructs with no WT-observed reconstruction target.
4. Implement deterministic uniform 40% masking and construct-balanced masked L1.
5. Start candidate/null from one common state; pretrain only the candidate encoder/decoder; prove the null and both residual heads remain unchanged before supervised training.
6. Train both point models under identical cell order and paired dropout RNG using method-balanced signed-delta L1.
7. Freeze point models and fit the frozen V10 asymmetric residual family without point gradients.
8. Emit prediction-only registered artifacts with candidate, null and required comparator distributions.
9. Implement complete-universe merge, score-once scorer and mechanical qualifier.

Acceptance: focused tests prove masking/data isolation, equal initialization and counts, pretraining gradient boundaries, paired downstream schedule, target invariance, point freeze, median constraint, full output, merge rejection of duplicate/missing folds and exact Gate replay.

## V14M2 — real-data smoke

- Folds 0 and 1, seed 0, 3 pretraining + 3 point + 3 calibration epochs.
- Use a persistent controller and currently safe GPU0–7.
- Produce prediction-only artifacts; do not compute scientific scores.
- Qualify only engineering invariants.

Acceptance: both folds complete, finite losses/gradients, exact parameter and initialization invariants, no held WT in pretraining, coverage 100%, failures/unexpected keys zero, target invariance and complete prediction-only merge PASS.

Failure handling: fix only protocol-preserving engineering defects; do not use smoke directions to modify the model.

## V14M3 — seed-0 twenty-fold screen

- Fixed seed 0 and folds 0–19.
- Fixed 200 pretraining + 40 point + 40 calibration epochs.
- Candidate and null share the frozen common initialization and downstream schedule per fold.
- Run persistent non-overlapping fold workers, with 900-second health checks.
- Until all twenty prediction artifacts exist, inspect only process health, log timestamps/sizes, artifact filenames and non-metric errors.
- Merge prediction-only artifacts once. Close training in a focused authority commit. Score once and qualify once.

Acceptance: exact `V14M3_TOP_JOURNAL_SCREEN_PASS` under all frozen gates. Anything else is terminal screen failure.

## V14M4 — fixed five-seed confirmation

Prerequisite: exact V14M3 PASS.

- Seeds 0–4, folds 0–19, unchanged 200+40+40 schedule and model.
- No seed, epoch, checkpoint or model selection.
- Equal-seed final mixture; repeat all feature41 and from-scratch-null gates.
- At least four seeds must independently preserve positive signed-delta and CRPS directions.

Acceptance: exact formal qualifier PASS. A formal PASS is still development-only pending a future sealed external amendment.

## V14M5 — freeze and handoff

- Freeze code, configs, prediction artifacts, score, qualification and generated tables.
- Clean-checkout replay validator, focused tests, merge, scorer replay and qualifier replay.
- Close training and external outcome access.
- PASS route: benchmark with a strong post-hoc development pretraining baseline.
- FAIL route: terminate the WT-profile pretraining family and return to benchmark/measurement work; no same-family V15.

## Resource and monitoring boundary

GPU0–7 may be used when memory is sufficient, including safe co-location. Never preempt, terminate, signal or alter unrelated work. OOM recovery may move a fold or reduce microbatch only. Persistent sessions are mandatory. Scientific configuration is immutable after V14M0.
