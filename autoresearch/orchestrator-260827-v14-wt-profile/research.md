---
slug: orchestrator-260827-v14-wt-profile
date: 2026-08-27
status: B5RP3_COMPLETE_SCORE_QUALIFIER_ONCE_AUTHORIZED
parent: orchestrator-260827-post-v13
---

# V14 WT-profile self-supervised pretraining

## Research objective

Determine whether outer-train-only masked WT-reactivity reconstruction learns task-relevant intra-construct context that improves full-construct unseen-puzzle signed-delta and probabilistic prediction beyond an identical larger architecture trained from scratch.

## Prior evidence

- V13 exact-mutant re-encoding produced only negligible increments over its matched replay null and is terminally closed.
- The complete post-V13 diagnostic closed noise-aware feature41 reweighting at the frozen 0.5% practical boundary and decisively falsified coherent sign-magnitude factorization.
- Its pre-registered route decision retained only WT-profile self-supervised pretraining as a meaningfully different capability route.

## Hypothesis

The WT profile contains transferable local and long-range context that is poorly learned from the small supervised mutant-effect pool. Masked reconstruction over outer-train WT constructs can shape the same downstream encoder before supervised fitting, improving point and residual-distribution estimates.

## Attribution design

Candidate and null are identical 5,117,874-parameter models initialized from the same state. Only the candidate encoder/decoder receives masked-WT pretraining. Both then receive the same feature41-anchored supervised residual training and V10 calibration. The pretraining decoder is unused downstream. This makes candidate-minus-null the identified pretraining increment, while null-minus-feature41 captures capacity and supervised architecture effects.

## Frozen experiment

- Data: OpenKnot M2 v4.5.2, split_v4 twenty-fold LOPO.
- Pretraining universe: outer-train WT constructs only; held puzzle WT and every mutant outcome excluded.
- Corruption: deterministic uniform 40% of observed WT positions; corrupted values/precision/observed tokens removed, explicit corruption token added.
- Objective: construct-balanced L1 reconstruction of standardized WT reactivity.
- Schedule: seed-0 screen, 20 folds, 200 pretraining + 40 point + 40 calibration epochs.
- Formal: seeds 0–4 only after exact screen PASS.
- No search over mask, model size, objective, epochs, calibration or seeds.

## Falsification and stopping

The hypothesis is falsified if the candidate misses any frozen V14M3 top-journal Gate, particularly the registered increments over the identical from-scratch null. Failure terminates the WT-profile pretraining family without Gate revision or same-family iteration. Success remains post-hoc development evidence and requires a separate sealed external amendment before broad claims.

## Progress log

- 2026-08-27: V14 architecture, objective, attribution null and gates selected before implementation and outcome access.
- 2026-08-27: contract validation passed; V14M1 implementation opened with real-data training and score access closed.
- 2026-08-27: 12 focused tests passed locally and remotely; V14M2 folds 0/1 engineering smoke opened with scientific score access closed.
- 2026-08-27: first smoke failed before artifact creation because `P20_Eterna` has zero WT-observed positions. Outcome-blind audit found histogram `{0: 1, 100: 159}` and no one-observed construct. The zero-target construct is now registered but excluded from the reconstruction objective; no target value is fabricated and all scientific settings remain frozen.
- 2026-08-27: pre-score code trace confirmed the V14 scorer uses position → mutant → method-balanced puzzle aggregation through `_puzzle_macro`; an imbalanced-method regression fixture was added before any V14 held score access.
- 2026-08-27: the reused held-prediction implementation was source-audited and guarded against indirect calls to mutant targets, target matrices or qualified masks; held outcomes remain scorer-only.
- 2026-08-27: V14M2 folds 0/1 completed the real-data 3+3+3 prediction-only smoke. The frozen merge and qualifier returned `V14M2_ENGINEERING_SMOKE_PASS`; every registered engineering invariant passed, no scientific score was computed and external outcome access remained closed. V14M3 seed-0 twenty-fold 200+40+40 score-blind training is now the sole runnable phase.
- 2026-08-28: V14M3 seed-0 twenty-fold prediction-only universe and its canonical complete unscored merge are complete. All 20 registered folds are present, with no missing, duplicate or unexpected artifacts, and the controller, runners and persistent session have exited. Training is now closed and the single complete-score-then-qualify authority is open under `V14_COMPLETE_MERGE_SCORE_ONCE_ONLY`; no V14 scientific score, qualification or new external outcome has yet been accessed.
- 2026-08-28: The single canonical score completed with 20/20 folds, full registered coverage, zero failures and zero unexpected keys. Qualification returned exact `V14M3_TOP_JOURNAL_SCREEN_FAIL` (9/24 frozen Gates passed; V14M4 not authorized). The candidate improved all four headline means over the identical from-scratch null and over feature41, but missed the frozen feature41 margins, lacked robust superiority over the terminal V10-V12 comparators and failed the 95% coverage and influence guardrails. This is post-hoc development evidence only, not formal, SOTA, publication-ready or externally replicated evidence.
- 2026-08-28: Training, screen reruns, partial-score access and new external outcomes remain closed. Exactly one first-matching router read is authorized over the three canonical bound paths under `POST_V14_FIRST_MATCHING_ROUTER_ONCE_ONLY`; its output does not yet exist and this authority commit does not itself authorize any downstream branch.
- 2026-08-28: The canonical first-matching router completed once with valid terminal inputs and selected branch `5`, classification `INDEPENDENT_CONSTRUCT_TRANSFER_LIMITED`. V14M4 and branch 6 are closed; the selected path requires the frozen branch-5 route probe before any P1 activation.
- 2026-08-28: The minimal B5RP0 source-projection dependency closure was integrated at `5660621` and passed 25 focused tests. B5RP0 now authorizes exactly one target-, score-, loss- and history-free projection of the 20 same-fold V13 candidate point checkpoints and 20 V14 candidate checkpoints into the canonical safe-source manifest. Training, held score, partial score and external outcomes remain closed; B5RP1 is not authorized.
- 2026-08-28: B5RP0 completed once with `POST_V14_BRANCH5_SAFE_SOURCE_MANIFEST_PASS`: 20 V13 and 20 V14 same-fold seed-0 checkpoints strictly loaded, the target-free canonical manifest passed its exact schema/path/fold audit, and no scientific performance claim was produced. This was an explicit CPU checkpoint-structure projection, not training or GPU validation.
- 2026-08-28: The remaining B5RP1 GPU route-probe runtime was integrated at `d9cbf77` and passed 25 focused tests. B5RP1 now authorizes only the fixed seed-0 folds 0-19 prediction universe under `POST_V14_BRANCH5_LINEAR_CROSS_CONSTRUCT_ROUTE_PREDICTION_ONLY`; controller and runner both require real CUDA, no CPU fallback or minimum-free-memory gate is permitted, and held/partial/external scores remain closed.
- 2026-08-28: B5RP1 completed its exact seed-0 folds 0-19 universe with 20 fold results, 20 prediction artifacts and 20 ridge artifacts, no missing, duplicate or unexpected artifacts, controller exit code zero and no remaining controller, runner or tmux session. Eight runner PIDs were observed on real NVIDIA devices at launch, no CPU fallback or narrow runtime error was recorded, and the canonical complete unscored merge passed an exact target-free schema, path, fold and invariant audit. No partial scientific content was inspected.
- 2026-08-28: Training and screen authority are now closed. Exactly one complete held-score read is authorized from the canonical B5RP1 merge under `POST_V14_BRANCH5_COMPLETE_MERGE_SCORE_ONCE_ONLY`; the complete score does not yet exist, partial score and external outcomes remain closed, and B5RP3 qualification is not yet authorized.
- 2026-08-28: The single B5RP2 scorer completed with exit code zero and canonical `BRANCH5_ROUTE_PROBE_COMPLETE_SCORE_PASS`: all 20 score rows are present, `complete_valid_score=true`, and there are no integrity errors. No metric direction was inspected before qualification; partial score and external outcomes remained closed.
- 2026-08-28: B5RP2 held-score authority is consumed and closed. B5RP3 now authorizes exactly one qualifier read of the canonical complete score under exact phase, score path and qualification output path bindings; all training, screen, held-score, partial-score and external-outcome authority remains closed.
