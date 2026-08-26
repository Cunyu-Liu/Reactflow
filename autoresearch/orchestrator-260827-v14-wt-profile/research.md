---
slug: orchestrator-260827-v14-wt-profile
date: 2026-08-27
status: SCREEN_AUTHORIZED
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
- 2026-08-27: before any V14M3 score access, `docs/plans/2026-08-27-post-v14-model-contingency.md` froze the post-screen routing logic. Exact PASS routes only to V14M4; point/transfer failure may support one cross-construct puzzle-set meta-context family, while a strict distribution-only failure may support one frozen-point median-preserving quantile residual. Neither proposal has training authority, and all other previously falsified families remain closed.
