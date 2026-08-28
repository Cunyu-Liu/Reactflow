# Independent RNet2 RND6 Formal Design

## Decision and boundary

RND6 is an inactive, pre-result implementation preparation. It must not become runnable merely because code or contract fields exist. Its only scientific predecessor is the exact canonical RND5 status `RNET_DISTILL_TOP_JOURNAL_DEVELOPMENT_SCREEN_PASS`; RND5 FAIL or INDETERMINATE closes this hypothesis without running RND6. All evidence remains `EXPOSURE_DISCLOSED_DEVELOPMENT_ONLY`: a formal PASS establishes fixed-seed stability on the already-consumed M2 development universe, not clean OOD performance, SOTA, external replication, publication readiness, or freedom from RibonanzaNet2 training-family exposure.

The frozen formal universe is seeds 0–4 crossed with outer folds 0–19, using the same split, paired candidate/null pretraining checkpoints, 40 point epochs and 40 calibration epochs as RND3. Every training or GPU validation task is CUDA-only and uses any supplied physical GPU through `CUDA_VISIBLE_DEVICES` to logical `cuda:0`; there is no minimum-free-VRAM Gate and no CPU fallback. Raw folds, checkpoints, predictions, assembly, score, and qualification stay under `/mnt/cunyuliu/reactflow_delta_independent_rnet_distill`. Code and commits remain in the `/home` project and are pushed immediately after each focused batch.

## Alternatives considered

1. **Recommended: equal-mixture plus matched-null seed stability.** Score the fixed equal-weight five-seed mixture against the unchanged RND5 Gates, then require each of the four headline metrics to have positive candidate-minus-matched-null direction in at least four of five individual seeds. This tests the central aligned-versus-shift17 attribution while retaining ensemble benefit and forbidding favorable-seed selection.
2. **Require every individual seed to pass the full RND5 Gate set.** This is substantially more stringent than the pre-frozen screen without a scientific reason, makes ensemble benefit irrelevant, and would turn seed noise into an arbitrary stop rule.
3. **Evaluate only the five-seed mixture.** This is simpler but cannot distinguish a stable representation effect from one favorable seed dominating the average, so it is too weak for a formal confirmation stage.

The recommended approach is frozen before RND5 is read. No threshold may be lowered, no extra seed appended, and no best seed, model, or threshold may be selected after score access.

## Authority architecture

`RND6` is only the umbrella name. The machine phase universe explicitly uses four mutually exclusive phases, prepared only on this inactive branch and never hot-patched into the running RND1 worktree:

- `RND6P` / `RNET_DISTILL_FIXED_SEEDS_0_TO_4_FORMAL_GPU_PREDICTION_ONLY`: GPU training/prediction plus target-free merge and assembly; score and qualification closed.
- `RND6S` / `RNET_DISTILL_FIXED_SEEDS_0_TO_4_FORMAL_SCORE_ONCE_ONLY`: all training closed; one complete formal score read; qualification closed.
- `RND6Q` / `RNET_DISTILL_FIXED_SEEDS_0_TO_4_FORMAL_QUALIFIER_ONCE_ONLY`: training and scorer closed; one qualifier run.
- `RND6T` / `RNET_DISTILL_FORMAL_TERMINAL_CLOSED`: no runnable phase and all mutation, training, score, qualification, external-outcome, and overwrite rights closed after canonical PASS, FAIL, or INDETERMINATE.

The prep branch may add an exact `inactive_formal_chain` map, future phase names, and `NOT_AUTHORIZED` gates while preserving the current RND1 phase, token, runnable set, permissions, and next action. Nothing in that map is activation. Only after RND1 has terminated may the reviewed prep commit be bound into active code, and only the exact canonical RND5 PASS may later authorize `RND6P`.

Each transition requires a focused authority commit and ledger event containing the canonical predecessor path, status, exact fold/seed count, exit code, score-access flags, CUDA evidence where applicable, and no-selection/no-external flags. Existing output forbids rerunning that stage. Prediction, score, and qualification rights never overlap.

## Prediction, merge, and assembly data flow

The RND6P controller creates the exact 100-task universe in seed-major/fold-major order and dynamically assigns one task at a time to each supplied physical GPU. It skips only a fold with its canonical result marker; any worker failure stops new dispatch, lets active workers settle, returns nonzero, and does not merge. With missing tasks and no GPU argument it returns 2. With no missing tasks it performs only target-free merge/assembly recovery and receives no GPU argument.

Each fold publishes the same seven-file atomic/result-last set as RND3. Candidate and shift17 null are retrained with the same seed-specific downstream RNG. Feature41 always reloads the single authoritative V10 seed-0 comparator for every formal seed; V8 point and historical V10 distribution fields are likewise fixed comparator data. The merger accepts exactly 100 unique `(fold, seed)` pairs, one exact 40-character Git commit, exact schedules and experiment ID, target-free schemas, actual `cuda:0` evidence, and no unexpected canonical basenames.

The assembler validates all 100 predictions before publishing anything canonical. Per fold it requires identical key order, biological keys, registration state, fixed Feature41/V8/historical-V10 fields, and finite values across seeds. Candidate/null points are arithmetic means. Candidate/null two-component mixtures are concatenated across five seeds with each component weight divided by five, producing ten-component equal-seed mixtures. Fixed Feature41 and historical-V10 comparators are copied exactly rather than redundantly expanded. All 20 assembled NPZ files plus their manifest publish by one atomic directory rename; the manifest records equal weight 0.2, no score access, and no seed selection.

## Score and qualification

Before reading M2 targets or the historical V14 score, the scorer validates the exact 100-fold-seed merge, exact 20-fold assembly, all target-free prediction schemas, canonical paths, and no existing score or qualification. It then scores the equal mixture and each individual seed using the same 20-puzzle aggregation, matched shift17 null, Feature41 comparator, V14 historical parent for signed/point/CRPS, and V10 historical distribution comparator used by RND4. One canonical score contains 20 mixture rows and five complete 20-row seed sets; it records no partial inspection, external access, model selection, or best-seed selection.

The qualifier first requires the exact canonical RND5 PASS and a complete formal score. The mixture must pass the unchanged full RND5 Gate dictionary: completeness, matched-null/Feature41/historical-parent relative margins, paired confidence intervals, positive-puzzle counts, 95% coverage, and maximum single-puzzle influence. It then computes individual-seed mean gain relative to the matched shift17 null for signed delta, point absolute, task CRPS, and distribution absolute; each metric must be positive for at least four of five seeds. Engineering/schema errors produce canonical INDETERMINATE with exit 2; a valid scientific FAIL writes canonical FAIL and exits 1; exact PASS exits 0. None is described as external or publication-ready evidence.

## Verification scope

Focused tests cover canonical CLI/path binding, RND6 seed/fold/schedule enforcement, fixed-comparator replay, atomic fold publication, exact 100-pair merge, mixed-commit rejection, equal-weight assembly math, preserved RNet-only fields, target-free validation before scoring, score/qualification no-overwrite, unchanged RND5 Gates, 4-of-5 seed stability, and exact authority transitions. Shell syntax and Python compilation cover only changed entrypoints. One combined focused suite runs at the milestone; no training, live scoring, M2 outcome read, `/mnt` artifact creation, or GPU validation occurs during inactive preparation.
