# ReactFlow-Delta Model Rescue v14 Amendment

**Status:** `V14M0_CONTRACT_FROZEN_IMPLEMENTATION_CLOSED`  
**Parent evidence:** `ec38e701f0528f8141e5724f334923dd49c266e4`  
**Route:** `WT_PROFILE_SELF_SUPERVISED_PRETRAINING_ONLY`

## 1. Scientific question

V14 asks one narrow question: can task-matched, outer-train-only masked WT-reactivity reconstruction improve mutation-effect prediction beyond an identical larger architecture trained from scratch?

The candidate and null use the same 5,117,874-parameter architecture, the same feature41 anchor, downstream signed-delta objective, calibration family, folds, seeds, epochs, initialization, batches and dropout stream. The sole identifying difference is that the candidate encoder first learns to reconstruct masked WT reactivity from the other WT positions. Capacity alone therefore cannot be credited as pretraining benefit.

This amendment does not alter any V1–V13 verdict. V13 remains `V13M3_TOP_JOURNAL_SCREEN_FAIL`, V13M4 remains closed, and exact-mutant re-encoding remains terminated. Post-V13 diagnostic arms A and C remain closed.

## 2. Frozen candidate and nested null

- Candidate: `v14_masked_wt_profile_pretrained_feature41_anchor`.
- Attribution null: `v14_from_scratch_feature41_anchor`.
- Both: width 256, 8 heads, 6 pre-norm relative-attention blocks, FFN width 1024, dropout 0.1, relative window 256, and a two-hidden-layer width-384 residual head added to the outer-train feature41 point.
- Exact total parameters: 5,117,874 each.
- Downstream-trainable parameters after freezing the reconstruction decoder: 5,117,105 each.
- Candidate pretraining may update only the encoder and reconstruction decoder. It cannot update the downstream residual head.
- Before supervised step one, candidate and null residual heads must be bitwise equal. The null must still equal its saved common initialization.

No width, depth, mask-rate, optimizer, loss, epoch, calibration-family or seed-subset search is allowed.

## 3. Task-matched self-supervision

For each outer fold, pretraining may read only the 152 outer-train WT constructs. The held puzzle WT profile and every mutant outcome are excluded. On each construct and epoch, 40% of WT-observed positions are selected uniformly without replacement using a deterministic seed/epoch/construct schedule. Their reactivity, precision and observed token are zeroed, a corruption indicator is set, and they are removed as attention keys. The target is the original construct-standardized WT reactivity at those masked positions.

The objective is mean absolute reconstruction error over masked positions with equal construct exposure. The screen/formal pretraining schedule is fixed at 200 epochs with AdamW, learning rate 3e-4, weight decay 0.01 and gradient clipping 5.0. Smoke uses three epochs. There is no early stopping.

## 4. Downstream task and calibration

The candidate and null are trained on the same outer-train puzzle×method cells using exact method-balanced signed-delta L1 for 40 epochs (three in smoke), Adam at 1e-3, zero weight decay and clipping 5.0. The final point is feature41 plus the neural residual. The reconstruction decoder is frozen and unused.

After point training, both point models are frozen. Each receives the exact V10 median-constrained asymmetric residual calibration for 40 epochs (three in smoke). Calibration cannot change point means.

Prediction artifacts remain prediction-only. Held mutant reactivity, held target error, held qualified target mask, scores, method ID, puzzle ID and dataset ID are forbidden model inputs. New external outcomes remain locked.

## 5. Evidence sequence

1. `V14M0`: freeze this amendment and machine authority; implementation and training closed.
2. `V14M1`: implement and prove parameter matching, data isolation, initialization, gradient and prediction invariants.
3. `V14M2`: folds 0/1, seed 0, 3+3+3 epoch real-data engineering smoke; no scientific scoring.
4. `V14M3`: seed 0, twenty-fold, 200+40+40 score-blind screen. No score is read before the complete prediction merge; score and qualification run once.
5. `V14M4`: only after exact screen PASS, run fixed seeds 0–4 and twenty folds without selection.
6. `V14M5`: freeze artifacts and return authority to the benchmark manuscript route.

## 6. Top-journal screen Gate

Every requirement below is conjunctive.

| Metric | vs feature41 | vs terminal prior | vs from-scratch null | positive puzzles |
|---|---:|---:|---:|---|
| signed-delta MAE | ≥12% | ≥2% vs V12 | ≥1.5% | 16 / 14 / 14 |
| point-absolute MAE | ≥7% | ≥2% vs V11 | ≥1% | 16 / 14 / 14 |
| CRPS | ≥5% | ≥2% vs V12 | ≥1.5% | 16 / 14 / 14 |
| distribution-absolute MAE | ≥15% | ≥2% vs V10 | ≥1% | 16 / 14 / 14 |

Each paired puzzle-level CI lower bound must exceed zero. Every headline comparison must retain positive direction under leave-one-puzzle-out analysis; no puzzle may contribute more than 20% of the total effect. Registered coverage must be 100%, failures and unexpected keys must be zero, and calibration coverage-error worsening may not exceed one percentage point.

## 7. Decision boundary

Exact `V14M3_TOP_JOURNAL_SCREEN_PASS` is required to open V14M4. Failure terminates the masked-WT-profile pretraining family: no same-family V15, no mask/epoch/width search and no Gate revision. A formal PASS remains `POST_HOC_DEVELOPMENT_PASS`; it does not establish external replication, SOTA, mechanism, practical utility or publication readiness.

GPU0–7 may be used when memory is sufficient, including safe co-location, but unrelated processes cannot be preempted, signalled or modified. Persistent sessions and 900-second low-frequency monitoring are required.
