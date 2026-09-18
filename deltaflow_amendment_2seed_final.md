# DeltaFlow Amendment: 2-seed final protocol (2026-09-19)

**User directive (verbatim)**: "就根据这两个已有的 seed 结果推进项目，以后也不可能重跑其他 seed 了"

This amendment supersedes the seed-related clauses of the Task 7 pre-registration
freeze (`deltaflow_preregistration_freeze.md`, F1/F2 seed protocol). All other
frozen quantities (metrics, estimator formulas, F3 constants, reference layers,
convergence epochs) remain verbatim.

## Changes

1. **F1 seed protocol**: final seed set = {0, 1}. No further seeds will ever be
   run. Assembly = equal 0.5/0.5 mixture over seeds 0/1 (was 0.2 x 5).
2. **F2 direction clause**: ">=4/5 seeds individually positive" -> "2/2 seeds
   individually positive". At P(single-seed positive)=0.8 the pass rate is 0.64
   (vs 0.737 for 4/5); at 0.9 it is 0.81 (vs 0.919) — the amended clause is
   stricter in per-set terms.
3. **F2 MDE independent tier**: sigma_assembly = sigma_d/sqrt(5) -> sigma_d/sqrt(2).
   Frozen MDE formula `(z_0.975+z_0.80) * sigma_assembly / sqrt(20)` (K=2.801585)
   with the frozen sigma_d (0.007395/0.009151/0.003287/0.003900) and frozen
   baselines (0.1773/0.1485/0.1272/0.1354) yields thresholds multiplied by
   sqrt(5/2) = 1.58114:

   | metric | MDE% independent (5-seed, frozen) | MDE% independent (2-seed, amended) | conservative (unchanged) |
   |---|---|---|---|
   | signed-delta MAE | 1.17 | **1.85** | 2.61 |
   | point absolute-delta MAE | 1.73 | **2.74** | 3.86 |
   | task CRPS | 0.72 | **1.14** | 1.62 |
   | distribution-absolute MAE | 0.81 | **1.28** | 1.80 |

   Conservative tier (rho=1) is seed-count independent and unchanged.
4. **F2 Step 2 measured tier**: still executed mechanically; with 2 seeds the
   realized rho estimate is noisy — registered as an audit caveat, not a gate.
5. **F3 joint metrics**: unchanged; per-seed scores summarized across the 2 seeds.
6. **Reference layers** (feature41 / V8 / historical V10): unchanged.
7. **Secondary attribution** (profile-permutation, teacher-permutation): unchanged
   in design; executed with 2 seeds.

## Power / risk registration (honest)

- MDE widened by x1.581 — the amended protocol is strictly lower-power than the
  pre-registered 5-seed plan. A PASS under this amendment is a stronger per-seed
  signal but a weaker evidence base than the 5-seed design would have been.
- Every report of this verdict must be labelled "amended 2-seed protocol
  (2026-09-19)".
- No further seeds will be run (user decision) — this is the final DeltaFlow
  protocol.

## Implementation

`score_deltaflow.py` (ACTIVE_SEEDS runtime check = (0,1), 0.5 assembly) and
`qualify_deltaflow.py` (seed universe (0,1), amended MDE table, 2/2 direction
clause) updated in the same commit as this document.
