# ReactFlow-Delta Model Rescue v3 Implementation Plan

**Goal:** Build one fold-legal disagreement-gated B1/MeanAligned residual model and
verify it against the unchanged original R2M4 dual-primary Gate.

**Architecture:** Train/freeze B1 and MeanAligned experts; use four-fold puzzle-grouped
inner OOF predictions from outer-train data to fit a fixed 95th-percentile two-bin convex
disagreement gate; freeze the blended mean; fit a zero-mean two-Gaussian CRPS residual.

## Batch 1 — contract and authority freeze (R3M0)

**Files:** v3 human/machine amendment, decision ledger, implementation plan, active
authority and contract test.

**Failure detected:** v3 accidentally changes v1/v2 terminal conclusions, opens training
before implementation tests, allows method/external/held-target inputs, or changes an
original R2M4 threshold.

**Action on failure:** fail closed at R3M1 and do not implement or train.

**Verification:** YAML parse; exact active paths; parent terminal statuses; endpoint-v7
input subset; v2-v3 formal threshold equality; training/external false.

## Batch 2 — model and crossfit gate implementation (R3M1)

**Files:** `model_rescue_v3.py`, `run_model_rescue_v3.py`, focused tests.

**Implementation:**

1. Reuse exact B1 and MeanAligned architectures/trainers without changing capacity.
2. Add deterministic four-fold puzzle-grouped inner crossfit ledger.
3. Generate prediction-only inner OOF rows; every outer-train puzzle appears once as
   inner held and never in the corresponding expert train set.
4. Compute exact hierarchy weights, fixed weighted q95 disagreement threshold and two
   weighted-median convex alphas.
5. Apply the frozen gate to final outer expert predictions.
6. Freeze blended mean and train the v2 two-scale zero-mean residual CRPS head.
7. Emit prediction-only v3 schema and scorer-separated result.

**Tests that change authorization:** inner-held exclusion, complete/disjoint inner key
universe, exact alpha fixture, q95 fixture, method/target invariance, alpha bounds,
calibration gradient isolation, point-mean invariance, full registered output, shard
duplicate/missing rejection.

**Failure handling:** any leakage, missing inner puzzle or mean mutation keeps training
closed. Performance direction is not tested on synthetic data.

## Batch 3 — mechanical qualification (R3M1)

**Files:** `qualify_model_rescue_v3.py` and focused fixtures.

**Implementation:** reuse the exact R2M3 screen and original R2M4 formal criteria;
add mandatory inner-crossfit coverage/invariance checks. Qualifier mechanically emits
PASS/FAIL from complete artifacts; no manual verdict field.

**Verification:** handcrafted single-condition failures, incomplete folds/seeds rejected,
and exact parity with v2 formal thresholds.

## Batch 4 — real-data smoke (R3M2)

**Input:** OpenKnot M2 v4.5.2 only. P01/P02, seed0, at most 3 epochs per expert/residual;
inner split reduced only in epoch count, never in puzzle exclusion semantics.

**Outputs:** checkpoints, inner ledger, prediction-only artifacts, engineering qualifier
under `/mnt/cunyuliu/reactflow_delta_model_rescue_v3/r3m2_real_smoke/`.

**Acceptance:** finite optimization, exact inner exclusion/coverage, prediction coverage
100%, failure/unexpected 0, target-invariance and residual point-mean identity. Smoke
scores are ignored and labeled `ENGINEERING_SMOKE_ONLY`.

## Batch 5 — seed0 20-fold screen (R3M3)

**Execution:** one frozen candidate, 20 outer folds, fixed four-fold inner crossfit,
40-epoch B1/MA experts and 40-epoch residual, seed0. Use GPU0--5 with disjoint fold
shards and persistent logs. Do not inspect partial scores.

**Acceptance:** complete fold/key universe and mechanical simultaneous signed-delta/
CRPS Gate PASS. If failed, freeze the candidate result; do not change threshold/bin/
feature inside this amendment.

## Batch 6 — original five-seed formal Gate (R3M4)

**Prerequisite:** exact `R3M3_SCREEN_PASS` only.

**Execution:** seeds0--4 × 20 outer folds; fixed experts, inner gate and residual;
no seed deletion or selection. Build one 5-seed B1 mixture and one 5-seed candidate
mixture, then run the unchanged original R2M4 qualifier.

**Acceptance:** qualifier status `R2M4_POST_HOC_DEVELOPMENT_PASS` and every CI,
practical-effect, positive-puzzle, LOO, influence, coverage and failure condition true.

## Batch 7 — artifact freeze and M6 handoff (R3M5)

Freeze code/config/checkpoints/predictions/inner ledgers/qualification and update claim
map. PASS route is `BENCHMARK_WITH_DISAGREEMENT_GATED_DEVELOPMENT_MODEL`; failure route
preserves benchmark mainline and the v3 negative result. Close training and keep external
outcomes locked in both cases.
