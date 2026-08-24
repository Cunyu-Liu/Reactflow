# ReactFlow-Delta Model Rescue v6 implementation plan

## V6M0 — contract and design freeze

Validate the human amendment, machine contract, decision ledger, design and active authority. Keep all execution closed until a focused commit exists.

## V6M1 — constrained ensemble cache

Implement an outcome-blind builder and qualifier. Tests must cover one-based SHAPE vectors, negative-to-zero transformation, null-to-`-999`, identical WT/mutant constraint fields, P20 all-missing exact fallback, corrected mutation coordinates, finite 12-feature tensors and zero mutant-outcome use. Run a small real smoke, then all 13,976 mutants in persistent CPU sessions.

## V6M2 — incremental fixed probe

Reuse the qualified v5 unconstrained cache. Fit direct+unconstrained baseline versus baseline+constrained features with the same weighted sufficient-statistics ridge. Generate all 20 prediction-only folds before merge, target join and the pre-frozen qualifier.

### Frozen implementation surface

No V6M2 executable may run until V6M1 has exact qualification status
`V6M1_OUTCOME_BLIND_CONSTRAINED_CACHE_PASS` and the active authority has a
focused V6M2 commit. The implementation consists only of:

- `scripts/reactflow_delta/model_rescue_v6_probe.py`;
- `scripts/reactflow_delta/run_model_rescue_v6_probe.py`;
- `scripts/reactflow_delta/merge_model_rescue_v6_probe.py`;
- `scripts/reactflow_delta/score_model_rescue_v6_probe.py`;
- `scripts/reactflow_delta/qualify_model_rescue_v6_probe.py`;
- focused unit and pipeline tests under `tests/reactflow_delta/`.

The v5 direct feature universe remains exactly 18 columns. The qualified v5
unconstrained ensemble universe remains exactly 12 columns. The V6 baseline is
their concatenation, with 30 columns. The V6 candidate adds the qualified v6
constrained ensemble universe, with 42 columns. Both models use the same
train-only weighted standardization and ridge `alpha=1`. The centered ridge
retains its train-only target mean; no intercept convention,
alpha, feature, target, fold or post-processing choice is searched. The two
fitted targets remain signed delta and absolute delta. Training weights
apply equal mass in the following hierarchy: qualified positions within mutant,
mutants within puzzle×method cell, and cells in the outer-training universe.
Missing outcomes are omitted, never set to zero. The held puzzle is absent from
all sufficient statistics, standardization and coefficients.

### Baseline replay invariant

The V6 baseline is mathematically the V5 candidate, not a newly defined
comparator. For every outer fold and registered biological key:

- V6 baseline signed-delta prediction must match the frozen V5 candidate
  signed-delta prediction within `atol=1e-12, rtol=0`;
- V6 baseline absolute-delta prediction must match the frozen V5 candidate
  absolute-delta prediction within `atol=1e-12, rtol=0`;
- the held puzzle, split, key order and registered row count must match;
- any mismatch closes V6M2 before scoring because the incremental estimand is no
  longer identified.

### Prediction-only artifacts

Each fold writes exactly one NPZ prediction ledger, one JSON model artifact and
one JSON fold result. The prediction schema contains:

- `schema_version`;
- `keys` and identical `biological_scoring_key`;
- `outer_fold`;
- `baseline_signed_delta` and `baseline_absolute_delta`;
- `candidate_signed_delta` and `candidate_absolute_delta`;
- `registered_status`.

It must not contain target, target error, qualified target mask, score, CRPS,
MAE or any per-puzzle effect. Every registered held mutant emits all 177
positions. Fold artifacts are complete only if keys are unique, finite, have no
unexpected rows, and cover the frozen held-puzzle universe.

### Merge, target join and qualification order

1. Run outer folds `0..19` in non-overlapping persistent shards.
2. Do not inspect partial predictions for a performance direction.
3. The merger rejects duplicate or missing folds, target-side fields, missing
   referenced artifacts and schema/key inconsistencies.
4. Write one complete unscored merged artifact.
5. Make a focused authority update opening one complete target join while
   keeping partial-fold score access false.
6. The scorer joins targets by biological key and reduces loss as position →
   mutant → method → puzzle.
7. The qualifier mechanically applies the frozen 1% incremental signed-delta
   gate, paired puzzle CI, positive-puzzle count, absolute-delta guardrail and
   prediction integrity checks.
8. Only exact eligibility PASS may authorize V6M3. A failure closes v6 without
   changing features, alpha or thresholds.

### Required tests before V6M2 authority

- constrained and unconstrained caches expose the same biological mutant
  universe and receiver length;
- cache lookup is independent of raw row ordering and uses puzzle, method,
  design position, ref and alt;
- V6 baseline replays frozen V5 candidate predictions;
- changing held target, target error or mask leaves every prediction unchanged;
- duplicating qualified positions within one mutant cannot change that mutant's
  total training mass;
- duplicating mutants within one method cannot change the method's total mass;
- all registered held positions are emitted, including positions without a
  qualified target;
- merge rejects 19/20 folds and duplicate fold IDs;
- score hierarchy distinguishes identical ref→alt alleles at different mutation
  positions;
- qualifier cannot produce PASS from an incomplete score artifact.

## V6M3 — neural implementation and smoke

Only after exact eligibility PASS, implement the primary and two mandatory identical-capacity controls. Freeze corrected B1, enforce zero-initialized residual identity and zero-mean calibration invariance, then run folds 0/1 with at most 3+3 epochs prediction-only.

## V6M4 — seed-0 complete screen

Run corrected B1, primary and both controls over all 20 LOPO folds. No partial scoring. The primary must pass the 5% dual-metric Gate and attribution CIs.

## V6M5 — fixed five-seed confirmation

Run seeds 0–4 without model, feature, epoch, seed or calibration selection. Assemble the unique mixtures and apply the same Gate.

## V6M6 — terminal handoff

Freeze either the complete negative result or `HIGH_EFFECT_POST_HOC_DEVELOPMENT_PASS`, close training and return to M6. External confirmation requires another sealed amendment.
