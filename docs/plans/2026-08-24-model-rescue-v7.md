# Model Rescue v7 Implementation Plan

## V7M0 — Contract freeze

- Add human amendment, machine contract, decision ledger, design, implementation plan and contract test.
- Validate YAML and frozen values.
- Commit locally and remotely before any RiNALMo install, weight download or inference.
- After focused PASS, update authority so V7M1 alone is runnable and score/training remain closed.

## V7M1 — Outcome-blind foundation cache

Files to add:

- `scripts/reactflow_delta/model_rescue_v7_dependency.py`
- `scripts/reactflow_delta/build_model_rescue_v7_dependency_cache.py`
- `scripts/reactflow_delta/qualify_model_rescue_v7_dependency_cache.py`
- `tests/reactflow_delta/test_model_rescue_v7_dependency.py`

Implementation order:

1. Implement pure NumPy/Torch log-odds transform and unit tests against hand-computed probabilities.
2. Implement registered WT/mutation sequence enumeration without mutant outcome fields.
3. Build the official Python 3.11 / PyTorch 2.1.0 / CUDA 11.8 / FlashAttention 2.3.2 runtime without altering the official attention code.
4. Implement official RiNALMo adapter fixed to `giga-v1` and A/C/G/U token indices obtained from the official alphabet.
5. Deduplicate WT and exact mutant sequences, run batched no-grad inference, and store dependency6 keyed by biological construct/mutation/receiver.
6. Qualify finite values, exact six-width, self-zero, method identity and full registered coverage.
7. Run a two-puzzle prediction-free engineering smoke, then the complete outcome-blind cache in a persistent session.

V7M1 acceptance: exact `V7M1_OUTCOME_BLIND_RINALMO_DEPENDENCY_CACHE_PASS`. No scientific score is read.

## V7M2 — Complete LOPO eligibility probe

Files to add:

- `scripts/reactflow_delta/model_rescue_v7_probe.py`
- `scripts/reactflow_delta/run_model_rescue_v7_probe.py`
- `scripts/reactflow_delta/merge_model_rescue_v7_probe.py`
- `scripts/reactflow_delta/score_model_rescue_v7_probe.py`
- `scripts/reactflow_delta/qualify_model_rescue_v7_probe.py`
- `tests/reactflow_delta/test_model_rescue_v7_probe.py`

Required behavior:

- reuse V5 and V6 caches and V6 fold prediction/model artifacts;
- baseline41 must replay V6 candidate at `atol=1e-12`;
- candidate adds dependency6 only;
- exact puzzle/method/mutant/position weights;
- prediction artifacts contain keys and predictions but no target or score;
- shard merge rejects duplicate or missing folds;
- only 20/20 folds authorize one complete target join and frozen qualifier.

Frozen implementation interface, confirmed without opening any score artifact:

- import `CANDIDATE_PROBE_FEATURE_NAMES`, `ConstrainedFeatureCache`,
  `accumulate_train_stats` and the full-construct prediction ordering from the
  existing v6 probe instead of independently rebuilding the 41-feature
  baseline;
- treat the v6 `candidate_signed_delta/candidate_absolute_delta` arrays as the
  replay reference and preserve their `biological_scoring_key` order exactly;
- index dependency6 by the same
  `(puzzle, method, design_pos, ref, alt)` biological key used by the qualified
  V5/V6 caches; use the v7 `full_pos` only to verify the corrected construct
  coordinate and source-zero invariant;
- retain the frozen `RIDGE_ALPHA=1.0`, train-only weighted standardization,
  signed/absolute two-output target and
  puzzle→method→mutant→qualified-position training weights;
- construct the 47-feature candidate by appending dependency6 to the replayed
  baseline41. No interaction expansion, feature selection, alpha search or
  replacement of existing columns is allowed;
- create held predictions for every registered construct position before any
  target join. V7 cache values may enter prediction; held target, target error,
  qualified target mask and score may not.

V7M2 acceptance: exact `V7M2_RINALMO_DEPENDENCY_SIGNAL_ELIGIBLE`. Anything else closes v7.

## V7M3–V7M4 — Neural operator and seed-0 screen

Implement only after V7M2 exact PASS:

- frozen corrected B1 adapter from exact R3C3-qualified artifacts;
- fixed dependency projection and residual MLP from the contract;
- equal-capacity zero and cyclic-shift controls;
- method-balanced L1 mean training and complete mean freeze;
- zero-mean two-Gaussian CRPS calibration;
- 20 prediction-only folds, complete merge, one score, frozen qualifier.

V7M4 acceptance requires the full five-percent dual-metric and attribution Gate. No hyperparameter response to V7M2 or partial V7M4 scores is allowed.

## V7M5 — Five-seed confirmation

- fixed seeds 0–4 and folds 0–19;
- no model, epoch, seed, calibration or checkpoint selection;
- unique five-seed mixture only;
- complete-before-score and full V7M4 Gate replay.

## V7M6 — Freeze and M6 handoff

- preserve all code, cache, predictions, scores and negative results;
- close foundation inference, training and score authority;
- update claim map with exact evidence status;
- return `M6/BENCHMARK_ROUTE_LOCKED` on failure or `M6/BENCHMARK_WITH_HIGH_EFFECT_POST_HOC_DEVELOPMENT_MODEL` on internal pass;
- do not open external outcome without a separate sealed amendment.

## Low-frequency operations

All long cache or fold universes run in persistent tmux sessions with complete logs. The existing hourly automation may monitor v3 and v7 as isolated authorities. Before a complete artifact universe exists, monitoring is limited to session existence, log timestamp, artifact filename count and non-metric errors.
