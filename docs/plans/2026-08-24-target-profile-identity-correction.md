# Target Profile Identity Correction Execution Plan

## TIC1 — Fix and qualification

- Replace raw-prefix indexing with canonical puzzle-method-mutation indexing.
- Remove the global mutation-suffix fallback.
- Add a fixture where raw prefixes differ from method labels and two constructs share the same mutation.
- Run focused accessor, data-isolation, baseline, v5 and v6 tests.
- Compare every corrected target/error array against its exact raw puzzle/method/mutation row on real M2.
- Acceptance: 13,976 present, 0 wrong target, 0 wrong error.

## Quarantine

- Preserve old v1-v6 and v3 artifacts.
- Mark all target-dependent legacy performance evidence scientifically invalid.
- Stop the invalid v3 remaining folds; do not merge the preserved 10 folds.
- Do not resume old checkpoints or only fill missing folds.

## V7M1

- Continue official RiNALMo outcome-blind cache because it does not call the target accessor.
- Require the existing V7M1 cache qualifier exact PASS.
- Do not infer model performance from the cache.

## TIC2 / corrected V7M2

- Reuse only qualified outcome-blind V5/V6/V7 caches.
- In one corrected 20-fold LOPO run, refit direct18, feature30, feature41 and dependency47 from outer-train targets.
- Freeze alpha=1, weighted standardization, signed/absolute targets and method-balanced weights.
- Generate full registered prediction-only ledgers and merge all folds before one score access.
- Compare dependency47 against corrected feature41 using the original V7M2 Gate.

## TIC3 / corrected neural stage

- Only after exact V7M2 eligibility PASS, rebuild corrected B1 folds 0-19 from scratch.
- Train the frozen dependency operator and two equal-capacity attribution controls.
- Apply the original V7M4 and V7M5 top-journal Gates without lowering thresholds.

