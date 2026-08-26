# Post-V13 route diagnostics implementation plan

1. Implement shared feature41 reconstruction, per-puzzle sufficient statistics, inverse-observation-variance weighting, and coherent signed-magnitude reconstruction.
2. Add prediction-only fold artifacts. No target/error/mask/score field may be serialized.
3. Add a merger requiring exactly folds 0–19, exact corrected feature41 replay and no duplicate key.
4. Add a complete-only scorer that joins targets after merge and applies the frozen position→mutant→method→puzzle aggregation.
5. Add a mechanical qualifier for the two frozen route-support Gates and deterministic both-pass rule.
6. Add focused tests for method balance, within-mutant reliability normalization, no held-error prediction dependency, feature41 replay, coherent reconstruction, complete-before-score and Gate boundaries.
7. After focused validation, open prediction-only execution, run the fixed 20 folds, merge, close fitting, open one complete score, score and qualify once.

