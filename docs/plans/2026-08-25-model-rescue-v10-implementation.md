# Model Rescue v10 implementation plan

1. Implement train-only standardization and extraction of the 201 features from
   the actually trained V8 direct path.
2. Implement the parameter-matched capacity-symmetric and median-constrained
   asymmetric heads, exact mixture-median invariant, CRPS and expected absolute
   output.
3. Implement one fold runner that trains all four frozen heads from shared
   common initialization and emits prediction-only artifacts.
4. Implement real folds0/1 smoke qualifier, complete 20-fold merge, one complete
   scorer and mechanical Gate qualifier.
5. Test exact V8/feature41 point replay, train-only transform, target invariance,
   symmetric nested-null identity, finite gradients, parameter counts, fair
   family/input permissions, complete output and shard merge integrity.
6. Only after focused tests pass, commit implementation and change authority to
   V10M1 real smoke. Scientific score access remains closed until V10M2 is
   complete and separately transitions to V10M3.
