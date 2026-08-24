# ReactFlow-Delta post-V9 autoresearch loop

Metric: V9-comparable method-balanced full-construct CRPS relative gain versus
the equi-calibrated corrected feature41 baseline, while preserving the frozen
V8 signed-delta and V9 distribution-derived absolute-delta improvements.

Verify: complete 20-puzzle LOPO prediction universe, followed by one frozen
score/qualifier run. No partial model scores may be read.

Incumbent: V9, with signed-delta MAE gain 8.36%, distribution-derived
absolute-delta MAE gain 9.81%, and CRPS gain 4.23%. V9 is terminal because the
pre-frozen CRPS threshold was 5%.

Current falsifiable question: does the exact held residual around the frozen
L1 point show puzzle-replicated mean/median separation or quantile asymmetry?
If yes, test a median-preserving asymmetric residual family against a
parameter-matched symmetric null. If no, test only the representation/capacity
symmetric null and do not claim an asymmetry mechanism.

Observed result: both pre-registered criteria passed. The method-balanced
mean-minus-median gap was +0.03452 (95% CI +0.02615 to +0.04289; 20/20 puzzles
same direction). Normalized q10/q50/q90 asymmetry was +0.23049 (95% CI +0.18823
to +0.27274; 20/20 same direction). The next iteration is therefore eligible
to test median-preserving asymmetric locations, but only against the frozen
parameter-matched symmetric null.
