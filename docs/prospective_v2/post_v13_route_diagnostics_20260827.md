# ReactFlow-Delta post-V13 route diagnostics

**Status:** prospective implementation and scoring rules frozen before either diagnostic is run.  
**Evidence ceiling:** post-hoc development route selection; this document cannot change V13 or create a model PASS.

## Why this diagnostic exists

V13 completed the full seed-0 20-fold screen and failed the frozen Gate. The exact-mutant candidate was nearly identical to its parameter-matched WT-replay null, so the exact-mutant representation family is closed. Structure deltas, contact propagation, sequence foundations, rank, residual calibration and shrinkage-gate capacity have also been tested and cannot be reopened by changing thresholds.

Two still-distinct explanations remain for the point-model ceiling:

1. all recent point models optimize method-balanced L1 while treating reported mutant measurement errors as equally reliable targets;
2. a single signed scalar head must simultaneously learn whether a response exists, its direction and its magnitude, even though feature41 already contains separately trained signed and absolute outputs.

The diagnostics below are deliberately linear and parameter-free. They are not intended to become the paper model. Their sole purpose is to decide whether either learning-problem change has enough cross-puzzle signal to justify one narrow neural amendment.

## Arm A: observation-noise-aware feature41 refit

The baseline exactly rebuilds the corrected 41-feature ridge. The candidate uses the same features, target universe and ridge alpha. The only difference is a train-only reliability multiplier

\[
w_{ui}=\frac{1}{e_{ui}^{2}+e_{\mathrm{WT},i}^{2}+0.05^{2}}.
\]

Within every mutant, the multiplier is normalized to mean one before applying the existing equal-mutant/equal-position cell weight. Thus no mutant or method receives extra total exposure; only the relative influence of positions within a mutant changes. Missing or nonpositive reported errors receive multiplier one. Held errors never enter prediction.

Arm A is supported only if both signed-delta MAE and coherent point-absolute MAE improve by at least 0.5%, both puzzle-level 95% CI lower bounds exceed zero, and both have at least 14/20 positive puzzles.

## Arm C: coherent signed-magnitude reconstruction

The existing feature41 ridge already produces an independently fitted signed output and absolute-magnitude output. Without fitting any new parameter, define

\[
\hat\Delta_{\mathrm{coherent}}
=\operatorname{sign}(\hat\Delta_{\mathrm{signed}})
\max(\hat m_{\mathrm{absolute}},0).
\]

This is compared with the ordinary feature41 signed point. It tests whether a future hurdle/signed-magnitude neural head has identifiable headroom before implementing one.

Arm C is supported only if signed-delta MAE improves by at least 0.5%, point-absolute MAE improves by at least 1%, both CI lower bounds exceed zero, and both have at least 14/20 positive puzzles.

## Execution and access boundary

All 20 LOPO prediction-only folds are generated before any diagnostic score is read. The merger rejects missing, duplicate or unexpected folds and refuses target-side fields. After a complete merge, authority is changed once to permit the complete score; then one scorer and one mechanical qualifier are run. Partial scores and external outcomes remain prohibited.

If exactly one arm passes, it is the only capability allowed to seed a new model amendment. If both pass, the deterministic rule chooses the larger minimum normalized Gate margin, with Arm A as an exact tie-breaker. If neither passes, both routes close; the only remaining untested candidate class is task-matched WT-profile self-supervised pretraining, which would require a separate contract and matched from-scratch control.

For the both-pass comparison, the four numeric margins are signed relative gain divided by its frozen minimum, point-absolute relative gain divided by its frozen minimum, signed positive-puzzle count divided by 14, and point-absolute positive-puzzle count divided by 14. The route margin is the minimum of these four values. The two CI-lower-bound checks are binary prerequisites and are not converted into a post-hoc numeric scale.

No result from this diagnostic is SOTA, external replication, mechanism evidence or publication readiness.

## Terminal result

The complete 20-puzzle score selected `WT_PROFILE_SELF_SUPERVISED_PRETRAINING_ONLY` and closed both tested routes.

- Noise-aware feature41 improved signed-delta MAE by 0.3887% with a positive 95% CI and 20/20 positive puzzles, but missed the frozen 0.5% practical threshold. Its point-absolute gain was 0.1374% with 13/20 positive puzzles, also below Gate.
- Coherent signed-magnitude reconstruction improved point-absolute MAE by 1.5185%, but worsened signed-delta MAE by 14.2568% with 0/20 positive puzzles. The factorization therefore breaks the primary point estimand.

Neither result authorizes a neural amendment of its family. The only route retained by the prospective decision rule is task-matched WT-profile self-supervised pretraining with an identical-architecture, identical-downstream-task from-scratch null. This conclusion cannot change V13 or establish a publication claim.
