# Post-V11 contingency — frozen before V11 score access

## Status and boundary

This document does not authorize V12 or any additional model. It freezes the diagnostic and decision logic before V11 held-score access so that a near miss cannot be converted into an improvised model family or a lowered Gate.

- `LOCATED_EVIDENCE`: V8 improved signed-delta MAE by 8.36% versus corrected feature41 but worsened point absolute-delta MAE.
- `LOCATED_EVIDENCE`: V10 improved task CRPS by 3.28% versus the fair feature41-asymmetric comparator and passed its other frozen Gates, but failed the 5% top-journal task-CRPS margin.
- `LOCATED_EVIDENCE`: V11 tests the unrun feature41-anchored context residual with an exact same-parameter unanchored null and reuses the fixed V10 median-constrained residual family.
- `ASSUMPTION`: the next useful model decision depends on whether V11 is limited by point optimization, cross-puzzle generalization, anchor identification, or conditional residual misspecification. The V11 headline score alone cannot distinguish these explanations.
- `PROHIBITED`: no partial-fold diagnostics, no new external outcome, no threshold relaxation, no best-puzzle or best-seed subset, and no new candidate training under V11 authority.

All diagnostics below may run only after the complete 20-fold V11M3 prediction universe has been merged and scored once under complete-score authority. They use the same method-balanced position→mutant→cell→puzzle estimator. The 20 held puzzles remain the only independent units for held-effect intervals.

The executable implementation is frozen in `scripts/reactflow_delta/diagnose_model_rescue_v11.py`. It requires the qualified complete merge, the one complete score artifact, and the mechanical V11M3 verdict. It cannot run while training or held-score access is closed, and it always records `new_model_authorized=false`.

## Predeclared diagnostic package

### D1 — Point residual learnability and anchor attribution

For every registered held position define

\[
r=\Delta-f_{41},\qquad
\hat r_A=\hat\Delta_{anchored}-f_{41},\qquad
\hat r_N=\hat\Delta_{unanchored}.
\]

Report, at puzzle level and then across 20 puzzles:

- residual MAE for zero residual, `hat_r_A`, and `hat_r_N`;
- signed association between `r` and each predicted residual;
- residual amplitude ratio `median(|hat_r|) / median(|r|)`;
- the anchored-minus-unanchored difference using the exact matched-null pairing.

`DECISION_RULE`: an anchor-specific residual signal requires anchored residual MAE gain over both zero residual and the unanchored null, CI lower greater than zero, and at least 14/20 puzzles in the same direction. Otherwise feature41 anchoring is not identified even if the full candidate beats an older baseline.

### D2 — Convergence versus representational failure

Use the already-recorded 40-epoch point histories; do not retrain. For each fold compute

\[
u_p=\frac{\operatorname{mean}(L_{31:35})-\operatorname{mean}(L_{36:40})}
{\operatorname{mean}(L_{31:35})}.
\]

`DECISION_RULE`: call the schedule visibly unfinished only if `u_p >= 1%` in at least 14/20 folds and the median `u_p >= 1%`. A negative final slope alone is insufficient. If this condition is absent, more epochs are not a supported rescue explanation.

After complete held scoring, final checkpoints may be evaluated on their own outer-training puzzles with the same estimator. Because the 20 outer-training sets overlap, this train result is descriptive and receives no independent-puzzle CI. A train-to-held relative-gain drop of at least five percentage points in at least 14/20 outer folds is treated as cross-puzzle overfit, not lack of parameter count.

### D3 — Outcome-blind regime localization

Decompose residual MAE and task CRPS only along bins fixed without held outcomes:

- absolute feature41 point: `[0,0.05)`, `[0.05,0.10)`, `[0.10,0.20)`, `[0.20,inf)`;
- absolute signed distance from edit: `0`, `1–5`, `6–20`, `>20`;
- region: `design` versus `other`;
- the eight registered experimental methods, reported separately without using method ID as a model input.

Each bin is first averaged within mutant and method cell, then puzzle. SNVs, positions, methods, or bins never increase independent N. A localized pattern may justify a mechanistic hypothesis only when its puzzle-level CI excludes zero and at least 14/20 puzzles agree; it does not itself authorize a routed expert.

### D4 — Fixed-median residual distribution adequacy

For feature41-asymmetric, anchored-asymmetric, and the unanchored null, report:

- probability integral transform summaries at puzzle level;
- 50%, 68%, 80%, 90%, and 95% central coverage errors;
- lower- versus upper-tail miss rates;
- conditional task-CRPS effects in the D3 outcome-blind bins;
- scale and allocation association with absolute residual error.

`DECISION_RULE`: a new distribution family is supported only if point Gates pass but task CRPS remains the sole failed headline Gate and one prespecified asymmetry/tail diagnostic has a 20-puzzle CI excluding zero with at least 14/20 puzzles in the same direction. Otherwise distribution complexity is not the identified bottleneck.

## Frozen interpretation of V11 outcomes

1. **All V11M3 Gates PASS:** open only the already-frozen V11M4 five-seed confirmation.
2. **Point and anchor attribution PASS; task CRPS alone FAILS:** V11 remains terminal FAIL. Run D4; only a replicated fixed-median residual misspecification can support a later distribution-only amendment. Do not free the point median.
3. **Signed point PASS; point absolute FAILS:** the sign/magnitude trade-off remains. V11 fails; do not relax the absolute Gate or hide it behind distribution-derived magnitude.
4. **Candidate point improves but matched-null attribution FAILS:** capacity/context may help, but feature41 anchoring is not identified. V11 fails as an anchored-residual contribution.
5. **Point Gates FAIL and D2 shows unfinished convergence:** a later convergence-only amendment may test the identical model for longer; architecture, capacity, loss, inputs, and null remain fixed. This document does not choose the longer schedule or authorize training.
6. **Point Gates FAIL with strong outer-train skill and a large D2 train-to-held gap:** treat the failure as cross-puzzle overfit. A later amendment may test train-only shrinkage of the neural residual around feature41; adding capacity is contraindicated.
7. **Point Gates FAIL without unfinished convergence, train-only residual skill, or stable D3 localization:** the feature41-anchored context-residual hypothesis is falsified on the consumed 20-puzzle development set. Stop this architecture family and return to the benchmark route.
8. **V11M3 PASS but V11M4 FAILS:** the seed-0 signal is unstable. Preserve the full negative formal result; do not select a seed subset or open another model from the best seed.

## Requirements for any later amendment

Any future amendment must name exactly one diagnostic-supported bottleneck, one exact candidate, one parameter/representation-matched null, fixed folds/seeds/epochs, and top-journal effect-size Gates before training. It must preserve V1–V11 verdicts, use a new worktree and artifact universe, keep external outcomes locked, and state an unconditional stop rule. No later model may be justified only by “V11 was close.”
