# ReactFlow-Delta post-V13 autoresearch ledger

## Mode and objective

- Mode: `orchestrator / optimize-metric`.
- User objective: materially improve the model before submission; a benchmark-only manuscript is not the desired endpoint.
- Current incumbent evidence: V13 is terminal `V13M3_TOP_JOURNAL_SCREEN_FAIL`; V13M4 is permanently closed.
- Current authority: M6, training closed, partial-score and external-outcome access closed.
- Success predicate: a newly frozen, genuinely distinct model family must first produce an exact complete seed-0 20-fold top-journal screen PASS and then an exact fixed-seeds-0-to-4 20-fold formal PASS. No threshold, seed, epoch, comparator or model-family selection is allowed after score access.
- Independent unit: held puzzle (`n=20`), with the frozen method-balanced full-construct estimand.

## Cross-version capability and falsification matrix

| Route | Added capability | Best complete evidence | What is ruled out |
|---|---|---|---|
| V1/P3 low-rank susceptibility | source-receiver low-rank operator | only small incremental skill; strict later rescue did not identify a stable low-rank mean increment | more rank search or rebranding low-rank factorization |
| V2 mean-first/calibration-second | exact L1 mean followed by zero-mean residual CRPS | mean gain 0.883%, below 1%; calibration gain positive with identical point | calibration can help, but the original mean objective is insufficient |
| V3 expert blending | B1/MeanAligned disagreement gate | post-hoc legal gate probe about 1.24% signed gain; corrected workflow did not become a qualifying terminal model | another disagreement-bin or gate-capacity search |
| V4 RNA-FM dual tower | large frozen sequence foundation plus mutation-conditioned tower | signed gain 0.227%, CRPS worsened | larger sequence backbone/foundation alone |
| V5/V6 Vienna ensemble change | exact mutant-WT thermodynamic deltas, then 2A3-constrained deltas | +0.634% then +0.116% signed increments | more Vienna descriptors, StructDelta or contact-proxy additions |
| V7 RiNALMo dependency | 650M pretrained directed WT-to-mutant dependency features | +0.038% signed, CI crosses zero | another frozen foundation dependency feature stack |
| V8/V9/V10 | feature41-anchored larger mean; zero-mean residual; asymmetric calibration | V8 signed +8.36% but absolute guardrail failed; V9 CRPS +4.23% missed 5%; V10 absolute +14.30%, CRPS +3.28% | joint mean/calibration, symmetric-only residuals, more calibration-head capacity |
| V11 | feature41-anchored WT-context neural residual | signed +9.80%, point-absolute +4.57%, CRPS +3.66%; only +0.05% signed vs unanchored and CRPS worse vs null | feature41 skip attribution and another WT-context residual of the same type |
| V12 | monotone distance/magnitude shrinkage of V11 residual | signed +10.08% but only +0.31% vs V11; point absolute worsened vs V11; gate within 1% of in-sample oracle | more shrinkage-gate capacity or coordinate tuning |
| V13 | shared WT/exact-mutant re-encoding anchored at feature41 | signed +9.88%, point absolute +5.26%, CRPS +3.92%, distribution absolute +12.65% vs feature41; only +0.12%, +0.22%, +0.10%, and -0.03% vs matched WT-replay null | exact-mutant re-encoding, wider/deeper same-family V14, extra epochs |
| historical structure propagation | Vienna BPP/contact message passing | random contacts matched/exceeded true contacts; generic concat beat EPRO | structure/contact message passing without a new identified signal |

## Confirmed bottlenecks

1. `feature41` is the dominant transferable estimator. It combines direct features with exact-mutant thermodynamic ensemble-change features; later neural models mostly reproduce or lightly reshape it.
2. The best neural point increment is localized away from the edit site and at non-negligible feature41 magnitude, but V12 showed that better shrinkage coordinates cannot provide the missing margin.
3. Exact-mutant sequence information is not the missing capability: V13 candidate and WT-replay null are statistically indistinguishable on all four headline metrics.
4. Probability calibration is useful but no longer the primary bottleneck: V9/V10 improved CRPS/absolute distribution, while the point estimator still misses the top-journal margin.
5. The remaining route must change the estimand-relevant learning problem, not merely capacity, context encoding, calibration family, structure proxy, rank or gate.

## Candidate capability audit

### A. Tail-balanced or measurement-error-aware point learning

- New ability: distinguish rare/high-signal mutation effects from noisy or near-zero observations during outer-train learning while keeping inference target-blind.
- Supporting evidence: the development failure atlas shows tail-specific weakness; reported mutant and WT errors exist but all V8/V11/V13 point objectives use unweighted L1 and do not model known observation noise.
- Non-duplication: unlike V12, this changes the learned expert, not a post-hoc shrinkage gate.
- Risk: weighting can move away from the unweighted MAE estimand and amplify noisy tails.
- Required diagnostic before amendment: show that feature41/V11 residual magnitude and relative model error have a reproducible relationship with train-legal reported uncertainty, and that an error-aware loss has cross-puzzle headroom under an outer-train-only probe.

### B. Task-matched WT-profile self-supervised pretraining

- New ability: learn a contextual WT reactivity representation by masked-profile reconstruction before mutation-effect fine-tuning.
- Supporting evidence: current V11/V13 encoders are trained only through mutant-response supervision on 19 puzzles; no current OpenKnot model-rescue run uses a task-matched WT-profile denoising objective.
- Non-duplication: V4/V7 are frozen sequence-only foundations; historical M2SL5 static pretraining was on an adjacent endpoint and failed, so it is cautionary rather than task-matched evidence.
- Risk: only 160 WT constructs, transductive use of held WT inputs must be disclosed, and the encoder may simply learn the identity shortcut because full WT reactivity is already an input.
- Required null: identical architecture and supervised budget with a mask-shuffled or from-scratch pretraining control that uses every parameter.

### C. Signed-magnitude/hurdle point factorization

- New ability: separately learn zero/near-zero probability, direction and effect magnitude, then combine them into the signed point prediction before zero-mean residual calibration.
- Supporting evidence: V8 improves signed MAE while worsening absolute magnitude; V9/V10 show strong distribution-derived absolute gains; the mismatch is consistent with a direct L1 head conflating occurrence, sign and magnitude.
- Non-duplication: SparseDelta used a probabilistic gate whose joint likelihood distorted the mean; the proposed factorization would be mean-first and explicitly optimized on the final point metrics.
- Risk: the evaluation-optimal signed median is not automatically recovered by separately optimized heads; an auxiliary-task gain may not survive the combined point metric.
- Required null: parameter-matched direct point model with the same encoder, hidden width, optimizer and compute, differing only in the frozen output factorization and auxiliary objectives.

## Completed route diagnostic

The frozen 20-puzzle diagnostic closed A and C. Noise-aware feature41 produced a statistically consistent but practically sub-threshold signed gain (0.3887%, 20/20, CI lower above zero) and only 0.1374% point-absolute gain (13/20). Coherent sign-magnitude reconstruction improved point-absolute by 1.5185% but worsened signed MAE by 14.2568% (0/20). Neither route is eligible for a neural amendment.

The deterministic qualifier selected B: task-matched WT-profile self-supervised pretraining. Its required null is the identical downstream architecture trained from scratch with the identical supervised task and budget. Structure/contact, exact-mutant, foundation-feature concatenation, rank, calibration capacity, shrinkage gates, noise-aware reweighting and signed-magnitude factorization are closed.

## Units remaining

1. Create the isolated WT-profile pretraining amendment/worktree/artifact universe.
2. Implement candidate, identical-architecture from-scratch null, runner, merger, scorer, qualifier and invariants.
3. Complete real-data prediction-only smoke.
4. Complete score-blind seed-0 20-fold screen.
5. If and only if exact screen PASS, complete fixed five-seed formal confirmation.
