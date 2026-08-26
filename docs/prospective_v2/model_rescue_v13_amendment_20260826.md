# ReactFlow-Delta Model Rescue v13 Amendment

## 1. Contract status and final goal

V13 is an isolated, narrow amendment for **feature41-anchored exact-mutant counterfactual re-encoding**. It does not reopen or rewrite V1–V12. In particular, V12 remains terminal `V12M3_TOP_JOURNAL_SCREEN_FAIL`, V12M4 remains permanently closed, and the shrinkage-gate family remains terminated.

### Final Goal

- **Scientific question:** after preserving the strongest train-only feature41 point anchor, does a shared-weight exact-mutant encoder provide a stable, attributable improvement in unseen-puzzle full-construct mutation-response prediction?
- **Hypothesis:** a WT-only context plus late mutation token cannot represent the mutation-induced change in every receiver context. The difference between shared encoder states, `H_mut-H_wt`, supplies this missing counterfactual representation while the feature41 anchor prevents the small dataset from forcing the network to relearn the dominant point estimator.
- **Primary candidate:** `v13_feature41_anchored_exact_mutant_contrast`.
- **Nested null:** `v13_feature41_anchored_wt_replay_null`.
- **Task/data/split:** OpenKnot M2 v4.5.2, exact target-profile identity, 20-puzzle LOPO split_v4, evaluator_v2 method-balanced full-construct scoring.
- **Point objective:** exact method-balanced signed-delta L1.
- **Probability objective:** point frozen first; then the unchanged V10 median-constrained asymmetric residual family, trained by exact method-balanced Gaussian-mixture CRPS.
- **Top-journal success:** the candidate must clear every pre-frozen point, probability, attribution, integrity and stability Gate in seed-0 screening and fixed five-seed confirmation.
- **Failure:** any Gate failure terminates exact-mutant re-encoding; no V14 variant, size search or threshold change is allowed from this family.
- **Evidence ceiling:** an internal PASS is only `POST_HOC_DEVELOPMENT_PASS`; external replication, SOTA, mechanism, practical value and publication readiness remain unestablished.

## 2. Immutable parent evidence

- V4 terminal negative remains unchanged: a 35–45M mutation-conditioned response/pair model without the later feature41 anchor did not improve CRPS and improved signed-delta only 0.2267%.
- V11 terminal negative remains unchanged: the feature41 anchor produced strong gains over feature41, but missed its pre-frozen top-journal and matched-null requirements.
- V12 terminal negative remains unchanged: a monotone residual shrinkage gate produced only 0.3076% signed gain over V11, worsened point-absolute versus V11, and did not meet CRPS/distribution Gates.
- The post-V12 finding `MONOTONE_PRODUCT_ADEQUATE` permanently closes more shrinkage-gate capacity.

## 3. Frozen model

### 3.1 Shared encoder

The candidate and null use the V11 context encoder without increasing its size:

- width 192;
- 8 attention heads;
- 4 relative-attention blocks;
- FFN width 768;
- dropout 0.1;
- WT local inputs: sequence one-hot, WT reactivity, WT precision, observed mask, normalized position and region;
- exact-mutant pass changes only the registered sequence nucleotide at corrected full coordinate; all WT measurements remain the same baseline anchor;
- the same encoder parameters process WT and the second sequence pass.

The second pass is batched in fixed microbatches of 64 mutants. Microbatch size is an engineering constant, not a scientific hyperparameter; OOM recovery may reduce it without changing predictions in evaluation mode, but the original and recovered value must be recorded.

### 3.2 Point head

For each mutant and receiver position, the head receives:

- WT source hidden;
- WT receiver hidden;
- second-pass minus WT source hidden;
- second-pass minus WT receiver hidden;
- normalized signed source–receiver distance;
- ref/alt one-hot;
- feature41 point.

The head is fixed to two width-256 hidden layers with GELU, LayerNorm after the first layer, dropout 0.1 and a zero-initialized scalar output. Final point:

`point = feature41 + residual`.

Candidate and null differ only in the second sequence tensor:

- candidate: registered exact-mutant sequence;
- null: WT sequence replayed with identical shape through the same encoder.

Both still receive the same legal mutation metadata, so the contrast isolates re-encoding rather than mutation identity.

### 3.3 Training

- initialize candidate and null identically from scratch for every fold and seed;
- no V11 checkpoint warm-start;
- exact method-balanced signed-delta L1;
- Adam, learning rate `1e-3`, weight decay `0`;
- 40 epochs, no early stopping;
- gradient clipping `5.0`;
- every outer-train puzzle×method cell visited once per epoch in deterministic seed/epoch order;
- missing target never becomes zero;
- held outcome/error/qualified mask never enters prediction;
- no rank, gate, structure proxy, method ID, puzzle ID, teacher or foundation feature.

### 3.4 Frozen probability stage

After point training:

- freeze all point parameters;
- fit the unchanged V10 `MedianAsymmetricResidual` family separately for candidate, null and feature41 comparator;
- inputs and hidden width remain exactly those frozen in V11/V12;
- 40 epochs, Adam `1e-3`, weight decay `0`;
- exact method-balanced Gaussian-mixture CRPS;
- predictive median must equal the frozen point to tolerance `1e-7`;
- calibration gradients must not reach point parameters.

## 4. Prediction-only schema

Schema: `reactflow_delta.model_rescue_v13_prediction.v1`.

Every registered mutant×construct position contains keys, fold, seed, candidate ID, point, component locations/scales/weights, registered status and checkpoint paths. It contains no target, target error, qualified target mask, loss or score.

Required invariants:

- complete registered key universe;
- exact corrected target identity;
- candidate and null parameter names/count/initial state identical;
- candidate/null second-pass sequence is their only point-path difference;
- ref/alt mutation changes exactly one registered nucleotide for candidate and zero for null;
- null hidden difference is zero in evaluation mode within `1e-7`;
- point is invariant to held target/error/mask;
- median equals point within `1e-7`;
- coverage 100%, failure 0, unexpected keys 0.

## 5. Phase graph

```text
V13M0_CONTRACT_FROZEN
  -> V13M1_IMPLEMENTATION_AND_INVARIANTS_PASS
  -> V13M2_REAL_DATA_ENGINEERING_SMOKE_PASS
  -> V13M3_TOP_JOURNAL_SCREEN_PASS / V13_TERMINAL_FAIL
  -> V13M4_FIXED_FIVE_SEED_CONFIRMATION_PASS / V13_TERMINAL_FAIL
  -> V13M5_ARTIFACT_FREEZE_AND_MAIN_CONTRACT_HANDOFF
```

### V13M2 smoke

- real OpenKnot folds 0/1, seed 0;
- point and calibration 3 epochs each;
- prediction-only; no scientific score;
- verifies finite optimization, input-ablation identity, point freeze, target invariance, merge integrity and complete output.

### V13M3 screen

- seed 0, all 20 LOPO folds;
- point 40 epochs, calibration 40 epochs;
- candidate and null both trained; feature41/V10/V11/V12 terminal comparators loaded only after the complete prediction universe is merged;
- 20/20 before any held score or partial direction access;
- one scorer run and one qualifier run.

All Gates are conjunctive:

1. **Signed-delta point**
   - relative gain vs feature41 at least 12%;
   - relative gain vs terminal V12 at least 2%;
   - relative gain vs nested null at least 1.5%;
   - paired puzzle CI lower > 0 for all three;
   - positive puzzles at least 16/20, 14/20 and 14/20 respectively.
2. **Point absolute-delta**
   - relative gain vs feature41 at least 7%;
   - relative gain vs terminal V11 at least 2%;
   - relative gain vs nested null at least 1%;
   - paired puzzle CI lower > 0;
   - positive puzzles at least 16/20, 14/20 and 14/20.
3. **Task CRPS**
   - relative gain vs fresh fair feature41-asymmetric at least 5%;
   - relative gain vs terminal V12 at least 2%;
   - relative gain vs nested null at least 1.5%;
   - paired puzzle CI lower > 0;
   - positive puzzles at least 16/20, 14/20 and 14/20.
4. **Distribution-derived absolute-delta**
   - relative gain vs feature41 at least 15%;
   - relative gain vs terminal V10 at least 2%;
   - relative gain vs nested null at least 1%;
   - paired puzzle CI lower > 0;
   - positive puzzles at least 16/20, 14/20 and 14/20.
5. **Stability/integrity**
   - leave-one-puzzle effects remain positive for every headline contrast;
   - no puzzle contributes more than 20% of total effect;
   - 68% and 95% coverage absolute error do not worsen by more than 1 percentage point versus the matched fair comparator;
   - registered coverage 100%, failure 0, unexpected keys 0;
   - no partial score access and no model/threshold selection.

### V13M4 formal confirmation

Only exact V13M3 PASS opens V13M4. Run seeds 0–4, 20 folds, fixed 40+40 epochs, candidate/null/feature41 across the same fold×seed universe. Assemble equal-seed mixtures; do not delete failed seeds or select a seed subset. Repeat all feature41 and nested-null Gates above on the unique five-seed mixtures, with at least 4/5 individual seeds positive for signed-delta and CRPS. The terminal V11/V12/V10 seed-0 comparisons must already have passed at V13M3 and remain historical context rather than pretending to be five-seed comparators.

## 6. Stop and handoff

- V13M3 FAIL: close V13M4 permanently, preserve complete negative results, terminate exact-mutant re-encoding, return M6.
- V13M4 FAIL: preserve instability result, terminate exact-mutant re-encoding, return M6.
- V13M4 PASS: status only `POST_HOC_DEVELOPMENT_PASS`; freeze the model as a benchmark development candidate and require a new sealed external amendment for stronger claims.
- No failure permits changing width, depth, objective, epochs, null, Gate or calibration; no V14 from this family.

## 7. Resource and access boundaries

- physical GPU0–7 may be used when memory is sufficient and may co-locate without interfering with other processes;
- no preemption, termination, signaling or mutation of unrelated jobs;
- persistent sessions and at least 900-second low-frequency monitoring during scientific runs;
- OOM permits only changing GPU or lowering inference/training microbatch size while preserving exact sample/cell order and scientific protocol;
- no new external outcome;
- no new split;
- no hash/checksum scaffolding, compatibility layer, feature flag or unrelated refactor.

