# Model Rescue v7 Design: Directed RiNALMo Dependency Operator

## Decision

Build one outcome-blind RiNALMo-Giga exact-SNV dependency cache and place a complete, fixed linear LOPO eligibility probe in front of any neural training. Do not reuse the failed V4 embedding-difference design and do not expand the V5/V6 thermodynamic descriptor family.

## Data flow

```text
registered WT sequence + source/ref/alt
  -> exact mutant sequence
  -> frozen RiNALMo-Giga unmasked WT/mutant MLM logits
  -> ACGU probabilities at every receiver
  -> signed target-base log-odds shifts
  -> fixed dependency6 cache
  -> V7M2: baseline41 vs baseline41+dependency6 weighted ridge
  -> only if eligible: frozen corrected B1 + dependency edge operator
  -> frozen mean + zero-mean residual calibration
  -> evaluator_v2 complete method-balanced score
```

## Why this representation

The target assay measures how one exact mutation changes a full receiver profile. A contextual embedding is a state representation; the required object is an intervention response. The dependency transform computes that response before supervised training and preserves direction from the actual mutation source to each receiver.

The six-dimensional basis contains the signed four-base response, the response for the WT receiver nucleotide, and the official maximum-absolute dependency summary. It is intentionally small and fixed. The neural operator receives the same six values; no hidden-layer or feature-subset search is permitted.

## Attribution

The V7M2 baseline exactly replays V6 candidate predictions. The later primary shares corrected B1 and trainable capacity with two controls. Zero dependency tests capacity; half-length cyclic receiver shift retains the marginal dependency feature distribution while destroying coordinate alignment. A primary result is attributable only if it beats both controls on CRPS and signed-delta MAE with paired CI lower above zero.

The 32-dimensional distance input is fixed before any V7 score: for raw integer
`d = receiver_index - source_index` and `k = 0,...,15`, concatenate
`sin(d / 10000^(2k/32))` and `cos(d / 10000^(2k/32))` in increasing `k` order.
It has no learned parameters and no alternative normalization or frequency grid.

For formal seeds, corrected B1 is a fold×seed object rather than a shared
post-hoc checkpoint. Seed 0 reuses the matching exact R3C3-qualified artifact;
seeds 1–4 refit the identical frozen B1 algorithm from scratch on outer-train.
Within one fold×seed, baseline, primary and both attribution controls share that
same B1 checkpoint.

## Main risks

1. RiNALMo dependencies encode structural contacts but not 2A3 reactivity direction. V7M2 would fail quickly.
2. Exact OpenKnot sequences may have appeared in pretraining. Exposure remains unknown and blocks external/SOTA claims even after an internal pass.
3. Unmasked MLM outputs can be overconfident. Float32 log-odds and fixed epsilon address numerical stability; they do not justify changing the scientific definition.
4. The real M2 universe contains 160 distinct WT sequences and 13,976 distinct
   registered mutant sequences, so the full cache requires 14,136 RiNALMo
   sequence inferences and 13,976 unique dependency edges. Exact-sequence
   deduplication remains implemented but gives no cross-method reuse on this
   dataset. OOM changes only batch size or GPU placement.

## Success decision

- V7M2 below 1% incremental signed-delta gain: terminate.
- V7M2 pass but V7M4 below 5% dual-metric gain or control attribution: terminate.
- V7M5 pass: freeze as a high-effect post-hoc development model; external confirmation remains separate.
