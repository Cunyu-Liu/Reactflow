# ReactFlow-Delta Model Rescue v4 Design

**Status:** FROZEN BEFORE IMPLEMENTATION

**Date:** 2026-08-23

**Evidence scope:** post-hoc development on OpenKnot M2; no new external outcome access

## 1. Scientific purpose

Model Rescue v4 tests one narrow architectural hypothesis: a mutation-effect model needs an explicit, mutation-conditioned receiver representation and a learned two-dimensional interaction state. The corrected B1 family encodes one WT context and concatenates the edit-site hidden state, receiver hidden state, signed distance, and ref/alt token. It therefore cannot directly represent how an exact mutant changes receiver context or how mutation-source information enters an RNA-wide pair field.

The sole primary candidate is a Mutation-Conditioned Dual-Tower Response Transformer. It combines a one-dimensional response tower with a two-dimensional pair tower, uses exact-mutant sequence information through frozen paired RNA-FM embeddings, and predicts a full-construct signed-delta mean. Probability calibration is fitted only after the mean model is frozen and all residual components have the same location, so calibration cannot move the point mean.

This amendment does not revise the terminal conclusions of Model Rescue v1 or v2 and does not modify the in-progress v3 diagnostic baseline. A v4 result is not external, SOTA, mechanism, practical-utility, or publication evidence unless a later independent external amendment closes those claims.

## 2. Frozen foundation model

- Primary foundation: `ml4bio/RNA-FM` at Git commit `348951516e0963d22bbb33b3c9fc18c89081d38e`.
- Loader: `fm.pretrained.rna_fm_t12()`.
- Architecture disclosed by the official project: 12 layers, 640 hidden dimensions, approximately 99M frozen parameters.
- Pretraining corpus: approximately 23.7M RNAcentral100 non-coding RNA sequences.
- Use: deterministic final-layer nucleotide embeddings for both WT and exact registered mutant sequences.
- Training: all RNA-FM parameters remain frozen; no OpenKnot outcome gradient can enter RNA-FM.
- Exposure qualification: self-supervised sequence exposure is disclosed. Exact sequence overlap with OpenKnot is unknown and no no-overlap claim is allowed.
- RiNALMo is a foundation-only sensitivity comparator, not a selectable primary foundation.
- RibonanzaNet is an exposure-disclosed task comparator only; it cannot be described as a clean primary foundation because of its OpenKnot/Ribonanza data relationship.

Embeddings are precomputed from CSV columns `id`, `puzzle`, `method`, and `sequence` only. Reactivity targets, target errors, target masks, method outcomes, and external outcomes are not loaded by the cache builder. Cache manifests record checkpoint source, Git revision, layer, embedding width, sequence keys, and shapes. No checksum file is added.

## 3. Primary architecture

### 3.1 Inputs

For each WT construct, the 1D WT input contains nucleotide one-hot, normalized WT reactivity, WT reported error, WT observation indicator, full-construct position, design-region indicator, and the frozen 640-dimensional RNA-FM WT embedding. For each exact mutant, the mutation-conditioned path additionally receives ref/alt one-hot, the corrected full-sequence mutation coordinate, signed source-to-receiver distance, and the frozen per-position difference `E_mutant - E_WT`.

The model never receives held mutant reactivity, held error, qualified target mask, held score, puzzle identity, method identity, external outcome, or any outcome-derived structural annotation in its prediction path.

### 3.2 Towers

The trainable primary configuration is fixed:

- sequence width: 512;
- attention heads: 8;
- WT sequence blocks: 5;
- mutation-response sequence blocks: 5;
- feed-forward width: 2048;
- pair width: 128;
- pair axial blocks: 5;
- dropout: 0.10;
- foundation input width: 640;
- expected trainable capacity: 35M–45M, verified mechanically before real-data smoke.

The WT tower produces nucleotide states. The pair tower initializes an `L x L x 128` state from left/right sequence projections, nucleotide-pair identity, clipped relative distance, WT-reactivity pair features, and low-rank projections of frozen RNA-FM embeddings. Each pair block performs row attention, column attention, and a transition MLP. Pair state supplies per-head bias to the WT sequence attention.

For a mutation at source position `p`, the response tower is initialized from receiver state, source state, ref/alt, signed distance, paired foundation delta, and both `P[p,i]` and `P[i,p]`. Five response blocks update every receiver while using the shared pair state as attention bias. The output head emits one signed-delta mean for every registered mutant and every construct position. Prediction output is not masked by target availability.

### 3.3 Mean and probability objectives

The mean stage optimizes exact method-balanced signed-delta L1:

`position -> mutant -> puzzle-method cell -> equal cell mean`.

The optimizer is AdamW with learning rate `2e-4`, weight decay `1e-2`, cosine decay, 5% warmup, BF16 when supported, gradient clip `1.0`, and 80 fixed epochs. There is no early stopping, epoch search, loss-weight search, hidden-size search, or architecture search. All model families use the same loss and schedule.

After mean training, the entire mean model is frozen. A two-component conditional Gaussian scale mixture is trained for 40 epochs by exact Gaussian-mixture CRPS. Both component locations equal the detached signed-delta mean. Calibration gradients cannot enter the mean or RNA-FM. The final distribution is therefore `Delta = mu_delta(x) + epsilon` with conditional residual expectation zero.

## 4. Required attribution controls

Five fixed model families enter the seed-0 screen:

1. `corrected_b1`: corrected-coordinate B1 comparator trained by the same evaluator universe.
2. `v4_dual_tower_rnafm`: sole primary candidate.
3. `v4_dual_tower_scratch`: identical dual tower without foundation embeddings.
4. `v4_rnafm_only`: paired frozen RNA-FM embeddings with a small receiver head and no pair tower.
5. `v4_capacity_matched_sequence_null`: same paired RNA-FM inputs and comparable trainable capacity, but no pair tower and no `P[p,i]/P[i,p]` source-row/column features.

The capacity-matched null uses additional functional 1D response blocks rather than unused parameters. Its trainable parameter count must be within 5% of the primary model. If this cannot be achieved without changing the frozen functional design, engineering qualification fails before scientific training.

No family is selected from these controls. They are all reported. The primary candidate remains fixed regardless of scratch, foundation-only, or null results.

## 5. Evaluation and decisions

Development uses the corrected OpenKnot M2 v4.5.2 universe, fixed split-v4 20-fold leave-one-puzzle-out, and evaluator-v2 method-balanced scoring. The seed-0 screen completes all 20 folds before any score is read. Formal confirmation uses seeds 0–4 with no seed deletion or subset selection.

The top-journal development gate requires the primary candidate versus corrected B1 to achieve at least 5% relative improvement in both full-construct CRPS and signed-delta MAE, paired puzzle-level 95% CI lower bounds above zero, at least 16/20 positive puzzles for each metric, positive leave-one-puzzle-out effects, no puzzle contributing more than 20% of either aggregate effect, 100% registered coverage, zero failures and unexpected keys, and no more than one percentage point worsening of 68% or 95% coverage error. It must also beat the capacity-matched null and RNA-FM-only comparator on both metrics with paired CI lower bounds above zero.

An internal pass is only `HIGH_EFFECT_POST_HOC_DEVELOPMENT_PASS`. A top-journal evidence chain additionally requires a new, independently frozen external amendment with at least 3% improvement on both metrics, cluster-level CI lower bounds above zero, and consistent direction in at least 75% of top-level study/publication units.

## 6. Compute and concurrency

v3 retains its existing artifacts, sessions, code, gates, and terminal history. Under the later 2026-08-23 owner authorization, v4 may use GPU0–7 whenever the selected card has sufficient available memory, including co-location with existing tasks. It never preempts, terminates, signals, or modifies unrelated processes. GPU availability changes placement and timing, not model family, epochs, seeds, losses, thresholds, or gates.
