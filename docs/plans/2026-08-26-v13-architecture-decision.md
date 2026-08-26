# ReactFlow-Delta V13 score-blind architecture decision

**Recorded:** 2026-08-26, after V12 terminal freeze and before any V13 implementation or outcome access.

**Evidence qualification:** `ARCHITECTURE_DECISION_ONLY_NO_MODEL_RESULT`.

## 1. Controlling evidence

- `CONFIRMED_FACT`: V12 is terminal `V12M3_TOP_JOURNAL_SCREEN_FAIL`; V12M4 is permanently closed. V13 does not modify that verdict.
- `CONFIRMED_FACT`: the post-V12 route is `TERMINATE_SHRINKAGE_GATE_CAPACITY_ROUTE`. It rules out more residual-gate capacity, not a new input representation with a separate amendment.
- `CONFIRMED_FACT`: V11's point encoder is WT-only. Mutation identity enters only after WT encoding in a generic source/receiver concatenation head.
- `CONFIRMED_FACT`: V4 used a large mutation-conditioned response tower and frozen exact-mutant RNA-FM differences, but did not preserve the later-identified feature41 point anchor. V4 improved signed-delta MAE by only 0.2267% and worsened CRPS.
- `CONFIRMED_FACT`: V11 established that the feature41 anchor is necessary for the current residual family, while V12 established that further shrinking that residual is not the missing capacity.
- `REASONED_INFERENCE`: the only untested combination with a distinct capability is to keep feature41 as the main estimator and expose a small residual model to a trainable, shared-weight WT/exact-mutant representation difference.

Ribonanza documents that M2 measurements are full reactivity profiles under sequence mutations, so exact-mutant sequence is a task-matched, outcome-blind input rather than an auxiliary label: <https://pmc.ncbi.nlm.nih.gov/articles/PMC10925082/>. M2-REEFFIT reports that single mutations can induce global profile changes and recurrent alternative patterns, which supports testing a mutation-conditioned representation but does not establish that the proposed model will work: <https://pmc.ncbi.nlm.nih.gov/articles/PMC4643908/>.

## 2. Independent candidate generation

### Idea A — feature41-anchored shared WT/exact-mutant re-encoding

- `IDEA`: run the same compact V11 encoder on WT and exact-mutant sequences, form receiver-wise and source-wise hidden differences, and use them only to predict a residual added to feature41.
- `NEW_CAPABILITY`: receiver context is recomputed after the nucleotide substitution; V11 cannot do this because all hidden states are WT-only.
- `PREDICTION`: the exact-mutant candidate beats an identical WT-replay null and the strongest terminal point/distribution comparators on complete LOPO puzzles.
- `FALSIFIER`: no stable gain over the WT-replay null, or failure of any pre-frozen top-journal point/probability Gate.

### Idea B — feature41-anchored source-conditioned affine modulation

- `IDEA`: replace V11's concat head with FiLM-like source/mutation modulation of receiver states.
- `ADVERSARIAL_FINDING`: V11's two-layer nonlinear head can already approximate multiplicative interactions, and V4's five-block mutation-conditioned response tower was stronger than a small affine modulator but failed. This is likely a reparameterization/capacity test rather than a new identifiable input capability.
- `DECISION`: rejected for V13.

### Idea C — latent full-profile state templates

- `IDEA`: predict mutation-specific weights over a small set of full-profile response templates.
- `ADVERSARIAL_FINDING`: the biological precedent is real, but the repository lacks prospective evidence that reusable latent states transfer across held puzzles. It also overlaps prior low-rank/operator failures and risks learning method/batch modes.
- `DECISION`: rejected until an outcome-blind or separately prospective state-reuse diagnostic supports it.

## 3. Selected hypothesis and exact null

`DECISION`: select Idea A as the sole V13 candidate.

The candidate and null have identical architecture, parameters, initialization, optimizer, epochs, dropout stream, inputs outside the second sequence pass, calibration family, folds and seeds.

The WT and second encoder pass also share the same dropout mask within each paired call. This makes the stochastic encoder comparison a common-random-numbers contrast rather than allowing independent dropout noise to masquerade as mutation-induced representation change.

- Candidate second pass: exact-mutant sequence with the registered corrected ref→alt change.
- Null second pass: WT sequence replayed in the same batch shape and through the same encoder.
- Both models still receive ref/alt one-hot, signed distance and feature41 in the residual head.
- Therefore the candidate-versus-null contrast identifies the incremental value of mutation-conditioned re-encoding, not mutation identity, parameter count, compute budget or the feature41 anchor.

The null intentionally makes the hidden difference zero. That is the nested hypothesis being tested; no unused ballast or separate null network is added.

## 4. Why this is not V4 repeated

V4 attempted to learn the complete point mean with a 35–45M response/pair model and frozen RNA-FM delta. V13 instead:

1. preserves the feature41 point as an immutable skip anchor;
2. uses the compact V11 encoder and trains only an incremental residual;
3. shares weights exactly between WT and mutant passes;
4. attributes the exact-mutant increment with the same model run on WT twice;
5. uses the same post-point median-constrained calibration family for candidate and null.

If V13 fails, the project will have tested both an unanchored high-capacity mutation-conditioned model and an anchored compact exact-mutant contrast. That is sufficient to terminate this representation route.

## 5. Decision boundary

- No architecture, loss, width, depth, batch, epoch, seed, calibration or threshold search is permitted.
- No V13 training is authorized by this document.
- Implementation begins only after the human amendment, machine contract, ledger and active authority are frozen in an isolated worktree.
- Scientific scoring is forbidden until the complete 20-fold prediction-only universe exists.
- A near miss remains a FAIL.
