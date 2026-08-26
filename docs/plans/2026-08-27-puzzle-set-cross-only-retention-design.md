# Puzzle-Set V5 zero-preserving cross residual and capability-retention design

**Status:** `PRE_SCORE_IMPLEMENTATION_DESIGN_ONLY_NO_TRAINING_AUTHORITY`

**Evidence boundary:** this correction was designed before any Puzzle-Set
training or score access. It changes no V14 code, authority, artifact, Gate or
running process and uses no external outcome.

## 1. Confirmed failure modes in the previous implementation

The position-deranged matched null in commit `f07f4b3` fixes the former
Q/K identifiability and attention-dropout mismatch. Three independent performance
risks remain in the implementation-only P1 model:

1. `mixed` still contains the focal query through a residual connection, while
   the point head separately receives focal V14 source/receiver states,
   feature41 and the frozen V13 point. The incremental head can therefore learn
   a focal-only correction and ignore non-focal WT context completely.
2. The new context layers receive 3,800 masked-WT updates with AdamW at
   `3e-4`, then a newly initialized Adam immediately updates the same layers for
   760 mutation-target steps at `1e-3`. Resetting optimizer moments and raising
   the learning rate 3.33-fold makes overwriting the newly learned cross-context
   representation a predictable risk.
3. Removing the explicit query residual is necessary but not sufficient for a
   zero-preserving cross representation. Learned projection biases, the summary
   construction, attention/FFN/output-normalization biases and query-conditioned
   attention can produce a nonzero tensor even when every raw non-focal input is
   zero. Without a paired raw-zero reference, `mixed != 0` does not by itself
   prove causal dependence on non-focal inputs.

These are code-path facts, not claims about held performance.

## 2. Frozen V5 zero-preserving cross representation

For every focal construct and full-sequence position, both arms form one focal
query and exactly eight non-focal-derived K/V tokens:

- seven individual tokens, one for each non-focal construct;
- one summary token formed from those seven tokens and their missingness-aware
  mean, spread and support, without a dedicated learned summary token;
- candidate tokens use the registered coordinate;
- null tokens use the same constructs at the fixed circular shift 17;
- the focal construct is excluded from K/V in both arms;
- attention-weight dropout remains zero;
- the attention output does not add the focal query as a value residual.

For each focal query, V5 additionally builds a raw-zero non-focal reference.
Before either learned projection, all seven reference hidden states, WT
reactivities and observed indicators are set to zero. The real focal query is
reused element by element in the actual and reference paths. Both paths then
execute the complete, shared computation:

```text
raw seven non-focal streams
  -> individual projection and non-focal summary construction
  -> Q/K/V projection and attention
  -> FFN residual
  -> output normalization
```

Let `V` denote the seven actual raw non-focal streams, `V0` their raw-zero
counterpart, and `F(q,V)` the full computation above with focal query `q`. The
returned representation is

\[
c(q,V)=F(q,V)-F(q,V_0).
\]

In training mode, the actual and reference passes replay exactly the same
dropout draw and restore the post-call RNG state to that produced by one
ordinary `F` evaluation. Consequently learned projection biases, summary
constants, attention/FFN/output-normalization biases and every query-only term
cancel algebraically. For every focal query and arbitrary learned parameters,

\[
V=0\Longrightarrow c(q,V)=0
\]

element by element. A nonzero `c` therefore establishes causal dependence on
the non-focal raw inputs under this operator; it does not establish that the
dependence is predictive or useful.

The output remains one `[8, L, 256]` nonfocal-dependent tensor, so downstream
data, prediction and calibration interfaces do not change. Candidate and null
retain identical parameters, initialization, query construction, K/V count,
raw-zero reference, compute shape and optimization; only registered versus
fixed wrong-position non-focal alignment differs. The operator identifier is
`POSITION_ALIGNED_CROSS_CONSTRUCT_CONSENSUS_V5`.

## 3. Frozen paired point residual

Let `b` denote the existing focal V14, mutation, distance, feature41 and parent
features, and let `c` denote the zero-preserving nonfocal-dependent
source/receiver pair. The point
increment is changed from a free head `h(b,c)` to the shared-head contrast

\[
g(b,c)=h(b,c)-h(b,0),
\qquad
\widehat\Delta=\widehat\Delta_{V13}+g(b,c).
\]

Both head evaluations use the same parameters and the same dropout mask. The
post-call RNG state is the state after one head evaluation, preventing the
reference computation from consuming a second stochastic stream. Therefore,
even in training mode,

\[
c=0\Longrightarrow g(b,c)=0
\]

element by element. Base-only, parent-only or focal-only corrections cancel
algebraically; any deviation from the V13 parent must depend on the
nonfocal-dependent representation. This is a causal-path invariant, not a claim
that a nonzero representation improves prediction. The existing zero
initialization and `1e-7` parent-replay checks are retained.

## 4. Frozen point optimization schedule

The scientific exposure budget remains 40 epochs, 19 outer-train puzzles per
epoch and one optimizer step per puzzle:

- epoch 0: point head only at `1e-3`; context parameters are frozen and receive
  no gradients;
- epochs 1--39: the same Adam continues with two groups, point head at `1e-3`
  and context at `3e-4`;
- the context learning rate is inherited exactly from masked-WT pretraining,
  not selected from a grid;
- weight decay remains zero and gradient clipping remains `5.0`;
- point-head moments continue across the boundary; context optimizer state is
  first created when its gradients become legal;
- each arm still executes 760 point optimizer steps and exposes every cell to
  exactly 40 target passes;
- the point head receives 760 updates and context receives 741 supervised
  updates.

Candidate and null use identical epoch boundary, parameter groups, puzzle
order, random stream and update counts. No early stopping or history-based
checkpoint selection is introduced.

## 5. Outer-train-only retention diagnostic

A diagnostic uses only the nineteen outer-train `{puzzle, contexts}` records,
the same frozen final decoder and deterministic mask epoch 200. P1M3/P1M4
pretraining uses mask epochs 0--199, so the diagnostic mask is not a training
mask. The P1M2 engineering smoke records its actual 0--2 training range while
using the same disjoint diagnostic epoch; it cannot turn the diagnostic into a
scientific result.

It reports puzzle-balanced masked-position L1 for:

- the initial context state with the final decoder;
- the post-pretraining context state;
- the post-point context state.

For the candidate only, it also reports

\[
R=\frac{L_{initial}-L_{postpoint}}
        {L_{initial}-L_{postpretraining}}.
\]

`L_postpretraining < L_initial` establishes that pretraining created held-mask
reconstruction ability, and `R > 0` establishes that point fitting retained
some of that gain. A P1M3/P1M4 PASS therefore requires both candidate booleans
for every fold-seed run. This is a train-only pre-score Gate: a negative
candidate diagnostic freezes a complete negative merge and leaves held scoring
closed. The null receives the identical diagnostic and
complete continuous report, but is not required to learn correct alignment.
The diagnostic cannot select a checkpoint, learning rate, epoch, candidate or
Gate and cannot access the held puzzle.

## 6. Verification and falsifiers

No future real-data run is eligible unless focused tests establish all of the
following:

- focal construct values are absent from every cross K/V tensor;
- both arms use query `[8L,1,256]` and K/V `[8L,8,256]`;
- each raw-zero reference zeros all seven non-focal hidden/reactivity/observed
  streams before learned projection while reusing the focal query exactly;
- actual and reference paths both traverse individual/summary projection,
  attention, FFN residual and output normalization;
- arbitrary nonzero learned biases and arbitrary focal queries still produce
  exact elementwise zero when actual raw non-focal inputs equal the raw-zero
  reference;
- actual/reference cross-block passes share dropout randomness and advance the
  RNG by exactly one ordinary cross-block evaluation;
- Q/K/V gradients are finite and nonzero in both arms;
- a registered-coordinate non-focal counterfactual changes candidate
  representation at that coordinate and changes null representation only at
  the shifted coordinate; this proves dependence, not utility;
- a zero V5 representation cancels the point residual exactly in train and
  evaluation modes;
- actual/reference point-head passes share dropout randomness;
- initial and post-pretraining parent replay remain within `1e-7`;
- warmup leaves every context tensor bitwise unchanged while changing the head;
- the first legal joint update gives context projection and Q/K/V finite,
  nonzero gradients;
- head/context learning rates and 760/741 update counts are exact;
- the retention diagnostic is deterministic, parameter- and mode-preserving,
  target-free and held-puzzle rejecting.

The two direct V5 regression tests are
`test_zero_nonfocal_cross_reference_cancels_bias_and_focal_query` and
`test_paired_cross_block_reuses_dropout_and_advances_rng_once`. They must remain
mechanical invariants and cannot be counted as model-performance evidence.

Any failure of raw-zero construction before projection, exact focal-query
reuse, full-path actual/reference matching, shared stochastic draws, one-step
RNG advancement or exact raw-zero cancellation falsifies V5 and closes real-data
eligibility. Omitting the paired cross reference cannot be interpreted as a
weaker equivalent implementation.

If the complete future candidate cannot pass its already frozen performance and
candidate-versus-null Gates, this family terminates. Neither this correction nor
the retention diagnostic authorizes wider models, extra epochs, a second shift,
loss-weight search or a post-hoc Gate change.

## 7. Explicitly deferred objective change

The current method-balanced signed-delta L1 and frozen-point CRPS objectives are
kept unchanged in this experiment. A simple weighted addition of absolute-delta
loss is not used: for wrong-sign predictions it can create flat or conflicting
gradients. A separate predeclared multi-objective method would require its own
matched current-objective null and amendment; mixing it into this representation
test would destroy attribution.
