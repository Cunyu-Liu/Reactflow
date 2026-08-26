# V5 position-aligned, zero-preserving puzzle-set context design

**Status:** `PRE_SCORE_IMPLEMENTATION_DESIGN_ONLY_NO_TRAINING_AUTHORITY`

**Evidence boundary:** this design was completed while V14M3 remained
score-blind. No V14 loss, partial fold score, per-puzzle effect, Gate direction,
or external outcome was read.

## 1. Why the global construct token is insufficient

The first parent-preserving P1 implementation compressed every WT construct to
one observed-position mean token. That token can represent a puzzle-wide regime,
but the same vector is appended to every receiver position. It therefore cannot
tell the incremental head what the other seven designs show at the mutation
source coordinate or at a particular receiver coordinate. Because the endpoint
is a full-construct profile, this is a direct representational bottleneck rather
than a generic request for more capacity.

## 2. Outcome-blind coordinate audit

A metadata-only audit of the 160 registered WT rows in OpenKnot M2 v4.5.2 found:

- every construct has full length 177;
- every puzzle has exactly eight WT constructs;
- within each puzzle all eight constructs have the same `sub_start` and
  `sub_end` coordinates;
- within each puzzle all eight rows carry one shared target structure;
- the eight sequences are not near duplicates: mean pairwise full-sequence
  Hamming distance ranges from approximately 81.7 to 91.0 positions across
  puzzles.

Thus full-sequence position is a registered shared coordinate frame inside a
puzzle. At a fixed coordinate, the eight WT profiles provide different designed
sequences and measurements for the same target-position role. This makes
position-aligned cross-construct conditioning legal and more explicitly
coordinate-indexed than global mean pooling. No mutant outcome column was used for this
audit.

## 3. Alternatives

### A. Retain one global token per construct

Rejected. It is inexpensive and permutation equivariant, but it supplies no
receiver-specific or source-specific cross-construct information. Increasing
its width would not repair that missing indexing ability.

### B. Position-aligned cross-construct attention

Selected. For every full-sequence coordinate independently, the eight frozen
V14 hidden states and their observed flags form an unordered set. For each focal
construct, its state supplies the single query and is excluded from K/V. The
candidate query sees exactly seven registered-coordinate non-focal individual
tokens plus one non-focal summary with no dedicated learned token. The matched
null
executes the same projection, eight-token K/V attention, FFN and normalization,
but circularly shifts all seven non-focal streams by 17 full-sequence positions
while leaving the focal query registered. Thus both arms expose every Q/K/V
parameter to the same number of legal non-focal tokens; only correct
cross-construct coordinate alignment differs.

The focal query is never added back to the attention output. V5 also subtracts
the complete cross block evaluated on a raw-zero non-focal reference, removing
learned constants and query-only terms. The resulting source and receiver
states enter one shared point head as a paired
contrast, `h(base,cross)-h(base,0)`. The two evaluations use the same dropout
draw, so the frozen V14 local source/receiver features, signed distance,
mutation identity, feature41 and frozen V13 parent cannot by themselves create
an increment.

### C. Target-structure graph alignment

Rejected for this experiment. It would introduce a second new capability and a
new metadata dependency after several structure/contact routes already produced
negative or sub-threshold evidence. Full-position alignment directly tests the
registered eight-design relationship with a cleaner null.

## 4. Frozen architecture

For hidden states \(H\in\mathbb R^{8\times177\times256}\), concatenate the WT
observed indicator and project each individual state:

\[
I_{c,j}=W_p[H_{c,j};O_{c,j}]+W_qR_{c,j}.
\]

Here `R` is an individual, missingness-aware four-vector containing the safe WT
value and observed support; it contains no cross-construct or focal-deviation
statistic. For focal construct \(c\), pool the other seven individual tokens and
add their WT mean, population spread and support fraction to form one summary
\(S_{-c,j}\) without a dedicated learned summary token.

For each focal construct and position, the one-token query and eight-token K/V
set are

\[
Q_{c,j}=I_{c,j},\qquad
K/V_{c,j}=\{I_{d,j}:d\ne c\}\cup\{S_{-c,j}\}.
\]

The focal token is never a K/V. Candidate uses the seven registered-position
non-focal states and their summary. Null keeps the focal query at position
\(j\) and takes each of the other seven states from
\((j-17)\bmod L\), which is the value produced at position \(j\) by the fixed
`torch.roll(..., shifts=17)` operation, then rebuilds the summary from those
shifted non-focal values. No construct-order or method embedding is present.
The fixed shift is independent of puzzle, method, target, fold and seed.
Internal attention-weight dropout is zero in both arms; the matched FFN and
output-side residual dropouts remain `0.1`. Within-construct positional context
has already been encoded by the frozen six-layer V14 encoder.

The actual tokens above are not yet sufficient to define a zero-preserving
representation because learned biases and query-conditioned attention can be
nonzero without raw non-focal input. V5 therefore creates a reference before
either learned projection by setting all seven non-focal hidden states,
reactivities and observed indicators to zero. The real focal query is reused
element by element. Let `F(q,V)` include individual and summary projection,
Q/K/V attention, the FFN residual and output normalization. The returned state
is

\[
C_{c,j}=F(Q_{c,j},V_{c,j})-F(Q_{c,j},V^{0}_{c,j}).
\]

There is deliberately no `Q + attention` residual. In training mode the actual
and reference calls use the same dropout draw and leave RNG advanced by exactly
one ordinary `F` call. This cancels learned projection biases, summary
constants, attention/FFN/output-normalization biases and every query-only term.
For arbitrary focal query and learned parameters, raw non-focal `V=0` therefore
implies exact elementwise `C=0`. A nonzero `C` demonstrates causal dependence on
non-focal inputs, not that the dependence is predictive or useful. Candidate
and null use identical raw-zero references, non-focal support, summary interface
and trainable parameter families; only registered versus fixed wrong-position
coordinates differ. The frozen identifier is
`POSITION_ALIGNED_CROSS_CONSTRUCT_CONSENSUS_V5`.

Before mutation-effect fitting, the set operator receives one narrow,
outer-train-only initialization stage. For each outer-train puzzle and each
eligible focal WT construct, the V14 corruption schedule deterministically
masks 40% of its observed positions. The frozen V14 encoder sees the masked
focal profile; the candidate mixer may use the other seven aligned WT profiles,
whereas the null receives the other seven WT profiles only at the fixed
wrong-position coordinates. A temporary `LayerNorm(256) -> Linear(256, 1)`
decoder reconstructs the masked focal WT values with L1 loss. Candidate and
null start from identical mixer/decoder states and use the same masks, puzzle
order, AdamW settings, matched residual/FFN dropout stream and epoch count. The
V14 encoder and shared point head remain bitwise unchanged. The
769-parameter decoder is frozen and never enters point prediction or residual
calibration. This initializes the new cross-construct operator using the exact
relationship that the WT-only audit established, without giving either arm
mutant outcomes.

For mutation source \(s\), receiver \(i\), and focal construct \(c\), define
`base` as the frozen focal V14 source/receiver features, signed distance,
mutation identity, feature41 and frozen V13 parent. Define
`cross=[C_{c,s},C_{c,i}]`. One shared head produces the paired increment

\[
g(x)=h([\mathrm{base},\mathrm{cross}])-h([\mathrm{base},0]).
\]

The two calls replay the same dropout mask in training and advance the random
stream as one ordinary head call. Therefore base-only behavior cancels exactly;
if the cross state is zero, \(g(x)=0\) even after arbitrary head updates. The
head's final layer is zero initialized and the prediction remains

\[
\widehat\Delta=\widehat\Delta_{V13}+g(x).
\]

Each arm has 6,171,697 total parameters: 4,767,280 frozen V14 encoder
parameters and 1,404,417 trainable position-aware mixer/head parameters. The
V13 parent is evaluated outside the module and never optimized. The temporary
769-parameter reconstruction decoder is a stage-specific training tool and is
not counted in the final prediction model.

Point fitting keeps the existing method-balanced signed-delta L1 objective and
uses one fixed warmup plus joint schedule. Epoch 0 updates only the shared point
head at learning rate `1e-3`; the context path is frozen and must remain bitwise
unchanged. Epochs 1–39 continue the head at `1e-3` and update the pretrained
context path at `3e-4`. Adam uses zero weight decay and gradient clipping `5.0`,
with no early stopping or checkpoint selection. Frozen-point residual
calibration then uses the existing puzzle-balanced CRPS objective; its gradients
cannot return to the point model. MGDA, absolute-delta auxiliaries, learned loss
weights and other objective changes are deferred to a separate amendment so
this experiment identifies representation rather than a compound loss change.

After point fitting, one outer-train-only retention diagnostic evaluates the
initial, post-pretraining and post-point context snapshots using the same final
frozen decoder and deterministic mask epoch `200`, which is disjoint from the
training mask epochs. It accepts only `{puzzle, contexts}` from the nineteen
outer-train puzzles, reports per-puzzle reconstruction L1 and then averages
equally over puzzles. Candidate qualification requires both
`pretraining_established` (post-pretraining mean L1 below initial) and
`retention_positive` (post-point state preserves a positive fraction of that
gain). The null receives the identical report, but its two retention booleans
are not Gates. The diagnostic performs no checkpoint, learning-rate,
architecture or threshold selection.

## 5. Verification and failure meaning

The implementation must establish:

- all eight constructs have one exact length before mixing;
- construct permutation changes only the corresponding construct axis;
- candidate representation changes under a registered-coordinate non-focal
  counterfactual; this verifies dependence only, not predictive value;
- the null focal output is invariant to a registered-position non-focal
  counterfactual when the corresponding shifted input remains fixed;
- candidate and null each expose seven non-focal individual K/V tokens plus one
  non-focal summary token, exclude the focal token from K/V, and have
  finite nonzero Q, K and V gradients on the same nondegenerate probe;
- neither arm adds the focal query as an attention residual;
- raw-zero references set all seven non-focal hidden/reactivity/observed streams
  to zero before learned projection and reuse the actual focal query exactly;
- actual and reference both traverse individual/summary projection, attention,
  FFN residual and output normalization;
- with arbitrary focal query and deliberately nonzero learned biases, raw-zero
  actual input cancels the V5 representation exactly in train and evaluation
  modes;
- actual/reference cross-block passes share one dropout draw and advance RNG as
  exactly one ordinary cross-block evaluation;
- attention-weight dropout is exactly zero in both arms;
- candidate/null state, total parameters, trainable parameters, input universe,
  optimizer, and random initialization match exactly;
- candidate summary statistics use only the seven registered-coordinate
  non-focal constructs, while null summaries use only their fixed
  17-position-shifted values;
- both arms replay the frozen V13 parent within `1e-7` before training;
- masked-WT batches contain only the nineteen outer-train `{puzzle, contexts}`
  records, exclude the held puzzle, and cannot carry mutant cells or targets;
- candidate and null use identical deterministic 40% masks, 200-epoch budgets,
  initialization and optimizer steps, including the real seven-eligible-context
  P20 case;
- pretraining changes the set operator but leaves the frozen V14 encoder and
  point head bitwise unchanged, preserves parent replay within `1e-7`, and
  freezes the 769-parameter decoder before point fitting;
- the shared point head computes `h(base,cross)-h(base,0)` under one replayed
  dropout draw, and zero cross cancels the residual exactly in train and
  evaluation modes even after nonzero head updates;
- point epoch 0 updates only the head at `1e-3` while context stays bitwise
  unchanged; epochs 1–39 update the head at `1e-3` and context at `3e-4`, for
  exactly 760 head and 741 context updates per arm;
- after the zero-initialized output layer has received its first update, a
  second supervised backward pass gives finite nonzero gradients to the
  construct projection and cross-construct attention while the frozen encoder
  still has no gradients;
- after two deterministic optimizer updates, perturbing only a non-focal WT
  value at the registered coordinate changes the candidate focal prediction by
  more than `1e-6`, whereas the matched null remains bitwise invariant when its
  shifted-coordinate input is unchanged;
- the frozen V14 encoder receives no gradient and remains bitwise unchanged;
- P20_Eterna stays in all 177 aligned context positions with observed flag zero
  and no fabricated supervised cell;
- the fixed outer-train-only epoch-200 retention diagnostic uses one final
  frozen decoder for initial/post-pretraining/post-point snapshots, leaves all
  model state, gradients and modes unchanged, and qualifies the candidate only
  when both `pretraining_established` and `retention_positive` are true; null
  retention is report-only.

Constructing the reference after projection, changing the focal query between
passes, skipping any layer in the reference path, using a second stochastic
draw, advancing RNG twice or failing exact raw-zero cancellation falsifies V5
and closes real-data eligibility. The point-head contrast cannot repair a
non-zero-preserving representation upstream.

This architecture is still implementation-only. If a future complete P1
candidate does not beat its position-deranged, zero-preserving V5 null under
the predeclared attribution Gate, the result means that aligned cross-design WT
context has not provided usable incremental signal under this frozen model and
protocol; the response is to terminate the family, not return to global
pooling, widen the mixer, or add a structure graph.

The counterfactual and gradient checks establish executable model capability,
not benchmark performance. They prevent a negative real-data result from being
misread when the intended cross-construct path was never trainable, but they do
not count as scientific evidence for P1.

### Outcome-blind real-input alignment audit

Before any P1 mutant outcome was scored, the reproducible WT-only audit at
`/mnt/cunyuliu/reactflow_delta_puzzle_set_meta_context/preflight/wt_alignment_audit_20260827.json`
compared registered coordinates with fixed circular wrong-position controls
`[1, 17, 43, 89]`. It used only WT sequence, WT reactivity and WT observation
masks. Its artifact explicitly records `mutant_outcome_used=false` and
`external_outcome_accessed=false`.

- mean pairwise sequence identity is only `0.510674`, so this is not a
  near-duplicate sequence ensemble;
- design-region same-position pair correlation exceeds the wrong-position
  control by `0.372299` on average and is positive in `20/20` puzzles;
- design-region leave-one-construct consensus correlation exceeds the control
  by `0.507915` on average and is positive in `20/20` puzzles;
- the corresponding full-construct increments are `0.359160` and `0.496006`,
  again positive in `20/20` puzzles;
- P20's sole design-region WT-missing construct receives observed aligned
  context from the other constructs at `100%` of its missing design positions.

This input audit supports the position-aligned inductive bias and rules against
terminating P1 for lack of aligned WT context. It does not establish that the
context predicts mutation response: only a complete candidate-minus-null P1M3
comparison can answer that question.

## 6. Proposed complete experiment and top-journal Gate

The executable implementation is fixed to a seed-0, folds 0–19 score-blind
screen with 200 masked-WT pretraining epochs, 40 point epochs and 40 calibration
epochs. Each fold uses the same fold's frozen V13 candidate seed-0 point and
frozen V14 candidate seed-0 encoder. The persistent controller only schedules
missing folds, preserves complete artifacts and moves interrupted prediction,
point, decoder and residual outputs aside before a clean fold restart. It
merges only after every worker exits and the full fold universe is present.

One pretraining epoch visits each outer-train puzzle once, yielding exactly
`200 x 19 = 3,800` WT-only puzzle updates per arm. It has no mutant target
exposure. One point epoch then visits every outer-train puzzle once and exposes
every available supervised cell once; 40 epochs retain the historical 40 target
exposures per cell and yield 760 puzzle-level Adam updates per arm. The first 19
updates (epoch 0) are head-only at `1e-3`; the remaining 741 updates train the
head at `1e-3` and context path at `3e-4`. Residual calibration adds another 760
puzzle-level updates with the point model frozen. The fold artifact records all
stage counts, histories, learning rates, warmup invariance and retention
diagnostics separately. The point stage does not inflate to 320 epochs merely
to match cell-level update count, because that would reuse each mutant target
eight times more often and confound the new capability with a much larger
outcome-exposure budget.

The inactive score-once path joins targets only after that merge. Before any
score access, the candidate retention report must have both
`pretraining_established=true` and `retention_positive=true`; failure terminates
qualification, while the null report never selects or blocks the model. An
eligible candidate must then simultaneously satisfy the existing top-journal
margins:

- signed-delta MAE: at least 12% versus feature41, 2% versus terminal V12,
  2% versus its frozen V13 parent and 1.5% versus the matched null;
- point absolute-delta MAE: at least 7% versus feature41, 2% versus terminal
  V11, 2% versus its frozen V13 parent and 1% versus the matched null;
- CRPS: at least 5% versus feature41, 2% versus terminal V12 and 1.5% versus
  the matched null;
- distribution absolute-delta MAE: at least 15% versus feature41, 2% versus
  terminal V10 and 1% versus the matched null;
- every paired comparison has puzzle-level 95% CI lower bound above zero;
- feature41 comparisons are positive on at least 16/20 puzzles and every
  historical-parent, frozen-parent and matched-null comparison on at least
  14/20;
- every headline comparison stays positive under leave-one-puzzle-out and no
  puzzle contributes more than 20% of the total effect;
- coverage is 100%, failure and unexpected keys are zero, and 68%/95%
  calibration error cannot worsen by more than one percentage point.

If and only if that complete P1M3 screen passes exactly, P1M4 is a new,
independent 20-fold by five-seed universe. Seed 0 is rerun rather than copied
from the screen. Each seed independently trains both the correctly aligned
zero-preserving V5 candidate and the position-deranged zero-preserving V5 matched null, each
with seven non-focal individual K/V tokens plus the summary, under the same
`200 + 1 head-only + 39 joint + 40 calibration` schedule. The formal point
prediction is the arithmetic mean of the five seed points. Candidate and null
probability predictions are ten-component Gaussian mixtures: each seed contributes its two
components and exactly one fifth of the total probability mass. CRPS, coverage
and distribution-derived absolute delta are recomputed from those ten
components; they are not averages of five single-seed scores.

The formal mixture must repeat every P1M3 margin, confidence interval,
positive-puzzle, leave-one-puzzle-out, influence, coverage and calibration
Gate above. It additionally requires candidate signed-delta and CRPS mean gains
against feature41 to be positive in at least four of five individual seeds.
All 100 fold-seed predictions must have complete key universes; no failed seed
may be deleted and no seed subset may be selected. The fixed V13 parent remains
a point comparator and feature41 probability scores remain frozen references;
neither is fabricated as a five-seed distribution. A ten-component mixture is
not required to have its median at the arithmetic mean point, so no such
post-hoc invariant is asserted.

The implementation-only P1M4 controller, assembler, scorer and qualifier are
present but cannot run under V14 authority. A future exact P1M3 PASS must be
followed by a focused authority commit that first opens P1M4 training and, only
after the complete prediction-only assembly, closes training and issues
`PUZZLE_SET_FORMAL_COMPLETE_SCORE_ONCE_ONLY`. This code path does not itself
authorize P1M4 or read any outcome.

These thresholds are implementation proposals frozen before any puzzle-set
outcome is scored. They do not authorize training, scoring or P1M4; a future
amendment must adopt them unchanged before real-data access.
