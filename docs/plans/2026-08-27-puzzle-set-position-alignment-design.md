# Position-aligned puzzle-set context design

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
position-aligned cross-construct conditioning both legal and materially more
informative than global mean pooling. No mutant outcome column was used for this
audit.

## 3. Alternatives

### A. Retain one global token per construct

Rejected. It is inexpensive and permutation equivariant, but it supplies no
receiver-specific or source-specific cross-construct information. Increasing
its width would not repair that missing indexing ability.

### B. Position-aligned cross-construct attention

Selected. For every full-sequence coordinate independently, the eight frozen
V14 hidden states and their observed flags form an unordered set. Candidate
attention can exchange information across all eight constructs at that
coordinate. The matched null executes the same projection, attention, FFN, and
normalization but masks every off-diagonal construct edge.

The focal incremental head receives both the aligned mixed state at the mutation
source and the aligned mixed state at each receiver. The frozen V14 local source
and receiver states, signed distance, mutation identity, feature41, and frozen
V13 parent point remain explicit inputs.

### C. Target-structure graph alignment

Rejected for this experiment. It would introduce a second new capability and a
new metadata dependency after several structure/contact routes already produced
negative or sub-threshold evidence. Full-position alignment directly tests the
registered eight-design relationship with a cleaner null.

## 4. Frozen architecture

For hidden states \(H\in\mathbb R^{8\times177\times256}\), concatenate the WT
observed indicator and project each aligned state:

\[
Z_{c,j}=W_p[H_{c,j};O_{c,j}].
\]

For each position \(j\), apply the same eight-token attention block:

\[
\widetilde Z_{:,j}=
\operatorname{SetBlock}(Z_{:,j}).
\]

Candidate uses full eight-by-eight attention. Null uses an identity attention
mask. No construct-order or method embedding is present. Position is the batch
axis of the set block, so the cross-construct operator cannot mix position
\(j\) with a different position \(k\); within-construct positional context has
already been encoded by the frozen six-layer V14 encoder.

For mutation source \(s\), receiver \(i\), and focal construct \(c\), the
trainable increment uses

\[
g(x)=\operatorname{MLP}[
f_{V14}(c,s,i),
\widetilde Z_{c,s},
\widetilde Z_{c,i},
\widehat\Delta_{V13}(c,s,i)
].
\]

Its final layer is zero initialized and the prediction remains

\[
\widehat\Delta=\widehat\Delta_{V13}+g(x).
\]

Each arm has 6,170,417 total parameters: 4,767,280 frozen V14 encoder
parameters and 1,403,137 trainable position-aware mixer/head parameters. The
V13 parent is evaluated outside the module and never optimized.

## 5. Verification and failure meaning

The implementation must establish:

- all eight constructs have one exact length before mixing;
- construct permutation changes only the corresponding construct axis;
- candidate focal output depends on non-focal input at the same coordinate;
- that dependency is exactly zero at all different coordinates;
- null focal output has zero dependency on every non-focal construct;
- candidate/null state, total parameters, trainable parameters, input universe,
  optimizer, and random initialization match exactly;
- both arms replay the frozen V13 parent within `1e-7` before training;
- after the zero-initialized output layer has received its first update, a
  second supervised backward pass gives finite nonzero gradients to the
  construct projection and cross-construct attention while the frozen encoder
  still has no gradients;
- after two deterministic optimizer updates, perturbing only a non-focal WT
  profile changes the candidate focal prediction by more than `1e-6`, whereas
  the matched null remains bitwise invariant;
- the frozen V14 encoder receives no gradient and remains bitwise unchanged;
- P20_Eterna stays in all 177 aligned context positions with observed flag zero
  and no fabricated supervised cell.

This architecture is still implementation-only. If a future complete P1
candidate does not beat its block-diagonal null under the predeclared
attribution Gate, the result means that aligned cross-design WT context lacks
usable incremental signal; the response is to terminate the family, not return
to global pooling, widen the mixer, or add a structure graph.

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
screen with 40 point epochs and 40 calibration epochs. Each fold uses the same
fold's frozen V13 candidate seed-0 point and frozen V14 candidate seed-0
encoder. The persistent controller only schedules missing folds, preserves
complete artifacts and moves interrupted partial outputs aside before a clean
fold restart. It merges only after every worker exits and the full fold
universe is present.

One epoch visits every outer-train puzzle once and therefore exposes every
available supervised cell once; 40 epochs retain the same 40 target exposures
per cell as the historical point protocol. This yields 760 puzzle-level Adam
updates per arm rather than 6,080 cell-level updates. The implementation records
both counts explicitly. It does not inflate to 320 epochs merely to match update
count, because that would reuse each target eight times more often and confound
the new capability with a much larger outcome-exposure budget.

The inactive score-once path joins targets only after that merge. Candidate
must simultaneously satisfy the existing top-journal margins:

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

These thresholds are implementation proposals frozen before any puzzle-set
outcome is scored. They do not authorize training, scoring or P1M4; a future
amendment must adopt them unchanged before real-data access.
