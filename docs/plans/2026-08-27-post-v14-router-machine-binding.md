# Post-V14 first-matching router machine binding

**Status:** `PRE_V14_SCORE_FROZEN_MACHINE_BINDING_NO_TRAINING_AUTHORITY`

**Evidence boundary:** this binding was written while V14M3 remained score-blind.
No V14 fold result, loss, partial metric, complete score, Gate direction or new
external outcome was read. It mechanically implements the contingency frozen at
`98d8eb519fb7c69d8a489a44ff72380204d6599c`; it does not change a V14 Gate.

## Focal question

Which first matching post-V14 branch follows from one complete V14M3 score and
qualification, and what single residual diagnostic controls branch 6 without
choosing a statistic after seeing V14 outcomes?

## Located evidence

- `LOCATED_EVIDENCE`: the V14 qualification already contains all twelve
  candidate comparisons, including relative gain, 20-puzzle CI, direction
  count, leave-one-puzzle-out direction and influence.
- `LOCATED_EVIDENCE`: branch 5 additionally needs four from-scratch-null versus
  historical point margins. The complete V14 score contains the required null,
  feature41, V11 and V12 puzzle rows; the existing `paired_summary` estimator is
  sufficient and no new threshold is needed.
- `LOCATED_EVIDENCE`: post-V11 D4, frozen before V14 at commit
  `7468f1e066c7e1f80aae326bcadc41f0349f172a`, already defines a fixed-point
  predictive-tail diagnostic with exactly twenty held puzzles, a two-sided 95%
  Student-t CI and a minimum 14/20 same-direction rule.
- `LOCATED_EVIDENCE`: V9 also defines normalized residual-quantile asymmetry,
  but its historical eligibility helper does not itself require all twenty
  finite puzzles and it diagnoses residual quantiles rather than the already
  fitted V14 predictive distribution.

## Router decision

`DECISION`: `scripts/reactflow_delta/route_post_v14_model_contingency.py` is the
single executable first-matching router. It accepts only one complete V14M3
score and one exact V14M3 qualification, computes the four missing null
historical summaries with the frozen estimator, evaluates branches in order
`1,2,3,4,5,6,P3`, writes one atomic artifact and refuses overwrite.

The runtime does not trust the supplied qualification as a second source of
truth. It validates the exact 20-row score fields used by this router, calls the
existing `qualify_model_rescue_v14.qualify(score)`, requires the supplied and
recomputed qualification dictionaries to be structurally equal, and routes
from the recomputed dictionary. JSON booleans and integers are type-exact; for
example, integer `1` is not accepted as a Gate boolean.

The machine predicates are the literal implementation of section 4 of the
contingency plan:

1. exact screen PASS opens only V14M4;
2. invalid identity, universe, coverage, aggregation, finite score or
   provenance is an audit repair branch;
3. all candidate-versus-feature41/historical comparisons pass, but at least one
   candidate-versus-null attribution comparison fails;
4. all candidate-versus-null comparisons pass, but at least one registered
   historical point margin fails;
5. candidate and null both miss at least one registered historical point
   margin, or candidate point direction/influence stability fails;
6. every signed and point-absolute comparison passes, while CRPS or
   distribution-absolute remains the failed headline family;
7. otherwise stop model rescue (`P3`).

First-match precedence is controlling when raw predicates overlap. No branch
may authorize both P1 and P2.

Branch 2 is deliberately not a scientific contingency result. Invalid score,
qualification, identity, universe, coverage, aggregation or provenance writes
an `ENGINEERING_EVIDENCE_AUDIT` artifact with status
`POST_V14_TERMINAL_INPUT_AUDIT_FAILURE`, returns nonzero, and does not emit or
select scientific `P3`. Its only allowed action is repair of the identified
engineering fault in the same frozen V14 universe.

## Branch-6 primary diagnostic

Two pre-existing candidates were considered without V14 outcomes:

- post-V11 D4 lower-minus-upper 90% tail miss;
- V9 normalized residual-quantile asymmetry.

`DECISION`: bind exactly one primary, `LOWER_MINUS_UPPER_TAIL_MISS90`, from
post-V11 D4. It most directly tests the branch-6 assumption that a fitted
conditional distribution has directional tail misspecification after point
Gates pass, already enforces all twenty independent puzzle units and requires
no point update. V9 quantile asymmetry is not a route-controlling alternative;
it cannot be selected later because it looks more favorable.

For each held puzzle, let the registered V14 candidate mixture define

```text
F_i(y_i) = sum_k mixture_weight_ik * Phi((target_i - location_ik) / scale_ik)
```

Use the frozen hierarchy `equal method -> equal mutant within method -> equal
qualified position within mutant`. Define

```text
lower_tail_miss90 = weighted mean[ F_i(y_i) < 0.05 ]
upper_tail_miss90 = weighted mean[ F_i(y_i) > 0.95 ]
S_puzzle = lower_tail_miss90 - upper_tail_miss90
```

Across exactly 20 held puzzles, compute

```text
mean(S) +/- t(0.975, 19) * sd(S, ddof=1) / sqrt(20)
```

The diagnostic is exact eligible only when the CI lies wholly on one side of
zero and at least 14/20 puzzle values have that same sign. A CI touching zero or
at most 13 puzzles in either direction is a valid exact diagnostic FAIL and
routes to P3. Fewer than twenty finite puzzle statistics is a measurement audit
failure: it raises before output and is not scientific evidence for P3.

## Assumptions and adversarial review

- `ASSUMPTION`: asymmetric 90% tail misses reflect residual distribution shape
  not repaired by the existing median-preserving V10 family.
- `ALTERNATIVE_EXPLANATION`: tail imbalance could be residual point bias or
  development-set noise. Branch 6 is therefore available only after every
  signed and point-absolute Gate passes; even a positive diagnostic grants only
  a later focused P2 amendment, not a result claim.
- `MEASUREMENT_FAILURE`: missing keys, non-finite values, fewer than twenty
  puzzles, nonpositive scales, weights that do not sum to one, or any point
  mutation stop the diagnostic before it writes an artifact. The condition
  remains an engineering/evidence audit failure and cannot be recorded as a
  scientific diagnostic FAIL or P3 result.
- `GENERALIZABILITY_LIMIT`: the same twenty development puzzles have been
  repeatedly consumed. A diagnostic PASS is route eligibility only, never
  external confirmation, SOTA, mechanism or publication evidence.

## Authority effect

This document and its runtime have no training authority. Both CLIs require the
focused active-contract path explicitly. They resolve every input and output
path and require exact equality with the canonical absolute paths in the active
authority. The supplied contract path must also equal the executing repository's
own `configs/reactflow_delta/active_contract.yaml`; that pointer must have schema
`reactflow_delta.active_contract.v14`, project task
`reactflow_delta_model_rescue_v14`, and `runnable_phases: [V14M3]`. A copied or
alternate YAML cannot issue either token. The router records those same resolved CLI score and qualification
paths in `source_artifacts`. Branch 6 first requires a complete V14 terminal
handoff and a router artifact selecting exact branch `6`; the diagnostic
recomputes the entire expected router from the same score and qualification and
requires full dictionary equality, not only matching branch metadata.

The router authority issued only after canonical V14 score and qualification
exist must keep `current_phase=V14M3`, close training, partial score and external
outcome access, and bind exactly:

```yaml
held_score_read_allowed: POST_V14_FIRST_MATCHING_ROUTER_ONCE_ONLY
next_allowed_action: RUN_SINGLE_POST_V14_FIRST_MATCHING_ROUTER
post_v14_router_authority:
  runtime_authority_token: POST_V14_FIRST_MATCHING_ROUTER_ONCE_ONLY
  complete_score_path: /mnt/cunyuliu/reactflow_delta_model_rescue_v14/v14m3_screen_seed0/v14m3_complete_score.json
  qualification_path: /mnt/cunyuliu/reactflow_delta_model_rescue_v14/v14m3_screen_seed0/v14m3_qualification.json
  router_output_path: /mnt/cunyuliu/reactflow_delta_model_rescue_v14/v14m3_screen_seed0/post_v14_first_matching_route.json
```

The output path is the only once-only identity: the runtime refuses overwrite
and adds no lock, digest or retry framework. After that canonical artifact is
written, the terminal authority commit closes the router token before any next
branch action.

The branch-6 diagnostic is read-only: it may join target only after complete
score access, may not train, update or rewrite point/distribution predictions,
and writes one diagnostic artifact. Exact diagnostic PASS can support a later
focused P2 authority; a valid exact diagnostic FAIL routes to P3. Invalid input
writes no diagnostic artifact and remains an engineering/evidence audit
failure.

The later focused diagnostic authority must bind exactly:

```yaml
held_score_read_allowed: POST_V14_BRANCH6_TAIL_DIAGNOSTIC_ONCE_ONLY
next_allowed_action: RUN_SINGLE_POST_V14_BRANCH6_TAIL_DIAGNOSTIC
post_v14_branch6_diagnostic_authority:
  runtime_authority_token: POST_V14_BRANCH6_TAIL_DIAGNOSTIC_ONCE_ONLY
  complete_unscored_merge_path: /mnt/cunyuliu/reactflow_delta_model_rescue_v14/v14m3_screen_seed0/v14m3_complete_unscored_merge.json
  complete_score_path: /mnt/cunyuliu/reactflow_delta_model_rescue_v14/v14m3_screen_seed0/v14m3_complete_score.json
  qualification_path: /mnt/cunyuliu/reactflow_delta_model_rescue_v14/v14m3_screen_seed0/v14m3_qualification.json
  router_path: /mnt/cunyuliu/reactflow_delta_model_rescue_v14/v14m3_screen_seed0/post_v14_first_matching_route.json
  m2_csv_path: /mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/openknot_m2/OK7a_M2_data.v4.5.2.csv
  diagnostic_output_path: /mnt/cunyuliu/reactflow_delta_model_rescue_v14/v14m3_screen_seed0/post_v14_branch6_tail_diagnostic.json
```

The canonical output remains under `/mnt/cunyuliu`; it is never committed to
Git. A valid diagnostic FAIL may return a nonzero process status after writing
its exact FAIL artifact and must not be rerun. After either valid PASS or valid
FAIL output, the terminal commit closes the diagnostic token.
