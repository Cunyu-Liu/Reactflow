# Independent RNet distillation screen result

- Qualification status: `RNET_DISTILL_TOP_JOURNAL_DEVELOPMENT_SCREEN_FAIL`
- Gate passed: `false`
- Integrity passed: `true`
- Evidence ceiling: `EXPOSURE_DISCLOSED_DEVELOPMENT_ONLY`
- Publication ready: `false`
- Recorded at: `2026-08-29T15:30:00+08:00`

## Provenance and fixed universe

- Experiment ID: `RND3_RNET_DISTILL_COMPLETE_SEED0_PREDICTION_ONLY`
- Authority branch: `codex/reactflow-delta-independent-rnet-distill-20260828`
- Finalizer source commit: `14b83d98fdfdefd1e1e819ad181d5380f09104a9`
- Source run commits: `fe8c36ccb63de89e0383baccf453a3b6cb6413cf`
- Fold universe: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]`
- Seed universe: `[0]`
- Downstream schedule: `40+40` point/calibration epochs
- Training devices recorded by canonical fold results: `['cuda:0']`
- GPU names recorded by canonical fold results: `['NVIDIA A100-PCIE-40GB', 'NVIDIA A100-PCIE-40GB MIG 1g.5gb']`
- Earliest fold start: `2026-08-29T02:37:10.424717+00:00`
- Latest fold finish: `2026-08-29T05:03:39.889222+00:00`

## Canonical records

- Complete target-free merge: `/mnt/cunyuliu/reactflow_delta_independent_rnet_distill/rnd3_screen_seed0/rnet_distill_complete_unscored_merge.json`
- Complete score: `/mnt/cunyuliu/reactflow_delta_independent_rnet_distill/rnd3_screen_seed0/rnet_distill_complete_score.json`
- Qualification: `/mnt/cunyuliu/reactflow_delta_independent_rnet_distill/rnd3_screen_seed0/rnet_distill_qualification.json`
- Exact per-fold runner commands: recorded in the canonical merge `/mnt/cunyuliu/reactflow_delta_independent_rnet_distill/rnd3_screen_seed0/rnet_distill_complete_unscored_merge.json` under `folds[*].command`; not duplicated into the decision ledger.

## Frozen Gate record

| Gate | Result |
| --- | --- |
| candidate_coverage95_in_frozen_interval | FAIL |
| distribution_absolute_gain_vs_feature41_ge_frozen_minimum | FAIL |
| distribution_absolute_gain_vs_historical_parent_ge_frozen_minimum | FAIL |
| distribution_absolute_gain_vs_matched_null_ge_frozen_minimum | FAIL |
| duplicate_or_unexpected_artifacts_eq_0 | PASS |
| exact_fold_count_20 | PASS |
| failed_rows_eq_0 | PASS |
| historical_parent_ci_lower_each_gt_zero | FAIL |
| historical_parent_positive_puzzles_each_ge_14 | FAIL |
| matched_null_ci_lower_each_gt_zero | FAIL |
| matched_null_positive_puzzles_each_ge_14 | FAIL |
| max_single_puzzle_effect_fraction_all_comparisons_le_0_20 | FAIL |
| point_absolute_gain_vs_feature41_ge_frozen_minimum | FAIL |
| point_absolute_gain_vs_historical_parent_ge_frozen_minimum | FAIL |
| point_absolute_gain_vs_matched_null_ge_frozen_minimum | FAIL |
| prediction_and_score_integrity | PASS |
| registered_prediction_coverage_eq_1 | PASS |
| signed_delta_gain_vs_feature41_ge_frozen_minimum | FAIL |
| signed_delta_gain_vs_historical_parent_ge_frozen_minimum | FAIL |
| signed_delta_gain_vs_matched_null_ge_frozen_minimum | FAIL |
| task_crps_gain_vs_feature41_ge_frozen_minimum | FAIL |
| task_crps_gain_vs_historical_parent_ge_frozen_minimum | FAIL |
| task_crps_gain_vs_matched_null_ge_frozen_minimum | FAIL |

## Failure or indeterminacy reasons

- Failed Gate: `candidate_coverage95_in_frozen_interval`
- Failed Gate: `distribution_absolute_gain_vs_feature41_ge_frozen_minimum`
- Failed Gate: `distribution_absolute_gain_vs_historical_parent_ge_frozen_minimum`
- Failed Gate: `distribution_absolute_gain_vs_matched_null_ge_frozen_minimum`
- Failed Gate: `historical_parent_ci_lower_each_gt_zero`
- Failed Gate: `historical_parent_positive_puzzles_each_ge_14`
- Failed Gate: `matched_null_ci_lower_each_gt_zero`
- Failed Gate: `matched_null_positive_puzzles_each_ge_14`
- Failed Gate: `max_single_puzzle_effect_fraction_all_comparisons_le_0_20`
- Failed Gate: `point_absolute_gain_vs_feature41_ge_frozen_minimum`
- Failed Gate: `point_absolute_gain_vs_historical_parent_ge_frozen_minimum`
- Failed Gate: `point_absolute_gain_vs_matched_null_ge_frozen_minimum`
- Failed Gate: `signed_delta_gain_vs_feature41_ge_frozen_minimum`
- Failed Gate: `signed_delta_gain_vs_historical_parent_ge_frozen_minimum`
- Failed Gate: `signed_delta_gain_vs_matched_null_ge_frozen_minimum`
- Failed Gate: `task_crps_gain_vs_feature41_ge_frozen_minimum`
- Failed Gate: `task_crps_gain_vs_historical_parent_ge_frozen_minimum`
- Failed Gate: `task_crps_gain_vs_matched_null_ge_frozen_minimum`

## Canonical paired summaries

| Comparison | Comparator mean | Candidate mean | Mean gain | Relative gain | 95% CI | Positive puzzles | Max single-puzzle fraction |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| distribution_absolute_vs_feature41 | 0.141267359694 | 0.135531038192 | 0.00573632150209 | 0.0406061351647 | [0.00402437285652, 0.00744827014765] | 19 | 0.104716634941 |
| distribution_absolute_vs_historical_parent | 0.131147770762 | 0.135531038192 | -0.0043832674297 | -0.0334223555935 | [-0.0055781555991, -0.0031883792603] | 1 | 0.111493095318 |
| distribution_absolute_vs_matched_null | 0.136445694551 | 0.135531038192 | 0.000914656359172 | 0.00670344610127 | [-0.000810277069636, 0.00263958978798] | 12 | 0.479841421838 |
| point_absolute_vs_feature41 | 0.154866203674 | 0.147481020336 | 0.00738518333838 | 0.0476875080759 | [0.00486406147271, 0.00990630520405] | 18 | 0.147761174538 |
| point_absolute_vs_historical_parent | 0.142370261183 | 0.147481020336 | -0.00511075915233 | -0.035897659454 | [-0.00916263135099, -0.00105888695368] | 3 | 0.37099899867 |
| point_absolute_vs_matched_null | 0.148671653142 | 0.147481020336 | 0.00119063280596 | 0.00800847223258 | [-0.00184062990954, 0.00422189552145] | 11 | 1.00218978227 |
| signed_delta_vs_feature41 | 0.191159730016 | 0.176795070195 | 0.0143646598206 | 0.0751448007352 | [0.0106503073603, 0.018079012281] | 20 | 0.0933732043081 |
| signed_delta_vs_historical_parent | 0.169841206187 | 0.176795070195 | -0.00695386400794 | -0.0409433267936 | [-0.0104055366868, -0.00350219132905] | 1 | 0.172497116598 |
| signed_delta_vs_matched_null | 0.179862786259 | 0.176795070195 | 0.00306771606353 | 0.0170558686838 | [-0.000461060572652, 0.00659649269971] | 11 | 0.281528950342 |
| task_crps_vs_feature41 | 0.130579558952 | 0.127135622905 | 0.00344393604788 | 0.0263742355657 | [0.00235651246016, 0.00453135963561] | 20 | 0.102399433709 |
| task_crps_vs_historical_parent | 0.124625602239 | 0.127135622905 | -0.00251002066556 | -0.0201404897587 | [-0.00382964764124, -0.00119039368989] | 3 | 0.224759681234 |
| task_crps_vs_matched_null | 0.127653721341 | 0.127135622905 | 0.000518098436476 | 0.00405862383825 | [-0.000434740978596, 0.00147093785155] | 12 | 0.461629550054 |

## Canonical calibration

```json
{
  "coverage95": {
    "candidate": 0.9224401131331866,
    "frozen_interval": [
      0.94,
      0.96
    ],
    "within_interval": false
  }
}
```

## Claim boundary

Allowed claims:

- The complete fixed RND5 development screen did not pass; RND6 formal confirmation was not run.
- The result is `EXPOSURE_DISCLOSED_DEVELOPMENT_ONLY` on the disclosed, repeatedly consumed development benchmark.

Prohibited claims:

- Clean out-of-distribution evidence is not established.
- Independent external replication is not established.
- State of the art is not established.
- Publication readiness is false.
- Training loss, smoke output, prediction coverage, and engineering checks are not scientific conclusions.
