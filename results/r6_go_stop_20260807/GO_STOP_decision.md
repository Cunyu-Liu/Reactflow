# GO/STOP Terminal Decision — run `r6_go_stop_20260807`

**Overall decision: `STOP`**  
**Route: `STOP_METHOD_ROUTE`**

Independent disk-evidence adjudication of §13.4 P0 gates (contract §13.2 R6). Any non-PASS / missing gate => STOP. Manual override to GO is not permitted.

## Gate verdicts

| Gate | Status | Primary evidence |
|---|---|---|
| AUTHORITY_CLOSED_PASS | PASS | contract_path=/home/cunyuliu/reactflow_delta_goal_20260729/configs/reactflow_delta/active_contract.yaml; sentinel_path=/home/cunyuliu/reactflow_delta_goal_20260729/configs/reactflow_delta/authority_epoch_14.sentinel.yaml; bundle_path=/home/cunyuliu/reactflow_delta_goal_20260729/configs/reactflow_delta/authority_epoch_14.bundle.sha256 |
| ASSET_DISPOSITION_1024_OF_1024 | PASS | jsonl_path=/home/cunyuliu/reactflow_delta_goal_20260729/data_registry/d0x_v2/asset_disposition_20260807.jsonl; summary_path=/home/cunyuliu/reactflow_delta_goal_20260729/data_registry/d0x_v2/asset_disposition_20260807.summary.json; n_rows=1024 |
| PRIMARY_MASK_V2_PASS | PASS | primary_pairs_path=/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/d1x_v2/d1x_v2_canonicalization_20260807T1830+0800/primary_pairs_v2.jsonl; summary_path=/home/cunyuliu/reactflow_delta_goal_20260729/data_registry/d1x_v2/d1x_v2_summary.json; n_primary_pairs=7961 |
| GROUP_ATOMS_AND_PUBLICATION_SPLIT_PASS | PASS | overlap_path=/home/cunyuliu/reactflow_delta_goal_20260729/data_registry/d2x_v2/overlap_report.json; split_path=/home/cunyuliu/reactflow_delta_goal_20260729/configs/reactflow_delta/split_v2.yaml; publication_overlap:train_vs_validation:n_intersection=0 |
| OLD_TEST_RETIRED_NEW_TEST_UNTOUCHED | PASS | split_path=/home/cunyuliu/reactflow_delta_goal_20260729/configs/reactflow_delta/split_v2.yaml; retired_test={'studies': ['16SFWJ'], 'status': 'DEVELOPMENT_CONSUMED', 'reason': 'old d2x test (pmid_25183835); cannot be re-used as confirmatory'}; new_test={'studies': ['SL5CV2', 'SL5HKU', 'SL5MER'], 'publication': 'UNKNOWN_PUBLICATION:SL5CV2', 'untouched': True, 'confirmatory_blocker': {'blocked': True, 'reason': 'frozen RMDB snapshot does not assert PMID per asset. The only CERTIFIED publications are the OLD split manifest entries, ALL of which were exposed to development (old train/val/test). New v2 studies resolve to UNKNOWN_PUBLICATION:<study> and cannot be certified as distinct untouched confirmatory publications from frozen metadata alone.', 'untouched_certified_publications': [], 'exposed_certified_publications': ['pmid_24469816', 'pmid_25183835', 'pmid_25303992', 'pmid_25883046', 'pmid_29446752', 'pmid_35982307', 'pub_RNAPuzzle18_daslab'], 'unknown_publication_studies': ['5SRRNA', 'ADDRSW', 'AGSARNA', 'CL1LIG', 'GLYCFN', 'HC16M2R', 'MDLOOP', 'RNAPZ6', 'RNASEP', 'SL5CV2', 'SL5HKU', 'SL5MER', 'SRPDIV', 'TRNAPH'], 'satisfies_ge3_untouched': False, 'recommendation': 'resolve publication identity for new studies (e.g. via RMDB entry citation / d0r_accession_registry, where per-entry PMIDs exist -- the SL5 family carries pmid_38427602) or designate a prospective confirmatory alternative before confirmatory CI.'}} |
| CALLER_V2_FOLD_LOCAL_AND_RELIABLE | PASS | caller_path=/home/cunyuliu/reactflow_delta_goal_20260729/scripts/reactflow_delta/caller_v2.py; test_path=/home/cunyuliu/reactflow_delta_goal_20260729/tests/reactflow_delta/test_caller_v2.py; checks={'fold_local_guard_present': True, 'no_call_present': True, 'seal_present': True} |
| EVALUATOR_V2_REFERENCE_TESTS_PASS | PASS | evaluator_path=/home/cunyuliu/reactflow_delta_goal_20260729/scripts/reactflow_delta/evaluate_v2.py; test_path=/home/cunyuliu/reactflow_delta_goal_20260729/tests/reactflow_delta/test_evaluate_v2.py; tests_passed=True |
| P2_LEARNABILITY_GO | FAIL | verdict_path=/home/cunyuliu/reactflow_delta_goal_20260729/results/p2_v1_learnability_20260808/P2_learnability_verdict.json; verdict=STOP_METHOD_ROUTE; estimand_status_primary=UNIDENTIFIABLE |

## Blocking Phase 3

- `P2_LEARNABILITY_GO` — FAIL

## Recommendation

Phase 3 model-architecture iteration is BLOCKED. The R5 P2 verdict is STOP_METHOD_ROUTE: the primary binary-changer estimand is UNIDENTIFIABLE under the frozen caller_v2 / d1x_v2 data (caller/null-calibration artifact: 3 changers / 3178 non / 3204 NO_CALL). Per §13.4 any non-PASS gate stops Phase 3. Fix requires per-study reactivity normalization + error recalibration to a common scale and a caller_v3/endpoint_v3 amendment under a new authority epoch (NOT a silent in-place change).

Machine manifest: `results/r6_go_stop_20260807/go_stop_terminal.json`

