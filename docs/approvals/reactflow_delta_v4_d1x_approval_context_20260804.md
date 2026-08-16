# ReactFlow-Delta V4 D1-X 批准上下文记录（待用户签核）

- 记录时间：`2026-08-04`（UTC+08:00）
- 会话角色：当前 Codex 任务中的用户
- 待批准修正案：`docs/contracts/amendments/reactflow_delta_v4_d1x_draft_20260804.yaml`
- 修正案状态：`DRAFT_AWAITING_USER_SIGNOFF`
- 修正案 `approval_binding`：`null`（尚未绑定批准记录）

## 一、为什么需要本次签核

上一份批准记录 `docs/approvals/reactflow_delta_v4_approval_20260803.yaml` 的
`explicit_denials` 明确包含 `D1_X_CANONICALIZATION`，且其批准上下文写明
"本批准不授权 D1-X、canonical dataset、split、normalization/threshold fitting、
任何模型或基线训练、sealed test 访问、跨项目导出、湿实验、push 或 PR"。

因此 D0-X 关闭后，项目处于 `STOP_AWAIT_D1X_AUTHORITY_AMENDMENT` 状态。
**继续 D1-X canonicalization 必须由用户新签核本 D1-X 修正案。**

## 二、D1-X 前置条件核验（全部满足）

| 前置 | 修正案引用 sha256 | 实际产物 sha256 | 状态 |
|---|---|---|---|
| d0x_terminal_manifest | `bb4b3cf822b6084f784a55ba8c820d177579307a88cc08f18cb486c47c0374bc` | `terminal_manifest.yaml` = 同值 | PASS |
| d0x_terminal_sentinel | `a157a4e28d13837b41c35a761796b78c06b5f6b4c365109cb0c864fe978a5794` | `D0X_CLOSED.yaml` = 同值 | PASS |
| d0x_checksum_ledger | `4355f29b4d4acd2c8fa1e7469a83915c2ac91dd3090f058ca21e25233e3a757f` | `D0X_CLOSED.yaml` 内引用 = 同值 | PASS |
| d0x_inventory_audit | `81678203825f567081e1186e6711407279ddb1c404d5ab3add966a6b4f4deb73` | `inventory_audit.json` = 同值 | PASS |
| d0x_artifact_inventory | `42401d689fc6e19461e58573f35e4436c1fced120a070e759461babc8bb0cefb` | `artifact_inventory.json` = 同值 | PASS |
| d0x_closure_commit | `ee9d656` | git HEAD = `ee9d656` | PASS |

D0-X 终态：`D0X_CLOSED` / `gate_result: PASS` / `scientific_gate_result: NOT_RUN`。

## 三、D1-X 授权范围（修正案 grant）

- `READ_ONLY_PREFLIGHT`
- `D0X_CANDIDATE_INVENTORY_READ`
- `PROFILE_LEVEL_CANONICALIZATION`
- `EXACT_REF_ALT_COORDINATE_VERIFICATION`
- `PARENT_LINEAGE_AND_DESIGN_GROUP_SEPARATION`
- `CONDITION_MATCH_AUDIT`
- `RAW_UPSTREAM_TRAIN_FROZEN_LAYER_RETENTION`
- `MISSING_INVALID_UNMEASURED_NULL_MASK_REASON`
- `PROVISIONAL_ROLE_ASSIGNMENT`
- `NOISE_AND_TAIL_AUDIT`
- `CONTROLS_ONLY_NORMALIZATION_AND_SCALE_FITTING`

## 四、明确不包含的授权（修正案 explicit_denials）

- `D2_X_SPLIT_BUILD`
- `TRAINING_DATASET_FINALIZATION`
- `MODEL_OR_BASELINE_TRAINING`
- `CONFIRMATORY_TEST_UNSEAL`
- `TEST_OUTCOME_FITTING`
- `CROSS_PROJECT_EXPORT`
- `WET_LAB`
- `PULL_REQUEST`
- `PUSH`（由用户指令 9 另行管辖）

## 五、需要用户确认的要点

请确认是否批准 D1-X。若批准，将把 `approval_binding` 绑定到本批准记录，
把修正案状态从 `DRAFT_AWAITING_USER_SIGNOFF` 置为 `FROZEN`，然后按
`run_id=d1x_canonicalization_20260804_v1` 在隔离干净 worktree 中执行 D1-X。

若用户回复"批准执行"或同义确认，将据此生成绑定记录并推进。