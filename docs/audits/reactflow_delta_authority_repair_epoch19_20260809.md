# Authority Integrity Repair — Epoch 19 (Phase 3 Fail-Closed Closure + Benchmark/Resource Pivot)

日期：2026-08-09　仓库：`/home/cunyuliu/reactflow_delta_goal_20260729`　分支：`codex/reactflow-delta-d0r`
授权：用户显式授权「修复 authority 完整性（terminalize、重绑 bundle/sentinel）」

## 1. 背景：审计 §2.1 的硬伤与当前真实状态

`ReactFlowDelta_M0X_strict_scientific_engineering_audit_20260807.md` §2.1 指出 authority 断裂
（旧快照：epoch 12 绑定 epoch 5、bundle 对 active manifest 校验失败）。但此后治理已通过
epoch 14–18 逐级重建 bundle+sha256+sentinel。本次修复前只读核查确认：

- **M0-X 已 TERMINAL / FAIL**（`M0X_FAILED_NO_PASS_SENTINEL`）——审计 §2.1 要求的「terminalize
  M0-X FAIL」**已经实现**，本次保持不变、不触碰冻结记录。
- 唯一仍开放（RUNNING / NOT_RUN / execution_authorized=true）的阶段是 **PHASE3-ARCH**（epoch 18）。

因此本次 authority 修复的实质动作是：**把最后一个开放阶段 PHASE3-ARCH 正式 terminalize 为
fail-closed 退役，pivot 到 benchmark/resource 路线，并重绑到新的 epoch 19 bundle+sentinel。**

## 2. 修复动作

- 新增 amendment：`docs/contracts/amendments/reactflow_delta_v4_epoch19_phase3_closure_20260809.yaml`
- 新增 approval：`docs/approvals/reactflow_delta_v4_epoch19_phase3_closure_approval_20260809.yaml`
- 更新 `configs/reactflow_delta/active_contract.yaml`：
  - authority epoch 18 → **19**；current_phase `PHASE3-ARCH` → **BENCHMARK-RESOURCE**
  - PHASE3-ARCH → `TERMINAL / FAIL / execution_authorized=false`，
    `terminal_sentinel_status=PHASE3_ARCH_FAIL_CLOSED_RETIRED`
  - 新增 BENCHMARK-RESOURCE 阶段（RUNNING，execution_authorized=true，training=false，gpu=false）
  - `training_allowed=false`（benchmark/resource 为 CPU-only 统计诊断，无模型训练）
  - governance_resolution 记录 Phase 3 全 fail-closed + pivot
  - integrity 指向 epoch 19 bundle/sentinel
- 生成 `authority_epoch_19.bundle.sha256`（6 成员）+ `authority_epoch_19.sentinel.yaml`

## 3. Phase 3 闭卷依据（已预注册、fail-closed）

三方案均在嵌套 leave-one-publication-out、5 paired seeds、capacity-matched generic 下执行，
因 paired publication-block bootstrap CI low ≤ 0 而 fail-closed 退役：

| 方案 | mean skill | generic mean skill | 裁决 |
|---|---|---|---|
| pair_v1 DeepSets | 0.684 | 0.679 | FAIL_CLOSED_RETIRED |
| exact_alt_v1 显式交互 | 0.722 | 0.679 | FAIL_CLOSED_RETIRED |
| epro_v2 修复传播 | 0.657 | 0.679 | FAIL_CLOSED_RETIRED |

依据合同 stop rule 转 benchmark/resource 路线，保留最简单 generic 为 development default，
**禁止 dev13+ free search**。闭卷文档：
`docs/audits/reactflow_delta_phase3_closure_fail_closed_20260809.md`。

## 4. 验收（独立重算，全部 PASS）

| 校验项 | 结果 |
|---|---|
| sentinel.active_manifest_sha256 == sha256(active_contract.yaml) | PASS |
| sentinel.bundle_ledger_sha256 == sha256(authority_epoch_19.bundle.sha256) | PASS |
| bundle 6 成员逐文件 sha256 重算全部一致（mismatch=0） | PASS |
| integrity 指针指向 epoch 19 bundle/sentinel | PASS |
| PHASE3-ARCH = TERMINAL/FAIL/exec_authorized=false | PASS |
| M0-X = TERMINAL/FAIL（冻结保留） | PASS |
| BENCHMARK-RESOURCE = RUNNING/exec_authorized=true/training=false | PASS |
| authority_epoch=19，current_phase=BENCHMARK-RESOURCE，training_allowed=false | PASS |
| **总体完整性裁决** | **PASS** |

## 5. 后续（需新授权）

- benchmark/resource 路线为 CPU-only 论文证据链（caller reliability / label shift /
  magnitude-vs-noise / per-publication uniformity），已入库。
- 任何新的**模型训练阶段**需新 authority epoch（epoch 19 之后）并保持 `training_allowed` 语义
  与 test-sealed 不变；解除该约束需用户显式授权。
