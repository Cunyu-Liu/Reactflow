# ReactFlow-Delta prospective-v2：development 结果 claim-evidence map (2026-08-13)

> 范围：P0–P3 development benchmark 结论。P4/P5 需外部数据/复现；P6 需 release token。
> 状态边界：`DEVELOPMENT_ONLY / SOTA_NOT_ESTABLISHED / PUBLICATION_NOT_READY`

## 主结论（仅 development benchmark）
| Claim | 状态 | 证据 |
|---|---|---|
| P2：direct 前瞻信号存在（outcome-blind, all-mutant, unseen-puzzle） | `DIRECT_DEVELOPMENT_LEARNABILITY_PASS` | 唯一 `Direct*`(reg_direct) vs `T*`：20-puzzle 95% CI lower = **+0.0079 > 0**；20/20 D_p 为正；sign-flip p=2e-6；LOP max shift 0.0011（`p2_direct_v2_result_20260813.json`） |
| P3：LRSO 增量技能（规范重跑后） | `LRSO_EXCEEDS_DIRECT_FOR_DEVELOPMENT`（合同 12.5 PASS） | 2026-08-15 规范重跑 `run_p3_lrso_v3.py`（trainable encoder + masked NLL + inner 4-fold validation + 五 seed mixture）：rank2 D_p^P3=+0.0147 [95% CI +0.0119,+0.0175]、rank4 +0.0155 [+0.0113,+0.0196]、rank8 +0.0154 [+0.0122,+0.0185]；ci_low_gt_0=True；20/20 puzzle 正向；sign-flip p=1.9e-6；LOO max shift ≤0.001（`p3_lrso_v3_result_20260815.json`）。v1/v2 的 `NO_INCREMENTAL_LRSO_SKILL` 已撤回（实现失败伪影：encoder detach 成固定特征；missing target 填 0 后误判全部可观测；单 seed/6 epoch/无 inner validation）。 |
| 横向对比（development） | `CONFIRMED_FACT`（结果） | reg_direct CRPS 0.2023（+5.92% vs ZeroResponse）；MLP 变体 0.2718（−26.4%，过拟合）（`horizontal_compare_p2_20260813.json`） |

## 禁止的表述（claim boundary）
- 不得写 SOTA / external generalization / practical importance / mechanism。
- 不得引用 v1/v2 的 `NO_INCREMENTAL_LRSO_SKILL`（已撤回）；P3 正式结论为 `LRSO_EXCEEDS_DIRECT_FOR_DEVELOPMENT`（开发集）。不得声称该优势已在外部可移植（未测；外部协议冻结于 direct candidate）。
- 不得把 P2 PASS 与 P3 结论混为"模型整体失败"——direct 信号是 P2 的合法正向结果。
- 不得声称 5-seed mixture 已部署（当前为单 seed v1；§9.1 五 seed 为正式部署要求）。

## 证据分类
- `CONFIRMED_FACT`：160 cells / 160 WT / 13976 SNV；P2/P3 统计量；横向表。
- `REPOSITORY_REPORTED_NOT_REPLAYED`：旧 M2 的 159/158 与历史效果数字（未重放为新 primary）。
- `NEW_EXPERIMENT_REQUIRED`：任何 external 确认。
- `EXPOSURE_UNKNOWN_DIRECT_REFERENCE`：RNet checkpoint（本地不可得）。
- `UNDERPOWERED_NOT_CONFIRMATORY_PREACCESS`（P4 风险）：无合格 disconnected 外部组件。

## 未决项（UNKNOWN_NOT_ASSERTED）
- 五 seed mixture 的正式部署数字（单 seed 为初步）。
- RNet static-delta 直接基线（需冻结 checkpoint）。
- 任何 external 统计/实际确认（需 owner 提供数据源）。
- 机制（无 external 复现 → `MECHANISM_NOT_ESTABLISHED`）。

## 下一步
- P4：owner 提供合格 external（否则 `PUBLIC_EXTERNAL_NOT_QUALIFIED`）。
- P6：development 结果复现包 + 写稿（需 Phase5-6 token 与 owner 明确请求）。
