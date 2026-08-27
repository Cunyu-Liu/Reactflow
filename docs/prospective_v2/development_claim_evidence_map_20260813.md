# ReactFlow-Delta prospective-v2：development 结果 claim-evidence map (2026-08-13)

> 范围：P0–P3 development benchmark 结论。P4/P5 需外部数据/复现；P6 需 release token。
> 状态边界：`DEVELOPMENT_ONLY / SOTA_NOT_ESTABLISHED / PUBLICATION_NOT_READY`
>
> **SUPERSEDED HISTORICAL MAP — NOT CURRENT CLAIM AUTHORITY.** 20-puzzle development
> universe已被后续多轮模型选择和审计反复使用，因此下述P2/P3结果当前只能写成
> `HISTORICAL_POST_HOC_DEVELOPMENT_EVIDENCE`，不能再写成prospective confirmation。
> V13M3已terminal FAIL；V14尚无terminal scientific verdict。

## 2026-08-22 Model Rescue v2 终局（覆盖后续模型救援表述）

`CONFIRMED_FACT`：在固定 seed-0、20-puzzle LOPO screen 中，MeanAligned 的
signed-delta MAE 相对 B1 改善 `0.8833%`，16/20 puzzles 正向，但未达到预注册的 `1%`
Mean Gate；严格零均值 residual calibration 在 point mean 完全不变时取得
`+0.00547832` CRPS gain，20/20 puzzles 正向并通过 Calibration Gate。由于双 Gate
必须同时通过，终局为 `MODEL_RESCUE_V2_FAIL / CALIBRATION_BASELINE_ONLY`，R2M4 未运行。

允许表述：零均值残差校准在 consumed-development screen 中改善概率评分且不改变 point
mean。禁止表述：MeanAligned 已建立 mutation-effect predictor improvement、v2 已通过
模型救援、五 seed confirmation、external replication、SOTA、mechanism 或 publication
readiness。证据：`docs/prospective_v2/audit/r2m3_qualification_20260822.json`。

## 主结论（仅 development benchmark）
| Claim | 状态 | 证据 |
|---|---|---|
| P2：direct历史development信号 | `HISTORICAL_POST_HOC_DEVELOPMENT_EVIDENCE` | 唯一 `Direct*`(reg_direct) vs `T*`：20-puzzle 95% CI lower = **+0.0079 > 0**；20/20 D_p 为正；sign-flip p=2e-6；LOP max shift 0.0011（`p2_direct_v2_result_20260813.json`）。该universe已被反复消费，不能支持新的prospective claim。 |
| P3：LRSO历史development增量（规范重跑后） | `HISTORICAL_POST_HOC_DEVELOPMENT_EVIDENCE` | 2026-08-15 规范重跑 `run_p3_lrso_v3.py`（trainable encoder + masked NLL + inner 4-fold validation + 五 seed mixture）：rank2 D_p^P3=+0.0147 [95% CI +0.0119,+0.0175]、rank4 +0.0155 [+0.0113,+0.0196]、rank8 +0.0154 [+0.0122,+0.0185]；ci_low_gt_0=True；20/20 puzzle 正向；sign-flip p=1.9e-6；LOO max shift ≤0.001（`p3_lrso_v3_result_20260815.json`）。v1/v2 的 `NO_INCREMENTAL_LRSO_SKILL` 已撤回；本行不构成external、SOTA或publication qualification。 |
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
- 本节是2026-08-13历史计划，不是当前authority。当前只允许完成V14 canonical
  terminal流程及冻结router；新external outcome仍被合同拒绝。
- 代码任务经聚焦验证后按当前用户指令及时push；public release和submission需独立资格。
