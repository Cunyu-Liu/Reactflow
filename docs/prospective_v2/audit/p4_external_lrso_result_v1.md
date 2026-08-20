# Final-LRSO External Cluster Validation — RESULT (exploratory, 2026-08-18)

> 依据：`提示词/ReactFlow_Delta_strict_scientific_engineering_audit_20260817.md`
> §9.1 决策树、§9.2 解决方案 3（"外部候选必须是最终 LRSO"）、§13 P0-4/P0-6。
> 协议：`docs/prospective_v2/p4_external_lrso_frozen_protocol_20260818.md`
> 执行：`run_p4_external_lrso_v1.py`（A100, cuda:6, ~47 min，含 dev 内层 epoch 选择 + 5-seed 最终训练 + 外部评分）。
> 原始产物：`/mnt/cunyuliu/prospective_v2_p4_lrso_20260818/p4_external_lrso_result.json`
> （本地副本 `docs/prospective_v2/audit/p4_external_lrso_result_v1.json`）。

## 1. 冻结设置（outcome-blind，先于任何外部 outcome access）

| 项 | 值 |
|---|---|
| 候选 | **最终 LRSO**（rank=2，5-seed 等权 Gaussian mixture，cfg lr=1e-3/wd=0/Student-t） |
| 训练 | ALL development OK7a_M2（20 puzzles, 13,976 SNV）；epoch=50 由 dev 级 puzzle-grouped 内层 4-fold early-stopping 选择（inner CRPS 0.1885） |
| 基线 | ZeroResponse（WT-anchor 预测，fixed scale 0.3） |
| 外部图 | 冻结 outcome-blind manifest（24 direct_external 组件, 3,237 SNV） |
| 对齐 | **seqpos-correct**（offset=26；修复审计 P4-M1 的 index-0 错位） |
| K_required_planned | 9 |
| K_preaccess / K_joint | 24 / **2**（SL5 PNAS 2024；Ribonanza 2024） |
| 模型 | 5 个 checkpoint 已保存（final_models/*.pt，sha256 记录于 result JSON） |

## 2. 结果

### 2.1 component 级（仅透明性描述，非独立单位）
- K_eff = 24（无 attrition）
- D_vs_zero mean = **+0.0307**，95% CI [+0.0152, +0.0461]（df=23）
- 该数字把 21 个 Ribonanza 组件当作独立单位，**高估精度**（audit P0-4）；不作为推论单位。

### 2.2 cluster 级（primary exploratory，K_joint=2，df=1）
| study cluster | n 组件 | cluster-macro D_vs_zero |
|---|---|---|
| study_sl5（SL5 PNAS 2024, batch 2023-06-06） | 3 | **+0.00075**（[-0.0025, +0.0014, +0.0034]）≈ 0 |
| study_ribonanza（Ribonanza 2024, batches 2023-08-01/10-31） | 21 | **+0.03496**（20/21 正） |
| **K=2 cluster-macro mean** | 2 | **+0.0179**，95% CI **[-0.199, +0.235]**（df=1，宽度因 K=2 巨大） |

- **LOSO**：leave-out SL5 → 剩 Ribonanza +0.035；leave-out Ribonanza → 剩 SL5 +0.00075。
- **unknown_study_components: []**（24/24 已归入两 study）。

## 3. 解读（fail-closed）

1. **component 级正信号主要由单一 study（Ribonanza cluster，21 组件共享 2 个测序批）驱动**；
   SL5 study（独立 publication、独立 batch）的 cluster-macro D ≈ 0（+0.0008）。
2. **cluster 级复制未成立**：2 个独立 study 的方向不一致（SL5 ≈ 0，Ribonanza 正）；
   K=2 下 95% CI [-0.199, +0.235] 完全无信息量。即便只看点估计，跨 study 也不一致。
3. **结论**：最终 LRSO 的外部低秩增量信号**没有在两个独立 study 上复制**。
   既不是 confirmatory（K=2 < 9），也**不足以作为 "development replication 信号一致" 的弱证据**——
   SL5 一翼不复制。
4. 与 audit §9.1 决策树对齐：外部顶层独立 cluster 不足（K=2）且最终 LRSO 未在独立 cluster 上复制
   → **不走 "broad generalization / LRSO 主线" 主张**；转向 development method paper / benchmark /
   direct 路线（§9.1 决策树"否"分支），除非获得新的 provenance-resolved、development-disconnected、
   非 Das-lab 体系的独立 study/batch 数据。

## 4. 状态

- `verdict = DEVELOPMENT_REPLICATION_EXPLORATORY`
- `confirmatory_eligibility = NOT_ESTABLISHED`（K_joint=2 < K_required=9；SL5 不复制）
- `practical_importance = NOT_ESTABLISHED`
- 已消费外部 outcome（本 run locked_outcome_access_count=1，独立于旧 P4/P5/P5b 的消费记录）；
  旧 P4/P5/P5b 因 P4-M1 错位已判 INVALID（见 `external_qualification_v1.md` §6）。
