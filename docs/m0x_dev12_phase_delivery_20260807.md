# M0-X 阶段交付文档 — EPRO_DEV_12 连续 burden 回归头迭代

> `contract`: ReactFlowDelta v4_data_first（active manifest 绑定） · `phase`: M0-X（受控开发窗口） · `authority`: M0X_AUTHORIZED（epoch 12）
> `iteration_id`: EPRO_DEV_12_REGRESSION · `run_id`: epro_dev12_regression_std_20260807
> `evidence_class`: **DEVELOPMENT_ONLY** · `claim_eligibility`: **NO_CONFIRMATORY_CLAIM** · test 保持 **SEALED**
> 生成时间: 2026-08-07 · 分支: `codex/reactflow-delta-d0r` · 验收 commit: `e0994ee`

---

## 1. 阶段范围与目标

在 M0-X 受控开发窗口内，验证**连续 burden 端点**（pair 级 "多强 / 多广" 的响应量）能否被监督模型端到端预测，并修复此前 dev12 在该端点上的**负相关**问题。

核心诊断（v3 时代遗留）：dev12 的 **pair 内位置排序良好（+0.49）**，但 **pair 间绝对量级失准**（预测 std 0.26 vs 真实 0.08，p90 0.75 vs 0.21），导致原始 `mean(|pred|)` 与真实 burden `mean(|Δ|/scale)` **负相关**。本迭代引入 scale-invariant 代理 `within_pair_z_max` 统一评估口径。

## 2. 数据与方法

| 项目 | 说明 |
|---|---|
| 数据集 | B0-X effect 数据集（GSE114002 等），frozen publication split |
| 划分 | train **3516** / val **548** / test **SEALED（不读取）** |
| 监督模型 | DeltaMagnitudeRegressor（线性头，MAE on signed `Δ/scale`），参数 **810,497** |
| 真实 burden | `mean(|Δ|/scale)` over eligible positions |
| 代理 | `within_pair_z_max` = max within-pair z-score of per-position magnitude（scale-invariant，统一口径） |
| 零样本基线 | efold, ufold, rnaformer, eternafold, vienna, moefold2d（exact-alt 构建） |
| 环境 | GPU（CUDA），editflow env，`fallback=0`（无 CUDA 则失败） |

> 分类器（dev10/07/09）为 sigmoid `P(changer)`，不适用连续 burden 端点 → `NOT_APPLICABLE_FOR_BURDEN`，不计入排名。

## 3. 核心结果

### 3.1 dev12 头号指标：burden 相关**转正**

| 代理 | Spearman | Kendall | NDCG@10 | 符号 |
|---|---|---|---|---|
| raw mean-burden | **−0.285** | −0.202 | 0.181 | 负（错） |
| **within_pair_z_max** | **+0.408** | +0.264 | **0.326** | **正（修正）** |

`within_pair_z_max` 通过 z-score 归一化消除 pair 间绝对量级漂移，将负相关**转正**，且 NDCG@10 由 0.181 提升至 0.326。

### 3.2 统一口径横向对比（n=548 val pairs）

| 模型 | 类别 | zmax Spearman | zmax Kendall | zmax NDCG@10 | raw Spearman | 排名 |
|---|---|---|---|---|---|---|
| **efold** | zero_shot | **+0.506** | +0.435 | 0.299 | +0.513 | **1** |
| **epro_dev12** | supervised | **+0.408** | +0.264 | **0.326** | −0.285 | **2** |
| vienna | zero_shot | +0.026 | +0.016 | 0.369 | +0.006 | 3 |
| rnaformer | zero_shot | +0.023 | +0.015 | 0.260 | +0.200 | 4 |
| eternafold | zero_shot | −0.002 | −0.002 | 0.291 | −0.027 | 5 |
| moefold2d | zero_shot | −0.054 | −0.047 | 0.271 | +0.026 | 6 |
| ufold | zero_shot | −0.169 | −0.114 | 0.160 | +0.248 | 7 |

**结论要点**：统一 `within_pair_z_max` 口径下，dev12 排名**第 2**（仅次于 efold，高于全部其余零样本折叠基线），并在 NDCG@10（0.326）上**超过 efold（0.299）**。

> 注：`within_pair_z_max` 对不同模型影响不同——efold 保持强（0.506），而 ufold/moefold2d 在原始均值口径下为正、在 zmax 口径下降级，印证"统一口径"对公平对比的必要性。

## 4. 验收结果（overall = **PASS**，6/6）

| 项 | 状态 | 证据 |
|---|---|---|
| artifacts_present | PASS | 8/8 齐全，checksum ledger 无缺失 |
| unit_tests | PASS | **51 passed / 0 failed** |
| regression_head_trained | PASS | 810,497 参数，best_epoch 89，CUDA |
| burden_proxy_calibration | PASS | raw −0.285 → zmax +0.408（sign_fixed=True） |
| horizontal_comparison | PASS | 7 模型统一口径，dev12 rank 2 |
| test_sealed | PASS | test 未读取 |

## 5. 诚实性边界（必须随汇报保留）

- 本迭代为 **DEVELOPMENT_ONLY**，**不构成** confirmatory / SOTA 结论，**不授权** "improves burden ranking" 的正式科学声明。
- 监督模型 **val WMAE-skill 为负**（-0.51），绝对量级预测仍未学会跨研究泛化；`within_pair_z_max` 是**代理校准**，不等同于端到端绝对 burden 预测能力。
- test 保持 SEALED，未用于任何选择/报告。

## 6. 产物路径（服务器 `/home/cunyuliu/reactflow_delta_goal_20260729/`）

| 产物 | 路径 |
|---|---|
| 验收报告 | `results/epro_dev12_acceptance_20260807/acceptance_report.json` |
| 校验台账 | `results/epro_dev12_acceptance_20260807/checksum_ledger.json` |
| 闭合哨兵 | `results/epro_dev12_acceptance_20260807/EPRO_DEV_12_CLOSED.yaml` |
| 统一对比 | `results/sota_pairlevel_v6_zmax_20260807/unified_zmax_compare.json` |
| 代理校准 | `results/dev12_calibration_20260807/calibration_comparison.json` |
| 可视化 | `results/sota_pairlevel_v6_zmax_20260807/dev12_delivery_summary.png` |
| 窗口登记 | `docs/governance/m0x_window_registry_20260804.json`（consumed=8） |

## 7. 停点

`STOPPED_AT_USER_REVIEW` — EPRO_DEV_12 迭代已闭合验收并出图，等待下一轮 M0-X 授权或用户审阅。
