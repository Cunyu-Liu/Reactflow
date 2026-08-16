# ReactFlowDelta P2v5 条件 magnitude 可学习性 GO 审计

- **审计对象**: conditional-magnitude learnability gate (endpoint_v5, authority epoch 17, Route B)
- **审计日期**: 2026-08-08
- **治理合同**: `docs/contracts/ReactFlowDelta_M0X_strict_scientific_engineering_audit_20260807.md`
- **执行服务器**: `/home/cunyuliu/reactflow_delta_goal_20260729`
- **Run**: `results/p2_v5_magnitude_20260808`
- **裁决**: **GO**（fail-closed）

## 1. 背景与科学问题

P2 主任务（binary-changer publication-macro AUPRC）在 endpoint_v4 下经 paired
publication-block bootstrap 判定为 **STOP**（fail-closed，epoch 16，已冻结）。该 STOP
不因后续 pivot 被覆盖或淡化为成功。

科学判定焦点随后转移到 **独立的 conditional-magnitude estimand**（合同 §4.2
conditional 行 / §9.2 conditional head）：对 caller_v3 判定的真实 changer
（C_i=1），仅凭 allowed inputs（WT sequence + exact single-nucleotide mutation +
allowed WT experimental reactivity state + condition，**不读 mutant profile**）
预测 profile-level |Δr| magnitude 是否可跨 publication 学习。

## 2. Endpoint_v5 定义（Route B，conditional magnitude）

- **unit**: matched WT–exact-single-mutant pair（TRUE CHANGERS only, C_i=1）
- **label**: `y_i` = 对真实 changer pair，在 ELIGIBLE positions 上的
  mean_i |mutant_react[i] − wt_react[i]|（raw reactivity 绝对变化幅度）；
  权重 `w_i` = 纳入均值计算的 ELIGIBLE position 数
- **score**: pair-level magnitude `hat{y}_i`（regression head 直接输出）
- **metric**: conditional WMAE skill 及 CI
- **baseline**: train-fold changer 加权均值（constant predictor，禁止 held-out 泄漏）
- **resampling**: publication 为 outer unit；paired publication-block bootstrap；
  permutation 用 (b+1)/(B+1)

## 3. 执行协议

- 嵌套 leave-one-publication-out（outer unit = publication，18 个 publication）
- caller_v3（empirical-scatter noise recalibration）判定真实 changer
- regression head（MLP / DeepSets）在 **GPU** 上训练，5 个确定性 seed
- 主任务 endpoint_v4 STOP 全程保持冻结

## 4. 结果（held-out TRUE CHANGERS，跨 publication pooled）

| 模型 | mean skill | CI low（min over seeds） |
|------|-----------|--------------------------|
| trivial (baseline) | 0.0（定义） | — |
| linear | 0.308 | 0.256 |
| gbm | 0.319 | 0.261 |
| p2_mlp | 0.602 | 0.315 |
| **deepsets** | **0.677** | **0.323** |

- 最佳模型：**deepsets**，mean skill = 0.677，全部 5 seed 为正
- paired publication-block bootstrap CI 下界 = 0.323（> 0）
- 方向一致性（seed 0）：10/10 个含真实 changer 的 held-out publication 全部 positive
- estimand_status = **IDENTIFIABLE**

判定标准（fail-closed）逐项满足：
1. `n_seeds>=5_all_models`: true
2. `estimand_identifiable_skill_numeric`: true
3. `best_skill_gt_0`: true
4. `best_bootstrap_ci_low_gt_0`: true

=> **P2_CONDITIONAL_MAGNITUDE = GO**

## 5. 治理冲突与定案（epoch 17，Route B vs Route C）

**冲突**：已执行的 `run_p2_v5.py` 实现的是 **conditional magnitude (Route B)** 且判定
GO；但服务器磁盘上的 epoch-17 治理（endpoint_v5 / amendment / active_contract）曾被
覆盖为 **position granularity (Route C)**——这是用户未选择的并行分支，其 run
（`run_p2_position_v5.py`）独立存在。

**定案步骤**（用户显式授权按 Route B 定案）：
1. 备份 Route C 治理产物到 `docs/contracts/amendments/_unselected_routeC_epoch17/`
   （`endpoint_v5_routeC_position_granularity.yaml` + amendment + approval），**保留不删除**。
2. 恢复 `configs/reactflow_delta/endpoint_v5.yaml` 为 conditional magnitude (Route B) 规范。
3. 创建 Route B 定案治理产物：
   - amendment: `reactflow_delta_v4_epoch17_endpoint_v5_conditional_magnitude_20260808.yaml`
   - approval: `reactflow_delta_v4_epoch17_endpoint_v5_conditional_magnitude_approval_20260808.yaml`
4. 更新 `active_contract.yaml` -> `CONDITIONAL_MAGNITUDE_EPOCH17_ENDPOINT_V5_GO_TERMINAL`，
   `route_selected = ROUTE_B_CONDITIONAL_MAGNITUDE_GO`。
5. 重建 `authority_epoch_17.bundle.sha256` + `authority_epoch_17.sentinel.yaml`，
   重生成 `endpoint_v5.sha256`。

## 6. 不变量与边界

- 主任务 endpoint_v4 STOP = **FROZEN_EPOCH16_PRESERVED**（未覆盖）
- Route C = **SUPERSEDED_UNSELECTED_PARALLEL_BRANCH_KEPT**（保留，不删除，不混淆）
- 测试 split 保持 sealed；test/holdout 数据未用于训练
- baseline 仅来自 train-fold changer（无 held-out 泄漏）
- 无隐藏失败、无降低门槛、无种子重试作弊、无伪造数据
- scientific_claim_authorized = false（conditional magnitude 为可 SOTA 化 estimand，
  但 wet-lab/真实数据独立验证尚未授权）

## 7. 下一步含义

conditional-magnitude GO 为后续 conditional 层建模 / SOTA 对比提供方法门禁。任何
"improves |Δr| magnitude prediction" 的主张仍需以 predicted/internal proxy 限定词，
直到独立 multi-region oracle / wet-lab 验证完成。
