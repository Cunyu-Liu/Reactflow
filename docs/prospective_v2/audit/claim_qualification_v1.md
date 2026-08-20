# Claim Qualification v1 — audit P0-1

> 依据：`提示词/ReactFlow_Delta_strict_scientific_engineering_audit_20260817.md`（§2、§9.3、§10.3、§13 P0-1）。
> 目的：把旧 headline 主张与它们实际能支持/不能支持的证据资格对齐；工程 PASS 不得自动产生 scientific/mechanism/SOTA PASS。
> 状态：本文件只读判定旧 artifact 资格，并把当前 P0 协议（统一 evaluator + distinct baselines + target-invariance + Direct*/K_rank=0 runner）标为 `CURRENT_PROTOCOL`。

## 1. 旧 headline 主张必须立即降级/删除

| 当前表述 | 处理 | 依据 |
|---|---|---|
| “RFD-LRSO exceeds direct due to low-rank susceptibility” | 降为：同架构 selected-rank 相对 rank0 在 development 上有小增量 `+0.00180`；低秩不是主要性能来源 | 20-fold `p2_v3_scores_merged.json` |
| “529 independent external components” | 删除，改为 “529 WT anchors across 7 dataset files; higher-level independent N unresolved” | audit §2.1, §9.3 |
| “pre-specified 20% negligible threshold” | 删除，明确为 post-hoc exploratory threshold | audit §2.1, §9.3 |
| “MECHANISM_EVIDENCE_PASS” (combined) | 降为 `MECHANISM_NOT_ESTABLISHED`；combined 仅 exploratory synthesis | audit §2.1, §9.3 |
| “two independent external data sets” | 只有补齐 publication/study/batch 后可用 “independent” | audit §9.3 |
| “external transportability of project method” | 删除；纠正 seqpos 后的最终 LRSO 在 `K_joint=2` 两个 study clusters 上未一致复制 | `p4_external_lrso_result_v1.*` |
| “spec-compliant P2/P3 PASS” | 改为 `POST_HOC_DEVELOPMENT`；evaluator/comparator/nested null 已修复，但 20 puzzles 已全部 development-consumed | model-rescue contract v1 |

## 2. 主张—证据表（重建后）

| 主张 | 所需证据 | 当前证据状态（2026-08-20） |
|---|---|---|
| 任务存在 development learnability | 正确 estimand 下 CRPS 与 signed-Δ MAE 都优于合法基线 | `RESOLVED_DEVELOPMENT_PASS`：ridge CRPS vs WT-anchor +0.02677；M1 重算表明 rank0/selected-rank 的 signed-Δ MAE gain vs WT 分别为 +0.01221/+0.01298，均 20/20 puzzle 为正 |
| 低秩项带来增量（rank-positive 胜 rank0） | 同架构/同 likelihood/同 seed/同 epoch 下 CI lower>0 | `RESOLVED_PASS`：20-fold 正式运行完成。selected-rank vs K_rank=0 主 null D_p=+0.00180 (CI +0.00031..+0.00328, ci_low_gt_0=True, sign-flip p=0.016)。低秩增量真实但量级小（网络 stack 贡献 +0.0153，低秩仅 +0.0018） |
| 外部 study-level 泛化 | publication/study/batch 聚类后最终 LRSO 复制 | `NOT_ESTABLISHED`：7 数据集/718 anchors → K_joint=2（SL5 + Ribonanza 两 study）；seqpos-correct 最终 LRSO 已完成，SL5 cluster effect 约 +0.00075、Ribonanza cluster 约 +0.03496，cluster CI 跨零且 LOSO 显示不一致；只能作 consumed exploratory evidence |
| 机制：跨位置低秩 susceptibility | rank0 + source/receiver randomization + 独立 negative control | `NOT_ESTABLISHED`：rank0 nested null 已通过（selected-rank vs rank0，D_p=+0.0018）；最终 LRSO external 未一致复制；source/receiver randomization 与新的独立 negative control 仍缺失 |
| 旧外部 direct 结果（P4_EXTERNAL_STATISTICAL_PASS, D=+0.0410） | 正确的 seqpos 对齐 | `INVALID`（审计发现 P4-M1）：旧 P4/P5/P5b 外部评分用 index-0 对齐，但 rdat seqpos 从 X27 开始（偏移 26），特征与评分位置错位；该数字不可用于任何结论 |

## 3. 新协议交付物与验收状态（远端 model-rescue branch，2026-08-20）

| 审计项 | 交付物 | 验收 | 状态 |
|---|---|---|---|
| P0-2 冻结 estimand + 手算 fixture | `scripts/reactflow_delta/evaluator_v2.py`；`tests/reactflow_delta/test_estimand_v2.py` | 5/5 通过；method-balanced≠pooled；key 精确配对；missing≠0 | `PASS` |
| P0-3 基线独立性与 P2 资格重算 | `scripts/reactflow_delta/baseline_v1.py`；`tests/reactflow_delta/test_distinct_baselines_v2.py` | 5/5 通过；object identity 不同；predictions 不同；train_median 真由 train fold 计算；RFDDirectRank0 精确禁用低秩项 | `PASS` |
| P0-5 prediction/score 分离 + target-invariance | `run_p3_lrso_v3.py` held/inner 路径改 WT-mask-only；`prediction_v3.py` schema；target-invariance 测试 | 结构 + 行为测试通过；prediction ledger 不含 target；target pattern 变化不改 prediction | `PASS` |
| P0-7 Direct*/rank0 正式重算 | `run_p2_v3.py` + sharded 20-fold run | `/mnt/cunyuliu/prospective_v2_p2v3_sharded/p2_v3_scores_merged.json`；20 folds；40 prediction-only OOF ledgers；rank0 与 selected-rank 同 cfg/epochs/seeds | `POST_HOC_DEVELOPMENT_COMPLETE` |
| P0-4/6 external qualification | `joint_dependency_component_v1.py` + corrected LRSO run | 7 datasets/718 anchors → K_joint=2；旧结果 seqpos-invalid；corrected final LRSO study-level replication fails | `EXPLORATORY_NOT_CONFIRMATORY` |

## 4. model-rescue M1 结论与 M2 边界

1. M1 failure atlas 已完成；旧“signed-Δ 差于 WT”只能限定为 P2-v2 ridge，不得用于当前 rank0/rank-positive 网络。
2. low-rank CRPS gain `+0.001796` 中 mean/scale Shapley 分别为 `+0.001226/+0.000570`；其 signed-Δ MAE 增量仅 `+0.000769`（约 0.32%），低于 2% practical gate。
3. near-zero 位置上 low-rank 使 signed-Δ MAE 恶化 `-0.002745`；因此 M2 的主救援候选是 aligned-direct 与 SparseDelta，L2-aligned 仅作 fixed control。
4. ViennaRNA 结构 probe 对 signed-Δ 显著恶化且 absolute-Δ CI 跨 0；StructDelta 已按预冻结规则退出 M2。
5. 新 confirmatory external 仍必须在 outcome-blind metadata 上达到 `K_joint_new>=9`；现有 K_joint=2 永久为 consumed exploratory。
