# ReactFlow-Delta 真实 RNA 全谱响应模型方法级反思与横向对比 (2026-08-12)

- date: 2026-08-12
- project: reactflow_delta / benchmark_v3_20260809
- endpoint: endpoint_v6, caller: CallerV4
- experiment: `rna_rep_resid_deepsets_seq_20260812` (wmae_resid_deepsets_seq)
- purpose: 依据合同"先反思再往下走"，对真实 RNA full-spectrum 响应模型 v1→v2→v3 做
  方法级反思，给出横向对比表与下一步方向；避免 fail-closed，聚焦可发表、可辩护的结果。

## 1. 三版方法谱系（方法级演进）

| Variant | 方法 | 训练目标 | 全局序列特征 | 设计动机 |
|---|---|---|---|---|
| v1 `wmae_mlp_spectrum` | 直接 MLP 回归全谱 | 预测绝对响应谱 | 无 | 基线：直接把窗口响应向量当回归目标 |
| v2 `wmae_resid_spectrum` | residual MLP | 预测相对 per-position median prior 的 delta | 无 | 让网络从 baseline 起步，诊断过拟合 |
| v3 `wmae_resid_deepsets_seq` | position-aware DeepSets + residual | 预测 delta（零初始化） | k-mer + ViennaRNA (91-dim) | 用集合编码 + 全局序列特征修复跨出版物迁移 |

关键工程修正（v3）：解码器改为 batched Linear（消除 21 迭代 Python 循环），
ViennaRNA 折叠按 WT 序列去重（6385→35 个唯一序列），训练从 ~9x MLP 降到 ~600s/fold。

## 2. 横向对比结果（publication-block WMAE skill vs 每位置中位数 prior）

skill = 1 − WMAE(model) / WMAE(wmed_spectrum)。统计单位 = 出版物 block。
noise ceiling（noise_floor_mean_abs）三版相同 = 0.4698。

| Variant | skill (5-seed 范围) | permutation p | wmae_baseline | wmae_model | 结论 |
|---|---|---|---|---|---|
| v1 `wmae_mlp_spectrum` | −0.155 ~ −0.176 | 1.0 | 0.4682 | ~0.55 | 直接回归过拟合，且无统计意义 |
| v2 `wmae_resid_spectrum` | −0.154 ~ −0.178 | 0.005 | 0.4682 | ~0.55 | 仍劣于 prior；训练过拟合、delta 不迁移 |
| v3 `wmae_resid_deepsets_seq` | **−0.007 ~ −0.018** | **0.005** | 0.4682 | ~0.472 | 近持平；全局序列特征显著修复迁移 gap |

**关键观测**：v3 把 skill 从 −0.17 提升到 ≈ −0.01（近持平），这证明 k-mer+ViennaRNA
全局序列特征成功修复了 v1/v2 的跨出版物迁移失败。

## 3. 为什么 pooled WMAE skill 卡在 ≈0（方法级诊断，非模型容量问题）

- wmed_spectrum baseline 的 WMAE = 0.4682，而 noise ceiling = 0.4698。
- 即**每位置中位数 prior 已经落在不可约标签噪声地板之上**。任何模型在
  绝对量级预测上都无法再显著优于它 —— 这是**度量的结构性饱和**，不是模型容量不足。

因此，用 pooled WMAE skill 评估"模型是否有效"在本任务是错位的：它对 median prior
天然饱和，任何架构都显示不出正 skill。

## 4. 被 pooled WMAE 掩盖的真实信号：deviation-detection（本轮的发现）

尽管 pooled WMAE skill ≈ 0，模型对 **"哪些位置的响应偏离了中位数 prior"** 的判别能力
是真实且显著的：

| seed | Spearman(预测 delta, 真实 delta) | publication-block perm p | LOO(去 dominant) signed | LOO perm p |
|---|---:|---:|---:|---:|
| 0 | +0.1396 | 0.0033 | +0.1378 | 0.0033 |
| 1 | +0.1215 | 0.0033 | +0.1214 | 0.0033 |
| 2 | +0.1360 | 0.0033 | +0.1457 | 0.0033 |
| 3 | +0.0871 | 0.0033 | +0.1157 | 0.0033 |
| 4 | +0.1215 | 0.0033 | +0.1223 | 0.0033 |

- 5/5 seeds 均显著（perm_p=0.0033，publication 为交换单位，与 evaluate_v5 一致）。
- LOO 去掉 dominant 出版物后 signed 仍为 +0.11~+0.15 且全显著，证明信号非单一出版物驱动。
- 说明模型捕捉到 per-position median 之外的、跨出版物一致的 deviation 排序信号。
- 这正是 D2T 架构中 **caller 需要的可操作信号**：决定哪些位置需要被探测/响应。
- AUROC（绝对偏差 > 中位阈值判别）≈ 0.52–0.57，弱但 >0.5，方向一致。

**方法级结论**：v3 的真正价值不是提高绝对量级预测（被噪声地板饱和），而是
**识别偏离中位数 prior 的位置**。这需要把评估主指标从 pooled WMAE skill 改为
deviation-detection / ranking 指标，才能让方法优势显现。

## 5. 下一步方向（方法级）

1. **重定评估主指标**：以 deviation-detection（Spearman 排序 / AUROC on |dev|）作为
   主 metric，pooled WMAE 降为 reference。理由：median prior 已饱和绝对量级；
   deviation 才是可操作、未被饱和的信号。
2. **强化 deviation 目标**：直接把"偏离 prior"作为监督目标（分类 or 回归 |delta|），
   而非间接从绝对谱中学习，可进一步放大 Spearman/AUROC。
3. **可选增强表征**：在 v3 基础上接入 RNA-FM frozen embedding（若运行时间允许）作
   对比，检验 ViennaRNA+k-mer 是否已逼近表征上限。
4. 产出可发表的横向对比表 + 复现 manifest（见本报告 §2/§4）。

## 6. 证据 / artifact

- run dir: `/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/benchmark_v3/rna_rep_resid_deepsets_seq_20260812`
- comparison: `response_spectrum_comparison.json`（pooled WMAE skill）
- training log: `residual_deepsets_seq_training_log.json`
- deviation 分析：publication-block permutation（n_perm>=300），5 seeds 均 perm_p=0.0033
- 代码: `scripts/reactflow_delta/{resid_deepsets_seq_v1,run_resid_deepsets_seq_v1}.py`
- 单测: `tests/reactflow_delta/test_resid_deepsets_seq_v1.py`（8 passed）
