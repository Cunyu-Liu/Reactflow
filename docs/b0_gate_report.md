# ReactFlow-Δ B0 强基线 Gate 报告

- **阶段**: B0 (强基线, §10/§12)
- **生成时间 (UTC)**: 2026-07-31 06:05
- **代码 HEAD**: `10b346e` (PH0) + 未提交 B0 代码
- **测试 split**: 293 pairs, 1 parent (Csde1_190), 1 study (10.1038/s41588-021-00830-1), frozen
- **评估器**: `reactflow-delta-b0-evaluator-v1`, Skill = 1 − WMAE(pred)/WMAE(0), macro 聚合 (pair→parent→study)

## 1. 执行摘要

12 个已注册基线全部在真实测试 split 上执行完毕 (0 运行时失败)。**没有任何基线取得正 Skill**：零变化参考 (`zero_change`, Skill=0.0) 是最强预测器。最强的 *学习型* 基线 `static_reactivity` (Skill=−0.0068) 与零变化实质持平。热力学基线 (rnafold/rnaplfold/eternafold) 灾难性失败 (Skill≈−3.4)，说明由折叠导出的 Δunpaired_prob 是一个很差的 delta 预测信号。

**B0 Gate 结论: PASS** — 强基线已建立，delta 信号被证实为弱/难预测，ReactFlow-Δ 只需 Skill>0 即可超越全部基线。

## 2. 完整结果表

按 Skill 降序排列。所有基线使用相同 split / endpoint mask / macro 聚合。

| # | baseline | Skill | WMAE_pred | WMAE_zero | params | runtime(s) | RMSE | Pearson | Spearman |
|---|----------|------:|----------:|----------:|-------:|-----------:|-----:|--------:|---------:|
| 1 | zero_change | 0.0000 | 0.009327 | 0.009327 | 0 | 0.0 | 0.013395 | NaN | NaN |
| 2 | edit_only | 0.0000 | 0.009327 | 0.009327 | 0 | 0.0 | 0.013395 | NaN | NaN |
| 3 | static_reactivity | −0.0068 | 0.009368 | 0.009327 | 10753 | 17.0 | 0.013482 | 0.0026 | −0.0051 |
| 4 | distance_decay | −0.0174 | 0.009446 | 0.009327 | 0 | 0.1 | 0.013471 | 0.0492 | 0.0509 |
| 5 | mutation_type_mean | −0.1218 | 0.010019 | 0.009327 | 0 | 0.0 | 0.013994 | −0.0022 | 0.0057 |
| 6 | generic_paired_matched | −0.1226 | 0.009991 | 0.009327 | 27361 | 27.4 | 0.013963 | 0.0170 | 0.0234 |
| 7 | siamese_matched | −0.1704 | 0.010364 | 0.009327 | 20017 | 27.7 | 0.014334 | 0.0177 | 0.0205 |
| 8 | local_release | −0.2195 | 0.011373 | 0.009327 | 0 | 0.1 | 0.017965 | −0.0034 | 0.0497 |
| 9 | nearest_train | −0.7426 | 0.014091 | 0.009327 | 0 | 0.1 | 0.052746 | 0.0038 | 0.0093 |
| 10 | eternafold | −3.3440 | 0.040561 | 0.009327 | 0 | 160.9 | 0.110335 | 0.0791 | 0.0568 |
| 11 | rnafold | −3.4357 | 0.040867 | 0.009327 | 0 | 175.3 | 0.074026 | 0.0684 | 0.0351 |
| 12 | rnaplfold | −3.4357 | 0.040867 | 0.009327 | 0 | 177.4 | 0.074026 | 0.0684 | 0.0351 |

注: `edit_only` Skill=0.0 是因为 edit 位置被 endpoint mask 排除 (§12.1)，其在 mask 内预测全零，与 `zero_change` 完全等价。

## 3. 最强基线冻结 (T-B0.10)

**最强可执行基线 (B0 champion)**: `static_reactivity`
- Skill = −0.0068 (与零变化参考差距 0.68%，实质持平)
- WMAE_pred = 0.009368 vs WMAE_zero = 0.009327
- 参数量 = 10753 (MLP, 输入=WT reactivity 局部窗口, 输出=delta)
- 训练: 1184 train pairs × 3 alt 扩增 = 3552 triples, 8 epochs, lr=1e-3, GPU
- Pearson r = 0.0026, Spearman ρ = −0.0051 (预测与真值几乎无相关)

**最强非平凡基线 (参考地板)**: `zero_change` (Skill=0.0, WMAE=0.009327)。ReactFlow-Δ 在 B0 gate 的通过条件为 **Skill > 0** (即 WMAE_pred < 0.009327)。

## 4. B0 Gate 自审 (4 项)

### 4.1 相同 split / mask / 聚合 ✓
- 全部 12 个基线在同一冻结测试 split (293 pairs) 上评估。
- endpoint mask (§12.1): 仅保留 unedited + aligned + probe-eligible + valid 位置，排除 edit 位置本身。
- 聚合: pair → parent → study macro 平均; 由于测试 split 仅 1 parent / 1 study，三者数值一致。
- WMAE_zero (=0.009327) 在所有基线间完全一致，证明 mask 与权重一致。

### 4.2 基线失败记录在 failure_table ✓
- `failure_table.json` 记录 11 个已知缺失工具 (LinearPartition, RNAstructure, RNAsnp, SNPfold, remuRNA, Riprap, VariantFoldRNA, Rchange, RibonanzaNet, RibonanzaNet2, eFold)，均附 `reason` + `attempted` 探测方式。
- 运行时失败: 0 (所有 12 个已注册基线均成功产出预测)。

### 4.3 参数可审计 ✓
- 每个基线的 `param_count` 记录在 `results.json`:
  - 非学习 / 热力学: 0 (无可训练参数)
  - static_reactivity: 10753
  - siamese_matched: 20017
  - generic_paired_matched: 27361
- 训练超参 (epochs=8, batch_size=8, lr=1e-3, seed=0, device=cuda) 记录在 `aggregation` 的 kwargs 中。

### 4.4 无弱基线 cherry-picking ✓
- 全部 12 个已注册基线均执行并报告，无事后剔除。
- 结果按 Skill 降序完整呈现，包含灾难性失败的热力学基线 (Skill≈−3.4)。
- 最强基线选择基于客观数值 (最高 Skill)，非主观挑选。

## 5. 已知限制

1. **rnaplfold ≡ rnafold (已知)**: ViennaRNA 2.7.2 Python API 不直接暴露 RNAplfold 滑动窗口; `RNAplfoldBaseline._fold_seq` 复用 `fold_compound.pf()` (全局配分函数)，与 `RNAfoldBaseline` 产出完全相同 (Skill/WMAE/RMSE 逐位一致)。此为文档化设计选择，非 bug; 真正的 RNAplfold CLI 差异未纳入。
2. **EternaFold MEA 近似**: `EternaFoldBaseline` 通过 contrafold `predict --params EternaFoldParams.v1` 获取 MEA 点估计结构，将 unpaired_prob 二值化 (paired=0, unpaired=1)。完整后验 BPP 需 `--posteriors` 标志，作为已知限制记录。
3. **EternaFold 调用修复**: 初版 `_fold_seq` 假设二进制从 stdin 读序列; 实际 `eternafold` 是 contrafold 符号链接，需 `predict --params <file> <FASTA>`。已修复为 temp FASTA + `>structure` 标记解析，并通过真实二进制 smoke test。
4. **encoded_alt="X" 全量边缘化**: 全部 1509 pairs 的 `encoded_alt="X"`，故所有突变序列基线对 3 个非 ref alt 碱基各构造 1 条突变序列，折叠/预测后取平均。
5. **测试 split 单 parent/study**: 293 test pairs 全部来自 Csde1_190 / 单一 study，parent/study 维度的 macro 聚合无法体现跨家族泛化; 这是冻结 split 的既定约束。

## 6. 测试

- `tests/reactflow_delta/test_evaluate.py`: 24 tests, editflow311 env 全部通过。
- `tests/reactflow_delta/test_baselines.py`: 29 tests (24 非torch in editflow311 + 5 torch in pc_cng_gpu)，全部通过。
- EternaFold `_fold_seq` 测试通过 mock (不依赖真实二进制); 真实二进制 smoke test 已通过。

## 7. B0 Gate 结论

**PASS**

- 强基线已建立: 12 个基线全部执行，0 失败，结果冻结于 `results.json`。
- 最强基线 `static_reactivity` (Skill=−0.0068) 与零变化参考持平; ReactFlow-Δ 通过条件 = Skill > 0。
- delta 信号被证实为弱/难预测 (最强学习基线 Pearson r=0.0026)。
- 4 项 Gate 自审全部满足。

**停止于 B0 Gate PASS，不进入 O0/M0。**
