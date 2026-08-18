# M2 response-spectrum 最终提交清单（Final Submission Checklist）

- 生成日期：2026-08-18（Asia/Shanghai）
- 分支：`codex/reactflow-delta-benchmark-v3-20260809`
- HEAD commit：`2766ff5`
- 远端：`git@github.com:Cunyu-Liu/Reactflow.git`（push 成功 ✅）
- 提交包目录：`/mnt/cunyuliu/m2_spectrum_submission_final_20260818/`
- headline 产物：`/mnt/cunyuliu/m2_gbdt_3way_ensemble_matched_20260818/m2_masked_eval_report.json`

## 1. Headline（design-level LOO，matched 272,988 positions，158 designs）

| 模型 | WMAE skill | WMAE | 95% CI | perm_p | 说明 |
|---|---|---|---|---|---|
| baseline (wmed_spectrum median) | — | 0.7537 | — | — | 序列无关先验 |
| plain residual MLP | +8.88% | 0.6867 | (0.0832, 0.0954) | 0.0033 | 早期模型 |
| position-aware (v3) | +10.41% | 0.6752 | (0.0977, 0.1114) | 0.0033 | per-position heads |
| attn 1-layer (v4) | +12.32% | 0.6608 | (0.1158, 0.1308) | 0.0033 | self-attention |
| attn 2-layer (v5) | +12.63% | 0.6585 | (0.1190, 0.1345) | 0.0033 | 2-layer |
| multi-depth ensemble | +12.75% | 0.6576 | (0.1190, 0.1363) | 0.0033 | 0.25·v4+0.75·v5 |
| **3-way deep ensemble** | **+12.84%** | **0.6569** | **(0.1215, 0.1357)** | **0.0033** | 0.15·v3+0.20·v4+0.65·v5 |
| **GBDT cross-arch + 3-way deep（NEW headline）** | **+14.06%** | **0.6477** | **(0.1341, 0.1482)** | **0.0033** | leak-free 31-dim + blend a=0.5 |

**blend vs 3-way deep**：pooled **+1.21pp**（per-design +1.11pp，66% positive）；LOO-exclusion：mean **+1.21pp**，range [+1.03, +1.24]pp，**158/158 folds positive**。

## 2. 数据来源（authentic, non-circular）

| 数据 | 路径 | 说明 |
|---|---|---|
| M2 实验 | `OK7a_M2_data.v4.5.2.csv`（OpenKnot，160 designs，13,839 changers，272,988 windowed positions） | 真实 SHAPE 数据 |
| v3 posaware OOF | `/mnt/cunyuliu/m2_response_spectrum_posaware_20260813/keyed_predictions_m2_posaware.jsonl` | 5-seed mu-ensemble |
| v4 attn 1-layer OOF | `/mnt/cunyuliu/m2_response_spectrum_attn_gpu3_20260815/keyed_predictions_m2_attn.jsonl` | 5-seed mu-ensemble |
| v5 attn 2-layer OOF | `/mnt/cunyuliu/m2_response_spectrum_attn_v5_deep_20260815/keyed_predictions_m2_attn.jsonl` | 5-seed mu-ensemble |
| 热力学折叠 | ViennaRNA 2.7.x（MFE + BPP partition function） | 序列派生，全合法 |

**合法性声明**：GBDT 特征只用 WT reactivity/error + WT/mutant 序列 + 突变身份（ref/alt）；**绝不用 mutant reactivity（即目标 y）作为特征**——已用测试强制（test_m2_gbdt_features_v1.py）。

## 3. 方法链（本 headline 的 adopted lever）

| Lever | 增益 | 显著性 | 依据 |
|---|---|---|---|
| MFE 热力学结构特征（per-position） | GBDT 自身 +11.95% | LOO 稳健 | m2_gbdt_features_v1 |
| 跨架构 GBDT + deep 集成（error decorrelation） | **+1.21pp vs 3-way deep** | **perm p=0.0033，LOO-exclusion 100% positive** | m2_masked_eval_from_oof |

## 4. Fail-closed 证据（本 headline 无关、独立验证）

| Direction | Result | Conclusion |
|---|---|---|
| Student-t NLL loss (v6) | +10.64% | 不优化 WMAE 指标，closed |
| 4-way ensemble incl. v6 | +12.78% (attn-heavy) | 无增益，closed（error corr 0.98-0.99） |
| global-sequence features / DeepSets full LOO | 无增益 / 资源不可行 | closed |

（完整 fail-closed 表见 `docs/paper/position_aware_method_chapter.md` §4）

## 5. Honesty note（已入文档）

早期全量版（277,451 行）报 +14.08% / +1.64pp，但混入 4,463 行未匹配位置（deep 被降为 median），虚增 blend-vs-deep 增益。**最终采用 matched-only（272,988 行）的 +14.06% / +1.21pp**，与官方 3-way 报告同覆盖、同 158 designs——公平可发表。

## 6. 复现命令（服务器 worktree）

```bash
cd /home/cunyuliu/reactflow_delta_worktrees/benchmark_v3_20260809/scripts/reactflow_delta
# 本地/服务器单元测试
python -m pytest test_m2_gbdt_features_v1.py test_m2_gbdt_ensemble_v1.py \
  test_m2_masked_eval_from_oof.py test_m2_spectrum_submission_table_v1.py -q
# 完整重跑（含 BPP 特征构建 ~30-50 min + GBDT LOO ~14 min）
bash run_m2_gbdt_3way_ensemble.sh        # 全量版（作审计基线）
# matched-only 诚实评估（从 OOF 直接计算，秒级）
python m2_masked_eval_from_oof.py \
  --oof /mnt/cunyuliu/m2_gbdt_3way_ensemble_20260818/m2_gbdt_ensemble_oof.npz \
  --out /mnt/cunyuliu/m2_gbdt_3way_ensemble_matched_20260818
# 提交表
bash run_m2_submission_final.sh
```

## 7. 测试状态

- 新增/修改测试：test_m2_gbdt_features_v1（leak-free 强制）、test_m2_gbdt_ensemble_v1（mask 对齐）、test_m2_masked_eval_from_oof（mask 检测 + block CI/perm）、test_m2_spectrum_submission_table_v1（提交表结构）
- 本地 `9 passed`（features+ensemble）+ `3 passed`（masked-eval）+ `2 passed`（submission table）
- 服务器完整套件在提交时一并运行验证
