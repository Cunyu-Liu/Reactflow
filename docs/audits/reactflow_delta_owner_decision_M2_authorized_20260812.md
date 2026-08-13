# ReactFlow-Delta Owner Decision — M2 Residual Method Work Authorized (2026-08-12)

- date: 2026-08-12
- project: reactflow_delta / benchmark_v3_20260809
- decision type: owner authorization / scope correction
- owner: 用户（owner）

## 1. 决定

Owner 明确授权：**M2 残差方法工作作为"已授权的方法"纳入项目**（原始表述：
"帮我从 OpenKnot M2 数据集加载数据，把 N 扩充到 100 以上，重新跑一遍残差学习 +
per-position median prior 的模型看看效果"；并确认"M2 残差工作作为'已授权的方法'纳入，
我允许的"）。

本决定**覆盖** Phase 2 硬停止（`METHOD_ROUTE_STOP_LEARNABILITY_NOT_ESTABLISHED`）对
M2 残差方法探索的默认禁令，授予该**特定方法工作**的 owner 级授权。

## 2. 授权范围

- OpenKnot M2 数据集成（`m2_data_v1.py`、`m2_caller_v1.py`），以 (puzzle×method)
  design 为交换单元，N=159 ≥ 100；
- 残差学习 + per-position median prior 全谱响应模型（`run_response_spectrum_m2_v1.py`，
  含 5 seeds）及其在 A100 GPU 上的 LOO 训练；
- 对比与显著性/诊断分析（`compare_m2_spectrum.py`、`analyze_m2_significance.py`、
  `diagnose_m2_design.py`）。

## 3. 授权后结果状态（方法开发证据）

- held-out 残差模型相对 per-position median prior 的改进：
  - design 级 sign-flip permutation **p < 1e-5**（z=11.12）；
  - Wilcoxon signed-rank p = 1.7e-27；单样本 t p = 8.9e-54；
  - bootstrap 95% CI [0.0490, 0.0578]（不含 0）；Cohen's d = 1.88；
  - 97.5% designs（154/158）improved；5 seeds 一致。
- 位置诊断（OK7a_M2_P12_Shujun_shape_struct2seq，100 changers）：残差模型在全部
  21 个窗口位置优于 baseline，改进集中在编辑位点（k=10 改进 0.5375）并随距离衰减。

## 4. 未被本决定覆盖的边界（仍生效，fail-closed）

- `confirmatory_test_outcome_access_allowed = false`：**不得打开 confirmatory test
  outcome/label/prediction**；training on confirmatory test 仍禁止。
- Phase 4 locked test 仍被阻塞：`test_statistical_sufficiency = NOT_ESTABLISHED`
  （尚无 untouched、provenance-confirmed 的 confirmatory 出版物集合；统计设计 N≥6）。
- `SOTA_NOT_ESTABLISHED` 维持：不得写 first/world-first/无竞争/领域 SOTA。
- 本结果属于**方法/开发证据**，不是发表释放；投稿路线仍需
  `AUTHORIZE_PHASE5_6_PUBLICATION_RELEASE`。

## 5. 证据

| 证据 | 路径 / 标识 |
|---|---|
| 显著性结果 | `/mnt/cunyuliu/m2_response_spectrum_20260812/significance_result.json` |
| 对比结果 | `.../compare_result.json` |
| 位置诊断 | `.../diagnose_P12_shape_struct2seq.json` |
| keyed predictions | `.../keyed_predictions_m2_spectrum.jsonl` (sha256 `54790e34…`) |
| 运行 manifest | `.../response_spectrum_m2_manifest.json` (wall 32,309 s) |
| 提交 | `03df1a0` (pushed to `Cunyu-Liu/Reactflow` `codex/reactflow-delta-benchmark-v3-20260809`) |
