# REPRODUCE — ReactFlow-Delta prospective-v2 one-click replay (P6)

> 从 clean checkout 重放 P0–P5 主结果。`run_replay_v1.py` 一次性输出 `REPLAY_CONSISTENT` 即通过验收（合同 12.8）。
> 数据/产物路径为运行机器上的绝对路径；代码在仓库内。P2/P3 为 artifact 重放（由保存的 per-position predictions 重算主估计，不重训——GPU 全量重训 >8h 且结果已被该方式复算验证）；P4/P5/P5b 为 fresh 重放（重新执行 frozen 协议）。

## 1. 环境
```bash
# 远程机器（Ubuntu, 已装 miniconda）
conda activate editflow   # python 3.10, numpy 1.26.4, scipy 1.15.3, torch 2.5.1+cu121, pandas 2.1.4, matplotlib 3.8.4, pytest 8.1.1
# 或从 docs/prospective_v2/submission/environment.yml 重建
```

## 2. 代码
```bash
cd /home/cunyuliu/reactflow_delta_worktrees/prospective_v2_20260813
git checkout codex/reactflow-delta-prospective-v2-20260813
export PYTHONPATH=.:src
```

## 3. 一键重放
```bash
mkdir -p /mnt/cunyuliu/prospective_v2_p6_20260814
python scripts/reactflow_delta/run_replay_v1.py \
  --dev-csv  /mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/openknot_m2/OK7a_M2_data.v4.5.2.csv \
  --rdat-dir /mnt/cunyuliu/reactflow_delta_raw/rmdb/rdat_tierA_20260730 \
  --components /mnt/cunyuliu/prospective_v2_p4_20260813/p4_external_components.json \
  --locked-p2 docs/prospective_v2/p2_direct_v2_result_20260813.json \
  --locked-p3 docs/prospective_v2/p3_lrso_v2_result_20260813.json \
  --locked-p4 /mnt/cunyuliu/prospective_v2_p4_20260813/p4_external_result.json \
  --locked-p5 /mnt/cunyuliu/prospective_v2_p4_20260813/p5_mechanism_result.json \
  --locked-p5b /mnt/cunyuliu/prospective_v2_p4_20260813/p5b_mechanism_result.json \
  --p5b-components /mnt/cunyuliu/prospective_v2_p4_20260813/p5b_external_components.json \
  --locked-p5-combined results/p5_combined_artifacts_20260813/p5_combined_meta_result.json \
  --p2-held-rows /mnt/cunyuliu/prospective_v2_p2_preds_20260813/p2_held_position_rows.jsonl \
  --replay-out /mnt/cunyuliu/prospective_v2_p6_20260814/replay \
  --out /mnt/cunyuliu/prospective_v2_p6_20260814/replay_report.json
```
预期输出：`verdict: REPLAY_CONSISTENT`，`all_reproduced: true`。
注：P5b 重放是 fresh（重新执行 frozen 协议），在 NEW 独立组件集（M2RFOK/M2RFPK）上运行，locked outcome access count = 2。
注：P5_COMBINED 是 report-level 聚合重放（`run_p5_combined_meta_v1.evaluate_combined` 对两个 locked per-set 报告重算），校验 overall verdict = MECHANISM_EVIDENCE_PASS、529 组件总数、6 条合取子标准全 PASS、4 条 caveats 数量、per-set fail-closed verdict 保留。该步骤不触碰 raw rdat，不产生新 outcome access。

## 4. 主表/图与 cards（可选，直接读 locked 产物）
```bash
python scripts/reactflow_delta/generate_p6_tables_figures_v1.py \
  --p2-result docs/prospective_v2/p2_direct_v2_result_20260813.json \
  --p3-result docs/prospective_v2/p3_lrso_v2_result_20260813.json \
  --horizontal docs/prospective_v2/horizontal_compare_p2_20260813.json \
  --p4-result /mnt/cunyuliu/prospective_v2_p4_20260813/p4_external_result.json \
  --p5-result /mnt/cunyuliu/prospective_v2_p4_20260813/p5_mechanism_result.json \
  --p5b-result /mnt/cunyuliu/prospective_v2_p4_20260813/p5b_mechanism_result.json \
  --p5-combined-result results/p5_combined_artifacts_20260813/p5_combined_meta_result.json \
  --calib-result /mnt/cunyuliu/prospective_v2_p4_20260813/p4_calibration_result.json \
  --replay-report /mnt/cunyuliu/prospective_v2_p6_20260814/replay_report.json \
  --out-dir /mnt/cunyuliu/prospective_v2_p6_20260814/out
```
（cards 见 build_p6_cards_v1.py 与 docs/prospective_v2/p6_environment.json）
产出：Table 1–4 含 Table 3b/3c/3c_claim_map，Fig1–6 含 Fig6 P5 combined claim map。

## 5. 复现了什么
| 阶段 | 重放类型 | 复现项 |
|---|---|---|
| P2 | artifact（无重训） | per-puzzle D_p2 + 20-puzzle CI（由 975,600 held rows 重算，rel tol 1e-3）|
| P3 | artifact（无重训） | rank2/4/8 20-puzzle CI + verdict（v3: LRSO_EXCEEDS_DIRECT_FOR_DEVELOPMENT）|
| P4 | fresh | 重跑 frozen 外部协议（verdict + CI）|
| P5 | fresh | 重跑 frozen 机制 contrasts（verdict）|
| P5b | fresh | 重跑 NEW 独立集机制协议（verdict + very-far CI + K_eff）|
| P5_COMBINED | report-level 聚合 | overall verdict MECHANISM_EVIDENCE_PASS + 529 comps + 6 conjuncts + 4 caveats + per-set fail-closed 保留 |

## 6. 已知边界
- P2 的 P20 per-puzzle D 有 ~1e-4 浮点聚合路径差异（rel tol 1e-3 内，verdict/CI 不变）。
- 全量 GPU 重训（P2/P3 训练本身）未在 replay 中重跑；per-position predictions 为 locked 产物，replay 验证其评分/统计层可复现。
- 结果一致性的最终裁决依据 `run_replay_v1.py` 的 `all_reproduced`。

## 7. P3 规范重跑（run_p3_lrso_v3.py，2026-08-15 → 08-16 完成，PASS）
> v1/v2 的 `NO_INCREMENTAL_LRSO_SKILL` 已撤回（实现未按冻结规格）。v3 规范重跑 2026-08-15 14:35 → 08-16 15:29（~24.9h，GPU cuda:3，A100-40GB，20 fold × ranks {2,4,8}，HP-selected cfg lr=1e-3/wd=0/Student-t，torch.compile on）PASSED 合同 12.5。结果：rank2 D_p^P3=+0.0147 [95% CI +0.0119,+0.0175]、rank4 +0.0155 [+0.0113,+0.0196]、rank8 +0.0154 [+0.0122,+0.0185]；ci_low_gt_0=True；20/20 puzzle 正向；sign-flip p=1.9e-6；LOO max shift ≤0.001。LRSO 开发集增量技能已确立；外部可移植性未测（外部协议冻结于 direct candidate）。

v3 关键特性：trainable WT encoder（无 detach）；missing target 永不等于 0（mask 在填充前从 raw 算）；inner 4-fold puzzle-grouped 验证选择 {lr,wd,likelihood} 与 early-stopped epoch（max 200, patience 20）；五 seed {0..4} 等权 Gaussian mixture CRPS；scale softplus+floor 正参数化。

```bash
# 最终执行的命令（2026-08-15）
cd /home/cunyuliu/reactflow_delta_worktrees/prospective_v2_20260813
conda activate rna_junction_preorganization_v1_1
export PYTHONPATH=.:src
nohup python scripts/reactflow_delta/run_p3_lrso_v3.py \
  --m2-csv  /mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/openknot_m2/OK7a_M2_data.v4.5.2.csv \
  --out-dir /mnt/cunyuliu/prospective_v2_p3_20260815 \
  --device cuda:3 --rank 2,4,8 --max-epochs 200 --patience 20 \
  --no-inner-select --compile \
  > /mnt/cunyuliu/prospective_v2_p3_20260815/run_p3_v3.log 2>&1 &
# 输出 p3_lrso_v3_result.json；按合同 12.5 用 ci_rank_{2,4,8}.ci_low_gt_0 裁决
# `--no-inner-select` 使用预先一次性 HP 选择（lr=1e-3, wd=0, Student-t, inner CRPS 0.18982）
# 而非每 fold 再做 6h 网格搜索；`--compile` 启用 torch.compile 加速。
```
- 结果 artifact: `docs/prospective_v2/p3_lrso_v3_result_20260815.json`（本地 worktree）和 `/mnt/cunyuliu/prospective_v2_p3_20260815/p3_lrso_v3_result.json`（远端 raw）。
- P3 重跑 replay 验证：`_replay_p3` 从 locked artifact 重算 20-puzzle CI 并正确得出 `LRSO_EXCEEDS_DIRECT_FOR_DEVELOPMENT` verdict（CI 精度 1e-12）。
