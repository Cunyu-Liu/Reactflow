# REPRODUCE — ReactFlow-Delta prospective-v2 replay (P6)

> `run_replay_v1.py` **不运行 P0/P1**。默认只重放内部开发产物 P2/P3；仅显式加入 `--external` 后才可能进入 P4/P5/P5b/P5_COMBINED。
> 数据/产物路径为运行机器上的绝对路径；代码在仓库内。P2/P3 为 artifact 重放（由保存的 per-position predictions 重算主估计，不重训——GPU 全量重训 >8h）；P4/P5/P5b 为 fresh 外部重放，P5_COMBINED 为报告级聚合。
>
> **当前外部重放被拒绝。** 脚本固定读取仓库内 `configs/reactflow_delta/active_contract.yaml`；授权检查先于任何输出目录创建和回放输入读取。只有顶层和 `authorization` 内的 `new_external_outcome_access_allowed` 均为布尔值 `true`，且 `authority.current_runnable_phase` 精确等于 `P6_EXTERNAL_REPLAY` 时，`--external` 才会继续。2026-08-27 当前合同两处权限均为 `false`、阶段为 `V14M3`，因此不得执行外部命令。

## 1. 环境

```bash
# 远程机器（Ubuntu, 已装 miniconda）
conda activate editflow   # python 3.10, numpy 1.26.4, scipy 1.15.3, torch 2.5.1+cu121, pandas 2.2.2
# 或从 docs/prospective_v2/submission/environment.yml 重建
```

## 2. 代码

```bash
cd /home/cunyuliu/reactflow_delta_worktrees/model_rescue_v14_staging_20260827
git checkout codex/reactflow-delta-model-rescue-v14-staging-20260827
export PYTHONPATH=.:src
```

## 3. 默认：仅内部 P2/P3 artifact 重放

该命令在当前合同下可运行；它不会调用 P4/P5/P5b/P5_COMBINED，也不需要它们的参数。

```bash
python scripts/reactflow_delta/run_replay_v1.py \
  --locked-p2 docs/prospective_v2/p2_direct_v2_result_20260813.json \
  --locked-p3 docs/prospective_v2/p3_lrso_v3_result_20260815.json \
  --p2-held-rows /mnt/cunyuliu/prospective_v2_p2_preds_20260813/p2_held_position_rows.jsonl \
  --out /mnt/cunyuliu/prospective_v2_p6_20260814/internal_replay_report.json
```

预期报告只含 `P2`、`P3`，`replay_mode: internal_artifact_only`。`REPLAY_CONSISTENT` 仅表示这两个历史 artifact 的评分/统计重算一致。

## 4. 外部 P4/P5/P5b/P5_COMBINED（当前禁止，仅保留授权后命令）

只有上述三项合同条件同时成立后，才可执行以下命令。脚本先读取固定的 canonical authority，再校验全部必需参数；拒绝时不创建 `--replay-out`/`--out`，也不读取回放输入。

```bash
python scripts/reactflow_delta/run_replay_v1.py \
  --external \
  --locked-p2 docs/prospective_v2/p2_direct_v2_result_20260813.json \
  --locked-p3 docs/prospective_v2/p3_lrso_v3_result_20260815.json \
  --p2-held-rows /mnt/cunyuliu/prospective_v2_p2_preds_20260813/p2_held_position_rows.jsonl \
  --dev-csv  /mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/openknot_m2/OK7a_M2_data.v4.5.2.csv \
  --rdat-dir /mnt/cunyuliu/reactflow_delta_raw/rmdb/rdat_tierA_20260730 \
  --components /mnt/cunyuliu/prospective_v2_p4_20260813/p4_external_components.json \
  --locked-p4 /mnt/cunyuliu/prospective_v2_p4_20260813/p4_external_result.json \
  --locked-p5 /mnt/cunyuliu/prospective_v2_p4_20260813/p5_mechanism_result.json \
  --locked-p5b /mnt/cunyuliu/prospective_v2_p4_20260813/p5b_mechanism_result.json \
  --p5b-components /mnt/cunyuliu/prospective_v2_p4_20260813/p5b_external_components.json \
  --locked-p5-combined results/p5_combined_artifacts_20260813/p5_combined_meta_result.json \
  --replay-out /mnt/cunyuliu/prospective_v2_p6_20260814/replay \
  --out /mnt/cunyuliu/prospective_v2_p6_20260814/replay_report.json
```

授权后的预期输出：`replay_mode: external_authorized`、`verdict: REPLAY_CONSISTENT`、`all_reproduced: true`。
注：P5b 重放是 fresh（重新执行 frozen 协议），在 NEW 独立组件集（M2RFOK/M2RFPK）上运行，locked outcome access count = 2。
注：P5_COMBINED 是 report-level 聚合重放（`run_p5_combined_meta_v1.evaluate_combined` 对两个 per-set 报告重算）；它本身不触碰 raw rdat，但仍只能经显式 `--external` 和同一合同授权进入。

## 5. 主表/图与 cards（历史命令；当前不要读取外部 locked 产物）

```bash
python scripts/reactflow_delta/generate_p6_tables_figures_v1.py \
  --p2-result docs/prospective_v2/p2_direct_v2_result_20260813.json \
  --p3-result docs/prospective_v2/p3_lrso_v3_result_20260815.json \
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

## 6. 阶段与实际入口映射

| 阶段 | 重放类型 | 复现项 |
|---|---|---|
| P2 | 默认内部 artifact（无重训） | per-puzzle D_p2 + 20-puzzle CI（由 975,600 held rows 重算，`P2_PUZZLE_RTOL=0.1`）|
| P3 | 默认内部 artifact（无重训） | 从 v3 `rank_d_p3` 重算 rank2/4/8 的 20-puzzle CI + verdict（`LRSO_EXCEEDS_DIRECT_FOR_DEVELOPMENT`）|
| P4 | 仅 `--external` fresh | 重跑 frozen 外部协议（verdict + CI）|
| P5 | 仅 `--external` fresh | 重跑 frozen 机制 contrasts（verdict）|
| P5b | 仅 `--external` fresh、可选 | 同时提供 `--locked-p5b` 与 `--p5b-components` 后重跑独立集机制协议 |
| P5_COMBINED | 仅 `--external` report-level、可选 | 同时提供 P5b 成对输入和 `--locked-p5-combined` 后重算跨集合聚合 |
| P0/P1 | 无入口 | `run_replay_v1.py` 不运行这两个阶段 |

## 7. 已知边界

- P2 的 P20 per-puzzle D 因历史 NaN 分母 artifact 有约 8% 相对差异；代码锁定 `P2_PUZZLE_RTOL=0.1`，20-puzzle CI 与 verdict 不变。
- 全量 GPU 重训（P2/P3 训练本身）未在 replay 中重跑；per-position predictions 为 locked 产物，replay 验证其评分/统计层可复现。
- `REPLAY_CONSISTENT` / `all_reproduced=true` 只说明对应历史计算可以重放；历史 replay **不能恢复、提升或改写任何已失败、已撤回或当前关闭的 qualification**，也不是新的科学结果。

## 8. P3 规范重跑（run_p3_lrso_v3.py，2026-08-15 → 08-16 完成，PASS）
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
