# DeltaFlow Task 7 预注册冻结（Phase 2）

> 生成：2026-09-10；执行人：codex agent（用户指令下的自动化执行）
> 依据：用户已确认的 Phase 1 冻结表（`phase1_analysis/task5_freeze_decision.md`，2026-09-08 生效为 Phase 2 预注册基线）
> 本文件为 Task 7.1 冻结 commit 的内容主体。**冻结后不得回改**；DFLOW3 正式 5-seed 运行（seeds 0–4 × folds 0–19 双臂）以本文件为准。
> 零编造声明：本文件全部数值逐字引自冻结表 F1–F8 与 DFLOW2 实测产物（/mnt/cunyuliu/reactflow_delta_deltaflow_dflow2_seed0）。

---

## F1. seed 协议（逐字引用）

- seeds 0–4 × folds 0–19；5-seed 等权 0.2 mixture assembly；无 best-seed、无 extra seed（上限 8 不启用）。

## F2. 四类主指标门槛（逐字引用，双档 MDE）

主比较 gate：DeltaFlow vs 分解式 null（同容量、去噪头对角化）。四类主指标同时满足三条方为 PASS：
1. assembly 级 paired 95% CI 改进方向显著（N=20，t 分布 df=19，t₀.₉₇₅=2.093024）；
2. ≥4/5 seeds 单独方向为正（每 seed 自身 20-fold assembly 聚合改进量为正）；
3. 装配级聚合改进% ≥ 判定档 MDE 数值。

| 指标（vs 分解式 null） | MDE%（独立档，primary） | MDE%（保守档，audit） |
|---|---|---|
| signed-delta MAE | 1.17 | 2.61 |
| point absolute-delta MAE | 1.73 | 3.86 |
| task CRPS | 0.72 | 1.62 |
| distribution-absolute MAE | 0.81 | 1.80 |

qualifier-once 机械执行：Step 1 独立档 primary 判定 → Step 2 实测复核（ρ、σ_assembly、实测档）→ Step 3 分级（PASS strong / PASS with caveat / FAIL）。

## F3. 联合结构指标门槛（逐字引用，与主指标分开报告、不互相补底）

| 联合指标 | PASS | MARGINAL | FAIL |
|---|---|---|---|
| profile shape correlation（per-mutant Pearson 中位） | ≥0.60 | ≥0.546 | <0.546 |
| distal localization top-1 | ≥0.066（3×随机 0.0219） | ≥0.044 | <0.044 |
| distal localization top-3 | ≥0.109（3×随机 0.0362） | ≥0.072 | <0.072 |
| energy score（vs 自身边际置换 null） | 优于 null ≥1% | — | <1% |
| joint coverage（PCA2） | ∈[0.30, 0.50] | — | 区间外 |

MARGINAL 不计 gate；五指标全部出档位。

## F4. 组件清单（逐字引用 + 实现绑定）

| 组件 | 冻结值 | 实现（commit） |
|---|---|---|
| Stage 1 point 机器 | V14 同款（256/6/8/1024）无 masked-WT pretraining；40+40 epochs | run_deltaflow_fold.py（V14PointModel 复刻） |
| Stage 2 CFM 生成器 | rectified flow；pair-representation 去噪器（1D conv + self-attention + 三角乘法）；条件 c 含 RNet2 teacher + Stage-1 输出 | deltaflow.py（commits 222cf45/fe29512） |
| matched null | 同容量/参数量/初始化/训练协议/推断步数，唯一差异去噪头对角化 | deltaflow.py make_flow_pair（dda8437 修复后） |
| 训练配置 | AdamW lr 2e-4、wd 0.01、batch 16（mutant 级）、clip 1 | run_deltaflow_fold.py |
| ODE 采样 | Euler 50 步；16 draws/臂（边际统计聚合） | deltaflow.py euler_flow_sample |
| V14 预训练/B5 | 不进主线 | 未使用 |

## Stage-2 终 epoch：冻结 = 100

依据（DFLOW2 实测，20 folds 池化）：
- candidate 池化 loss：epoch 1 = 0.3979 → epoch 40 = 0.2697 → epoch 100 = 0.2424（全程降 39.1%）；
- **epoch 99 仍存在 trailing-10-window 相对降幅 ≥1% 的窗口（41–100 区间 19/60 个窗口 ≥1%）——100 epochs 仍未进入持续平台**，收敛规则在 epoch 40 的触发为短暂中途平台误报（epoch 45–50 窗口降幅回升至 2.13%/1.50%）；
- null 臂同形：epoch 1 = 0.4267 → epoch 100 = 0.3033（28.9%），最后 ≥1% 窗口在 epoch 93；
- 20/20 folds candidate 末 epoch loss < null 末 epoch loss（起点即分化的结构性差距，无任何 fold 例外）；
- 时间预算：100 epochs 下 5 seeds × 20 folds × 双臂网格可在本周内完成；>100 会突破交付窗口且末段增益 <2%/10-epoch。
- **epoch 100 为本次冻结值；若 DFLOW3 需更多 epoch（本冻结禁止），须走 amendment 而非静默改动。**

## 其余冻结项（逐字引用）

- F5 外部数据双轨（轨 A C01/C02/C04 符号级；轨 B C03 semi-external；泄漏禁用清单）——Phase 3 执行时逐字引用。
- F6 定案输入、F7 已登记风险、F8 移交义务——按冻结表原文有效。
- 数据：OpenKnot M2 v4.5.2（13976 mutants、EXACT_PUZZLE_METHOD_MUTATION）；切分 split_v4（seed 20260813）；CUDA fail-fast；产物 /mnt/cunyuliu；代码 /home。

## 运行计划（冻结后执行）

1. DFLOW3：seeds 1–4 × folds 0–19（seed0 复用 DFLOW2 产物；DFLOW2 层级 = screen，DFLOW3 = 正式 5-seed）——grid controller（63e3710）调度。
2. 唯一 canonical unscored merge（每 seed 一份，merge-once）。
3. score-once：四类主指标 vs null（assembly 级）+ 联合指标（deltaflow_joint_metrics.py，F3 口径）+ 参考层。
4. qualifier-once：F2 Step 1–3 机械判定 + F3 档位，一次性，结果无论好坏如实登记。

## Git 绑定

- 分支：codex/reactflow-delta-deltaflow-impl-20260908（GitHub Cunyu-Liu/Reactflow）
- 冻结前 commits：222cf45 → fe29512 → dda8437 → c012e3f → 63e3710 → a9c8e24
- 本冻结 commit 即预注册时戳；此后代码层面的任何协议改动均违反本冻结。
