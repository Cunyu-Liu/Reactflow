# ReactFlow-Delta: A Matched WT–Single-Mutant Reactivity-Probing Resource and a Rigorous Negative Result on Cross-Publication Mutation-Effect Prediction (Draft)

状态：`DRAFT_FOR_REVIEW`（资源/负结果路线交付物，非投稿稿；所有数字均可由
`results/phase3_*_20260809/*report.json` 与 `phase3_diagnostic_table.tsv` 复现）
日期：2026-08-09　Authority：epoch 18（Phase 3 失败处置→benchmark/resource）
合同：`ReactFlowDelta_M0X_strict_scientific_engineering_audit_20260807.md`

---

## 1. Introduction (scope of this draft)

本稿只回答一个可证伪问题：**在 provenance-complete、mask 正确、publication-disjoint、
test 未消费的 matched WT–single-mutant probing benchmark 上，mutation-induced 实验反应性
response 是否可被跨 publication 预测；若可，pair-aware 与受控非局部传播是否带来超过最强
同信息条件基线的稳定增益。** Phase 3 的三种授权架构方案均在此 endpoint 上 fail-closed。
本稿给出数据/测量层的诊断解释，并把数据集本身作为一个可复用资源正式化。

不主张 SOTA、不主张 confirmatory 跨 publication 正向增益（`scientific_claim_authorized=false`）。

## 2. Methods

### 2.1 Data resource
- 源：1024 个 frozen assets（`data_registry/d0x/`），解析出 matched WT–single-mutant probing
  pairs。本文域（pool，剔除 test studies）为 **6385 对、18 个 publication**。
- 每对 = (WT reactivity, single-nucleotide mutant reactivity, 条件, exact ref/alt/pos)；
  不含 mutant profile 作为输入（信息权限干净）。
- 预测 endpoint：`pair_magnitude` = eligible positions 上的 mean |mut − WT| reactivity；
  `C_i` 由 train-only replicate-aware caller 定义（CallerV3）。

### 2.2 Caller（标签来源）
- frozen CallerV3（seed 20260807），在全部 pool training replicate groups 上拟合。
- 单元可靠性：global ICC(1,1) = **0.686**；per-group ICC 中位 0.827（p10 0.062，min −0.034，
  max 1.0），阈值 0.5；仅 **41** 个 ≥2-replicate 组可用于噪声估计。

### 2.3 诊断度量
- **NO_CALL 率**：caller 因结构或可靠性 < 阈值无法打标的 pair 比例。
- **changers 率**：被调用 pair 中 label=1 的比例（跨 publication 的标签域漂移）。
- **magnitude-vs-noise 比**：逐 position |delta| / within-WT replicate 噪声 std；
  报告 <1× 与 <1.96× 噪声的比例。
- **feature 域漂移**：generic [WT,Mut,cond] 特征 per-publication 均值相对 pool 训练均值的
  standardized mean difference（SMD）。

## 3. Results

### 3.1 Caller 覆盖率低
- **50.2%**（3204/6385）pair 为 **NO_CALL**——caller 无法打标。
- 被调用 pair 的总体 changers 率 **0.753**（多数被调用 pair 即 changer；类别高度不平衡）。

### 3.2 强 publication 标签漂移
- 18 个 publication 中 **8 个没有任何可调用 pair**（全部 NO_CALL）。
- 10 个可调用 publication 的 changers 率从 **0.048（HC16M2R）到 1.0**（pmid_25883046 /
  pmid_35982307），**max/min = 20.7×**。

### 3.3 magnitude 信号近 replicate 噪声底
- **44.7%** 的 (pair, position) 突变效应 **低于 1× within-WT replicate 噪声**；**61.95%** 低于
  1.96× 噪声。ratio 稳健中位数 1.23（p25 0.41, p75 3.65）。
- 分布重右尾（部分 replicate position 近零方差 → 极端 ratio，p99~1e15）；故只用稳健分位数与
  计数比例作结论，不使用均值（该重尾由近零方差 position 造成，不影响 below-1x 计数）。
- **29.7%** 的 pair 无噪声估计（WT 为 singleton，无 ≥2 replicate）。
- 主导 publication pmid_29446752（占被评估 positions 的 ~76%）41.8% below-1x；
  其余可评估 publication 达 66–85%。

### 3.4 中等 feature 域漂移
- generic 特征 overall mean abs SMD = **0.327**（per-publication 0.138–0.375）。

### 3.5 Phase 3 架构负结果（endpoint_v5, 同容量 generic 对比）
| 方案 | candidate mean skill | generic mean skill | diff CI low | 裁决 |
|---|---|---|---|---|
| PairHeadV1 (DeepSets) | 0.684 | 0.668 | <0 | FAIL-CLOSED |
| exact-alt 显式交互 | 0.722 | 0.679 | <0 | FAIL-CLOSED |
| 修复 EPRO 传播 | 0.657 | 0.679 | −0.226 | FAIL-CLOSED |

三种方案全部在 paired publication-block bootstrap 的 CI 下界 >0 门槛下不胜同容量 generic。

### 3.6 负结果在 publication 层一致（uniformity check）
以方案三为例，对每个 held-out publication 计算 EPRO vs generic 的 skill 差
（LOOCV 下 per-publication skill = per-fold skill；`run_phase3_per_pub_skill.py`，
`results/phase3_per_pub_skill_20260809/per_pub_skill.json`）：

- 9 个可分析 publication 中，EPRO **在所有 5 seeds 都胜 generic 的只有 2 个**（pmid_25883046 340
  对、RNASEP 75 对，均小样本），且非跨 publication 一致。
- 最大 publication（pmid_29446752，8960 对）diff=+0.017，仅 60% seed 为正（不显著）。
- overall diff mean = **−0.020**（EPRO 平均略差）；min −0.56，max +0.76。
- 结论：没有任何 publication 给出 EPRO 相对 generic 稳定、跨 seed 一致的增益——负结果在
  publication 层一致，非仅 aggregate 假象。

## 4. Discussion / 解释

三条独立证据链共同解释架构负结果，且都落在数据/测量层而非「模型不够复杂」：

1. caller 覆盖率低（50% NO_CALL）+ 被调用类别高不平衡（changer 占 75%）；
2. 标签域跨 publication 强漂移（changer 率 20.7×，8/18 publication 无可调用样本）；
3. magnitude 目标近 replicate 噪声底（45% below 1×，62% below 1.96×）。

在此组合下，任何同容量候选相对 generic 的增量都落在 publication 级噪声内，无法产生
稳定、可跨 publication 复现的增益。因此本稿不推进架构搜索，而是：
- 正式化该 **benchmark/resource**（matched WT–mutant probing，含 caller 与噪声口径）；
- 给出诚实的 **negative result** 及其数据层归因。

## 5. 边界与限制（诚实声明）
- 本稿为 `DRAFT_FOR_REVIEW`，未做正式 claim freeze；`scientific_claim_authorized=false`。
- 未读取 test 样本/标签，未做 confirmatory 跨 publication 正向主张。
- 大型 .npz / 权重不入库；代码、报告、表已入库（`codex/reactflow-delta-d0r`）。
- 若未来获得新授权，可在此资源上重开预注册的 confirmatory 实验，但不得解封 test。

## 6. 复现
- 三个诊断：`run_phase3_benchmark_resource.py`、`run_phase3_noise_floor.py`、
  `build_phase3_diagnostic_table.py`（命令见各脚本 `--help`）。
- 主证据表：`results/phase3_diagnostic_20260809/phase3_diagnostic_table.{json,tsv}`。
- 架构结果：`results/phase3_{v1,scheme2,scheme3}_20260809/`。
