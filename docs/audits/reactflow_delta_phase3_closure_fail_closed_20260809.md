# ReactFlow-Delta Phase 3 (Model Architecture Iteration) — FAIL-CLOSED CLOSURE

审查日期：2026-08-09
Authority：epoch 18（`reactflow_delta_v4_epoch18_phase3_approval_20260808.yaml`）
合同：`ReactFlowDelta_M0X_strict_scientific_engineering_audit_20260807.md` §9 / §12 Phase 3
Endpoint：endpoint_v5（conditional WMAE skill vs trivial）
分支：`codex/reactflow-delta-d0r`（HEAD `dc5530d`）

## 1. 阶段目标与授权范围

Phase 3 只检验三种明确能力（§12 Phase 3）：pair-level 对齐、exact-alt × WT-state、
可控 nonlocal 传播；识别其独立增量，并只在 **P2 gate 通过后** 才做架构研究（§9.1, 943）。
三种方案逐一在 nested leave-one-publication-out 上与**同容量 generic concat 基线**
（[WT, Mut, cond]，`build_scheme2_features(use_wt_anchor=True)`）比较，
paired publication-block bootstrap CI 下界 > 0 为成功门槛（§12 Phase 3 验收标准）。

## 2. 三种方案执行与裁决（全部 fail-closed）

| 方案 | 代码 | candidate mean skill | generic mean skill | diff CI low min | 裁决 |
|---|---|---|---|---|---|
| 一：PairHeadV1 (DeepSets) | `models/pair_v1.py`, `run_phase3.py` | 0.6838 | 0.6679 | < 0 | `FAIL_CLOSED_RETIRED` |
| 二：exact-alt WT/mutant 显式交互 | `models/pair_v2.py`, `run_phase3_scheme2.py` | 0.7220 | 0.6785（concat） | < 0 | `FAIL_CLOSED_RETIRED` |
| 三：修复 EPRO 非局部传播 | `models/epro_v1.py`, `run_phase3_scheme3.py` | 0.6565 | 0.6785 | −0.2263 | `FAIL_CLOSED_RETIRED` |

- 方案三另有消融：epro_local（无传播）0.6496 < epro 0.6565（弱 PASS），
  epro_random（随机 contact）0.6589 ≥ epro 0.6565（FAIL）——真实 base-pair contact
  传播未提供可测增量，非局部传播假设未被支持。
- 全部 5 seeds、多个 held-out publications（方案三 10 个 retained publications），
  CI 下界均 ≤ 0。无一在预注册标准下胜过同容量 generic。

## 3. 处置（§9.2/9.3/9.4 与 §12 Phase 3 失败处理）

按合同预注册处置：

1. **退役架构主张：** EPRO propagation、structure/contact 增量、two-stage response head
   已公平证伪（§9.5「待公平证伪」→ 退役）。
2. **保留最简单 generic 作为 development 默认 winner：** 不使用任何更复杂候选；不再开
   dev13+ 自由架构搜索（§12 Phase 3 失败处理：`全部不胜则使用最简单generic并转benchmark/resource`）。
3. **转入 benchmark/resource/negative-result 路线：** 按 §9.2/9.3 失败处理
   「分析 caller reliability 和 domain shift」「重点转向数据 reliability 或资源论文」。
   Phase 4（SOTA 对比）因无 winner 且前置（≥3 untouched publications、confirmatory test
   未暴露）在本次审计中未闭合，不自动进入。

## 4. 本记录的边界

- 本记录是 Phase 3 的**可审计阶段产物**，不改动 active_contract/bundle/sentinel；
  正式 authority 再绑定需用户按治理流程处理（审计 §2.1 已提示 authority 完整性需修复）。
- 三个方案结果/heldout 留在 `results/phase3_{v1,scheme2,scheme3}_20260809/`（不入库，大 .npz）。
- 代码与测试已提交（`2dd103c`, `7c92fda`, `dc5530d`）。

## 5. 下一阶段（benchmark/resource 路线）

首个交付物：caller reliability + domain-shift 表征分析（`run_phase3_benchmark_resource.py`），
回答「三种架构为何全部不胜 generic」——区分是 caller 标签不可靠、publication 域漂移，
还是任务处于噪声底。该分析为资源/负结果论文提供证据，且无需新架构授权。

### 5.1 表征结果（`results/phase3_benchmark_resource_20260809/benchmark_resource_report.json`）

**Caller 可靠性（frozen global caller, seed 20260807）：**
- global ICC(1,1) = 0.686；per-group ICC median 0.827，但 p10 = 0.062、min = −0.034、max = 1.0（范围极大）。
- **50.2% 的 pair 为 NO_CALL**（3204/6385）——caller 因结构或单元可靠性 < 阈值（0.5）无法打标。
- 被调用 pair 的 changer 率 0.753（高；多数被调用 pair 即 changer）。

**Publication 标签漂移：**
- 18 个 publication 中 **8 个没有任何可调用 pair**（全部 NO_CALL），在 held-out 评估中不贡献样本。
- 10 个可调用 publication 的 changer 率从 0.048（HC16M2R）到 1.0（pmid_25883046 / pmid_35982307），
  **max/min 比值 20.7×**——标签分布在 publication 之间高度异质，单一模型难以跨 publication 泛化。

**Feature 域漂移（generic [WT,Mut,cond]，SMD vs pool 训练均值）：**
- overall mean abs SMD = 0.327；per-publication 0.138–0.375，部分 publication max_abs_smd > 2.3。

### 5.2 解读（为何三种架构全 fail-closed）

1. 半数 pair 不可被 caller 打标（NO_CALL），且被调用 pair 中 changer 占 75%——
   conditional-magnitude endpoint 建立在覆盖率低、类别高度不平衡的标签上。
2. changer 率跨 publication 20× 异质 → 标签空间域漂移大，无架构可在 publication 内产生
   稳定、可跨 publication 复现的增量。
3. feature 域存在中等漂移（SMD≈0.33）。
4. 综合：任务处于「低覆盖率标签 + 强标签漂移 + 中等特征漂移」组合，任何同容量候选相对
   generic 的增益都落在 publication 级噪声内——与三种方案 CI 下界均 ≤0 一致。

结论：Phase 3 的负结果由数据/测量层属性解释，而非「模型不够复杂」。这支持合同预注册的
**benchmark/resource/negative-result 路线**，且不应继续架构自由搜索。
