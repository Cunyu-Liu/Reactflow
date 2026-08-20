# External Qualification v1 — audit P0-6

> 依据：`提示词/ReactFlow_Delta_strict_scientific_engineering_audit_20260817.md` §2.1/§4.2/§7.3/§9.2 解决方案 3、§13 P0-4/P0-6。
> 输入：`joint_dependency_component_v1.py`（实质化 union-find 图）、`external_provenance_registry_v1.csv`、`recompute_external_k_joint_v1.py` 输出 `external_k_joint_v1.json`。
> 只读重判：不改旧 P4/P5/P5b 原始 outcome aggregate；仅按最高层独立单位重映射资格与 power。

## 1. K_joint 重判（全部已消费外部数据）

| 单位 | N |
|---|---|
| N_rows（mutant×position 行） | 110,141 |
| N_SNV | 110,141 |
| N_WT_anchor（组件） | 718 |
| N_dataset（rdat 文件） | 7 |
| N_batch（NovaSeq 测序批） | **3** |
| N_study | **2** |
| N_publication | **2** |
| K_joint（最高层合并簇） | **2** |

- 3 个测序批：2023-06-06 SL5 M2-seq；2023-08-01 DasBigLib0-15k；2023-10-31 OneMil2 BigLib2-1M。
- 2 个 study/publication：**SL5 研究**（Tertiary folds of the SL5 RNA, PNAS 2024, PMID 38427602）与 **Ribonanza 研究**（Ribonanza: deep learning of RNA structure through dual crowdsourcing, bioRxiv 2024, DOI 10.1101/2024.02.24.581671）。
- 所有 7 个数据集均来自 **Das-lab（Stanford）M2-seq 2A3-MaP 体系**。BigLib2 OneMil2 的 5 个 rdat 分片（M3SARS + M2RFOK + M2RFPK×3）共享同一测序批，必须合并为 1 个 cluster。

## 2. 旧 P4/P5/P5b aggregate 资格重判（只读 remap）

| 旧 claim | 旧 K 计数 | cluster 级重判 | 新资格 |
|---|---|---|---|
| P4 "24 direct_external components" | K_preaccess=24 | K_joint(P4 子集)=2（SL5 3 个 + Ribonanza 21 个） | `DEVELOPMENT/REPLICATION`：cluster 级 N=2，达不到 confirmatory power |
| P5b "694 new independent components" | K_preaccess=694 | K_joint(P5b 子集)=1（全部 OneMil2 BigLib2 同一批） | `DEVELOPMENT/REPLICATION`：cluster 级 N=1 |
| P5b "505 K_eff" 的 t-CI | n=505 | 最高层独立单位=1（同批） | t-CI 严重高估精度；只能作为 component 级描述，不能作为 study 级推论 |
| P5 combined "529 independent" | 529 | K_joint≤2 | **删除 "independent" 表述**；改为 "529 WT anchors across 7 dataset files; 2 study clusters; K_joint=2" |
| P4/P5/P5b "external statistical PASS" | — | cluster 级 power 不足（K=2 < K_required=9） | 降为 `DEVELOPMENT_REPLICATION_ONLY`；无 confirmatory 资格 |

## 3. Power 重判

- Phase 4 验收（audit §12）：`K ≥ max(K_required_power, 9)` 个最高层 study/batch clusters，来自 ≥2 个独立 study/publication lineage；cluster-macro CI 与 LOSO 均通过。
- 现有已消费外部数据：**K_joint=2**（2 studies，勉强满足 "≥2 lineage"），但 **K=2 << K_required=9**（power 严重不足）。
- 结论：**现有外部数据不能作为新 confirmatory set**（audit §9.2 解决方案 3："已经打开的 P4/P5/P5b 只能标 consumed development/replication；任何新 confirmatory threshold 必须在新数据 outcome access 前冻结"）。
- 若继续 Phase 4，必须获得**新的、provenance-resolved、development-disconnected、非 Das-lab 体系的 study/batch 数据**，且在 outcome access 前冻结阈值。

## 4. 对最终 LRSO 外部验证的资格约束

- 最终 LRSO（per-fold selected rank）在已消费外部 cluster 上运行，只能标记 `DEVELOPMENT_REPLICATION_EXPLORATORY`，报告 cluster 级（K=2）均值与 LOSO，但**不得作为 confirmatory PASS**。
- 若 K=2 cluster 级效应方向一致、且 LOSO 稳定，可作为"development replication 信号一致"的弱证据；不足以下 confirmatory 结论。

### 4.1 实际 LRSO 外部 exploratory 结果（2026-08-18，`run_p4_external_lrso_v1.py`）

| 项 | 结果 |
|---|---|
| 候选 | 最终 LRSO rank=2, 5-seed, cfg lr=1e-3/wd=0/student_t, epoch=50（dev 内层选择） |
| K_eff | 24（无 attrition） |
| component 级 D_vs_zero（透明性） | +0.0307（CI [+0.0152, +0.0461]，df=23；非独立单位） |
| **cluster 级（K_joint=2, df=1）** | **SL5 = +0.0008（≈0, n=3）；Ribonanza = +0.0350（n=21）**；K=2 mean +0.0179，CI [-0.199, +0.235] |
| LOSO | leave-out SL5 → +0.035；leave-out Ribonanza → +0.0008 |
| 判定 | `DEVELOPMENT_REPLICATION_EXPLORATORY`；**跨 study 复制不成立（SL5≈0）** |

- **结论**：component 级正信号几乎全部来自 Ribonanza cluster（21 组件共享 2 个测序批）；
  独立 SL5 study 的 LRSO 增量 ≈ 0。**最终 LRSO 的低秩外部增量未在两个独立 study 上复制**，
  因此连"development replication 信号一致"的弱证据也不成立。
- 与 audit §9.1 决策树对齐 → 走向"否"分支：development method paper / benchmark / direct 路线，
  不声称 broad generalization；除非获得新的非 Das-lab、development-disconnected 独立 study/batch 数据。
- 旧 P4/P5/P5b 因 P4-M1（seqpos 错位）判 INVALID（见 §6），本 LRSO exploratory 结果是当前唯一可引用的外部数字。

## 5. 状态

`EXTERNAL_CONFIRMATORY_ELIGIBILITY: NOT_ESTABLISHED`（K_joint=2 < 9；无新的独立 study/batch 数据）。旧外部结果全部降级为 development/replication；最终 LRSO 的外部运行仅 exploratory。

## 6. 审计发现 P4-M1：旧 P4/P5 外部评分存在 seqpos 错位（2026-08-18 新增）

> 依据：`run_p4_external_lrso_v1.py` 的实证检查（seqpos 起点 = X27 → 偏移 26）。
> 所有 3 个 direct_external rdat（M2SL5/M3SARS/15KLIB）的 `seqpos` 均从 X27 开始，
> reactivity 数组长度 = n_seqpos，即 **reactivity[k] ↔ 序列 index (seqpos[k]-1) = k+26**，
> 而非 index 0。

- **旧 P4/P5/P5b 评分错位**：`run_p4_external_v1.py`（及同期的 P5/P5b 外部评分）
  用 `wt_react[i]` / `mut_react[i]`（i = shared_region 序列 index）作为序列位置 i 的
  reactivity，但真实映射是位置 i+26。后果：
  1. 候选模型（ridge/LRSO 特征）的 reactivity 读数与距离特征错位 ~26，特征被破坏；
  2. "shared region（WT==mut）"保证被破坏：被评分的位置实际对应序列位置 i+26，
     该处可能是 pad/barcode（WT≠mut）差异位置；
  3. 真实被观测窗口 [26, 165) 的位置 139..164 从未被评分，而窗口外的 pad 位置 0..25
     被当作可观测位置评分。
- **实证**（M2SL5 `SL5_SARS_CoV_2_0G-A`，编辑位序列 index 31）：seqpos-correct 对齐下
  WT reactivity 0.206 → mutant reactivity **-1.240**（强突变响应）；index-0 对齐下
  0.438 → 0.360（无显著响应）。确认 index-0 对齐丢失了真实突变信号。
- **处理**：
  - 旧 P4 `P4_EXTERNAL_STATISTICAL_PASS`（component-macro D_vs_zero=+0.041）**不作为
    replication 证据**（特征错位 + 评分位置无效）；即使按 audit §2 已降级为
    `DEVELOPMENT_REPLICATION_ONLY`，现在进一步判定该数字本身不可用于任何结论。
  - 旧 P5/P5b 机制分析同样存在该错位风险；P5/P5b 的机制结论保留 `MECHANISM_NOT_ESTABLISHED`，
    不引用其绝对 CRPS/D 数字。
  - 最终 LRSO 外部验证（`run_p4_external_lrso_v1.py`）强制 seqpos-correct 对齐，
    并已在 `p4_external_lrso_frozen_protocol_20260818.md` §6 冻结该要求。
- **影响**：现有已消费外部数据仅能提供"development replication 信号一致"的弱证据，
  且必须以 seqpos-correct 对齐重算后才可引用（即本次 LRSO exploratory 运行）。
