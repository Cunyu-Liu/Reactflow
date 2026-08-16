# Step 4: StructRMDB / RASP v2.0 / Ribonanza RMDB v2 可行性评估

**日期**: 2026-08-02
**评估者**: Trae Agent
**目标**: 评估三个外部 RNA 结构数据库作为 ReactFlow-Δ 预训练/辅助数据的可行性

---

## 1. 数据库概览

| 维度 | StructRMDB | RASP v2.0 | Ribonanza RMDB v2 |
|------|-----------|-----------|-------------------|
| **发表** | CSBJ 2025-12 (PMID:41439023) | NAR 2024-11 (DOI:10.1093/nar/gkae1117) | NAR 2018 (PMID:30053264) |
| **URL** | http://www.rnamd.org/StructRMDB/ | http://rasp2.zhanglab.net/ | http://rmdb.stanford.edu |
| **数据量** | 880K+ 修饰位点 | 18 物种, 18 种实验方法 | 800+ entries, 134K RNAs |
| **数据类型** | 预测的结构变化（m6A/Ψ/A-to-I 修饰 → 二级结构差异） | 实验性 transcriptome-wide 结构探测信号 | 实验性化学映射数据（RDAT 格式） |
| **格式** | Web 界面查询/下载 | 批量下载 (bed/bedGraph) | RDAT 文件 |
| **许可** | 免费学术使用 | CC BY-NC 4.0 | 免费学术使用 |
| **已下载?** | 否 | 否 | 部分（1024 RDAT 中 113 个为 Ribonanza 类别） |

---

## 2. 逐库评估

### 2.1 StructRMDB

**内容**: 首个聚焦 RNA 修饰对二级结构影响的数据库。整合 880,000+ 修饰位点（m6A、假尿苷 Ψ、A-to-I 编辑），覆盖 9 个物种的 pre-RNA 和 mature RNA。使用 RNAstructure 和 ViennaRNA 预测修饰前后的二级结构变化，并通过 4 种评分方法（Similarity/Relative/Distance/SMC）量化结构差异。

**与 ReactFlow-Δ 的关联性**:
- **D4（编辑干预数据）**: 修饰位点 = 天然"编辑"，提供 WT→modified 的结构差异对。直接支持 ReactFlow-Δ 的 H3 假设（"直接预测编辑造成的结构差异"）。
- **D2（显式二级结构）**: 预测的二级结构可作为弱标签数据。
- **局限**: 结构是**预测的**而非实验测量的；>700bp 序列准确率降至 61%；仅覆盖 3 种修饰类型（m6A/Ψ/A-to-I），不含 SHAPE/DMS 等探针数据。

**可行性评估**:
- **下载难度**: 低。Web 界面支持批量下载，无需认证。
- **格式兼容**: 需要编写解析器将修饰位点 + 结构变化评分转为 ReactFlow-Δ 的 pair 格式（WT 序列 + 修饰位置 + delta_structure_score）。
- **数据重叠**: 与现有 RMDB RDAT 数据无直接重叠（RMDB 是探针数据，StructRMDB 是修饰→结构预测）。
- **推荐用途**: D4 层预训练（edit→structure delta 预测的弱监督数据）。**不适用于** D1（探针数据）或 D3（实验性 ensemble）。

**推荐**: ⚠️ **有条件推荐**。下载用于 D4 预训练，但需明确标注为"预测标签"而非"实验标签"，并限制序列长度 ≤700bp。

---

### 2.2 RASP v2.0

**内容**: Transcriptome-wide RNA 二级结构探测数据图谱。覆盖 18 个物种（动物/植物/细菌/真菌/病毒），整合 18 种实验方法（DMS-seq、SHAPE-Seq、SHAPE-MaP、icSHAPE 等）。提供结构探测信号（structure probing signals）的基因组浏览器可视化。

**与 ReactFlow-Δ 的关联性**:
- **D1（化学探针数据）**: 直接补充 RMDB 的探针数据。RASP 提供 transcriptome-wide 视角，而 RMDB 侧重单个 RNA 构建体的详细映射。两者互补。
- **D0（预训练序列）**: transcriptome-wide 数据覆盖大量天然 RNA 序列。
- **关键优势**: 实验性数据（非预测），与 ReactFlow-Δ 的 H2 假设（"化学探针作为结构分布观测"）直接对齐。

**可行性评估**:
- **下载难度**: 低。http://rasp2.zhanglab.net/download/ 提供批量下载，CC BY-NC 4.0 许可。
- **格式兼容**: RASP 使用 bed/bedGraph 格式（基因组坐标 + 探测信号），需转换为 ReactFlow-Δ 的 per-position reactivity 数组。需要基因组注释（GTF）将坐标映射到转录本序列。
- **数据重叠**: 部分实验可能与 RMDB 重叠（同为结构探测数据），需要去重审计。但 RASP 的 transcriptome-wide 视角是 RMDB 所缺的。
- **推荐用途**: D1 层扩充（大规模 transcriptome-wide 探针预训练）。

**推荐**: ✅ **强烈推荐**。高价值（实验性探针数据）、低下载成本、直接补齐 D1 短板。优先下载。

---

### 2.3 Ribonanza RMDB v2

**内容**: Stanford RMDB v2（Yesselman et al. 2018），包含 800+ 条目、134,000 个天然和工程 RNA 的化学映射数据。以 RDAT 格式存储。Ribonanza RNA Folding Kaggle 竞赛基于此数据集。

**与 ReactFlow-Δ 的关联性**:
- **D1（化学探针数据）**: 这是 ReactFlow-Δ 已有的核心数据源。
- **sota doc §6.3 明确列出**: Ribonanza/Ribonanza2 作为 D1 层数据，约 200 万条 chemical mapping profiles。

**已处理状态**:
- Step 2 已下载并解析全部 1024 个 RMDB RDAT 文件（commit 945fd49）。
- 其中 113 个文件归类为 "ribonanza" 类别。
- 提取了 10,424 candidate relations（43 parents, 18 studies）。
- **但**: RMDB RDAT 文件（1024 个）只是 Stanford RMDB 的子集。完整的 Ribonanza2 Kaggle 数据集（~200 万 profiles）**尚未下载**。

**可行性评估**:
- **下载难度**: 中。Ribonanza2 完整数据集在 Kaggle 上，需要 Kaggle API 认证。
- **格式兼容**: Kaggle 数据集使用 CSV/H5 格式，需编写解析器转为 ReactFlow-Δ 格式。
- **数据重叠**: 1024 RDAT 文件中已包含 113 个 Ribonanza 条目。Kaggle 完整数据集规模更大（~200 万 profiles vs 1024 文件中的 ~10K relations）。
- **推荐用途**: D1 层大规模预训练（masked reactivity pretraining）。

**推荐**: ✅ **推荐**。但需要 Kaggle API 认证。建议作为下一步下载任务，sota doc §6.3 已明确要求"Kaggle dataset version / model version / competition terms / hash / H5 schema"必须在下载时固化。

---

## 3. 综合建议

### 优先级排序

| 优先级 | 数据库 | 理由 | 预计工作量 |
|--------|--------|------|-----------|
| **P0** | RASP v2.0 | 实验性探针数据、免费批量下载、直接补齐 D1 短板、无认证门槛 | 低（下载 + 解析器） |
| **P1** | Ribonanza2 Kaggle | sota doc 已明确要求、~200 万 profiles 规模最大、但需 Kaggle 认证 | 中（Kaggle API + H5 解析） |
| **P2** | StructRMDB | 有价值但为预测数据（非实验）、仅适用于 D4 弱监督 | 低（下载 + 解析器） |

### 行动计划

1. **立即**: 下载 RASP v2.0 数据（http://rasp2.zhanglab.net/download/），编写解析器转为 ReactFlow-Δ 格式，进行与 RMDB 的去重审计。
2. **短期**: 配置 Kaggle API 凭证，下载 Ribonanza2 完整数据集，固化版本/hash/schema。
3. **中期**: 下载 StructRMDB 数据，解析为 D4 pair 格式（WT + modification → delta_structure），限制 ≤700bp 序列。

### 风险与约束

- **RASP 格式转换**: bed/bedGraph → per-position reactivity 需要基因组注释和转录本映射，可能有坐标不一致问题。
- **Ribonanza Kaggle 条款**: 需确认竞赛条款允许用于非竞赛用途的模型训练。
- **StructRMDB 预测质量**: >700bp 准确率 61%，必须限制序列长度或标注质量分层。
- **去重**: RASP 与 RMDB 可能有实验重叠，必须进行 source-level 去重审计（类似 D2 的 lineage verification）。

---

## 4. 结论

三个数据库均对 ReactFlow-Δ 有价值，但优先级不同：
- **RASP v2.0** 是最高性价比的下一步（实验性数据 + 免费下载 + D1 直接补齐）。
- **Ribonanza2 Kaggle** 是规模最大的数据源，但需要 Kaggle 认证和 H5 格式处理。
- **StructRMDB** 提供独特的修饰→结构 delta 数据，但为预测数据，适用于 D4 弱监督预训练。

**本次评估为只读可行性分析，未下载数据。** 实际下载需用户确认后执行。
