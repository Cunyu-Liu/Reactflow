> [!CAUTION]
> **SUPERSEDED / 仅作历史记录（2026-07-29）。**
> 本 V2 文件不再是可执行合同。当前唯一有效合同为
> `docs/contracts/ReactFlow分阶段执行提示词_ReactFlowDelta科研合同_v3_EPRO_20260729.md`
>（SHA-256：`3efcc1504208d8089236dfe4e7d41553741441d3b86b6174c8b5af52d614ec10`）。
> V2 原始字节保留在 Git commit `e8080d4bc23c1513ee12dd1feef7a08145a08c5a`
>（原始 SHA-256：`5d2dc9e2ac0e6b8c6355791f4ff95958b2e9ab5722d2d2eba49c6578a3e87c13`）；
> 下文仅作为 historical engineering assets / historical negative evidence，不得用于启动任务、降低 Gate 或支撑新科学主张。

# ReactFlow-Δ Experimental Structural Response Implementation Plan

> **For Codex/Claude:** REQUIRED SUB-SKILL: Use `executing-plans` to execute this plan task-by-task.  
> 中文名称：ReactFlow-Δ 成对实验结构响应科研合同与分阶段执行提示词  
> 合同版本：V2.0  
> 冻结日期：2026-07-29  
> 文献检索截止：2026-07-29  
> 适用仓库：`Cunyu-Liu/Reactflow`  
> 当前只读现场：`/home/cunyuliu/reactflow_c1_3_stage_20260722`  
> 建议新工作树：`/home/cunyuliu/reactflow_delta_goal_20260729`

---

## 0. 文档权威性与状态

本文件是 ReactFlow 后续科研与工程执行的最高优先级合同，完整取代旧版
`ReactFlow分阶段执行提示词.md` 中以下主线：

- “先追逐静态 RNA 二级结构 SOTA，再进入 intervention”；
- 自动追加 3 seed / 10 seed；
- 依赖新湿实验或外部专家标签；
- 用 teacher-generated BPP、预测结构或 quick evaluation 代替实验真值；
- 未经数据可行性审计直接扩展大模型。

旧 C1-0 至 C1-6 的代码、报告和产物不得删除，但全部降级为：

> **historical engineering assets / historical negative evidence**

它们不能自动继承为 ReactFlow-Δ 的科学证据。

### 0.1 已冻结的决策

1. PCCNG 冻结，不再投入新增训练、数据或论文资源。
2. ReactFlow 为唯一主项目。
3. 不要求用户提供新湿实验。
4. 不要求用户寻找专家做标签。
5. 先完成数据可行性与质量审计，再决定是否训练小模型。
6. 探索阶段固定一个 seed，不自动追加多 seed。
7. 所有正式学习型训练必须使用 GPU。
8. 旧静态 C1-3 当前运行可自然结束，但不得打断、续 seed 或升级成新主线。
9. 每个可独立验收的工程任务完成后，必须形成聚焦 commit 并 push 到 GitHub 任务分支。
10. 任何失败只能通过“保留证据后进入预声明的下一分支”前进，不能通过降低 Gate、改主指标或隐藏失败来“前进”。

### 0.2 当前远端事实快照

快照日期：2026-07-29。此段只记录现场，不授权修改现场。

- 分支：`trae/c1-3-static-scale`
- HEAD：`2cdf9faf02f075b6f9289e84411a1ae60ff8d45a`
- 工作区存在未提交修改和备份文件，禁止新路线直接在该工作区继续堆叠。
- 旧 C1-3 v4 正以固定 seed 0、3 GPU 运行。
- 已完成的旧 checkpoint 全量评测：
  - validation MEA F1：约 `0.4013`
  - test MEA F1：约 `0.4049`
  - novel MEA F1：约 `0.4004`
  - EternaFold 同协议约 `0.7039 / 0.7036`
- 旧 `gate_audit.json` 仍引用 50 样本 quick evaluation 的约 `0.80`，与全量评测冲突；quick evaluation 不得作为科学结论。
- 旧注册表有 317,039 条记录，但：
  - 307,641 条来自 `efold_train`；
  - 256,951 条是 proxy profile；
  - 只有 60,088 条标记为 real profile；
  - 最大长度只有 256 nt；
  - 只有 29 个真实 Rfam family；
  - 70,144 个所谓“clan”主要是组件或伪 clan，不是 70,144 个真实 Rfam clan；
  - `structure_similarity` 合并计数为 0；
  - reads、SNR、原始重复信息没有保留。

结论：旧数据量、旧 split PASS 和旧静态训练不能证明新科学问题可做。

---

# 一、最终科学问题

## 1.1 精确定义

给定：

- 野生型或母本 RNA 序列 \(x\)；
- 编辑/突变后序列 \(x'\)；
- 编辑掩码 \(m\)；
- 实验条件 \(c\)，包括 probe、温度、配体、缓冲体系、体内/体外和批次等；

学习：

\[
G_\theta(x,x',m,c)\rightarrow \Delta r
\]

其中：

\[
\Delta r = r(x',c)-r(x,c)
\]

\(r\) 是真实实验测得的 DMS、SHAPE/2A3 或兼容 chemical probing reactivity。

### 主问题

> 在严格的跨研究、跨母本外推下，联合建模“母本—突变—条件”的模型，是否比对两条序列分别做静态预测再相减，更准确地预测突变引起的实验结构响应，尤其是未编辑位点上的远端、非线性和补偿性响应？

### 主协议

**P1：sequence-only**

\[
G_\theta(x,x',m,c)\rightarrow \Delta r
\]

用于没有目标 RNA 实验 profile 时的变体筛选。

### 次协议

**P2：WT-anchored**

\[
G_\theta(x,x',m,c,r_{WT})\rightarrow \Delta r
\]

用于已有一个 WT profile、但没有条件为每个 mutant 做实验的场景。

P1 是论文主要部署场景；P2 是实用增强场景。二者必须分表报告，不能混合。

## 1.2 为什么这个问题有价值

静态折叠回答“每条序列各自可能是什么结构”，但实际设计和变体解释更关心：

- 这次编辑是否真的改变实验可见的结构状态；
- 改变发生在编辑附近还是远端；
- 是否发生 partner switch 或构象重排；
- 双突变是否补偿或救回 WT 响应；
- 预测是否可靠，何时应该拒绝给出结论。

独立静态预测再相减会把两个静态模型的误差相减，并默认两次预测误差相互独立。联合响应模型可以显式使用：

- 两条序列共享的母本背景；
- 编辑位置、编辑类型和编辑距离；
- 条件与 probe；
- WT 与 mutant 的交互；
- “大部分未编辑位点应保持稳定，但少数位置可能发生远端重排”的稀疏响应结构。

这个问题的应用价值包括：

- mRNA/UTR 最小编辑的结构风险筛选；
- riboSNitch 优先级排序；
- riboswitch、RNA 设计和补偿突变分析；
- 在无法进行新实验时，对公开突变扫描数据进行可复现的系统学习；
- 为后续 RNA 编辑优化器提供实验响应 oracle，而不是静态 teacher oracle。

---

# 二、文献调研结论：是否已经有人做过

## 2.1 检索方法与结论边界

本次检索覆盖 PubMed/PMC、期刊官网、DOI 页面、arXiv/bioRxiv 和官方数据库页面，关键词覆盖：

- RNA mutation / SNV / riboSNitch；
- structural disruption / structure change；
- paired WT-mutant SHAPE/DMS；
- direct response prediction；
- differential reactivity；
- sequence-to-reactivity；
- deep learning / foundation model；
- mutate-and-map / M2-seq。

结论是：

> **有人做过大量相邻问题，但截至检索截止日，没有发现一项工作同时满足“以 WT+mutant 序列和编辑为输入、以成对实验 Δreactivity 为监督、直接输出连续响应图、并以跨研究/跨母本无泄漏外推为核心评测”。**

这不是“世界首个”的证明。论文中在完成投稿前的更新检索之前，禁止写 “first” 或 “首次”。

## 2.2 最接近的既有工作

| 工作 | 已经解决什么 | 与本项目重合 | 仍留下的缺口 |
|---|---|---|---|
| [RNAsnp, 2013](https://pubmed.ncbi.nlm.nih.gov/23315997/) | 用热力学 ensemble 比较 WT 与 mutant，寻找最大局部变化 | 输入 WT/mutant，研究结构扰动 | 不从成对实验响应学习；不直接预测 Δreactivity |
| [Genome-wide riboSNitch benchmark, 2015](https://pubmed.ncbi.nlm.nih.gov/25618847/) | 用 PARS 实验标签评估静态折叠差分方法 | 强调实验 benchmark 和局部变化 | 主要评估规则/热力学方法，不是联合响应学习 |
| [classSNitch, 2017](https://doi.org/10.1093/bioinformatics/btx041) | 读取 WT 和 mutant 两条已测 SHAPE trace，用 167 个专家共识标签训练分类器；分析 2,019 对 trace | 使用成对实验 trace，分类 no/local/global change | 输入已经包含 mutant 实验结果，因此不是从序列预测未知 mutant 响应；依赖专家标签；未做跨母本深度外推 |
| [M2-seq, 2017](https://pubmed.ncbi.nlm.nih.gov/28851837/) | 通过 intentional/accidental mutations 测量结构扰动并推断配对 | 提供天然的突变—响应矩阵 | 目标是实验结构推断，不是从序列学习未测 mutant 的响应 |
| [MutaRNA, 2020](https://doi.org/10.1093/nar/gkaa331) | 可视化静态预测的 BPP 差异，并集成 RNAsnp/remuRNA | 分析 mutation-induced change | 仍是静态预测后差分 |
| [Riprap / RiboSNitchDB, 2020](https://doi.org/10.1093/nargab/lqaa057) | 从折叠算法 BPP 中寻找局部结构扰动，建立数据库 | 结构改变定位和分类强基线 | 无训练组件；实验仅用于验证，不直接拟合连续响应 |
| [VariantFoldRNA, 2025](https://doi.org/10.1093/nargab/lqaf066) | 将 SNPfold、remuRNA、RNAsnp、Riprap 扩展为全基因组流水线 | 已覆盖“可扩展 riboSNitch 工具”叙事 | 仍依赖热力学双折叠；没有联合实验响应模型 |
| [dStruct, 2019](https://doi.org/10.1186/s13059-019-1641-3) | 在有重复的两组 probing 数据中识别差异响应区域 | 可替代专家标签，提供统计 caller | 是分析已测数据，不预测新突变 |
| [Ribonanza / RibonanzaNet, 2024](https://pubmed.ncbi.nlm.nih.gov/38464325/) | 从单条序列预测 DMS/2A3 reactivity，并提供约两百万条测量 | 可分别预测 WT/mutant 再相减，是最强学习基线之一 | 不是以配对响应为目标；公开模型可能见过 RMDB 子集 |
| [eFold, 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC12935039/) | 用多源静态结构/化学探测数据提升静态二级结构预测 | 可提供静态 backbone 或差分基线 | headline 仍是静态 structure prediction；部分训练标签由 EternaFold/RNAstructure 生成 |
| [CHANRG, 2026 preprint](https://arxiv.org/abs/2603.22330) | 证明普通 held-out 排名会高估静态模型 OOD 泛化，强调结构去重和层级 split | 直接支持严格 OOD 设计 | 不研究 mutation-conditioned response |

## 2.3 真正可主张的创新

本项目不能主张：

- 首次研究 RNA 突变结构效应；
- 首次预测 riboSNitch；
- 首次使用机器学习分析 WT/mutant SHAPE；
- 首次建立可扩展变体结构工具；
- 首次用深度学习预测 chemical reactivity。

在证据成立时，可以主张：

1. **问题创新**  
   把任务从“两次静态折叠再相减”改写为“mutation-conditioned experimental response prediction”。

2. **标签创新**  
   主要监督是成对实验连续 \(\Delta r\)，而不是 teacher ΔBPP、静态结构差、专家主观标签或功能 proxy。

3. **评测创新**  
   把未编辑对齐位置上的 trans-effect 作为主要 endpoint，避免模型只靠读取突变碱基本身获得虚假优势。

4. **泛化创新**  
   同一母本的所有 mutant 必须留在同一 split；确认性测试按 publication/study 整体留出。

5. **方法创新**  
   比较联合交互模型与严格 matched-capacity 的 independent/Siamese difference 模型，直接检验“联合建模是否真的有价值”。

6. **可靠性创新**  
   显式预测不确定性，并用 no-edit/replicate noise ceiling 约束模型，报告 coverage-risk 而不只报告均值。

7. **数据资源创新（条件性）**  
   若 D0-D2 证明可行，可发布一个版本化、无泄漏、按条件分层的 RNA Structural Intervention Benchmark，暂定名 `RSIB-v1`。投稿前不得宣称是首个。

## 2.4 可发表性判断

### 当前判断

**值得做，但属于“数据 Gate 通过后值得做”，不是无条件值得大规模训练。**

### 发表层级

| 数据与结果状态 | 合理论文形态 | 可能目标 |
|---|---|---|
| 多研究、多母本、严格 external study test，联合模型显著优于全部差分基线 | 方法 + benchmark + 生物学案例 | NAR、Nature Communications、较强 Bioinformatics 方向 |
| 数据中等，至少一个严格跨研究 test，联合模型有稳定增益 | 方法/资源论文 | Bioinformatics、PLOS Computational Biology、NAR Genomics and Bioinformatics |
| 深度模型无明显增益，但形成高质量成对数据集、泄漏审计和强 benchmark | 数据/benchmark/negative-result 论文 | NAR Genomics and Bioinformatics、Database、GigaScience 等 |
| 有效数据只来自极少数母本或单一实验室 | 不能声称通用预测器 | 限定为 dataset analysis；不进入大模型 |

### 最大风险

不是“模型能否再加一层”，而是：

> RMDB/Ribonanza 中看起来有数百万 construct，但真正满足同 probe、同条件、可确定 WT 母本、编辑关系明确、profile 可比、能按研究留出的有效 pair 和独立母本可能很少。

经典 classSNitch 数据只有 17 个 RMDB entry、11 个 RNA、2,019 个 WT–single-mutant pair。RMDB 当前规模已扩展到 1,024 entries 和 4,556,825 constructs，但规模不能直接等同于有效干预 pair 数量。[RMDB 官方页面](https://rmdb.stanford.edu/)。

---

# 三、讨论漏洞复核与修复

## 3.1 “实验 reactivity 等于结构真值”——错误

Reactivity 是 probe、溶液条件、蛋白结合、批次和构象 ensemble 的联合观测，不是唯一二级结构真值。

修复：

- 主结果称为“实验结构响应”或“probing response”；
- 不把 Δreactivity 直接写成真实 ΔBPP；
- ΔBPP、pair gain/loss 只在有可信实验支持结构的子集上作为次任务；
- 不同 probe 不直接混成同一数值空间。

## 3.2 “突变位点的响应很容易预测，所以模型有效”——错误

突变本身会改变 nucleotide identity 和 DMS 可探测性。模型可能只学会“这个位置换了碱基”。

修复：

- 主 endpoint 只在未编辑且 probe eligibility 不变的对齐位置上计算；
- 编辑位置及其 eligibility 改变位置作为 secondary endpoint；
- 分别报告 local、mid-range、remote 距离区间；
- 必须有 `edit-only trivial baseline`。

## 3.3 “同一 mutational scan 随机拆行”——严重泄漏

同一 WT 被几百个 mutant 共享时，按 pair 随机拆分会让模型在 test 中看到母本。

修复：

- 最小 split 单位为 parent RNA；
- 确认性 split 单位为 study/publication；
- 同一实验板、同一 mutational library 和重复测量不能跨 split；
- exact、near-identity、parent、study 四层 group 均不得跨界。

## 3.4 “用每对 profile 的最优缩放最小化差异”——会抹掉真实效应

classSNitch 的历史归一化适合模拟人工 trace 比较，但如果为每对 WT/mutant 选择使差异最小的 multiplier，会把真实 global response 向 0 偏置。

修复：

- 保留原始、上游标准化和本项目标准化三层；
- test 的变换参数只能来自固定 protocol 或 training controls；
- 禁止用 test pair 自身选择“最小差异”缩放；
- 全局幅度与形状变化分开报告。

## 3.5 “专家标签不可得，所以任务不能做”——不成立

连续 \(\Delta r\) 本身就是监督信号，不需要专家。

替代方案：

1. 主任务直接回归连续 \(\Delta r\)；
2. 有重复时用 dStruct/DiffScan 类 replicate-aware 方法生成统计标签；
3. 无重复时不强行生成显著性标签，只保留连续目标和可靠性权重；
4. changer 分类阈值由 no-edit/replicate noise 分布冻结，不由人工目测决定；
5. classSNitch 专家共识只作为历史小型外部验证，不作为主训练标签。

## 3.6 “预训练模型越强越好”——可能污染 test

Ribonanza 论文明确包含从 RMDB 汇总的 profile。RibonanzaNet/RibonanzaNet2 可能已经见过候选 RMDB test sequence 或其近邻。

修复：

- 主科学结论使用 `from_scratch` 或可重建的 `train-only self-pretraining`；
- RibonanzaNet、RibonanzaNet2、RiNALMo、ERNIE-RNA 等作为 external-pretrained secondary arm；
- 无法重建预训练集合时标记 `unknown_contamination`；
- 外部预训练 arm 不得单独支撑 clean OOD 主张。

## 3.7 “做 ΔBPP 就等于有实验标签”——错误

用 RNAfold、EternaFold 或 RNAstructure 生成 WT/mutant BPP 后相减，仍是 teacher proxy。

修复：

- teacher ΔBPP 只能用于辅助损失、蒸馏或 baseline；
- 主要表格必须以实验 Δreactivity 为真值；
- teacher-only 结果必须标 `proxy-only`，不得写 experimental validation。

## 3.8 “不做多 seed 等于不需要不确定性”——错误

不追加多 seed 是资源决策，不代表可以忽略模型随机性。

修复：

- 开发与确认均使用预注册固定 seed；
- 数据不确定性通过 study/parent cluster bootstrap；
- 测量不确定性通过 replicate/error weighting；
- 模型随机性限制必须在论文 limitations 中披露；
- 不自动追加不同 seed；
- 如未来确需额外 seed，必须由用户单独书面批准，且不得在看到 test 结果后临时决定。

## 3.9 “只能前进”被误解为失败后继续烧 GPU——错误

本合同中的“只能前进”定义为：

> fail-forward = 冻结证据 → 解释失败 → 进入预声明的 fallback；绝不通过降低 Gate、删除失败、改 test、换主指标或回到已冻结主线来制造成功。

允许为了恢复工程正确性回到已知良好 commit，但必须：

- 保留失败 commit/patch/log；
- 写 root-cause；
- 不覆盖历史产物；
- 从新的 run ID 继续。

这类技术恢复不属于科学路线后退。

---

# 四、可证伪假设与论文主张

## 4.1 主假设 H1

在冻结的 leave-study-out test 上，联合模型：

\[
G_\theta(x,x',m,c)
\]

在未编辑对齐位置的 trans-response 指标上优于最强可执行的：

\[
F(x',c)-F(x,c)
\]

差分基线。

## 4.2 次假设

- H2：联合模型在 remote response 上的增益大于 local response。
- H3：WT-anchored 模型显著优于 sequence-only，但 sequence-only 仍优于 zero-change。
- H4：显式 edit mask 和 cross-sequence interaction 比 matched-capacity Siamese difference 有增益。
- H5：uncertainty 能识别跨研究、跨 family 和低质量 profile 的失败。
- H6：双突变/补偿突变子集中，模型能正确排序 rescue 程度。

## 4.3 零结果仍然有价值的条件

如果 H1 不成立，但完成以下内容，项目仍可前进为 benchmark 论文：

- 公开数据被完整审计并形成可重建 pair registry；
- 证明独立差分和联合学习都受到哪些数据/条件限制；
- 有严格跨 study split；
- 有强基线、噪声上界和污染审计；
- 结果诚实，不把 negative result 改写成成功。

## 4.4 禁止性主张

在对应证据前禁止：

- “首次”；
- “SOTA”；
- “通用 RNA 突变效应模型”；
- “真实结构变化”；
- “因果预测”；
- “适用于任意 mRNA/任意编辑”；
- “无需实验验证即可指导临床或治疗设计”；
- “超过专家”；
- “已解决 RNA 动态结构预测”。

---

# 五、数据合同

## 5.1 数据优先级

### Tier A：主监督候选

- RMDB 中明确的 mutate-and-map、M2-seq、mutate-map-rescue；
- 同一研究内、同一条件下测得的 WT 和 single-mutant profiles；
- 有明确 parent、edit、probe、condition、sequence 和 reactivity；
- 有重复或 measurement error 时优先。

RMDB 当前数据以 RDAT 版本化发布，内容为 CC0，可通过官方 GitHub releases 获取。[RMDB About](https://rmdb.stanford.edu/about/)。

### Tier B：扩展监督候选

- Ribonanza 中能被可靠重建为同批次、同条件、单编辑近邻的实测 sequence pair；
- OpenKnot/Eterna libraries 中有明确设计 lineage 的 mutant family；
- PARS allele-specific riboSNitch 数据；
- 公开 riboswitch 变体、补偿突变和配体条件数据。

Tier B 必须显式标记：

- designed-neighbor 还是真实 WT-mutant；
- 是否有共同 parent；
- 是否存在 batch confounding；
- 是否能用于主 test。

### Tier C：只作辅助或 baseline

- Ribonanza 单序列 reactivity；
- eFold 静态训练数据；
- bpRNA、ArchiveII、PDB、Rfam；
- RNAfold/EternaFold/RNAstructure teacher BPP；
- synthetic mutations；
- 功能 assay 但无 matched probing 的数据。

Tier C 不得成为主干预真值。

## 5.2 不可变原始层

每个下载文件必须记录：

- `source_name`
- `source_version`
- `source_url`
- `retrieved_at`
- `license`
- `sha256`
- `size_bytes`
- `upstream_id`
- `publication_doi/pmid`
- `raw_path`
- `download_status`
- `parser_version`

原始文件只读。任何清洗都写入新层，禁止覆盖 raw。

## 5.3 Construct schema

每条 construct 至少包含：

- `construct_id`
- `source_entry_id`
- `study_id`
- `publication_id`
- `laboratory_id`
- `parent_id`
- `design_lineage_id`
- `sequence_raw`
- `sequence_normalized`
- `length`
- `probe`
- `probe_chemistry_version`
- `temperature`
- `ligand`
- `ligand_concentration`
- `buffer`
- `in_vivo_in_vitro`
- `batch_id`
- `replicate_id`
- `reactivity`
- `reactivity_error`
- `coverage/read_depth`
- `snr`
- `valid_mask`
- `probe_eligibility_mask`
- `normalization_method_upstream`
- `quality_flags`

缺失字段不得伪造。使用 `null + missing_reason`。

## 5.4 Pair schema

每个 pair 至少包含：

- `pair_id`
- `wt_construct_id`
- `mut_construct_id`
- `parent_id`
- `study_id`
- `edit_type`
- `edit_positions`
- `wt_alleles`
- `mut_alleles`
- `edit_count`
- `alignment_cigar`
- `condition_match_fields`
- `condition_match_status`
- `delta_reactivity_raw`
- `delta_reactivity_normalized`
- `unchanged_position_mask`
- `changed_position_mask`
- `probe_eligibility_unchanged_mask`
- `local/mid/remote_masks`
- `replicate_noise_estimate`
- `pair_quality_weight`
- `primary_eligible`
- `exclusion_reasons`

## 5.5 第一版范围

第一版只纳入：

- substitution；
- edit_count = 1；
- WT/mutant 长度相等；
- condition 完全匹配；
- probe 相同；
- 至少 60% 未编辑位置有合法、可比 reactivity。

以下延后：

- insertion/deletion；
- 多编辑；
- 跨 probe 直接回归；
- 体内与体外混合；
- 不明确 parent 的任意近邻配对。

双突变只进入独立 rescue 子集，不与 single-mutant 主训练集混合。

## 5.6 Probe 处理

- DMS 传统协议主要在可探测碱基上评分；四碱基 DMS 必须由协议元数据明确支持。
- SHAPE/2A3 使用各自 valid mask。
- 因突变导致 probe eligibility 改变的位置不进入主 trans-effect。
- 不同 probe 建立独立 normalization domain。
- 不允许把 DMS 与 SHAPE 数值直接拼成同一个无条件回归目标。

## 5.7 清洗顺序

1. 校验 RDAT/上游文件可解析。
2. 校验 sequence、position、reactivity 长度一致。
3. 规范 T/U，但保留 raw。
4. 确认 WT/mutant edit 关系。
5. 核对所有 condition。
6. 构建对齐与 eligibility mask。
7. 识别重复、技术重复、生物重复和 no-edit control。
8. 计算 profile missingness、SNR、coverage。
9. 计算 pair 可用位置比例。
10. 估计 replicate/no-edit noise。
11. 才能生成 normalized Δreactivity。
12. 最后才构建 split。

## 5.8 禁止的数据操作

- 不得用 test 统计量做归一化或阈值选择；
- 不得把同一 parent 的 mutant 拆到不同 split；
- 不得用模型预测值填充主标签；
- 不得把缺失 reactivity 当 0；
- 不得删除负结果 pair 只保留“大变化”；
- 不得按 observed change 大小选择 test；
- 不得把多个研究的 profile 无条件平均；
- 不得把 `efold_train` 数量当成真实 intervention 数据量；
- 不得把 MMseqs component 命名为真实 Rfam clan。

---

# 六、专家标签替代合同

## 6.1 主替代：连续监督

专家不再是项目依赖。

主训练目标：

\[
\Delta r_i=r_{mut,i}-r_{WT,i}
\]

主评测位置：

\[
i \in \text{unchanged aligned positions with unchanged probe eligibility}
\]

## 6.2 Replicate-aware 统计标签

当 WT 和 mutant 各自有足够重复时：

- 使用冻结版本的 dStruct 或等价 replicate-aware caller；
- 识别 differential reactive regions；
- 控制 FDR；
- 输出 `changer/non-changer` 和 region labels；
- caller 的参数只用 train/validation 冻结。

## 6.3 无重复数据

无重复时：

- 只回归连续响应；
- 使用上游 `reactivity_error` 或同 study controls 估计权重；
- 不声称统计显著 changer；
- 可用 noise-calibrated effect size 做 exploratory ranking，但不作 confirmatory label。

## 6.4 classSNitch 的角色

classSNitch 的 167 个专家共识标签仅用于：

- 外部历史 sanity check；
- 验证模型预测的 response summary 是否与人类 no/local/global 感知相关；
- 与历史方法保持可比。

不得：

- 用 classSNitch 对全部数据打伪标签后称为实验真值；
- 把 classSNitch 输入了 mutant trace 的结果与 sequence-only 模型直接当同类预测器比较。

---

# 七、数据可行性 Gate

## 7.1 Tier A：完整方法论文可行

同时满足：

- 至少 5 个独立 study/publication；
- 至少 20 个独立 parent RNA；
- 至少 5,000 个 primary-eligible single-mutant pair；
- 至少 2 个完全留出的 test study，每个至少 100 个 pair；
- 至少 3 个 study/parent block 有 replicate 或 no-edit controls；
- controls 总计至少 100 对，可估计噪声；
- 主 probe domain 中不存在一个 parent 占全部 pair 的 40% 以上；
- condition metadata 足以冻结主分析。

通过后：允许 B0、M0、M1。

## 7.2 Tier B：受限方法/benchmark 可行

同时满足：

- 至少 3 个独立 study；
- 至少 10 个 parent；
- 至少 1,000 个 primary-eligible pair；
- 至少 1 个完整留出 study，至少 100 个 pair；
- 有可审计的噪声估计。

通过后：

- 允许小模型和 benchmark；
- 论文必须限定适用域；
- 不得称通用模型。

## 7.3 Tier C：不允许深度模型 headline

任一情况成立：

- 少于 3 个独立 study；
- 少于 5 个 parent；
- 少于 500 个 primary-eligible pair；
- 无法可靠匹配 condition；
- test 无法按 parent/study 隔离；
- 标签主要由 teacher 生成。

处理：

- 不训练大模型；
- 不追加 seed；
- 转为 ReactFlow 数据资源/负结果/方法审计；
- 继续在 ReactFlow 内前进，不回到 PCCNG。

---

# 八、Split、污染与 benchmark 合同

## 8.1 Split 层级

由弱到强：

1. exact construct group；
2. parent RNA group；
3. design lineage / mutational library group；
4. study/publication group；
5. family/clan；
6. structure similarity。

确认性 test 优先使用第 4 层：leave-study-out。

## 8.2 冻结集合

- `train`
- `validation_parent_holdout`
- `test_study_holdout_1`
- `test_study_holdout_2`（Tier A 时必需）
- `family_ood`（exploratory）
- `rescue_subset`
- `replicate_control_subset`
- `classic_classsnitch_external`
- `pars_external_stress`

## 8.3 Test 隔离

test 冻结后：

- 不查看标签分布细节；
- 不用 test 选 checkpoint；
- 不用 test 选 normalization；
- 不用 test 选 probe；
- 不用 test 选距离 bins；
- 不用 test 决定是否换模型；
- 每个 confirmatory test 只允许一次主解封；
- 解封命令、时间、commit 和 config 写入 audit。

## 8.4 预训练污染

每个外部模型必须记录：

- 模型版本；
- weight SHA256；
- 训练数据描述；
- 是否包含 RMDB/Ribonanza/Rfam；
- exact overlap；
- parent overlap；
- identity overlap；
- family overlap；
- `clean / contaminated / unknown_contamination`。

主论文必须至少有一个 clean protocol：

- `from_scratch`，或
- 使用只含 train split 的自建预训练。

---

# 九、模型架构编排

## 9.1 架构原则

- 先小后大；
- 先 1D response 后 2D pair；
- 先证明联合建模价值，再增加 PairFormer；
- 参数和算力必须与数据规模匹配；
- 不因项目叫 ReactFlow 就强行使用 flow/diffusion；
- 不以 backbone 新颖代替科学问题新颖。

## 9.2 模型梯度

### M0-a：线性/局部基线

- mutation type embedding；
- 相对编辑距离；
- 局部序列 k-mer；
- condition token；
- linear / ridge / small MLP。

作用：验证数据是否存在可学习信号。

### M0-b：Matched Siamese baseline

- WT/mutant 共享 encoder；
- 分别编码；
- 输出差分；
- 不允许 cross interaction。

这是检验 joint model 的关键 matched-capacity baseline。

### M1：ReactFlow-Δ Small

建议初始约束：

- 参数量不超过 15M；
- max length 512；
- shared 1D encoder；
- 显式 edit token/mask；
- WT↔mutant 轻量 cross-attention；
- delta feature trunk；
- probe/condition embedding；
- per-position Δreactivity mean head；
- heteroscedastic uncertainty head；
- 可选 changer head。

不得在 M1 前直接启用全规模 PairFormer。

### M2：Sparse Pair-aware 扩展

仅当 M1 通过时：

- banded/sparse pair representation；
- edit-centered long-range candidates；
- ΔBPP 或 pair gain/loss auxiliary head；
- 不使用 dense \(L^2\) 全图作为默认；
- pair labels 必须来自可信子集。

## 9.3 损失

主损失：

- masked Huber 或可靠性加权 MAE；
- 只在合法主 mask 上计算。

允许的辅助项：

- swapped-pair antisymmetry；
- no-edit identity；
- heteroscedastic NLL；
- profile correlation/ranking；
- rescue ranking（仅 rescue 子集）。

禁止：

- 用 teacher ΔBPP 主导损失；
- 强制所有远端位置为 0；
- 用 test 调 loss weights；
- 同时堆叠过多损失而无法消融。

## 9.4 预训练路线

优先级：

1. from-scratch；
2. train-only self-pretraining；
3. 可重建且去除 test/near-hit 的 Ribonanza reactivity pretraining；
4. external pretrained secondary arm。

如果无法证明 Ribonanza/RMDB overlap 已排除，RibonanzaNet2 只能作 contaminated baseline/ablation。

---

# 十、强制横向对比

## 10.1 非学习基线

- `zero-change`：所有位置预测 0；
- `mutation-type mean`；
- `distance-decay`；
- `nearest measured mutant`（严格 train-only）；
- `edit-only trivial`。

## 10.2 热力学/结构差分基线

- ViennaRNA RNAfold / RNAplfold；
- LinearPartition；
- EternaFold；
- RNAstructure partition；
- RNAsnp；
- SNPfold；
- remuRNA；
- Riprap；
- VariantFoldRNA pipeline。

## 10.3 学习型静态差分基线

- RibonanzaNet；
- RibonanzaNet2；
- eFold 可执行 head；
- 其他模型只有在能冻结版本、权重和输入协议时才加入。

统一形式：

\[
\hat{\Delta r}_{independent}=F(x',c)-F(x,c)
\]

## 10.4 架构消融

- no edit mask；
- no condition token；
- no cross-attention；
- Siamese independent；
- WT-only；
- mutant-only；
- sequence-only vs WT-anchored；
- from-scratch vs external-pretrained；
- 1D vs sparse pair-aware。

## 10.5 不可直接比较的方法

classSNitch 和 dStruct 读取了 mutant 实验 trace，属于 experimental analysis oracle，不是 sequence-only predictor。它们必须放在单独表中。

---

# 十一、测评合同

## 11.1 主 endpoint

**Trans-response skill score**

对每个 parent/study block：

\[
\text{Skill}=1-\frac{\text{WMAE}(\hat{\Delta r},\Delta r)}
{\text{WMAE}(0,\Delta r)}
\]

仅在：

- 未编辑；
- 对齐；
- probe eligibility 不变；
- profile 有效；

的位置计算。

先对 parent 宏平均，再对 study 宏平均。禁止由 pair 数量最大的 parent 支配结果。

## 11.2 次指标

- weighted MAE / RMSE；
- Pearson / Spearman；
- sign accuracy；
- affected-position AUPRC；
- local/mid/remote 分层；
- no-change specificity；
- changer AUPRC；
- dStruct region overlap；
- uncertainty NLL；
- calibration error；
- coverage-risk；
- runtime、显存、参数量。

## 11.3 距离分层

距离阈值必须在 validation 前冻结，例如：

- edit site：0；
- local：1–10 nt；
- mid：11–50 nt；
- remote：>50 nt。

如果按结构距离评估，结构必须来自独立可信来源或明确标 `predicted`.

## 11.4 Rescue 子集

对 WT、single mutant A、single mutant B、double rescue：

- 计算 double mutant 到 WT 的 response distance；
- 比较模型是否正确排序 rescue；
- 不把任意 double mutant 都叫 compensatory；
- 必须有实验 lineage 或论文证据支持。

## 11.5 统计

- 预注册一个主模型和一个最强主基线；
- paired comparison；
- study/parent cluster bootstrap 95% CI；
- 必要时 study-level sign permutation；
- 次比较用 Holm correction；
- 同时报告 effect size 和 CI，不只报 P 值；
- 不用 pair-level 独立假设伪造样本量。

## 11.6 主 Gate

确认性 test 上同时满足：

1. ReactFlow-Δ 的 study-macro Skill > 0；
2. 相对最强可执行基线的 Skill 差异 95% cluster-bootstrap CI 下界 > 0；
3. 增益不是只来自一个 parent；
4. remote 子集不显著劣于 zero-change；
5. 结果可从冻结 commit/config/manifest 重建；
6. 无 test 调参和污染违规。

失败则进入 benchmark/negative-result 路线，不追加 seed 来碰运气。

---

# 十二、GPU、监控和时间管理合同

## 12.1 GPU-only 的精确定义

必须 GPU：

- 任何用于科学结果的模型训练；
- pilot、小模型、正式模型和 learned baseline fitting；
- 需要学习参数的 fine-tuning。

允许 CPU：

- 下载；
- 解压；
- RDAT parsing；
- QC；
- registry/split 构建；
- thermodynamic command-line baseline；
- 单元测试；
- 统计分析；
- 论文与文档；
- 极小 forward/backward 单元测试，但不得作为科学训练。

训练启动前必须 fail-closed：

- `torch.cuda.is_available() == true`
- 记录 GPU 型号、driver、CUDA、PyTorch；
- 记录实际 device；
- 如果模型参数落在 CPU，立即失败；
- 不允许自动回退到 CPU。

## 12.2 Run contract

每次训练必须有唯一 `run_id`，并提前创建：

- config snapshot；
- git SHA；
- data/split hashes；
- local structured log；
- metrics JSONL；
- system metrics；
- checkpoint dir；
- manifest；
- stop reason。

W&B 只能作副本，不能是唯一证据。

## 12.3 不频繁查看进度

禁止 busy polling 和数分钟一次 `tail`。

默认节奏：

1. 启动后 5–10 分钟做一次健康检查；
2. 此后最短 30 分钟检查一次；
3. 预计超过 6 小时的任务，稳定后改为 60 分钟；
4. 由日志告警触发的 NaN/OOM/进程退出可立即检查；
5. 不为“看看有没有涨一点”频繁跑全量 validation。

等待期间必须并行做无写冲突任务：

- 数据卡；
- source/citation 核验；
- baseline wrapper 单元测试；
- evaluator fixtures；
- 文档；
- contamination audit；
- failure matrix；
- 论文图表脚本。

不得同时修改同一文件或同一 artifact 目录。

## 12.4 安全停止

出现以下任一项，安全暂停当前 run 并保留证据：

- NaN/Inf；
- CUDA/device 异常；
- 连续 5 次项目级 validation 无推进且达到预注册资源上限；
- 磁盘/显存接近安全阈值；
- label/split 泄漏；
- 数据 hash 改变；
- checkpoint 无法恢复；
- 主 Gate 的前置条件被证伪。

安全暂停不是后退；盲目续跑才是违约。

---

# 十三、GitHub 合同

## 13.1 当前脏工作区保护

- 不在 `/home/cunyuliu/reactflow_c1_3_stage_20260722` 上启动新路线开发；
- 不覆盖未提交修改；
- 不删除 `.bak`；
- 不停止旧 v4；
- 旧 run 结束后生成只读 archive manifest；
- 新路线使用独立 clean worktree 和独立 artifact namespace。

## 13.2 分支

建议：

- `codex/reactflow-delta-r0`
- `codex/reactflow-delta-d0`
- `codex/reactflow-delta-d1`
- `codex/reactflow-delta-d2`
- `codex/reactflow-delta-b0`
- `codex/reactflow-delta-m0`
- `codex/reactflow-delta-m1`
- `codex/reactflow-delta-paper`

## 13.3 每个任务结束

“任务”定义为下面 todo 中一个可独立验收的 `T*` 单元，不是每敲一行代码。

任务完成必须：

1. `git status`；
2. 检查 diff；
3. targeted tests；
4. 必要的全量 tests；
5. artifact schema validation；
6. 检查无 secret/data/weight/cache；
7. 聚焦 commit；
8. push 当前任务分支；
9. 记录 commit SHA 和 GitHub branch/URL。

禁止：

- 直接 push main；
- 提交原始大数据、FASTQ/BAM/RDAT 全库；
- 提交 checkpoints；
- 提交 token、SSH key、`.env`；
- 把无关脏文件夹带进 commit；
- 因 GitHub 暂时不可用而丢失本地 commit。

push 失败时：

- 保留本地 commit；
- 记录错误；
- 任务状态为 `IMPLEMENTED_NOT_PUSHED`；
- 修复网络/权限后再 push；
- 不伪报完成。

---

# 十四、阶段路线图

## Phase R0：路线重置与旧资产封存

### 目标

保护正在运行的旧 C1-3，建立新工作树和权威合同。

### Todo

- [ ] T-R0.1 只读记录旧 checkout、HEAD、dirty files、process、GPU、artifact links。
- [ ] T-R0.2 等旧 v4 自然结束；不追加 seed。
- [ ] T-R0.3 生成旧结果 archive manifest，标注 historical-only。
- [ ] T-R0.4 建立 clean worktree 和 `codex/reactflow-delta-r0`。
- [ ] T-R0.5 将本合同复制到 repo `docs/contracts/`。
- [ ] T-R0.6 添加 supersession notice 到旧 Goal 文档，不删除旧文档。
- [ ] T-R0.7 测试、commit、push。

### Gate

- 新工作树 clean；
- 旧运行未被干扰；
- 合同 hash 固定；
- historical 与 new evidence 分离。

---

## Phase D0：公开成对响应数据可行性审计

### 目标

不训练模型，回答“到底有多少真实可用的 WT-mutant 实验 pair”。

### Todo

- [ ] T-D0.1 建立 source registry。
- [ ] T-D0.2 获取 RMDB 1,024 entries 的 metadata/index 和 release manifest。
- [ ] T-D0.3 识别 mutate-and-map、M2-seq、rescue、variant library entries。
- [ ] T-D0.4 对每类先抽样解析 3–5 个 RDAT，冻结 parser tests。
- [ ] T-D0.5 下载候选 RDAT，保存 checksum，不提交 raw。
- [ ] T-D0.6 解析 construct、annotation、sequence、reactivity、condition。
- [ ] T-D0.7 统计明确 WT、single mutant、double mutant、replicate、no-edit。
- [ ] T-D0.8 审计 Ribonanza/Ribonanza+ 中可构成同条件单编辑实测 pair 的数量。
- [ ] T-D0.9 构建候选 pair registry，不做最终 normalization。
- [ ] T-D0.10 输出来源×study×parent×probe×pair 数量矩阵。
- [ ] T-D0.11 给出 Tier A/B/C 预判。
- [ ] T-D0.12 测试、commit、push。

### 输出

- `data_registry/source_registry.jsonl`
- `data_registry/raw_manifest.json`
- `data_registry/construct_candidates.parquet`
- `data_registry/pair_candidates.parquet`
- `reports/d0_data_feasibility_audit.md`
- `artifacts/d0/data_feasibility_summary.json`
- `artifacts/d0/parser_fixture_results.json`

### Gate

- 所有数量来自可解析 artifact；
- 同一 construct 不重复计数；
- 明确区分真实 WT-mutant、designed neighbor、synthetic pair；
- 未开始任何 learned training。

---

## Phase D1：清洗、配对、噪声与标签合同

### 目标

把候选 pair 变成可用于科学训练的 primary-eligible pair。

### Todo

- [ ] T-D1.1 冻结 construct/pair schema。
- [ ] T-D1.2 实现 condition exact matching。
- [ ] T-D1.3 实现 substitution edit verification。
- [ ] T-D1.4 实现 alignment 和 probe eligibility mask。
- [ ] T-D1.5 识别重复、replicate、no-edit control。
- [ ] T-D1.6 实现 raw/upstream/project-normalized 三层数据。
- [ ] T-D1.7 估计每个 study/probe 的 noise ceiling。
- [ ] T-D1.8 有重复时运行 replicate-aware differential caller。
- [ ] T-D1.9 生成 exclusion reasons 和质量权重。
- [ ] T-D1.10 人工可手算 fixtures 覆盖突变位点、缺失、DMS eligibility、远端变化。
- [ ] T-D1.11 测试、commit、push。

### Gate

- fixture 100%；
- 不用 pair 自身最小化差异；
- 不把 missing 当 0；
- 每个排除均有机器可读 reason；
- 噪声估计不使用 frozen test。

---

## Phase D2：无泄漏 benchmark 与最终数据 Gate

### 目标

冻结 RSIB-v1 候选 benchmark 和是否允许训练的决策。

### Todo

- [ ] T-D2.1 parent/study/design-lineage group graph。
- [ ] T-D2.2 exact/identity/family/structure overlap。
- [ ] T-D2.3 leave-study-out split。
- [ ] T-D2.4 validation parent holdout。
- [ ] T-D2.5 rescue/control/external subsets。
- [ ] T-D2.6 预训练污染审计。
- [ ] T-D2.7 冻结 metrics、距离 bins、主比较和统计计划。
- [ ] T-D2.8 生成 encrypted/permission-protected test label path 或等价隔离。
- [ ] T-D2.9 判定 Tier A/B/C。
- [ ] T-D2.10 测试、commit、push。

### Gate

- 所有跨 split group overlap = 0；
- 至少满足 Tier B 才允许 learned model；
- 不满足则转 benchmark/data paper，不越级训练。

---

## Phase B0：强基线

### 前置

D2 ≥ Tier B。

### Todo

- [ ] T-B0.1 zero、mutation mean、distance decay。
- [ ] T-B0.2 RNAfold/RNAplfold、LinearPartition。
- [ ] T-B0.3 EternaFold、RNAstructure。
- [ ] T-B0.4 RNAsnp、SNPfold、remuRNA、Riprap、VariantFoldRNA。
- [ ] T-B0.5 RibonanzaNet/RibonanzaNet2 independent difference。
- [ ] T-B0.6 external weight/provenance/contamination manifest。
- [ ] T-B0.7 统一 evaluator、runtime、failure accounting。
- [ ] T-B0.8 冻结 strongest executable baseline。
- [ ] T-B0.9 测试、commit、push。

### Gate

- 每个 baseline 同 split、同 mask、同聚合；
- baseline 失败也必须计入 failure table；
- 不因运行困难挑弱基线。

---

## Phase M0：单 seed 小模型可学习性

### 目标

用最小 GPU 成本判断联合建模是否有信号。

### Todo

- [ ] T-M0.1 线性/MLP GPU baseline。
- [ ] T-M0.2 matched-capacity Siamese encoder。
- [ ] T-M0.3 ReactFlow-Δ Small。
- [ ] T-M0.4 单样本和小批量 overfit tests。
- [ ] T-M0.5 固定 seed、固定预算 pilot。
- [ ] T-M0.6 只在 train/validation 调试。
- [ ] T-M0.7 failure matrix。
- [ ] T-M0.8 测试、commit、push。

### Gate

- 模型 beats zero-change on validation；
- joint > matched Siamese，达到预注册最小 effect；
- 无 NaN/泄漏；
- 不自动追加 seed。

失败：

- 回到数据/表示诊断；
- 不扩大模型；
- 不解封 test。

---

## Phase M1：确认性联合响应模型

### 前置

M0 Gate PASS。

### Todo

- [ ] T-M1.1 冻结最终 small architecture。
- [ ] T-M1.2 sequence-only。
- [ ] T-M1.3 WT-anchored。
- [ ] T-M1.4 uncertainty/calibration。
- [ ] T-M1.5 必需消融。
- [ ] T-M1.6 rescue 子集。
- [ ] T-M1.7 资源和复杂度报告。
- [ ] T-M1.8 冻结 checkpoint，不看 test。
- [ ] T-M1.9 测试、commit、push。

### Gate

- validation 上联合模型优于 strongest baseline；
- calibration 可用；
- 结果不依赖单一 parent；
- 所有主张可由 artifact 重建。

---

## Phase M2：可选 pair-aware 扩展

仅当：

- M1 PASS；
- 有可信 pair-level 实验标签子集；
- 1D 响应已证明有价值。

否则永久跳过，不视为失败。

---

## Phase E0：冻结外部评测

### Todo

- [ ] T-E0.1 再审计 commit/config/data hashes。
- [ ] T-E0.2 一次性解封 primary test。
- [ ] T-E0.3 运行主模型和所有已冻结基线。
- [ ] T-E0.4 cluster bootstrap 和统计。
- [ ] T-E0.5 local/mid/remote、study、parent、probe 分层。
- [ ] T-E0.6 negative cases 和 uncertainty。
- [ ] T-E0.7 生成 immutable final manifest。
- [ ] T-E0.8 测试、commit、push。

### 禁止

- 看完 test 后改模型；
- 换 seed；
- 改主指标；
- 删除不利 study；
- 再训练后覆盖同一 run ID。

---

## Phase P0：论文与发布

### 论文主线

1. 静态差分并不等于实验响应；
2. 构建严格的成对实验 benchmark；
3. 联合模型是否优于 independent difference；
4. 哪些结构响应可预测，哪些不可预测；
5. 不确定性和 OOD；
6. 对 RNA 编辑/变体筛选的限制性价值。

### 必需材料

- dataset card；
- model card；
- license/provenance；
- frozen splits；
- baseline table；
- statistic plan/result；
- negative-result analysis；
- limitations；
- reproducibility manifest；
- code release；
- GitHub release tag。

不得在没有新实验的情况下暗示 prospective wet-lab validation。

---

## Phase I0：可选 mRNA-EditFlow 集成

只有 E0 主 Gate PASS 后才允许。

集成时：

- ReactFlow-Δ 只作结构响应 oracle；
- 输出 uncertainty；
- 高不确定样本拒绝评分；
- 不把模型分数写成实验结论；
- 下游只做公开数据 retrospective evaluation。

---

# 十五、下一阶段立即执行 Goal

以下文本可直接交给新的 Codex/trae 会话。

```text
你现在执行 ReactFlow-Δ 的 Phase R0 + D0，只做路线重置和公开成对实验响应数据可行性审计。

服务器：
ssh -p 22 cunyuliu@36.137.135.49

只读旧现场：
/home/cunyuliu/reactflow_c1_3_stage_20260722

建议新工作树：
/home/cunyuliu/reactflow_delta_goal_20260729

权威合同：
docs/contracts/ReactFlow分阶段执行提示词_ReactFlowDelta科研合同_v2_20260729.md

最高目标：
在不启动任何 learned training 的前提下，回答 RMDB、Ribonanza 和其他公开数据中到底有多少真实、条件匹配、可按 parent/study 隔离的 WT-single-mutant chemical probing pair，并根据合同判定 Tier A/B/C。

强制边界：
1. 不停止、修改或重启旧 C1-3 v4。
2. 不在旧 dirty checkout 上开发。
3. 不训练任何模型。
4. 不下载或提交无关大数据。
5. 原始数据只读，所有下载有 checksum。
6. 不把 designed near-neighbor 自动称为 WT-mutant。
7. 不把 teacher/proxy profile 称为实验真值。
8. 不把 construct 总数当成有效 pair 数。
9. 每个 T-R0/T-D0 任务完成后，测试、聚焦 commit 并 push 当前 GitHub 任务分支。
10. GitHub 不提交 raw RDAT、FASTQ、BAM、weights、checkpoints、cache 或 secret。

先做只读 preflight：
- pwd
- repo/HEAD/branch/dirty worktree
- active ReactFlow processes
- GPU processes
- disk
- artifacts symlink
- remote URL

然后依合同完成 T-R0.1 至 T-D0.12。

必须输出：
- source_registry.jsonl
- raw_manifest.json
- construct_candidates.parquet
- pair_candidates.parquet
- d0_data_feasibility_audit.md
- data_feasibility_summary.json
- parser_fixture_results.json

报告至少包含：
- 总 entry/construct 数；
- 候选 mutational entry 数；
- 明确 WT 数；
- single-mutant pair 数；
- double/rescue 数；
- replicate/no-edit control 数；
- study 数；
- parent 数；
- probe/condition 分布；
- primary-eligible 预估；
- 不能用的原因分布；
- Tier A/B/C 判断；
- 最大三个不确定性；
- 下一阶段是否允许 D1；
- commit SHA 与 GitHub branch。

Gate 未通过时：
- 不训练；
- 不降低阈值；
- 保留审计；
- 进入合同定义的数据/benchmark fallback。
```

---

# 十六、每个阶段统一验收提示词

```text
停止实现下一阶段。现在只验收当前阶段。

1. 核对权威合同版本和 SHA256。
2. 列出所有修改和新增文件。
3. 确认没有触碰未授权旧运行或无关脏文件。
4. 运行 targeted tests。
5. 运行必要的全量 tests。
6. 查找 placeholder、TODO、pass、mock、hard-coded metric。
7. 验证所有 JSON/JSONL/Parquet/YAML schema。
8. 验证 artifact 非空、可解析、可由命令重建。
9. 验证数据来源、license、checksum。
10. 验证 train/validation/test 的 parent/study overlap 为 0。
11. 验证 test 未被用于归一化、阈值、checkpoint 或模型选择。
12. 验证 experimental、proxy、teacher、synthetic 标签没有混写。
13. 验证 GPU-only training；若本阶段不训练则写 NOT APPLICABLE。
14. 对 Gate 每项给 PASS / FAIL / NOT RUN。
15. FAIL 时生成 failure matrix，不降低 Gate。
16. 检查 git diff 无 secret/data/weight/cache。
17. commit 并 push 当前任务分支。

最终只汇报：
- contract SHA
- branch
- commit SHA
- push status
- tests
- artifacts
- Gate
- blockers
- 下一阶段是否获准
```

---

# 十七、失败后的 fail-forward 提示词

```text
当前 Gate 未通过。禁止通过扩大模型、延长训练、追加 seed、改 test、改主指标或删除失败来制造 PASS。

先冻结：
- run/config/git/data/split hashes
- logs
- last usable checkpoint
- metrics
- system metrics
- failure evidence

建立 failure matrix：

A. 数据存在性
- 是否真有 WT-mutant pair
- parent 是否明确
- condition 是否匹配
- 独立 study/parent 是否足够

B. 标签
- reactivity/mask 是否对齐
- probe eligibility
- normalization
- replicate/noise
- missing 是否误作 0

C. 泄漏
- exact
- parent
- study
- design lineage
- pretraining

D. 基线
- zero-change
- thermodynamic difference
- RibonanzaNet difference
- matched Siamese

E. 优化
- 单样本 overfit
- 小数据 overfit
- gradient/NaN
- GPU/device
- checkpoint selection

F. 科学假设
- 是否根本不存在可学习的跨 parent 响应
- 增益是否只在 local/edit site
- remote response 是否低于 noise
- condition heterogeneity 是否主导

每个根因只设计一个最小可证伪实验，按信息增益排序，最多三个。

然后只允许进入合同中预先声明的下一分支：
1. 修复数据/评测错误；
2. 缩小到 Tier B 限定任务；
3. 转 benchmark/data/negative-result；
4. 安全停止该模型路线。

禁止返回 PCCNG，禁止恢复“静态 SOTA 即主论文”的旧叙事。
```

---

# 十八、论文前最终审计清单

- [ ] 文献检索更新到投稿日前 30 天内。
- [ ] 未发现完全同题工作，或已明确重写差异。
- [ ] 不使用 “first” 除非有可审计系统综述证据。
- [ ] 主标签是实验 Δreactivity。
- [ ] 编辑位置不主导主 endpoint。
- [ ] parent/study split 无泄漏。
- [ ] 外部预训练污染已标记。
- [ ] strongest executable baselines 完成。
- [ ] VariantFoldRNA/RNAsnp/Riprap/MutaRNA 已纳入相关工作与比较。
- [ ] RibonanzaNet independent difference 已纳入。
- [ ] classSNitch/dStruct 没被错误当作 sequence-only predictor。
- [ ] 主指标、主模型和主基线在 test 前冻结。
- [ ] cluster-bootstrap 单位是 study/parent。
- [ ] 未追加多 seed。
- [ ] GPU 训练证据完整。
- [ ] negative results 未删除。
- [ ] 论文数字自动来自 final artifacts。
- [ ] README、paper、model card、dataset card 数字一致。
- [ ] GitHub release 不含受限数据、权重或 secret。
- [ ] 无新湿实验时，结论明确限定为 retrospective public-data validation。

---

# 十九、核心参考文献与数据资源

1. Sabarinathan R, et al. RNAsnp: efficient detection of local RNA secondary structure changes induced by SNPs. *Human Mutation* (2013). [DOI 10.1002/humu.22273](https://pubmed.ncbi.nlm.nih.gov/23315997/)
2. Corley M, et al. Detecting riboSNitches with RNA folding algorithms: a genome-wide benchmark. *Nucleic Acids Research* (2015). [DOI 10.1093/nar/gkv010](https://pubmed.ncbi.nlm.nih.gov/25618847/)
3. Woods CT, Laederach A. Classification of RNA structure change by “gazing” at experimental data. *Bioinformatics* (2017). [DOI 10.1093/bioinformatics/btx041](https://doi.org/10.1093/bioinformatics/btx041)
4. Cheng CY, et al. RNA structure inference through chemical mapping after accidental or intentional mutations. *PNAS* (2017). [DOI 10.1073/pnas.1619897114](https://pubmed.ncbi.nlm.nih.gov/28851837/)
5. Lin J, et al. Identification and analysis of RNA structural disruptions induced by single nucleotide variants using Riprap and RiboSNitchDB. *NAR Genomics and Bioinformatics* (2020). [DOI 10.1093/nargab/lqaa057](https://doi.org/10.1093/nargab/lqaa057)
6. Miladi M, et al. MutaRNA: analysis and visualization of mutation-induced changes in RNA structure. *Nucleic Acids Research* (2020). [DOI 10.1093/nar/gkaa331](https://doi.org/10.1093/nar/gkaa331)
7. Kirven KJ, et al. VariantFoldRNA: a flexible, containerized, and scalable pipeline for genome-wide riboSNitch prediction. *NAR Genomics and Bioinformatics* (2025). [DOI 10.1093/nargab/lqaf066](https://doi.org/10.1093/nargab/lqaf066)
8. Choudhary K, et al. dStruct: identifying differentially reactive regions from RNA structurome profiling data. *Genome Biology* (2019). [DOI 10.1186/s13059-019-1641-3](https://doi.org/10.1186/s13059-019-1641-3)
9. Ribonanza consortium. Ribonanza: deep learning of RNA structure through dual crowdsourcing. (2024). [PubMed/PMC](https://pubmed.ncbi.nlm.nih.gov/38464325/)
10. de Lajarte AA, et al. Diverse database and machine learning model to narrow the generalization gap in RNA structure prediction. *Science Advances* (2026). [Full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC12935039/)
11. Chen Z, et al. Fair splits flip the leaderboard: CHANRG reveals limited generalization in RNA secondary-structure prediction. arXiv (2026). [arXiv:2603.22330](https://arxiv.org/abs/2603.22330)
12. Cordero P, et al. An RNA Mapping DataBase for curating RNA structure mapping experiments. *Bioinformatics* (2012). [DOI 10.1093/bioinformatics/bts554](https://pmc.ncbi.nlm.nih.gov/articles/PMC3496344/)
13. RNA Mapping Database. [Current database](https://rmdb.stanford.edu/) and [download/versioning documentation](https://rmdb.stanford.edu/about/).

---

# 二十、最终执行原则

1. **数据先于模型。**
2. **实验连续响应先于专家分类。**
3. **跨 study/parent 泛化先于随机 held-out。**
4. **联合模型必须战胜 matched independent difference，才有方法创新。**
5. **小模型先于 PairFormer。**
6. **clean protocol 先于外部预训练 headline。**
7. **GPU 训练，但不盲跑。**
8. **不频繁查看，但必须有健康监控。**
9. **等待时并行做无冲突的高价值工作。**
10. **每个任务测试、commit、push。**
11. **失败保留证据，只能 fail-forward。**
12. **没有湿实验和专家也能做，但必须诚实限定结论。**

本合同的最终成功标准不是“训练了更大的模型”，而是：

> 用公开、可审计、成对实验数据，严格回答联合 mutation-conditioned 模型是否真的比两次静态预测相减更能预测 RNA 的实验结构响应；如果答案是否定的，也形成一个可信、可发表、可复用的 benchmark 和负结果。
