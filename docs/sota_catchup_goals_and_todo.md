下面这版以 **ReactFlow 当前真实代码、实验结果和 mRNA-EditFlow 下游需求**为出发点，重新定义了静态结构预测、结构 ensemble、编辑干预与功能验证之间的关系。它不再把“堆一个 Evoformer 并达到某个单一 F1”当作终点，而是把顶刊主线收敛为一个可验证的科学问题。

# ReactFlow 最终版 SOTA Goal  
## 从弱 partner-class 生成器升级为探针校准、干预感知的 RNA 结构 Ensemble Foundation Model

**版本日期：2026-07-21**

**工作名称：ReactFlow 2.0 / ReactFlow-SEI**

> SEI = Structure prediction、Ensemble inference、Intervention modeling

**目标期刊层级：Nature Methods / Science Advances / Nature Machine Intelligence / Nature Biotechnology 交叉方向**

**项目原则：不降低科学目标，不虚报 SOTA；所有结论必须由同协议复现、严格 OOD 评估、结构干预实验和功能验证共同支撑。**

---

# 0. 执行摘要

ReactFlow 当前已经具备较完整的真实数据、数据质控、Rfam/MMseqs 切分、结构合法化、化学探针前向模型、冻结 RibonanzaNet2 特征、训练监控和评估审计体系。当前仓库不是概念草稿，而是一个可以运行的实验系统。

但当前核心模型仍然是：

- 每个位点预测 `L+1` 个 partner classes；
- 小型单层特征投影；
- 非对称双线性配对打分；
- 按行 softmax；
- CTMC 从离散 partner states 生成结构。

这一实现直接写在 `PairwiseDenoiser` 中。

在严格 MMseqs split 上，当前 base 与 frozen warm-start 的 novel-clan F1 分别约为 0.0267 和 0.0447；早期最优 exact-split novel-clan F1 约为 0.0624。 当前系统能够输出合法结构，但没有形成足够强的 pair recovery、长程建模和跨家族结构归纳能力。

因此，最终路线不是简单“继续扩大当前 DFM”，也不是彻底抛弃 ReactFlow 的生成式思想，而是进行一次明确的职责重构：

1. **静态结构预测主干**：改为确定性的、显式对称的 \(L\times L\) pair posterior predictor，负责达到同协议 SOTA。
2. **结构化解码器**：使用 nested DP 与 pseudoknot-compatible decoder 双轨输出合法结构。
3. **ReactFlow ensemble 模块**：从“静态结构主预测器”调整为“基于强 pair posterior 的合法结构 ensemble 生成器”。
4. **化学探针观测模型**：从简单的一阶未配对概率回归，升级为 probe、batch、replicate 和结构上下文感知的概率观测模型。
5. **干预式结构预测**：直接预测编辑前后的结构变化，而不只分别预测两张静态结构。
6. **mRNA-EditFlow 整合**：把 ReactFlow 变成 5′UTR–CDS–3′UTR 联合编辑中的结构 oracle，验证跨区域协同、最小编辑和功能收益。

论文的最终主问题定义为：

> **在家族严格隔离、结构标签稀缺、化学探针只提供 ensemble 投影的条件下，能否学习一个校准的 RNA 结构分布模型，不仅准确预测未知 RNA 家族的静态配对结构，还能预测少量序列编辑导致的远程结构重排，并指导保持蛋白不变的全长 mRNA 优化？**

---

# 1. 领域痛点：RNA 结构预测真正没有解决的是什么

## 1.1 高分 benchmark 不等于真实泛化

多数 RNA 二级结构数据集中，大量序列来自少数经典 ncRNA family。随机切分或仅按 sequence identity 切分，很容易让模型在训练和测试中看到相似的家族骨架。

eFold/RNAndria 的核心结论不是单纯提出一个更大的网络，而是证明：

- 只增加训练样本数量不足以获得跨家族泛化；
- 数据的长度、结构复杂度和 RNA 类型多样性决定 OOD 性能；
- 长 mRNA、病毒 RNA 和 lncRNA 与短 ncRNA benchmark 之间存在明显的分布鸿沟。

因此，ReactFlow 的主战场不能是随机切分下的平均 F1，而必须是：

- MMseqs-disjoint；
- Rfam family/clan-disjoint；
- structure-similarity-disjoint；
- time-censored；
- 长 RNA 和跨域测试。

---

## 1.2 静态 dot-bracket 不是 RNA 的完整真值

RNA 可以在多个相近能量构象之间转换。化学探针测量的是群体平均可及性，不是一张唯一结构图。

2026 年的单分子直接 RNA 测序与化学探针工作已经能够在细胞内区分 RNA 结构 ensemble，并应用于 SARS-CoV-2 基因组和真菌转录组。 更广泛的 ensemble 预测研究也指出：当前瓶颈不仅是生成多个构象，而是缺少可靠的 ensemble 真值、population 定量方法和统一比较指标。

所以，ReactFlow 原始的“结构分布而非单结构”问题具有真实科学价值，但必须建立在一个强静态 pair posterior 之上。弱静态模型生成再多样的样本，也只是生成多样化错误。

---

## 1.3 长 RNA 的关键问题不是单纯长度，而是跨区域耦合

全长 mRNA 包含：

- 5′UTR；
- CDS；
- 3′UTR；
- poly(A) 邻接区；
- 局部 stem-loop；
- 起始区 accessibility；
- 跨越数百乃至上千核苷酸的长程相互作用。

随机切窗会删除跨窗口碱基对，也会把一个完整结构域切成不真实的边界条件。

ReactFlow 当前已经支持 windowing，但仓库也明确承认跨窗口 pair 被省略，长 RNA 信息被割裂。

这恰好对应 mRNA-EditFlow 的科学问题：该项目希望联合编辑 5′UTR、CDS 和 3′UTR，同时保持阅读框和蛋白完全不变。 一个只会局部折叠的结构 oracle 无法判断：

- 5′UTR 编辑是否通过 CDS 区域产生远端影响；
- 同义密码子替换是否破坏 UTR–CDS 配对；
- 3′UTR 元件插入是否改变起始区 accessibility；
- 两个单独有害的编辑是否通过补偿性配对形成协同收益。

---

## 1.4 静态结构准确率与功能效应之间存在缺口

mRNA 的翻译效率、稳定性和免疫原性受到序列、修饰、结构和细胞条件共同影响。结构是重要因素，但不是唯一因素。

因此，顶刊级论文不能停留在：

> “ArchiveII F1 提升了 0.02。”

必须进一步证明：

> “模型准确预测了编辑引起的结构变化，而该变化能够解释或改善翻译、稳定性或调控功能。”

这构成 ReactFlow 与普通 RNA folding 模型之间最重要的差异。

---

# 2. 现有方法的局限与 ReactFlow 的切口

| 现有路线 | 已解决的问题 | 主要局限 | ReactFlow 的切口 |
|---|---|---|---|
| ViennaRNA、RNAstructure、EternaFold、MXfold2 | 热力学可解释、合法 nested structure | 参数化能量模型难覆盖复杂细胞环境和 OOD family | 学习数据驱动 pair prior，同时保留结构化解码 |
| UFold、eFold、RNAformer、PriFold | 强静态 contact prediction | 主要输出单结构或单张 BPP；对编辑诱导变化和 ensemble population 支持有限 | 在强静态主干上增加探针校准的 ensemble posterior |
| RNA-FM、RiNALMo、ERNIE-RNA | 从大规模无标签序列学习 RNA 表征 | foundation embedding 不自动等于准确配对；存在预训练污染风险 | 以 pair-aware trunk 和结构目标显式转化 foundation 表征 |
| RibonanzaNet 系列 | 大规模 chemical probing representation | 主要学习 sequence-to-reactivity；RibonanzaNet2 alpha 权重尚不是稳定的同行评议二级结构标准 | 使用其 single/pair representations，同时独立建立可审计主干 |
| 生成式 diffusion/flow | 可表达多峰结构分布 | 若基础 score 较弱，采样只会放大错误；普通 categorical state 破坏对称性 | 改成基于强 BPP 的合法 edge-set ensemble flow |
| mRNA 生成与 RL | 可优化翻译或稳定性代理 | 结构 oracle 通常为 MFE 或简单 predictor，且缺少编辑不确定性 | 提供编辑前后 \(\Delta\)BPP、ensemble shift 和结构风险 |

RiNALMo 已表明大规模 RNA LM 可以改善 unseen-family 结构泛化；其公开模型约 650M 参数，使用约 3600 万条 ncRNA 预训练。 ERNIE-RNA 则通过结构增强 attention representation 强化 RNA 表征。

但 foundation model 不能替代任务结构。ReactFlow 必须证明新增价值来自：

- pair-aware architecture；
- probing likelihood；
- ensemble inference；
- intervention modeling；

而不是仅仅换了一个更大的 encoder。

---

# 3. 最终科学假设与论文主张

## 3.1 核心假设

### H1：强对称 pair trunk 是可靠 ensemble 建模的前提

在相同训练数据和 backbone 下，显式 \(L\times L\) pair representation、triangle update 和 structured decoding 应显著优于当前 partner-class denoiser。

证据要求：

- static pair F1、MCC、AUPRC 显著提升；
- long-range recall 提升；
- pair probability calibration 改善；
- novel-clan 提升不能仅来自 in-clan 记忆。

---

### H2：化学探针作为结构分布观测，比普通 auxiliary regression 更有价值

探针数据应通过：

\[
p(y\mid S,x,k,c)
\]

进入模型，其中：

- \(S\)：结构或结构 ensemble；
- \(k\)：探针类型；
- \(c\)：实验条件、batch、细胞或 replicate；
- \(y\)：观测 reactivity。

模型不仅拟合平均 reactivity，还要拟合：

- replicate variance；
- probe-specific sensitivity；
- nucleotide identity；
- stem end、loop 和局部堆叠上下文；
- ensemble population uncertainty。

证据要求：

- held-out reactivity NLL、Pearson、Spearman 和 calibrated MAE 改善；
- static F1 不下降；
- ensemble diversity 不坍缩；
- calibration 优于仅做 sequence-to-reactivity regression 的模型。

---

### H3：直接预测编辑造成的结构差异，优于分别折叠编辑前后序列

给定原始序列 \(x\)、编辑后序列 \(x'\) 和 edit mask \(m\)，模型直接输出：

\[
P(S\mid x),\quad P(S'\mid x'),\quad
\Delta P=P(S'\mid x')-P(S\mid x)
\]

并预测：

- pair gain；
- pair loss；
- accessibility change；
- ensemble population shift；
- long-range structural impact；
- uncertainty。

证据要求：

- \(\Delta\)BPP correlation；
- pair gain/loss F1；
- compensatory rescue ranking；
- edit-effect calibration；
- 优于两个独立 fold 结果直接相减。

---

### H4：跨区域结构协同能够改善受约束 mRNA 编辑

对两个区域的编辑 \(A\) 和 \(B\)，定义结构或功能协同：

\[
\operatorname{Synergy}(A,B)
=
\Delta F(A+B)-\Delta F(A)-\Delta F(B)
\]

其中 \(F\) 可以是：

- 起始区 accessibility；
- 目标 BPP 保持程度；
- ensemble entropy；
- translation efficiency；
- mRNA half-life；
- 蛋白表达。

核心证据是：

- 单独编辑效果有限或有害；
- 联合编辑产生显著非加性收益；
- 模型提前预测该收益；
- 化学探针和功能实验共同验证。

---

# 4. 最终任务定义

ReactFlow 2.0 包含四个层次，而不是把所有目标混成一个 F1。

## Task A：静态二级结构预测

输入：

\[
x\in\{A,C,G,U\}^{L}
\]

输出：

- calibrated base-pair probability matrix \(B\in[0,1]^{L\times L}\)；
- nested structure；
- pseudoknot-compatible structure；
- pair uncertainty；
- optional pair type。

这是进入后续所有任务的基础 gate。

---

## Task B：探针条件结构 ensemble 推断

输入：

\[
(x,\;y^{DMS},\;y^{SHAPE/2A3},\;condition)
\]

输出：

- top-\(K\) 合法结构；
- 每个结构的 population；
- ensemble BPP；
- reactivity reconstruction；
- population uncertainty。

---

## Task C：编辑干预结构预测

输入：

\[
(x,\;x',\;edit\ mask,\;region\ labels)
\]

输出：

- pair gain/loss map；
- \(\Delta\)BPP；
- \(\Delta\)accessibility；
- 远端受影响区域；
- compensatory interaction；
- confidence/OOD score。

---

## Task D：受约束全长 mRNA 设计

输入：

- 原始全长 mRNA；
- 固定蛋白序列；
- 可编辑区域；
- 最大 edit budget；
- 目标结构或功能；
- 禁止 motif 和制造约束。

输出：

- 最小编辑候选；
- CDS 蛋白完全一致；
- 结构保持或定向重塑；
- 翻译、稳定性和免疫风险的 Pareto frontier；
- 每项候选的结构不确定性。

---

# 5. 最终模型架构

## 5.1 总体架构

```text
RNA sequence / optional probing / optional edit pair
                     │
                     ▼
       Multi-source RNA foundation encoder
       ├─ RiNALMo / ERNIE-RNA
       ├─ RibonanzaNet2 single + pair features
       └─ ReactFlow 自训练 long-mRNA encoder
                     │
                     ▼
       Single representation h_i
       Pair initialization z_ij
                     │
                     ▼
     Symmetric PairFormer / Evoformer trunk
       ├─ triangle multiplicative update
       ├─ triangle attention
       ├─ axial row/column attention
       ├─ single↔pair communication
       ├─ relative-distance / region features
       └─ long-range global routing
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
      Static BPP   Probe head   Edit-delta head
          │          │          │
          ▼          ▼          ▼
   Dual decoder   Ensemble     Cross-region
 nested / PK      edge flow     intervention
          │          │          │
          └──────────┼──────────┘
                     ▼
       ReactFlow structure oracle API
                     │
                     ▼
              mRNA-EditFlow / RL
```

---

## 5.2 Foundation encoder

不再只使用冻结的 single-token feature 加线性 adapter。

建议并行比较三条路线：

### Encoder E1：RiNALMo

优点：

- 大规模通用 RNA 表征；
- 已报告 unseen-family 二级结构泛化；
- 权重和训练脚本公开。

使用方式：

- frozen probe；
- LoRA；
- full fine-tuning；
- intermediate-layer weighted sum。

### Encoder E2：RibonanzaNet2

作用：

- 提供 probing-pretrained single representation；
- 提供 pairwise representation；
- 作为 chemical mapping teacher。

限制：

- 当前是 Kaggle alpha checkpoint；
- 版本可能变化；
- 不能把其 model-card 指标当作论文基线；
- 必须记录版本、checkpoint SHA256 和推理代码。

### Encoder E3：ReactFlow-mRNA-FM

由于算力充足，应训练一个针对全长 mRNA 和结构干预的自有 backbone，而不是永久依赖外部 alpha 模型。

预训练语料包括：

- RNAcentral；
- Rfam；
- GENCODE/RefSeq full-length transcripts；
- 病毒基因组与结构域；
- Ribonanza/Ribonanza2 chemical profiles；
- 去污染后的结构标签数据。

预训练目标：

1. span-masked nucleotide modeling；
2. codon-aware masked modeling；
3. 5′UTR/CDS/3′UTR region contrastive learning；
4. masked reactivity reconstruction；
5. pair-map distillation；
6. mutation consistency；
7. homolog contrastive learning；
8. local-to-global domain reconstruction。

最终主模型使用 gated fusion：

\[
h_i=\sum_m g_{i,m}h_i^{(m)}
\]

而不是简单拼接所有 embedding。

---

## 5.3 Pair initialization

对每对位置 \((i,j)\)，构建：

\[
z_{ij}^{0}
=
\operatorname{MLP}\left[
h_i,\,
h_j,\,
h_i\odot h_j,\,
|h_i-h_j|,\,
r_{ij},\,
c_{ij},\,
a_{ij}
\right]
\]

其中：

- \(r_{ij}\)：相对距离和方向；
- \(c_{ij}\)：AU/GC/GU compatibility；
- \(a_{ij}\)：foundation attention 或 pretrained pair feature；
- 额外加入 region pair，例如 5′UTR–CDS、CDS–3′UTR。

要求：

- \(z_{ij}=z_{ji}\) 通过构造保证；
- 不依赖后验正则勉强学习对称性；
- 对角线永久 mask；
- canonical-only 与 all-pair label 分开建模。

---

## 5.4 Pair trunk

推荐配置范围：

- 24–48 个 block；
- single width 512–768；
- pair width 128–256；
- 8–16 个 attention heads；
- bf16；
- gradient checkpointing；
- FlashAttention；
- FSDP/ZeRO。

每个 block 包含：

1. single self-attention；
2. single-to-pair outer update；
3. triangle multiplication outgoing；
4. triangle multiplication incoming；
5. triangle attention starting-node；
6. triangle attention ending-node；
7. pair transition；
8. pair-to-single attention；
9. 显式对称化。

RNAformer 的 axial attention 和二维 latent modeling、DEPfold 的全局结构化解码、PriFold 的 base-pair motif inductive bias 都应作为 matched-capacity ablation，而不是只选一个架构。

---

## 5.5 长 RNA 层次化模块

对于 \(L>1024\)，禁止简单随机切窗。

采用三层机制：

### 局部层

- overlap windows；
- 每个窗口保留局部 pair representation；
- 窗口边界至少 128–256 nt overlap。

### 全局层

- domain tokens；
- region tokens；
- top-\(K\) long-range anchor candidates；
- sparse global pair attention。

### 拼接层

- overlap consistency loss；
- parent-coordinate recovery；
- cross-window pair proposal；
- domain closedness audit；
- 全局 constrained decoder。

每个被切分的训练样本必须保存：

- parent transcript ID；
- parent coordinates；
- 被删除的真实跨窗 pair 数量；
- closed-domain score；
- overlap agreement；
- 原始长度和 region composition。

---

## 5.6 双轨结构解码

ReactFlow 当前已经实现 exact maximum-weight nested projection，复杂度为 \(O(L^3)\)。 该实现可以继续作为 nested decoder 的参考和 evaluator，但需要 GPU 化与 differentiable structured training。

### Decoder N：Nested

用于 canonical pseudoknot-free benchmark：

- weighted Nussinov；
- partition function / inside-outside；
- maximum expected accuracy；
- differentiable structured loss；
- optional thermodynamic energy fusion。

### Decoder P：Pseudoknot-compatible

用于包含 crossing pairs 的 benchmark：

- maximum-weight matching；
- dependency parsing formulation；
- crossing-aware edge selection；
- pseudoknot type head；
- one-partner constraint；
- 不强制 non-crossing。

所有结果必须明确标注：

- nested-only；
- pseudoknot-allowed；
- canonical-only；
- canonical+wobble；
- all annotated pairs。

禁止把不同 label/decoder 协议的 F1 放入同一列。

---

## 5.7 ReactFlow ensemble 模块

保留 flow matching，但更换状态空间和职责。

### 当前废弃形式

- 每个位点独立 partner class；
- uniform categorical initialization；
- 以 CTMC 结果作为主要静态预测；
- 采样后再修补非对称冲突。

### 新形式：legal edge-set flow

结构状态直接定义为合法 pair edge set：

\[
S=\{(i,j)\}
\]

基本 transition：

- add pair；
- remove pair；
- swap partner；
- stem extension；
- stem contraction；
- compensatory pair move。

每次操作满足：

- symmetric edge；
- one partner per nucleotide；
- min-loop；
- optional nested constraint；
- optional canonical mask。

初始化不再是 uniform noise，而是：

- static BPP posterior；
- DP posterior sample；
- thermodynamic ensemble；
- probe-conditioned posterior。

因此，生成式模块负责：

- alternative conformations；
- top-\(K\) sampling；
- uncertainty；
- probe-conditioned refinement；
- edit-induced ensemble shift；

不承担最基础的 pair classification。

---

## 5.8 化学探针概率观测模型

当前仓库已经正确保留 missing-value mask，并区分 DMS 与 2A3 的有效碱基范围。 

下一版观测模型定义为：

\[
y_{i,r}^{(k)}
\sim
\mathcal{D}
\left(
\mu_i^{(k)}(S,x,c),
\sigma_i^{2(k)}(S,x,c)
\right)
\]

其中：

- \(r\)：replicate；
- \(k\)：DMS、SHAPE、2A3 等；
- \(c\)：cell、buffer、temperature、batch；
- \(\mathcal D\)：Gaussian、Student-t 或 zero-inflated distribution。

特征包括：

- unpaired probability；
- base identity；
- loop type；
- stem end/fraying；
- local stacking；
- neighboring reactivity；
- modification status；
- experimental error；
- read coverage。

同时拟合：

- first moment；
- second moment；
- replicate likelihood；
- calibration；
- population entropy。

---

## 5.9 编辑干预头

对原始序列和编辑后序列进行共享编码，并建立 cross-sequence pair alignment：

```text
original x ──► shared encoder ──► BPP(x)
     │              │
 edit alignment     cross-attention
     │              │
edited x'  ──► shared encoder ──► BPP(x')
                     │
                     ▼
             Delta-pair decoder
```

输出四分类：

- unchanged-unpaired；
- pair gained；
- pair lost；
- partner switched。

加入以下结构一致性约束：

### Identity consistency

没有编辑时：

\[
\Delta B=0
\]

### Locality prior

小编辑通常应产生稀疏变化，但允许模型预测远端传播。

### Reverse consistency

从 \(x\to x'\) 和 \(x'\to x\) 的变化应互为相反。

### Compensatory rescue

若突变 A 破坏 pair，补偿突变 B 恢复 pair，则模型必须预测：

\[
\Delta(A)<0,\qquad
\Delta(A+B)\approx0
\]

### Region synergy

显式建模 UTR–CDS、CDS–UTR 和 UTR–UTR pair change。

---

# 6. 数据体系

## 6.1 数据分层

| 层级 | 数据类型 | 主要用途 |
|---|---|---|
| D0 | 无标签 RNA 序列 | foundation pretraining |
| D1 | 化学探针 profiles | probing representation 与 ensemble likelihood |
| D2 | 显式二级结构标签 | static BPP 和 decoder |
| D3 | 多构象/单分子 probing | ensemble population |
| D4 | WT–mutant / compensatory edits | intervention head |
| D5 | mRNA 功能标签 | translation、stability、expression |
| D6 | 自建前瞻实验 | 最终因果和功能验证 |

---

## 6.2 D0：无标签预训练序列

### RNAcentral

用途：

- ncRNA foundation pretraining；
- 多物种、多 family；
- family-balanced sampling。

必须记录：

- RNAcentral release；
- source database；
- taxon；
- sequence checksum；
- 与 benchmark 的相似度。

### Rfam

Rfam 用于：

- family/clan labels；
- covariance-model family；
- OOD split；
- family-balanced pretraining。

截至 2026 年初的 Rfam 15.1 已包含超过四千个 RNA family，可作为 clan-aware 数据治理基础。

### GENCODE / RefSeq

用于全长 mRNA：

- 5′UTR/CDS/3′UTR region modeling；
- codon-aware pretraining；
- mRNA-EditFlow 对接。

当前 mRNA-EditFlow 已经具有 GENCODE 和 RefSeq 的解析、CDS 验证和 region track。

---

## 6.3 D1：化学探针数据

### Ribonanza / Ribonanza2

用途：

- masked reactivity pretraining；
- probe-specific representation；
- sequence-to-reactivity teacher；
- ensemble likelihood。

Ribonanza 的公开工作包含约两百万条高通量 chemical mapping profiles，为大规模 probing pretraining 提供了基础。

RibonanzaNet2 alpha 与 Ribonanza2 数据卡中的具体规模、schema 和版本必须在下载时固化：

- Kaggle dataset version；
- model version；
- competition terms；
- hash；
- H5 schema；
- probe labels；
- read/SNR filters。

不得把动态 Kaggle 页面上的未固定描述直接写成论文不可变事实。

### RNAndria/eFold

Dryad 数据包含：

- 1,098 个 pri-miRNA；
- 1,456 个 human mRNA regions；
- ArchiveII；
- PDB；
- viral fragments；
- lncRNA；
- eFold train。

其发布版本约 368 MB，并明确列出结构和 reactivity 字段。

---

## 6.4 D2：显式二级结构数据

候选来源：

- bpRNA-1m / bpRNA90；
- RNAStrAlign；
- ArchiveII；
- RNASSTR；
- eFold train；
- RNA3DB-2D；
- PDB-derived structures；
- curated Rfam consensus structures。

RNASSTR 提供大规模、重新整理过的 RNA 二级结构数据文件，可作为统一数据源之一，但必须检查其与 bpRNA、RNAStrAlign 和 benchmark 的重叠。

### 关键规则

ArchiveII 不得既进入训练集又作为独立测试集。

所有来源先合并成一个 global registry，再统一去重和切分，不能各数据库内部去重后直接拼接。

---

## 6.5 D3：ensemble 数据

优先来源：

1. sm-PORE-cupine 单分子结构 ensemble；
2. transcriptome-scale ensemble mapping；
3. riboswitch 多状态数据；
4. DREEM/DMS-MaP multi-state datasets；
5. HIV TAR sequence-ensemble-function variants；
6. 自建 reporter ensemble。

sm-PORE-cupine 已展示利用单分子 chemical modifications 和 mixture clustering 区分细胞内结构 ensemble。

这类数据用于评估：

- ensemble cluster recovery；
- population fraction；
- alternative pair recall；
- population calibration；

不能仅转成一张平均 dot-bracket。

---

## 6.6 D4：编辑干预数据

需要建立专门的 **RNA Structural Intervention Benchmark，RSIB**。

样本类型：

- single mutation；
- double mutation；
- compensatory mutation；
- synonymous CDS mutation；
- UTR mutation；
- cross-region mutation；
- insertion/deletion；
- structure-preserving edit；
- structure-disrupting edit。

每条记录包含：

```text
wild-type sequence
edited sequence
edit coordinates
region labels
WT probing
mutant probing
WT ensemble
mutant ensemble
functional measurements
experimental condition
```

数据来源分三类：

### 公开整理

- riboswitch mutational scans；
- viral RNA mutational structure studies；
- TAR variants；
- RNA Mapping Database；
- published compensatory-mutation experiments。

### 合成训练数据

由可靠热力学或强 teacher 生成：

- 单点破坏；
- 补偿恢复；
- stem extension；
- long-range pair switch。

合成数据只能用于 curriculum 或预训练，不能作为最终科学验证。

### 自建前瞻实验

重点生成：

- 5′UTR–CDS；
- CDS–3′UTR；
- 5′UTR–3′UTR；
- 两个 CDS 区域之间的远程补偿编辑。

---

## 6.7 D5：mRNA 功能数据

### GSE114002

包含约 30 万条随机 5′UTR、自然变体和设计序列，并测量 mean ribosome load；同时包含未修饰、pseudouridine 和 m1Ψ 条件。

用途：

- 5′UTR translation prediction；
- modification-conditioned structure–translation关系；
- minimal edit ranking。

### GSE145046

包含超过一百万个 5′UTR variants，并测量 translation 和 stability 相关信号。

用途：

- structure/accessibility 与翻译、降解的关系；
- uORF/G-quadruplex 周围结构效应；
- 大规模 edit-effect benchmark。

### mRNABERT / PERSIST-seq 相关数据

mRNABERT 公开了预训练和下游数据，并整理了 233 条全长 mRNA reporter sequences，包含翻译和稳定性标签。

用途：

- full-length mRNA property prediction；
- UTR–CDS interaction；
- small-data transfer；
- downstream function head。

### RiboNN translation atlas

该资源整合了 3,819 个 ribosome profiling datasets、覆盖超过 140 个人和小鼠细胞类型，可用于构建 cell-conditioned TE evaluator。

### RNA stability enhancer 数据

近期研究筛选了 196,277 条病毒序列并鉴定能够提高 mRNA 稳定性和翻译的调控元件。

用途：

- 3′UTR element insertion；
- element–CDS context dependency；
- structure-aware enhancer ranking。

---

# 7. 数据获取与版本治理

## 7.1 Immutable raw registry

每个数据源必须保存：

- source name；
- release/version；
- acquisition date；
- DOI/accession；
- license；
- raw checksum；
- file size；
- schema；
- downloader version；
- preprocessing commit；
- known overlap。

Raw 数据永久只读。

---

## 7.2 Unified record schema

```json
{
  "record_id": "...",
  "sequence": "ACGU...",
  "source": "...",
  "parent_id": "...",
  "coordinates": [0, 256],
  "region_track": ["utr5", "cds", "utr3"],
  "pairs": [[i, j, "canonical"]],
  "pseudoknot_pairs": [],
  "reactivity": {
    "DMS": {
      "values": [],
      "errors": [],
      "mask": [],
      "replicate": "...",
      "condition": "..."
    }
  },
  "family": "...",
  "clan": "...",
  "cluster": "...",
  "release_date": "...",
  "quality_flags": []
}
```

---

# 8. 数据预处理

## 8.1 序列标准化

- upper-case；
- T→U；
- 仅 A/C/G/U 的主训练 track；
- ambiguity bases 保留到独立 robustness track；
- modified nucleotides 不可简单映射后丢失信息，必须保留 modification track；
- 多链、circular RNA 和 intermolecular structures 分开处理；
- CDS 验证 start、stop、frame 和 internal stop。

---

## 8.2 结构标签处理

对 PDB-derived RNA：

1. 使用固定版本 DSSR/FR3D 等工具提取 base pairs；
2. 至少两种 extractor 交叉比较；
3. 区分 canonical、wobble、noncanonical；
4. 区分 secondary contacts 与 tertiary contacts；
5. 对 extractor disagreement 建立 uncertainty mask；
6. 保留 pseudoknot bracket pages；
7. 删除 ligand-mediated 或多链配对，或放入独立任务。

eFold 的 PDB 处理使用了特定的单链过滤和 RNApdbee/DSSR 转换流程；ReactFlow 必须复现其 protocol 后，再构建自己的更新版 PDB 数据。

---

## 8.3 Chemical probing QC

当前代码的 missing mask、reads/SNR 和 MAD outlier 机制可以保留。

需要增加：

- replicate correlation；
- batch correction；
- primer/adaptor mask；
- base-specific probe mask；
- raw/normalized 双版本；
- error-aware weights；
- coverage floor；
- saturation detection；
- condition metadata；
- in vitro/in vivo 分轨；
- normalized profile 不跨数据集盲目共用标尺。

---

## 8.4 全局去重与防泄漏

按如下顺序建立 union-find contamination groups：

1. exact sequence；
2. reverse/complement 或 DNA/RNA normalization duplicate；
3. overlapping parent transcript windows；
4. MMseqs identity clusters；
5. BLAST local high-coverage similarity；
6. Rfam family；
7. Rfam clan；
8. structure similarity；
9. identical probing construct；
10. 同一 PDB chain 的不同裁剪。

然后整个 contamination group 只能属于一个 split。

---

## 8.5 Pretraining contamination audit

外部 foundation model 可能已经看过 benchmark 序列。仅对微调数据做 CD-HIT/MMseqs 不足以排除污染。

最终报告必须区分：

- external-pretrained；
- time-restricted self-pretrained；
- from-scratch；
- exact overlap；
- >80%、>60%、>40% identity；
- same family；
- same clan；
- same structural motif；
- unknown contamination。

SOTA 主结论至少要包含一个可审计的 time-restricted 或 from-scratch model，防止所有收益都无法与预训练记忆区分。

---

# 9. 训练策略

## Stage P0：结构表示预训练

任务：

- MLM/span corruption；
- masked pair reconstruction；
- teacher BPP distillation；
- masked reactivity prediction；
- family contrastive；
- region-aware mRNA modeling；
- mutation consistency。

---

## Stage P1：静态结构监督

损失：

\[
\mathcal L_{\text{static}}
=
\mathcal L_{\text{pair}}
+\lambda_{\text{f1}}\mathcal L_{\text{soft-F1}}
+\lambda_{\text{struct}}\mathcal L_{\text{structured}}
+\lambda_{\text{distill}}\mathcal L_{\text{teacher}}
+\lambda_{\text{cal}}\mathcal L_{\text{pair-calibration}}
\]

其中：

- `pair loss` 使用 class-balanced BCE/focal；
- 正负样本只在合法候选 pair space 中计算；
- family-balanced 和 source-balanced sampler；
- sequence-level macro 与 pair-level micro 同时优化；
- decoder threshold 只能在 validation 上确定。

---

## Stage P2：数据复杂度 curriculum

训练顺序：

1. 短、明确、nested ncRNA；
2. mixed families；
3. pri-miRNA；
4. long-range contacts；
5. viral domains；
6. human mRNA regions；
7. lncRNA；
8. pseudoknot；
9. full-length mRNA。

但 curriculum 不是简单先短后长。每阶段都必须保留旧数据 replay，防止 catastrophic forgetting。

---

## Stage P3：probe-calibrated ensemble

\[
\mathcal L_{\text{ensemble}}
=
\mathcal L_{\text{static}}
+\lambda_{\text{react}}\mathcal L_{\text{react-NLL}}
+\lambda_{\text{variance}}\mathcal L_{\text{replicate}}
+\lambda_{\text{entropy}}\mathcal L_{\text{max-ent}}
+\lambda_{\text{population}}\mathcal L_{\text{ensemble-pop}}
\]

先冻结强 static trunk，只训练 observation 和 ensemble 模块；随后 joint fine-tune。

---

## Stage P4：intervention training

\[
\mathcal L_{\text{edit}}
=
\mathcal L_{\Delta\text{pair}}
+\lambda_{\text{cycle}}\mathcal L_{\text{reverse}}
+\lambda_{\text{identity}}\mathcal L_{\text{no-edit}}
+\lambda_{\text{rescue}}\mathcal L_{\text{compensatory}}
+\lambda_{\text{synergy}}\mathcal L_{\text{cross-region}}
\]

训练 batch 需要包含：

- WT；
- single edit；
- double edit；
- compensatory edit；
- function label。

---

## Stage P5：mRNA-EditFlow 联合优化

ReactFlow 不再提供一个单一 MFE reward，而是输出 reward vector：

```text
structure preservation score
target accessibility score
long-dsRNA penalty
ensemble shift
cross-region synergy
pseudoknot risk
uncertainty penalty
OOD penalty
```

RL 目标：

\[
R
=
w_{\mathrm{func}}R_{\mathrm{function}}
+w_{\mathrm{struct}}R_{\mathrm{structure}}
-w_{\mathrm{edit}}C_{\mathrm{edit}}
-w_{\mathrm{risk}}U_{\mathrm{structure}}
\]

硬约束：

- protein identity = 100%；
- frame validity = 100%；
- forbidden motif violations = 0；
- edit budget satisfaction = 100%。

---

# 10. 测评协议

## 10.1 首要原则

任何外部论文数字只能标为 `cited_only`。

只有以下条件同时相同，才能进入主表：

- data version；
- split manifest；
- sequence preprocessing；
- label definition；
- decoder protocol；
- metric script；
- relaxed/exact matching；
- pseudoknot policy；
- seed protocol。

当前 ReactFlow 的 evaluator 已经支持 tiered F1/MCC、distance bins、reactivity correlation、calibration 和 cited/local 分栏，应保留为统一评估基础。

---

## 10.2 Static structure benchmarks

### 公共 benchmarks

- PDB-derived；
- ArchiveII；
- eFold viral；
- eFold lncRNA；
- human mRNA；
- Rfam interfamily；
- bpRNA-new；
- RNA3DB time-split；
- pseudoknot benchmark；
- CASP RNA targets 的 2D contacts。

### ReactFlow 主 benchmark

- MMseqs-disjoint test；
- MMseqs novel；
- Rfam family-disjoint；
- Rfam clan-disjoint；
- structure-disjoint；
- time-censored；
- long full-transcript domains。

---

## 10.3 Static metrics

必须报告：

- exact pair F1；
- relaxed pair F1；
- precision；
- recall；
- MCC；
- AUPRC；
- Brier score；
- pair ECE；
- sequence-level macro；
- pooled micro；
- family macro；
- median and worst-family；
- pair-count bias；
- legality rate；
- empty-structure rate；
- runtime；
- peak memory。

分层：

- length；
- RNA type；
- family/clan；
- pair distance；
- stem/loop；
- pseudoknot；
- canonical/noncanonical；
- confidence；
- sequence identity；
- structure similarity。

当前 short/medium/long distance-bin evaluator可以继续使用，但 long 应进一步拆成 24–63、64–255、≥256 nt。

---

## 10.4 Ensemble metrics

- held-out reactivity NLL；
- Pearson/Spearman；
- calibrated MAE；
- replicate coverage；
- BPP calibration；
- top-\(K\) native recall；
- ensemble diversity；
- population Jensen–Shannon divergence；
- cluster adjusted Rand index；
- energy–population consistency；
- condition sensitivity；
- mode collapse rate。

---

## 10.5 Intervention metrics

- \(\Delta\)BPP Pearson/Spearman；
- pair gain precision/recall/F1；
- pair loss precision/recall/F1；
- partner-switch accuracy；
- accessibility-change MAE；
- compensatory rescue AUROC/AUPRC；
- edit ranking NDCG；
- long-range impact localization；
- uncertainty–error correlation；
- cross-region synergy sign accuracy。

---

## 10.6 Downstream mRNA metrics

### 结构约束

- start-site accessibility；
- CDS target BPP KL；
- UTR target BPP KL；
- long dsRNA length；
- structural ensemble shift；
- edit count；
- normalized edit distance。

### 功能

- MRL；
- TE；
- half-life；
- protein expression；
- degradation；
- RBP motif preservation；
- innate immune proxy；
- manufacturability。

### 设计质量

- exact protein fraction；
- frame-valid fraction；
- hard-budget fraction；
- Pareto hypervolume；
- success per edit；
- diversity；
- OOD confidence。

---

## 10.7 统计协议

探索阶段：

- 3 seeds；
- family-cluster bootstrap；
- effect size。

论文阶段：

- 10 seeds；
- paired per-sequence comparison；
- family-cluster bootstrap 95% CI；
- permutation/sign-flip test；
- multiple-testing correction；
- pre-registered primary endpoints；
- failure cases 全量披露。

不能只汇报“generalization gap 小”。当前系统的 gap 很小，主要原因是 in-clan 和 novel-clan 都很低。真正的 retention 必须在 in-clan 已达到高准确率后才有意义。

---

# 11. 必须复现的 baseline

## Static

P0：

- RNAstructure；
- ViennaRNA；
- EternaFold；
- MXfold2；
- UFold；
- eFold；
- RNAformer；
- RiNALMo fine-tuned head；
- ERNIE-RNA fine-tuned head。

P1：

- PriFold；
- DEPfold；
- RNADiffFold；
- TVAE-RNA；
- RibonanzaNet2-derived structure head。

## Ensemble/probing

- RNAstructure + SHAPE/DMS restraints；
- ViennaRNA probing constraints；
- EternaFold constrained fold；
- RibonanzaNet reactivity predictor；
- MERGE-RNA；
- ReactFlow static-only；
- ReactFlow mean-only；
- ReactFlow full ensemble。

## Downstream mRNA

- LinearDesign；
- LinearDesign2；
- EnsembleDesign；
- codonGPT；
- UTailoR/UTR-LM 类方法；
- mRNABERT predictor；
- RiboNN；
- GEMORNA；
- mRNA-GPT；
- ProMORNA；
- 当前 mRNA-EditFlow。

预印本模型必须标注版本和同行评议状态，不能与正式发表模型混为一谈。

---

# 12. 分阶段执行计划

## Phase C1-0：可信度重置与 evaluator forensic audit

### 任务

- 官方 eFold checkpoint + 官方数据 + 官方 evaluator；
- 官方 checkpoint + ReactFlow evaluator；
- ReactFlow wrapper + 官方 evaluator；
- 修复 diagonal、index、pseudoknot 和 relaxed-F1 差异；
- 冻结 benchmark registry；
- 生成所有 benchmark checksum；
- 复核 ArchiveII 是否进入训练；
- 建立 canonical/all-pair 双协议。

### Gate

- 官方 eFold 结果与论文/官方 notebook 的差异不超过预注册容差；
- 两个 evaluator 在同一 prediction 上完全或近似一致；
- 未通过前禁止训练新 SOTA 主干。

---

## Phase C1-1：强静态 baseline

### 任务

- 实现 symmetric pair head；
- 实现 12-block compact PairFormer；
- 接入 RiNALMo/RibonanzaNet2 pair-aware feature；
- GPU nested decoder；
- 完成 eFold、RNAformer、RiNALMo same-split baseline；
- 在 L≤256 上进行 matched-capacity ablation。

### Gate

- ArchiveII/PDB 显著超过当前 ReactFlow；
- novel-clan F1 至少进入可用区间；
- pair-count 和 empty-structure bias 消失；
- long-range recall 不接近零。

这一阶段不保留当前 F1≥0.15 作为论文目标；0.15 只能视为“模型开始学习配对”的工程检查点。

---

## Phase C1-2：Full-scale static SOTA

### 任务

- 24–48 block PairFormer；
- multi-encoder gated fusion；
- data-diversity curriculum；
- long RNA hierarchy；
- pair calibration；
- nested/pseudoknot 双 decoder；
- multi-GPU full fine-tuning。

### 项目内部性能门槛

以下是 ReactFlow 自己的工程门槛，不应被描述为跨论文绝对可比数字：

- PDB-derived exact F1 ≥ 0.85；
- ArchiveII exact F1 ≥ 0.80；
- viral F1 ≥ 0.68，目标 ≥ 0.73；
- lncRNA F1 ≥ 0.40；
- novel-clan 达到或超过同 split 最强 baseline；
- 至少 3 个 OOD tier 显著优于最强 same-split baseline；
- pair ECE 和 Brier 显著优于未校准模型；
- legality rate = 100%。

### 失败动作

若 foundation encoder 很强但 pair trunk 无增益：

- 检查 label extraction；
- 检查 loss/negative space；
- 检查 decoder；
- 做 teacher-forcing oracle BPP 实验；
- 不继续无条件堆参数。

---

## Phase C1-3：长 RNA 与 pseudoknot

### 任务

- domain proposer；
- global anchor pair；
- divide-and-stitch；
- overlap consistency；
- pseudoknot matching decoder；
- sequence lengths 512、1k、2k、4k 分层。

### Gate

- 跨窗口真实 pair recall 明显提升；
- lncRNA/viral/human mRNA 不因长度增长坍缩；
- long-range recall 显著超过局部窗口 baseline；
- pseudoknot improvement 不损害 nested benchmark。

---

## Phase C2：Probe-calibrated structural ensemble

### 任务

- probe likelihood；
- replicate/noise model；
- edge-set flow；
- BPP-informed initialization；
- top-\(K\) structures；
- population prediction；
- single-molecule benchmark。

### Gate

同时满足：

- static F1 不显著下降；
- held-out reactivity NLL 改善；
- ensemble population calibration 改善；
- 生成多样性不是随机错误；
- single-molecule ensemble cluster recovery 优于 static/thermodynamic baseline。

若 probing loss 只提高 reactivity 拟合而降低结构准确率，则 probing 模式改为 conditional refinement，不强行进入 de novo 主干。

---

## Phase C3：Intervention-aware ReactFlow

### 任务

- 构建 RSIB；
- WT–mutant paired encoder；
- delta-pair head；
- compensatory loss；
- uncertainty；
- cross-region synergy；
- synthetic-to-real curriculum。

### Gate

- \(\Delta\)BPP 优于 independent-fold subtraction；
- pair gain/loss F1 显著提升；
- compensatory rescue ranking 优于 static baseline；
- long-range edit impact 可被定位；
- uncertainty 能识别失败编辑。

如果直接 delta head 不优于两次独立预测，则不能保留“intervention-aware”作为主创新，应回到数据和 paired supervision，而不是只增大模型。

---

## Phase C4：与 mRNA-EditFlow 整合

### 任务

- ReactFlow oracle API；
- differentiable BPP/Delta-BPP reward；
- structure-preserving RL；
- structure-remodeling RL；
- UTR-only、CDS-only、3′UTR-only、joint-region 对照；
- independent-region 与 joint-region 对照；
- minimal-edit Pareto optimization。

### 关键实验

1. 同义 CDS edit 是否保持目标结构；
2. 5′UTR edit 是否改善起始区 accessibility；
3. UTR 与 CDS 联合编辑是否产生非加性收益；
4. 蛋白完全不变时，结构 reward 是否改善 TE/stability；
5. 结构不确定性惩罚是否减少高风险候选；
6. ReactFlow reward 是否优于 MFE、RNAfold 和单点结构 predictor。

### Gate

- hard constraints 100%；
- joint-region 显著优于 independent-region；
- 单位 edit 的功能收益提升；
- 结构变化与功能变化具有可解释关系；
- 在独立功能 evaluator 和 wet-lab 中保持方向一致。

---

## Phase C5：前瞻性实验验证

### 最小实验设计

选择 3–5 条 mRNA，覆盖：

- 短 reporter；
- 中长 therapeutic CDS；
- 不同 5′UTR；
- 不同 3′UTR；
- 强与弱结构背景。

每条设计：

- wild type；
- local structure-disrupting edit；
- compensatory rescue edit；
- 5′UTR-only edit；
- CDS-only synonymous edit；
- 3′UTR-only edit；
- cross-region joint edit；
- model-low-confidence negative control。

每组至少有生物学重复。

### 结构测量

- DMS-MaPseq；
- SHAPE-MaP；
- 必要时单分子 long-read probing；
- edit 前后 BPP/ensemble comparison。

### 功能测量

- luciferase 或荧光 reporter；
- mRNA abundance；
- half-life；
- polysome/MRL；
- protein expression；
- 必要时 innate immune markers。

### 顶刊级关键证据

最强结果应是：

> 模型预测两个相距较远的区域存在补偿性结构协同；两个单独编辑均无收益或有害，联合编辑恢复目标 ensemble，并显著提高翻译或稳定性。该效应被 probing 与功能实验共同验证。

---

## Phase C6：论文、发布与外部盲测

### 论文主图建议

1. 领域 generalization 与 intervention gap；
2. ReactFlow 2.0 架构；
3. same-split static SOTA；
4. probing-calibrated ensemble；
5. mutation/compensatory benchmark；
6. cross-region mRNA design；
7. wet-lab causal validation；
8. uncertainty 和 failure map。

### 发布内容

- frozen benchmark registry；
- RSIB intervention dataset；
- model weights；
- static/ensemble/intervention API；
- mRNA-EditFlow integration；
- evaluator；
- dataset manifest；
- pretraining contamination report；
- all-seed results；
- negative results；
- reproducibility containers。

---

# 13. 代码迁移方案

## 继续保留

- `reactflow.data`；
- `reactflow.rfam_metadata`；
- MMseqs/Rfam split；
- provenance/checksum；
- `reactflow.constraints`；
- evaluator；
- distance-bin metrics；
- reactivity mask/QC；
- experiment watcher；
- final-result contract；
- reproducibility manifest。

## 降级为 legacy baseline

- 当前 `PairwiseDenoiser`；
- partner-class static prediction；
- uniform CTMC；
- single-token linear frozen adapter；
- static structure中的 categorical diffusion 主路线。

## 新增模块

```text
reactflow/
  backbones/
    rinalmo.py
    ernie_rna.py
    ribonanzanet2.py
    reactflow_fm.py
  pair/
    initializer.py
    pairformer.py
    long_context.py
    calibration.py
  decoders/
    nested_dp.py
    inside_outside.py
    pseudoknot_matching.py
  ensemble/
    observation_model.py
    edge_flow.py
    population.py
  intervention/
    paired_encoder.py
    delta_pair.py
    compensatory.py
    synergy.py
  downstream/
    mrna_oracle.py
    editflow_adapter.py
```

旧模型和新模型必须共享同一个 evaluator，避免通过重写测评代码人为获得提升。

---

# 14. 当前最大不确定性

## 14.1 二级结构“真值”本身不稳定

PDB-derived 结构取决于 extractor；probing-derived 结构又依赖折叠算法。解决方式：

- 保存原始证据；
- 多 extractor；
- soft/uncertain labels；
- consensus 与 disagreement 分层；
- 不把 probing-constrained prediction伪装成无偏实验真值。

---

## 14.2 预训练污染难以完全排除

RiNALMo、RNA-FM 和其他模型可能见过测试序列。解决方式：

- external-pretrained 与 self-pretrained 分表；
- time-censored model；
- contamination strata；
- from-scratch control；
- 不以单一 foundation backbone 结果作为唯一主结论。

---

## 14.3 RibonanzaNet2 版本和证据稳定性

当前 checkpoint 是 alpha 版本。解决方式：

- pin revision/hash；
- 保存本地代码；
- 不引用未正式发表 leaderboard；
- 与 RiNALMo/ERNIE-RNA/自训练 backbone 比较；
- 主创新不能依赖某个外部 alpha 模型。

---

## 14.4 化学探针无法唯一确定 ensemble

一阶或二阶 profile 仍可能由多个 ensemble 解释。解决方式：

- maximum-entropy prior；
- thermodynamic prior；
- replicate likelihood；
- orthogonal probe；
- single-molecule数据；
- calibrated uncertainty；
- 明确输出“可能 ensemble”，而不是假装唯一真值。

---

## 14.5 高静态 F1 不保证下游功能收益

结构模型可能在 benchmark 上优秀，但预测不到 translation 或 stability。解决方式：

- intervention benchmark；
- independent functional evaluator；
- causal compensatory controls；
- prospective wet-lab；
- 把功能收益作为独立主 endpoint。

---

## 14.6 全长 mRNA 结构具有条件依赖

体外、细胞内、不同修饰和不同温度下的结构并不相同。解决方式：

- condition token；
- modification track；
- in vitro/in vivo 分轨；
- 不宣称单一“真实全长结构”；
- 预测 \(p(S\mid x,c)\)，而不是只预测 \(S(x)\)。

---

# 15. 最终验收标准

ReactFlow 不能因为某个单一公开 benchmark 达到 F1 0.70 就宣告完成。

最终完成必须同时通过五个 gate：

## Gate 1：可信测评

- eFold 官方复现通过；
- 数据和 evaluator 冻结；
- 无训练测试泄漏；
- cited/local 完全分开。

## Gate 2：Static SOTA

- 同 split 超过最强 baseline；
- 10-seed 显著；
- long RNA 和 novel-family 均成立；
- calibration 和合法性达标。

## Gate 3：Ensemble 增值

- probing NLL 与 population calibration 改善；
- 不以牺牲 static accuracy 为代价；
- 多构象具有实验支持。

## Gate 4：Intervention 增值

- \(\Delta\)structure 优于 independent folds；
- compensatory rescue 和远端编辑效果可预测；
- cross-region synergy 可被识别。

## Gate 5：功能和实验闭环

- mRNA-EditFlow 的 joint-region design 有收益；
- protein/frame/edit-budget 约束完全满足；
- probing 验证结构变化；
- reporter/half-life/translation 验证功能；
- 至少一个前瞻性设计形成可重复的因果证据。

---

# 16. 最终论文定位

不建议把论文标题和摘要定位为：

> “A new Evoformer for RNA secondary structure prediction.”

这不足以区分 eFold、RNAformer、RiNALMo、PriFold 和其他现有方法。

推荐定位为：

> **ReactFlow: a probe-calibrated and intervention-aware RNA structural ensemble foundation model for constrained full-length mRNA design**

三个主贡献：

1. **静态基础能力**  
   在严格 family、clan、structure 和 time-disjoint 条件下达到同协议 SOTA。

2. **结构 ensemble 能力**  
   将 chemical probing 作为结构分布的概率观测，输出校准的 alternative conformations 和 population。

3. **编辑干预与功能能力**  
   直接预测最小编辑造成的远程结构重排，并指导 5′UTR–CDS–3′UTR 联合 mRNA 设计，通过 probing 与功能实验验证。

这一定位把领域痛点完整连成一条线：

```text
结构标签稀缺与家族泄漏
        ↓
静态模型在长 RNA / OOD 上不可靠
        ↓
单一 dot-bracket 无法描述真实 ensemble
        ↓
现有模型无法预测小编辑造成的结构变化
        ↓
mRNA 设计依赖不可靠或过于简单的结构 oracle
        ↓
ReactFlow：
强静态 pair trunk
+ probe-calibrated ensemble
+ intervention-aware delta structure
+ cross-region constrained mRNA design
        ↓
从“预测结构”推进到“预测并控制结构—功能关系”
```

这才是 ReactFlow 最有机会形成顶刊主线的新增价值。

这份版本可以直接作为新的 `ReactFlow_SOTA_Goal_Final_2026-07-21.md` 主规划文档；原有 C0/C1 的 partner-class 计划应保留为失败基线和方法演化记录，而不再作为主执行路线。