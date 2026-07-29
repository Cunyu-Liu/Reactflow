> [!IMPORTANT]
> **已由 ReactFlow-Δ V2 合同取代（2026-07-29；仅保留为历史工程资产与负结果证据）。**
> 本文件不再是可执行阶段计划，也不得用于启动训练、追加 seed、降低 Gate 或支撑新的科学主张。当前唯一有效合同为
> `docs/contracts/ReactFlow分阶段执行提示词_ReactFlowDelta科研合同_v2_20260729.md`
>（SHA-256：`5d2dc9e2ac0e6b8c6355791f4ff95958b2e9ab5722d2d2eba49c6578a3e87c13`）。

# C5 规划:Ribonanza2 Warm-start + eFold Head-to-Head

> 状态:**规划已定稿,待执行(C5.1 → C5.5)** · 日期 2026-07-07
> 上游依赖:C1–C4 实现内核已验收(合成 pilot 端到端跑通;采样器 100% 合法;guidance `η` 扫描单调性已证并 SymPy 验证;131 passed / 1 skipped,coverage 95.81%)。
> **诚实约定(沿用 ledger 两级标注):** ✅ 本轮由主源核实 · ◻️ 待核实(成文前逐条查证卷期页/ID/schema) · ⚠️ 风险点。

---

## 0. 两项已锁定的架构决策(本轮 AskUserQuestion)

1. **Warm-start 引擎 = 外部冻结编码器 + stdlib 头。**
   离线跑一次 RibonanzaNet2 / eFold(PyTorch/GPU)导出 per-nucleotide 与 pairwise 表征,带 provenance + SHA256 缓存成特征文件;`reactflow` 包本体**仍是纯标准库 + 手写反传**,只训练 DFM 分布头 + reactivity 一致性。
   - **保住**:reactflow 可审计性(不引入 PyTorch 到 import 图)、确定性 bit-for-bit。
   - **代价**:编码器不端到端可训(冻结);依赖一次性外部推理。
2. **评测切分 = 复用 eFold 官方测试集 + 叠加自建 Rfam-clan novel split。**
   - eFold 官方测试集口径可直接**引用其已发表 F1**(可比性最强);
   - 自建 Rfam-clan novel-clan holdout 承载 ReactFlow 的 **OOD 差异化卖点**(泛化鸿沟)。

> 这两条同时化解了「规模化 warm-start / 打败 eFold」与「纯 stdlib + 手写反传」两条硬约束的张力:**重活(100M encoder 前向)离线一次性做成"数据",训练侧保持轻量可审计。**

---

## 1. 已核实的外部事实(C5 的事实底座)

### 1.1 eFold / RNAndria — head-to-head 对象(✅ 本轮核实)
- **主源**:de Lajarte, Martin des Taillades, Aruda, …, Rouskin, *"Diverse database and machine learning model to narrow the generalization gap in RNA structure prediction"*, **Science Advances 2026, 12(9):eadz4967**, DOI `10.1126/sciadv.adz4967`(Epub 2026-02-25),PMID `41739924`,PMCID `PMC12935039`。
- **架构**:受 AlphaFold Evoformer 启发的双通路块(self-attention 处理序列表征 + ResNet/CNN 处理成对表征),两阶段训练(大规模预训练 → 特定数据集微调)。
- **官方报告 F1(可引用,标注来源)**:

  | 测试集 | eFold F1 | UFold | SPOT-RNA | 备注 |
  |---|---|---|---|---|
  | viral mRNA(长/多样) | **0.73** | 0.58 | 0.56 | 最具挑战 |
  | lncRNA | **0.44** | 0.16 | 0.26 | 最难,长非编码 |
  | 短 ncRNA(PDB) | ~0.9(经典算法量级) | — | — | 家族饱和,非差异化战场 |

  > 核心论点(与 ReactFlow 一致):**单纯扩大数据库规模不足以跨家族泛化**,需增加结构多样性/复杂度。这正是我们叠加 Rfam-clan novel split 的理由。
- **数据完全开放、可直接消费(✅)**:
  - **Dryad DOI `10.5061/dryad.79cnp5j95`**(2026-01-29 版,368.20 MB),明确 *"All data are freely available without restrictions"*。
  - 文件:`efold_train.json`(349.51 MB)、`archiveII.json`、`human_mRNA.json`、`lncRNA_nonFiltered.json`、`PDB.json`、`pri_miRNA.json`、`viral_fragments.json`。
  - **JSON schema**(逐条核实):`{sequence: "ACGU…", structure: [[i,j],…] (碱基对列表), shape: [归一化 SHAPE 浮点数列表]}`。
  - RNAndria 在线库 <https://rnandria.org/>;代码 <https://github.com/rouskinlab/efold>。
- **RNAndria 规模**:DMS-MaPseq 探测 4550 个人 mRNA 3′ 区 + 1292 个 pri-miRNA,质控(读深 >3000,AUROC >0.8)后得 **1456 mRNA + 1098 pri-miRNA** 高质量结构。

### 1.2 RibonanzaNet2 — warm-start 源(⚠️ 已解除为 ✅ 可得)
- **Kaggle model `shujun717/ribonanzanet2`**(PyTorch,**alpha / V1**),**License MIT**,**Fine-Tunable = Yes**,权重公开(创建 2025-03-31)。
- **规格(model card ✅)**:~**100M** 参数;**48** 个 transformer block,每块含 2 个对称 triangle multiplicative update;**序列表征 384** 维、**pairwise 表征 128** 维;从头训练。
- **训练数据**:30M RNA 100mer DMS/SHAPE profiles,来自 200B raw reads;4% holdout 作验证。
- **已有 2D 微调**:官方 notebook `rnet2-alpha-2d-structure-inference`(含伪结),以及 `ribonanzanet2-ddpm-training/inference`。→ Stage-A 可直接借其 2D 结构头 / pairwise 表征。
- **RibonanzaNet v1 权重**:Kaggle `shujun717/ribonanzanet-weights`(MIT),含 `RibonanzaNet.pt`(化学 mapping 预训练)/`-SS.pt`(PDB 二级结构微调,v0.9)等 —— 作为 warm-start 退路。
- ⚠️ 复现风险:MultiMolecule 页面提示复现原始指标存在潜在风险;alpha 版本可能更新。**我们只取其表征做冻结特征,不复述其指标。**

### 1.3 Ribonanza2 主干数据(◻️ 规模/许可成文前再核)
- Kaggle `rhijudas/ribonanza2-training-data`,data card 报 64M 序列、174GB H5,schema `reads/SNR/reactivity/error/norm/heatmap`,标 CC BY 4.0。
- **本项目已有** [data.py](file:///Users/bytedance/Documents/research/reactflow/src/reactflow/data.py) 的 CSV/H5 schema 检查、SN_filter、p90 归一化、inverse-error 权重、MAD 异常检测 —— C5.1 直接复用。

---

## 2. C5 数据流总览

```
                    [ 离线一次性 · PyTorch/GPU · 产出=数据 ]
RibonanzaNet2 (MIT, 100M) ─┐
eFold (rouskinlab/efold)  ─┼─► Stage-A 冻结特征导出器 (C5.2)
序列 (eFold Dryad + Ribonanza) ┘        │  per-nt h_i∈R^384, pairwise p_ij∈R^128
                                        │  + provenance + SHA256 缓存 (.npz/.jsonl)
                                        ▼
              [ reactflow 纯 stdlib + 手写反传 · 确定性 bit-for-bit ]
      C5.1 数据管线 SOP ──► clan-disjoint split + novel-clan holdout + 长度分桶
                                        │
      C5.3 reactflow.features ──► 读冻结特征 → 线性 adapter(手写反传)
                                        │  拼接到 PairwiseDenoiser 输入 (FEATURE_SIZE=8 → +冻结通道)
                                        ▼
              DFM 分布头 p_θ(S|x) + L_DFM + λ_r·L_react + L_thermo + guidance
                                        │
      C5.4 reactflow.evaluate ──► F1/MCC + Pearson/Spearman + ECE + 泛化鸿沟
                                        │
             head-to-head:eFold 官方测试集(引用 DOI 数字) + Rfam-clan novel(本地重算)
```

---

## 3. 分周期任务(对应 Task #15–#19)

### C5.1 数据管线 SOP(Task #15)✅ **代码已落地(合成 fixture);真实下载/去冗余暂缓**
**目标**:确定性、可复现、无家族泄漏的数据准备。
- **✅ 已实现(纯 stdlib,小 fixture 单测)**:
  - eFold/RNAndria JSON 读取器 [data.py](file:///Users/bytedance/Documents/research/reactflow/src/reactflow/data.py):`parse_efold_record` / `read_efold_json`(list 与 mapping 两种布局)/ `EfoldRecord`;严格解析 `structure:[[i,j],…]`(越界/自配对/长度不符即 raise),支持 `structure|pairs`、`shape|dms|reactivity` 别名与 one-based 偏移;`efold_pair_matrix` 桥接到 `pairs_to_matrix`。
  - `pairs_to_matrix` 新增于 [constraints.py](file:///Users/bytedance/Documents/research/reactflow/src/reactflow/constraints.py)(`matrix_to_pairs` 的逆),对角/越界即 raise,校验仍归 `validate_pair_matrix`。
  - 新模块 [splits.py](file:///Users/bytedance/Documents/research/reactflow/src/reactflow/splits.py):`build_split_manifest`(clan 为切分单位 → seeded Fisher-Yates → novel-clan holdout → 贪心 LPT 分 train/val/test)、`validate_split_leakage`(**clan 两两不相交** + **cluster 不跨 split** 双重断言)、长度分桶 `length_bucket_label`、`manifest_to_json`/`manifest_from_json`(load 时重校验)。
  - **验证**:纯 stdlib(import 时不加载 numpy/torch/h5py);同 seed 跨进程 bit-for-bit 一致;输入顺序无关;36 个新单测,全量 **167 passed / 1 skipped**,coverage **96.29%**(splits.py 99%)。
- **◻️ 暂缓(用户选择"先代码+小 fixture,暂缓大下载")**:真实 eFold 368MB `curl` Dryad、Ribonanza CSV/H5 `kaggle` 下载、MMseqs2/CD-HIT 离线去冗余(产出 cluster manifest 供 `splits` 只读)、Rfam clan 映射(记录 Rfam release 号)。下载命令备好(见下),留待管线验证通过后触发。
- **下载(显式命令,不隐式拉取)**:
  ```bash
  # eFold 结构数据(freely available)
  curl -L -o data/raw/efold/efold_bundle.zip \
    "https://datadryad.org/downloads/dataset/doi:10.5061/dryad.79cnp5j95"
  # Ribonanza CSV/H5(需 ~/.kaggle/kaggle.json)
  kaggle datasets download -d rhijudas/ribonanza2-training-data -p data/raw/ribonanza2
  kaggle competitions download -c stanford-ribonanza-rna-folding -p data/raw/ribonanza
  ```
- **完整性/有效性/标准化/特征工程**:全部复用 [data.py](file:///Users/bytedance/Documents/research/reactflow/src/reactflow/data.py) 已有契约(缺失 NaN 掩码不插补、reads>100 & SNR>1.0 门、MAD 异常、p90 归一化、GC/probe-mask/inverse-error 特征)。
- **可验证产物**:`split_manifest.json`(每条序列 → clan / cluster / bucket / split),附**自动泄漏校验**(断言任意 test clan ∉ train clans)。✅ 已实现并单测。
- **数学/复杂度**:去冗余 O(N²) 比对由 MMseqs2 近似到近线性;split 构建 O(N + C log C)、校验 O(N)。

### C5.2 Stage-A 外部冻结特征导出器(Task #16)
**目标**:把 100M encoder 的重活离线做成"数据"。
- **脚本位置**:`scripts/export_frozen_features.py`(**不进 `reactflow` 包 import 图**,避免 PyTorch 成为库依赖)。
- **加载**:RibonanzaNet2(Kaggle MIT alpha/V1)与 eFold(`rouskinlab/efold`)权重;对每条序列前向,导出:
  - per-nucleotide `h_i ∈ R^384`(序列表征);
  - pairwise `p_ij ∈ R^128`(成对表征)或 2D 结构 logits;
  - reactivity logits(DMS/2A3)——可作 `L_react` 的额外 warm 信号。
- **provenance**:每个特征文件头写 `{model_name, model_version, weights_sha256, commit, date, schema}`;整体 SHA256 缓存,避免重复推理。
- **格式**:`.npz`(数值)+ `.jsonl`(元数据),供 stdlib 侧 `numpy`-free 读取(用 stdlib `array` + 显式 dtype,或轻量二进制读取器)。
- ⚠️ 隔离:`reactflow` 单测**不**依赖 174GB 或 GPU;提供 **小 fixture**(几条短序列的伪冻结特征)驱动加载器单测。
- **数学**:纯前向,O(层数 · L² · d) 由外部框架承担;导出后训练侧只做 O(L·d) 读取。

### C5.3 冻结特征 → stdlib adapter → DFM 分布头(Task #17)
**目标**:在纯 stdlib 手写反传下消费冻结特征。
- **新模块 `reactflow.features`**:
  - 读 Stage-A 缓存 → 对齐到序列位点;
  - **线性 adapter** `a_i = W·h_i + b`(手写前向 + 手写反传),把 384 维冻结表征投影到 DFM 头输入维度;
  - 与现有手写特征(`FEATURE_SIZE=8`,见 [train.py](file:///Users/bytedance/Documents/research/reactflow/src/reactflow/train.py))**拼接**:输入维 `8 + d_adapter`;
  - **缺特征回退**:无 Stage-A 缓存时回退到 C3 纯手写特征(保证仓库在无外部权重时仍可跑 pilot)。
- **训练**:`PairwiseDenoiser`([model.py](file:///Users/bytedance/Documents/research/reactflow/src/reactflow/model.py))输入通道扩展;`L_DFM + λ_r·L_react + L_thermo`(C4 已实现)联合;adapter 参数随 DFM 头一起 SGD。
- **数学验证(硬约束)**:
  - adapter 反传梯度经 **SymPy 符号推导** + **有限差分** 双验证(沿用 C3 `1.8e-9` 量级标准);
  - 拼接层链式法则显式写出。
- **确定性**:固定 seed,bit-for-bit(冻结特征是常量输入,不破坏可复现性)。

### C5.4 评测协议 + eFold head-to-head(Task #18)
**目标**:诚实、可比、差异化的对比。
- **新模块 `reactflow.evaluate`**:
  - **结构准确度**:F1 / MCC(复用 [metrics.py](file:///Users/bytedance/Documents/research/reactflow/src/reactflow/metrics.py) 的 `f1_score` / `matthews_corrcoef`),分 in-clan / cross-clan / novel-clan 三档;
  - **reactivity 质量**:Pearson / Spearman(形状)+ 校准后 MAE(幅度,复用 `mean_absolute_error`);
  - **不确定度校准**:ECE(reliability diagram);
  - **泛化鸿沟** = F1(in-clan) − F1(novel-clan),对齐 eFold/Szikszai 口径。
- **两栏对比(诚实分栏,严禁混淆)**:
  1. **eFold 官方测试集**(viral mRNA / lncRNA / PDB / ArchiveII):**引用其 DOI 报告 F1**(0.73 / 0.44 / …),标注"引用自 `10.1126/sciadv.adz4967`";ReactFlow 在**同测试集同口径**跑出的数字单列。
  2. **自建 Rfam-clan novel split**:eFold(用 `rouskinlab/efold` 权重本地推理)与 ReactFlow **本地重算**,标注"本地重算"。
- ⚠️ **禁造假红线**:引用数字与本地重算数字**永不混列**;ReactFlow 在真实数据上训练完成前,SOTA 表对应格保持"pending / 待 C5 训练"。

### C5.5 测试 ≥90% + SymPy + 文档(Task #19)
- **新测试**:`test_features`(adapter 反传 FD、拼接维度、缺特征回退)、`test_evaluate`(三档指标闭式、ECE、泛化鸿沟符号)、`test_data_split`(clan-disjoint 无泄漏断言、长度分桶、eFold JSON 解析)。保持 **coverage ≥90%**。
- **SymPy 清单**:加 adapter 梯度检查(现 10 项 → 11 项),残差须为 `0`。
- **README 更新**:
  - SOTA 表填**真实数字**(两栏:引用 vs 本地重算,禁造假);
  - warm-start 原理图(Stage-A 冻结 → adapter → DFM 头);
  - C5 可复现命令(含 PyTorch **可选依赖**说明 + 外部权重 provenance);
  - Quality Gates / Next Cycle 更新。
- **research_plan.md 更新**:§8 C5 行标状态、§4 baseline 表补 eFold 实测口径、**§9 文献账本 eFold(`10.1126/sciadv.adz4967`)与 RibonanzaNet2(Kaggle MIT)◻️→✅**。

---

## 4. 验收标准(C5 "赢点成立"的可检验定义)

| 维度 | 通过判据 |
|---|---|
| 数据无泄漏 | `split_manifest` 自动校验:任意 test/novel clan ∉ train clans(单测断言) |
| warm-start 真实 | Stage-A 特征带可核 provenance(权重 SHA256 + model card 引用);加载器单测通过 |
| head-to-head 可比 | 在 **eFold 官方测试集同口径**报告 ReactFlow F1,与引用的 0.73/0.44 并列(分栏标注来源) |
| 差异化卖点 | 在 **Rfam-clan novel split** 报告泛化鸿沟;若 ReactFlow 鸿沟 < eFold 鸿沟 → 卖点成立 |
| 诚实 | 引用数字 vs 本地重算严格分栏;未训完的格标 pending,不编造 |
| 质量门 | coverage ≥90%;SymPy 全残差 0;确定性 bit-for-bit |

---

## 5. 风险与诚实局限(写进论文并防御)

- ⚠️ **冻结编码器 ≠ 端到端微调**:表征冻结可能限制上限;需消融"冻结 vs (若 C5 引入 PyTorch 可选路径)微调"。本 pilot 诚实标注为"冻结 warm-start"。
- ⚠️ **RibonanzaNet2 alpha 版本漂移**:权重可能更新;固定 SHA256 并记录版本,不复述其内部指标。
- ⚠️ **eFold 切分对齐**:其"long/diverse"测试集切分细节须精读 `github.com/rouskinlab/efold` 与 Dryad README 对齐,避免口径偏差(◻️ 成文前核)。
- ⚠️ **可辨识性**(承 §2.2.6):reactivity 只定边缘;靠 `L_DFM`/热力学缓解,诚实承认上限。
- ⚠️ **算力**:Stage-A 大规模前向需 GPU;首轮可先在 eFold Dryad 中等规模数据(非 64M 全量)跑通,标注为"缩减规模 warm-start pilot",全量留后续。

---

## 6. 文献账本(C5 相关,✅ 本轮核实)

- ✅ **eFold / RNAndria** — de Lajarte et al., *Sci Adv* 2026, 12(9):eadz4967, DOI `10.1126/sciadv.adz4967`,PMID `41739924`,PMCID `PMC12935039`;数据 Dryad DOI `10.5061/dryad.79cnp5j95`;代码 `github.com/rouskinlab/efold`;库 `rnandria.org`。官方 F1:viral mRNA 0.73、lncRNA 0.44。
- ✅ **RibonanzaNet2** — Kaggle `shujun717/ribonanzanet2`(PyTorch alpha/V1,MIT,fine-tunable);~100M 参数、48 blocks、seq rep 384 / pair rep 128;30M 100mer DMS/SHAPE、200B reads;Das lab(Stanford/HHMI)。基于 RibonanzaNet(bioRxiv `10.1101/2024.02.24.581671`)。
- ✅ **RibonanzaNet v1 权重** — Kaggle `shujun717/ribonanzanet-weights`(MIT):`RibonanzaNet.pt` / `-SS.pt`(PDB 2D 微调)等,作 warm-start 退路。
- ◻️ **Ribonanza2 training data** — Kaggle `rhijudas/ribonanza2-training-data`,64M/174GB/CC BY 4.0(规模/许可成文前再核)。
