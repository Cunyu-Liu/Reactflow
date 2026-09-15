# Comparator Registry Re-check Amendment (F8 Obligation 5)

> 生成：2026-09-15 ｜ 执行：Phase 2（Task 9 前夕）
> 性质：**流程合规补做**。Phase 1 冻结表 F8 第 5 条要求"direct 层 comparator 检索在预注册冻结（2026-09-13, commit 7bc771e）前重检一次"；实际执行时点为 2026-09-15（冻结后 2 天）。如实记录该时序偏差；本 amendment 不改变任何冻结数值/门槛/协议，仅更新 comparator registry 的登记内容。
> 检索方式：WebSearch（2 轮，共 8+6 条结果）+ 对最接近候选的全文核验。

## 结论：direct 层空白维持成立（含一处登记更新）

**direct 层任务口径**（本项目的比较目标）：给定 WT 序列 + WT 2A3-MaP profile（含误差/missing）+ 单个 exact SNV，预测全 construct 位置上的 signed reactivity-delta **联合分布**。

### 重检结果

1. **MERGE-RNA**（Sacco et al., arXiv:2512.20581, 2026-08）——物理基（最大熵）ensemble 模型，DMS 数据→结构系综。方向为 probing→structure（反问题侧），非 profile→profile 正问题。**已登记于 Phase 1 registry（adjacent 层），维持不变。**

2. **MIMIC**（Golkar et al., arXiv:2604.24506, 2026-04，Polymathic/Flatiron）——**新发现，登记更新**：多模态基础模型，§7 "Context-conditioned RNA reactivity prediction" 用实验语境（assay context）做语义条件化预测 RNA 化学探测 reactivity。核验判定：**adjacent 层（强）而非 direct 层**——其条件是实验语境而非 SNV，输出是单 reactivity profile 而非 mutation-response delta 分布；无 mutate-and-map 全 SNV 库交叉验证，无 matched-null 归因设计。但它是"reactivity 条件建模"侧最接近的已发表工作，论文相关工作章与对比讨论**必须引用并区分**（可预期审稿人会提出）。

3. 其余检索结果（ShapeRNA web server、eFold、VIRSE、RNA 二级结构 ML review、CodonFM、Townley pseudoknot design）均已在 Phase 1 registry 或与 direct 口径无重叠。

### 对论文的影响（登记，不改变计划）

- Task 11.4 文献定位章新增义务：与 MIMIC 的区分段（我们的任务 = profile→profile delta 联合分布 + 归因 null 设计；MIMIC = context→reactivity 语义条件 + 多模态补全范式）。
- direct 层"无 task-matched 可比物"的论文主张维持成立，但表述需从"完全空白"细化为"最接近工作为 MIMIC 的 context-conditioned reactivity 建模，无 mutation-response 联合分布任务先例"。

## 零编造声明

本 amendment 全部检索记录来自本轮 WebSearch/WebFetch 实际执行（2 轮搜索 + MIMIC 全文核验），无虚构条目。
