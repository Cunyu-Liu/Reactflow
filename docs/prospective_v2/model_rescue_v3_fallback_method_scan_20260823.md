# ReactFlow-Delta Model Rescue v3：非绑定后备方法扫描

**记录日期：** 2026-08-23
**证据资格：** `NOT_AUTHORIZED_NOT_GATE_EVIDENCE`
**适用条件：** 仅当完整 R3M3 20-fold artifact 合并、预冻结 qualifier 完成且当前 v3 未能进入或通过 R3M4 时，才允许据此起草新的窄 amendment。
**当前实验边界：** 本说明没有读取 R3M3 的任何部分 CRPS、signed-delta MAE、loss、per-puzzle effect 或 Gate 方向；不修改当前候选、seed、epochs、threshold、residual family 或 Gate。

## 1. 已确认的架构缺口

`CONFIRMED_FACT`：当前 B1/MeanAligned 并不对 exact mutant sequence 重新编码。

- `scripts/reactflow_delta/lrso_v1.py::WTContextEncoder.forward` 从 WT sequence、WT reactivity/error/mask、position 和 region 得到单一 WT context `H`。
- `scripts/reactflow_delta/model_rescue_v2.py::MeanAlignedModel.forward_mean_and_features` 只把 WT `H`、edit-site hidden、signed distance 和 ref/alt one-hot 拼接到 direct head。
- `scripts/reactflow_delta/run_model_rescue_v3.py::predict_expert_means` 对 B1 和 MeanAligned 都调用 `encode(ctx_cache[construct_id])`；每个 mutant 没有独立的 sequence encoder pass。

因此，现有模型能学习“WT 表征 + 突变 token → response”，但不能显式表达“同一共享 encoder 下，突变如何改变每个 receiver position 的上下文表示”。这是一项表征能力缺口，不是不确定度校准问题。

`CONFIRMED_FACT`：既有完整 development OOF failure atlas 显示 signed-delta 改善高度依赖 effect magnitude：`autoresearch/loop-260822-1700/cycle0-oof-diagnosis.json` 中，大效应 `|delta| > 0.20` 子集相对恶化约 2.51%，moderate 子集改善约 2.32%，near-zero 子集改善很大。该证据只用于提出“尾部学习不足”假设，不是新候选的有效性证据。

## 2. 三条窄后备路线

| 优先路线 | 明确瓶颈 | 最小架构/目标修改 | 直接依据 | 主要风险 | 最小证伪实验 |
|---|---|---|---|---|---|
| A. Shared-weight paired WT–mutant encoder | WT-only `H` 无法表达突变后 receiver context 的改变 | 保持 B1 encoder 宽度、层数和参数共享；构造 exact mutant sequence，以同一个 encoder 分别得到 `H_wt`、`H_mut`；mean head 只增加 `H_mut-H_wt` 与原 B1 features；仍使用 method-balanced signed-delta L1，校准仍在 mean freeze 后进行 | RibonanzaNet 的公开服务提供 sequence-conditioned mutate-and-map 计算；MutaRNA/remuRNA 以 WT 与 mutant ensemble 差异定义突变效应 | 两次 encoder pass 增加约 2 倍 encoder 计算；WT reactivity 作为 mutant encoder 的固定 anchor 必须严格解释；小样本下差分 hidden 可能过拟合 | 单 seed、20-fold、同容量共享权重 screen；与 B1 和“只把 alt token 加到 WT encoder”的同预算 null 比较；必须同时过原 mean/calibration Gate |
| B. Tail-balanced signed-delta expert | method-balanced L1 被大量 near-zero positions 主导，稀有大效应的梯度不足 | 保留 B1 backbone；只增加一个基于 outer-train signed-delta density 的 tail expert。候选必须通过合法、train-only gate 与 B1 组合，不替换 B1；density bin/kernel、权重截断和 gate 规则在 outcome screen 前冻结 | Deep Imbalanced Regression 提出 LDS/FDS；Balanced MSE给出连续标签失衡的统计修正。二者证明问题类型真实存在，但不证明适用于本任务 | 训练目标可能偏离全分布 MAE；稀有标签权重可放大噪声；如果 gate 使用 target magnitude 会构成泄漏 | 先在历史完整 OOF 上只做 train-only crossfit probe；正式候选只能使用 prediction-time legal features。若大效应改善但总体 signed-delta 或 CRPS 失败，立即终止 |
| C. Latent structural-state reweighting mean | 单一逐位置 direct mean 难以表达“一个 mutation 使整条 RNA 在少数反应模板间切换” | 不扩大 encoder；从 WT context 产生固定 2–3 个 full-profile delta templates，mutation head 只预测 simplex state weights；最终 mean 为模板加权和；概率 residual 仍严格零均值且后拟合 | M2-REEFFIT 的实验与模拟结果表明单突变可稳定 alternative structures，mutant profiles 可由共享状态及其 population reweighting解释 | 状态不可识别、模板置换、20 puzzles 数据不足、可能把 method/batch pattern 当结构状态；复杂度和科学主张风险最高 | 仅在完整 failure atlas 出现跨 mutation 重复的全局 profile modes 后开放；与等参数 low-rank profile head 比较。没有稳定跨 puzzle state reuse 则终止 |

## 3. 证据来源与可迁移边界

1. Yang et al., *Delving into Deep Imbalanced Regression*, ICML 2021：提出 label distribution smoothing 与 feature distribution smoothing，处理连续标签的非均衡分布。官方论文与代码入口：<https://proceedings.mlr.press/v139/yang21m.html>。
2. Ren et al., *Balanced MSE for Imbalanced Visual Regression*, CVPR 2022：从统计角度修正 imbalanced continuous regression，并提供高维回归实现。官方论文入口：<https://openaccess.thecvf.com/content/CVPR2022/html/Ren_Balanced_MSE_for_Imbalanced_Visual_Regression_CVPR_2022_paper.html>。
3. Townshend et al., *Ribonanza: deep learning of RNA structure through dual crowdsourcing*：论文把 M2 描述为对每个位点引入突变后观察整条 profile，官方 RibonanzaNet 服务同时提供 sequence-conditioned `Mutate-And-Map` 计算。`REASONED_INFERENCE`：这支持 exact mutant sequence 重新编码是可实现且与任务匹配的归纳偏置，但不证明本项目的 paired encoder 会提高指标。论文：<https://pmc.ncbi.nlm.nih.gov/articles/PMC10925082/>；官方服务：<https://ribonanza.stanford.edu/>。
4. Miladi et al., *MutaRNA*：以 WT 与 mutant 的 base-pair/unpaired probability 差异描述突变影响，支持成对表示的归纳偏置。论文：<https://pmc.ncbi.nlm.nih.gov/articles/PMC7319544/>。
5. Cordero & Das, *Rich RNA Structure Landscapes Revealed by Mutate-and-Map Analysis*：M2-REEFFIT 联合估计共享 alternative structures 与 mutant-specific populations，支持状态重加权假设。论文：<https://pmc.ncbi.nlm.nih.gov/articles/PMC4643908/>。

这些来源只支持“为什么值得测试”，不支持 ReactFlow-Delta 已经提升，也不能替代 OpenKnot 20-fold screen。RibonanzaNet 与 OpenKnot/Ribonanza 数据生态相关；任何预训练权重只能作为暴露已披露的 comparator，不能在未完成 exposure audit 时作为 headline candidate 或 SOTA 证据。

## 4. 结果出现后的唯一决策顺序

1. 先完成当前 R3M3 20/20 folds、合并和冻结 qualifier；当前候选若双 Gate PASS，则直接进入原始 R3M4，不启动后备路线。
2. 当前候选若 FAIL，只能在完整、已合并的 failure atlas 上判断失败结构：
   - 错误主要集中于稀有大效应：优先 B，A 为第二候选；
   - 错误跨 effect magnitude 且呈非局部 profile pattern：优先 A，B 为第二候选；
   - 只有出现可重复、跨 mutant 的整条 profile modes 时，才允许 C。
3. 新 amendment 一次只冻结一个主候选和一个同预算 null；保持原始 R2M4/R3M4 门槛，不通过改变阈值、增加 seed 选择或事后挑 puzzle 获得 PASS。
4. A/B/C 任一路线都必须延续 mean-first、calibration-second：概率损失不能回流到 point mean；CRPS-only 改善仍只能是 calibration baseline。

## 5. 明确排除

- 更大 backbone、更多 attention 层或 hidden size 搜索；
- 新 rank 搜索、恢复 LRSO/SparseDelta/StructDelta；
- teacher/foundation ensemble；
- 使用 partial R3M3 结果选择路线；
- 使用 held puzzle outcome、external outcome 或 target-derived magnitude 作为 prediction-time gate；
- 用新的后备说明修改当前 v3 合同、authority、candidate 或 Gate。

本说明的作用是减少当前 screen 失败后的停顿和开放式试错；它不是 v4 合同，也不授权训练。
