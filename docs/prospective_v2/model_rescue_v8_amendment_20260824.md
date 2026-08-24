# ReactFlow-Delta Model Rescue v8 Amendment

**终局：V8M2 absolute guardrail FAIL；signed mean breakthrough 保留；V8M3 永久不开放**

## 1. 为什么另立 v8

V7 的完整 corrected 20-fold 结果表明，RiNALMo dependency6 相对 strongest corrected feature41 的 signed-delta 相对改善只有 0.0378%，95% CI 跨零，因此该方向已经终止。与此同时，v3 的 10 个所谓 corrected experts 只修复了 design/full coordinate，生成时仍使用跨 construct 的 mutation suffix fallback，早于 exact puzzle×method×mutation target identity 修复。它们必须保留，但不能继续训练、补 fold 或用于科学评分。

V8 不修改 v3/v7 结论。它只回答一个尚未被正确实验回答的问题：在完全正确的 target identity 下，mean-first、method-balanced 的端到端神经均值学习是否有真实增量。

## 2. V8M1：新鲜 corrected experts

- 从已经通过 13,976/13,976 identity 审计的 accessor 开始；
- 20 个 outer LOPO folds、seed 0、40 epochs；
- 每个 fold 从头训练 B1 与 MeanAligned；
- 不 warm-start，不复用 v1/v2/v3 checkpoint 或 prediction；
- 输出 full registered construct-position prediction-only artifacts；
- 完整 20 folds 前不读取 held score；
- external outcome 始终锁定。

## 3. V8M2：一次性 mean signal screen

完整 expert rebuild PASS 后，统一比较 corrected feature41、corrected B1 和 corrected MeanAligned。MeanAligned 必须同时相对 feature41 和 B1 获得至少 1% signed-delta MAE 改善，两个 paired CI lower 都大于 0，且相对 feature41 至少 14/20 puzzles 正向，才能开放更大模型。

2026-08-24 23:26（Asia/Shanghai），V8M1 qualifier 在不读取任何 held score 的情况下确认 20/20 folds、exact target identity、fresh checkpoints、完整 registered key universe 和 prediction-only schema 全部通过，精确状态为 `V8M1_CORRECTED_EXPERT_REBUILD_PASS`。因此只开放一次完整 V8M2 target join；训练保持关闭，partial-fold score 继续禁止。

一次性完整评分显示，MeanAligned 相对 corrected feature41 的 signed-delta MAE 改善 8.36%，绝对 gain 的 95% CI 为 [0.01286, 0.01911]，20/20 puzzles 正向；相对 corrected B1 改善 3.89%，18/20 正向。该结果证明 mean-first 目标修复了 signed mean。与此同时，absolute-delta MAE 相对 feature41 恶化 1.77%，超过预冻结的 -0.5% guardrail，因此精确资格仍为 `V8M2_MEAN_SIGNAL_NOT_ELIGIBLE`，不得开放 V8M3。

Gate 后只读诊断进一步显示：MeanAligned 的 method-balanced `|mu|` 均值为 0.04284，而真实 `|Delta|` 均值为 0.18332，20/20 puzzles 均表现为幅度低估；feature41 的幅度均值为 0.18283。该诊断不改变 V8 FAIL，只为独立 V9 的零均值残差分布假设提供依据。

## 4. V8M3：固定大容量 residual model

只有 V8M2 PASS 才实现。模型固定为 feature41 base 加大容量 neural residual：d=192、8 heads、4 attention blocks、hidden=128。均值先用 method-balanced signed-delta L1 训练；均值冻结后再训练严格零均值的两组件条件 Gaussian scale mixture。不得搜索容量、层数、loss、epoch 或 mixture 数。

顶刊 screen Gate 不降低：相对 corrected feature41 的 signed-delta MAE 与 CRPS 均至少改善 5%，两个 CI lower 均大于 0，二者至少 16/20 puzzles 正向，LOO 保持正向，单 puzzle 贡献不超过 20%，coverage 100%，failure 0。

## 5. 不可协商边界

- V8M1 不读取 held score；
- 不访问 external outcome；
- 不复用任何 target-identity 修复前 artifact；
- 不继续 RiNALMo dependency6、rank、SparseDelta、StructDelta 或 teacher 搜索；
- engineering PASS 不等于性能 PASS；
- development PASS 不能自动生成 SOTA、external 或 publication-ready 主张。
