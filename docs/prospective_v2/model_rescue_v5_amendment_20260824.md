# ReactFlow-Delta Model Rescue v5 Amendment

**合同日期：** 2026-08-24

**当前阶段：** `V5M2_FIXED_WEIGHTED_RIDGE_ELIGIBILITY_PREDICTION_ONLY`

**父 HEAD：** `820f34fc693f933dba7db816ba5f62f101cf6986`

## 1. 地位与唯一目标

本 amendment 独立于仍在运行的 v3，也不修改 v1、v2 或 v4 的任何终局。v4 的 `MODEL_RESCUE_V4_FAIL` 永久保留；v5 不能用新的模型结果回写或美化它。

v5 只检验一个缺失能力：显式计算 exact mutant 与 WT 的 Boltzmann RNA secondary-structure ensemble 差异，并把这个固定、outcome-blind 的扰动量作为 corrected B1 的小型 residual branch。它不更换 foundation，不扩大 B1，不搜索 rank、深度、宽度、structure engine、feature subset、epoch 或 loss 权重。

核心科学假设是：2A3 reactivity 反映 nucleotide flexibility；因此 exact mutation 引起的 unpaired probability、pairing entropy 与 base-pair-probability field 改变，可能解释 corrected B1 尚未捕获的跨 puzzle signed-delta 残差。旧结构 probe 只使用 WT MFE graph distance 和 WT BPP，没有计算 mutant−WT ensemble change，所以旧负结果不能直接否定这一假设。

## 2. Outcome-blind ensemble cache

V5M1 使用 ViennaRNA 的 global McCaskill partition function，在 37°C 对每个 177-nt WT 与 exact mutant sequence 分别计算 base-pair probability。cache builder 只允许读取 `id`、`sequence` 和由 non-outcome metadata 已确认的 corrected full mutation coordinate；禁止读取 reactivity、reactivity_error、held mask、score 或 external outcome。

每个 registered mutant×receiver 固定生成十二个特征：delta unpaired probability、delta pairing entropy、mutant−WT source-receiver pair probability、WT 与 mutant source-receiver pair probability、receiver BPP-row L1/L2 change、upstream/downstream pairing-mass change、global BPP Frobenius change、ensemble-free-energy change和source unpaired-probability change。不得根据 outcome 删除或新增特征。

## 3. 先验资格 probe

任何神经残差训练前，V5M2 必须完成固定 20-fold LOPO weighted-ridge probe。baseline 使用距离、位置、region、mutation identity 和 WT context；candidate 只额外加入上述十二个 exact-ensemble-delta features。训练权重严格按 puzzle-method cell、mutant、qualified position 平衡；held puzzle 不参与标准化或拟合。

完整 20 folds 合并前不得读取部分 MAE、CRPS、per-puzzle effect 或 Gate 方向。只有 signed-delta MAE relative gain 至少 1%、paired CI lower 大于 0、至少 14/20 puzzles 正向，且 absolute-delta MAE 不恶化超过 0.5%，才允许进入神经残差实现。该 PASS 只表示 `STRUCTURE_DELTA_SIGNAL_ELIGIBLE`，不是模型、SOTA 或论文 PASS。

## 4. 唯一神经候选

若且仅若资格 probe 通过，候选固定为 `b1_exact_ensemble_delta_residual`：先按 corrected B1 的相同协议训练 B1；随后冻结 B1，使用 detached B1 source/receiver features 和十二个固定 structure-delta features，训练 hidden-64 的 `Linear→GELU→Linear` residual head，最后一层严格零初始化。residual stage 使用 method-balanced signed-delta L1、Adam `1e-3`、weight decay `1e-3`、40 epochs、clip `5.0`。均值冻结后再拟合严格 zero-mean 两 Gaussian residual calibration；校准不得改变 point mean。

## 5. 顶刊级 Gate 与停止条件

seed-0 20-fold screen 相对 corrected B1 必须在 CRPS 和 signed-delta MAE 上各至少改善 5%，两个 paired CI lower 都大于 0，两指标各至少 16/20 puzzles 正向，leave-one-puzzle effect 始终正向，单 puzzle 贡献不超过 20%，coverage 100%、failure 0、unexpected keys 0，68%/95% coverage error 恶化均不超过 1 个百分点。只有全部通过才开放固定 seeds 0–4 的五 seed confirmation。

资格 probe 失败则 v5 当场终止，不得用神经网络“挖信号”；seed-0 screen 或五 seed Gate 失败也立即返回 M6 benchmark route。任何内部 PASS 都只能标记 `HIGH_EFFECT_POST_HOC_DEVELOPMENT_PASS`。external replication、SOTA、mechanism、practical utility 和 publication readiness 仍需新的独立 outcome amendment。

## 6. 计算与隔离边界

v5 可使用 CPU 并行 cache，也可在不抢占、不终止、不发送信号给其他进程的前提下使用 GPU0–7 中显存足够的卡。v3 worktree、sessions、artifacts 和 authority 不得修改。完整阶段前只做低频文件名、session、日志更新时间和非指标错误检查。不得访问新的 external outcome。
