# ReactFlow-Delta Model Rescue v9 Amendment

**状态：V9M0 implementation/tests PASS；V9M1 两折工程 smoke 已授权，科学评分仍关闭**

## 1. 与 V8 的关系

V8 的终局不变：MeanAligned 在 exact target identity 下相对 corrected feature41 获得 8.36% signed-delta MAE 改善、20/20 puzzles 正向，但 absolute-delta MAE 恶化 1.77%，因此 V8M2 Gate FAIL，V8M3 永久不开放。

V9 不重训或改变该 signed mean。它只检验一个新的、由完整 V8 结果支持的假设：L1 学到的是接近零的条件中位数，导致 `|mu|` 系统性低估真实 magnitude；若在冻结均值后学习严格零均值的残差分布，则 signed point mean 保持不变，而分布的 `E|Delta|` 可以恢复 magnitude。

## 2. 公平比较

corrected feature41 与 V8 MeanAligned 都使用同一残差 head：输入为相同的 outcome-blind feature41 特征，加各自冻结的 signed mean 与其绝对值；结构、初始化规则、epoch、优化器、method-balanced CRPS 和零均值约束完全相同。两个 Gaussian component 的 location 都必须逐 key 等于各自 frozen mean，因此 calibration 不可能移动 signed point prediction。

## 3. 阶段

- V9M0：实现合同、模型、prediction schema、merge/scorer/qualifier 和测试；训练关闭。
- V9M1：仅 folds 0/1、seed0、3 epochs 的真实数据 prediction-only smoke；禁止科学评分。
- V9M2：smoke exact PASS 后运行 seed0、20-fold、40 epochs；全 20 folds 完整前禁止评分。
- V9M3：一次完整 join 后执行预冻结顶刊 Gate。
- V9M4：只有 Gate PASS 才进行 seeds0–4 的固定正式确认。

## 4. 顶刊级 Gate

候选必须同时满足：signed MAE 相对 feature41 至少 5%；distribution-derived absolute MAE 相对 feature41 的独立 absolute head 至少 1%；CRPS 相对同 family 校准的 feature41 至少 5%；三者 CI lower 均大于零；signed/CRPS 至少 16/20 puzzles 正向，absolute 至少 14/20；LOO 全正，单 puzzle 贡献不超过 20%，coverage 与 failure 完整，68%/95% calibration error 不比基线恶化超过 2 个百分点。

## 5. 边界

不访问 external outcome；不复用 target-identity 修复前 artifact；不改变 V8 Gate；不搜索 residual family、hidden size、component 数、epoch、loss weight 或阈值；未通过完整 Gate 时不得开启正式多 seed 或声称 SOTA。
