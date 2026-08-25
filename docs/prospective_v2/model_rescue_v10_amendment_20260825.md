# ReactFlow-Delta Model Rescue v10 Amendment

**状态：V10M1 folds0/1、seed0、3 epochs prediction-only engineering smoke PASS；V10M2 固定 seed0、20-fold、40 epochs score-blind screen 已授权；完整 20 folds 前科学评分关闭。**

## 1. 合同地位

V10 是独立阶段性 amendment，不回写 V9。V9 的终局保持
`V9M3_TOP_JOURNAL_SCREEN_FAIL`：signed-delta 与 absolute-delta 分别改善
8.36% 和 9.81%，CRPS 改善 4.23%，但未达到预冻结 5% 门槛，因此 V9M4
永久不开放。

V10 的唯一准入依据是 V9 终局后、按 outcome access 前冻结权重运行的
residual diagnostic。MeanAligned residual 的 mean-minus-median gap 为
`+0.03452 [95% CI +0.02615,+0.04289]`，quantile asymmetry 为
`+0.23049 [+0.18823,+0.27274]`，两项均为 20/20 puzzles 同方向。这证明
“对称零均值残差足够”与数据不符，但不保证新模型一定改善 CRPS。

## 2. 唯一科学问题

在冻结 V8 L1 点预测作为条件中位数的前提下，允许残差分布具有偏斜、
同时精确约束该点仍为 0.5 分位数，是否能够相对参数匹配的对称残差模型
进一步改善 full-construct、method-balanced CRPS。

V10 不扩大点预测 backbone，不重训 V8 点模型，不增加结构、rank、teacher
或 foundation ensemble。新增容量只位于 residual distribution，并由同输入、
同预算的 symmetric null 进行归因。

## 3. 冻结模型

所有新 residual heads 输入 244 维 train-only 标准化特征：feature41、冻结
point/abs(point)，以及 V8 实际训练路径产生的 source/receiver hidden、距离和
mutation one-hot 共 201 维。随机初始化且未被 V8 L1 训练使用的分支禁止进入。

必须运行四个 head：feature41-symmetric、feature41-asymmetric、
MeanAligned-symmetric、MeanAligned-asymmetric。前两者和后两者分别共享 point；
所有 head 共享输入权限、hidden256、两 Gaussian components、40 epochs、Adam
和 method-balanced CRPS。Symmetric 为 `244→256→3`、63,491 参数；asymmetric
为 `244→256→4`、63,748 参数，仅多 257 个 location-allocation 参数。

非对称模型约束 mixture CDF 在 frozen point 处精确为 0.5。`a=b=0.5` 必须
逐值恢复 symmetric null，因此 asymmetric-vs-symmetric 只识别偏斜位置能力。

## 4. 执行与 Gate

V10M1 只允许 folds0/1、seed0、3 epochs prediction-only smoke。V10M2 才允许
seed0、20-fold、40 epochs score-blind screen；完整 20 folds 前禁止查看任何
partial score。完整后 V10M3 只运行一次冻结 scorer 和 qualifier。

最终 candidate 固定为 MeanAligned-asymmetric，不能在结果出现后把 symmetric
升级为候选。PASS 必须同时达到：signed MAE 相对 feature41 至少 5%；distribution
absolute MAE 相对独立 feature41 absolute head 至少 5%；CRPS 相对公平的
feature41-asymmetric 至少 5%；CRPS 相对历史 V9 至少 1%；asymmetric 相对同
point 的 symmetric null 至少 1%，CI lower 大于零且至少 14/20 puzzles 正向；
其余 headline CI、puzzle direction、LOO、influence、coverage 和 failure Gate
按 machine contract 全部通过。

任一 Gate 失败即关闭 V10，不降低阈值、不增加 mixture components、不改 hidden、
不搜索 loss 或 epoch。只有 exact V10M3 PASS 才可开放固定 seeds0-4 正式确认。

## 5. 主张边界

V10 内部 PASS 最多建立 development-consumed LOPO 上的性能与 residual-asymmetry
增量证据。External replication、SOTA、mechanism、practical utility 和 publication
readiness 均需要后续独立证据，不能由内部显著性自动生成。

## 6. V10M1 工程记录与 V10M2 authority

首次 smoke 完成 fold0 后，fold1 在 held prediction 的 median invariant 检查处
中断。根因是 float32 下 inverse-normal CDF 与 normal CDF 在 allocation 边界的
数值往返误差，不是科学 Gate 或训练分数失败。四个 matched heads 的分布构造
统一提升为 float64；网络、输入、参数量、目标、epoch、候选和 Gate 均未改变。
fold0 被原样保留，只补跑 fold1。预冻结 smoke qualifier 随后机械给出
`V10M1_ENGINEERING_SMOKE_PASS`，且未读取任何科学分数或 external outcome。

因此唯一开放阶段为 V10M2。必须使用固定 controller 完成 seed0 的 20 个 folds；
完整 prediction-only universe 出现前不得运行 scorer、读取 partial metric 或修改
任何候选。V10M3 仍未授权，只有 20/20 完整后的一次预冻结 scorer 与 qualifier
可以裁决是否开放后续 formal confirmation。

V10M2 首次并行运行完成并保留 14/20 folds；其余6个folds在 held V8 point
重放检查处停止。只读身份诊断显示，重新运行同一 checkpoint 的最大差异为
`2.3841858e-7`，103,368 rows 中608 rows超过预冻结 `1e-7`，符合 CUDA
float32 的少数 ULP 漂移；未发现 key 或模型语义变化。V10 不放宽 replay
阈值，而是把已冻结的 V8 prediction artifact 定义为 held point 的权威来源，
按 biological key 原值读取。Outer-train point 仍由同一 checkpoint 计算，held
direct features 仍来自其实际训练路径。该修复不改变模型、训练、候选或 Gate；
只补跑缺失的 folds 2/3/9/10/16/17，14个已完成 artifacts 保持不变。
