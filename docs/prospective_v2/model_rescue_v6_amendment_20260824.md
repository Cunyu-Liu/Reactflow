# ReactFlow-Delta Model Rescue v6 Amendment

**合同日期：** 2026-08-24

**当前阶段：** `V6M1_OUTCOME_BLIND_CONSTRAINED_CACHE_RUNNING`

**远端父 HEAD：** `eb9fbdc33a03aeb12decc0d68575c18549395e71`

## 1. 合同地位

v6 是独立 amendment。v5 的 `MODEL_RESCUE_V5_FAIL`、v4/v2/v1 的终局以及仍在运行的 v3 均不可修改。v6 不降低 v5 的 1% eligibility Gate，也不训练被 v5 禁止的 neural branch。

v6 的唯一新增能力是用合法的 WT 2A3 input 约束 WT 与 exact-mutant 的 Boltzmann partition ensembles，测试 experiment-conditioned differential structure 是否比 sequence-only v5 features 提供更大的跨 puzzle signed-delta signal。

## 2. 固定 constraint protocol

- ViennaRNA `2.7.2`，37°C，global partition function；
- Deigan pseudo-energy，`m=1.8`、`b=-0.6`、`RNA.OPTION_PF`；
- OpenKnot 已归一化 WT 2A3 reactivity；有限负值固定 clamp 为 0；null 固定为 `-999`；
- Python vector 前置一个 `-999` dummy，以满足 ViennaRNA 一基坐标；
- WT 与 exact mutant 使用完全相同的 WT constraint vector；
- 不删除 mutant edit-site constraint；
- 不读取或使用 mutant reactivity、mutant error、target mask、score 或 external outcome；
- P20 Eterna 全缺失 WT profile 必须严格退化为 unconstrained partition ensemble；
- 不搜索 normalization、m、b、temperature、constraint method、feature subset 或 structure engine；cache 保留全部 12 个解释性通道。

在任何 v6 target 或 score access 之前的 outcome-blind basis audit 证明三个通道满足恒等式

\[
\Delta p_{\mathrm{unpaired}}+\Delta m_{\mathrm{upstream}}+\Delta m_{\mathrm{downstream}}=0.
\]

因此学习输入固定排除可由前两者精确重构的 downstream mass。该修正不搜索 outcome、不删除信息，也不要求重建 12-channel cache。

## 3. V6M2 eligibility

baseline 为 v5 的 18 个直接 covariates 加原有 12 个 unconstrained ensemble-delta features，以保证逐 key 重放 v5 candidate；candidate 只增加 11 维满秩 constrained basis。learner 固定为 train-only weighted standardized ridge alpha 1。

只有以下条件全部成立才开放 neural implementation：signed-delta relative MAE gain ≥1%、paired CI lower >0、至少 14/20 puzzles 正向、absolute-delta relative MAE 不恶化超过 0.5%、coverage 100%、failure 0、unexpected keys 0。完整 20 folds 前禁止 target join 和任何 partial score。

## 4. 唯一 neural primary 与 mandatory controls

primary 为 `b1_2a3_constrained_ensemble_residual`。corrected B1 按相同协议从头训练后冻结；residual 使用 detached B1 source/receiver features、11 维 unconstrained 独立基和 11 维 constrained 独立基。head 固定为 train-only normalization 后的双投影 hidden-64 fusion、GELU、zero-initialized scalar output。

两个 control 不是候选搜索：`b1_zero_structure_residual` 使用 22 个零 structure channels；`b1_unconstrained_ensemble_residual` 使用 11 个 unconstrained channels 加 11 个零 channels。三者输入宽度、参数量、训练预算、mean loss 和 calibration family 完全相同。

## 5. 顶刊级 Gate

seed-0、20-fold primary 相对 corrected B1 的 CRPS 和 signed-delta MAE 均须改善 ≥5%，两个 paired CI lower 均 >0，两指标均至少 16/20 puzzles 正向，leave-one-puzzle effect 始终正，单 puzzle 贡献 ≤20%，coverage 100%、failure 0、unexpected 0，68%/95% coverage error 恶化 ≤1 percentage point。

此外 primary 相对两个 mandatory controls 的 CRPS 与 signed-delta MAE paired CI lower 均须 >0。全部通过才允许 seeds 0–4 的无选择 confirmation。

任何内部 PASS 仍仅为 `HIGH_EFFECT_POST_HOC_DEVELOPMENT_PASS`。external、SOTA、mechanism、practical utility 和 publication readiness 均不自动建立。

## 6. 停止条件

V6M2、seed-0 screen 或 five-seed confirmation 任一失败即关闭 v6，不改变 threshold，不追加 feature、constraint strategy、head、seed 或 epoch。下一方向只能在新的独立 amendment 中提出。
