# ReactFlow-Delta Model Rescue v12 Amendment

## 1. 合同地位

V12 是独立的、后验诊断支持的开发 amendment。V1–V11 的终局保持不可变；尤其是 V11 仍为 `V11M3_TOP_JOURNAL_SCREEN_FAIL`，V11M4 永久不开放。V12 的任何结果不得回写或美化 V11。

本 amendment 只测试一个能力：利用 prediction-time 合法的绝对编辑距离和 feature41 绝对幅度，对冻结 V11 neural residual 进行连续、单调、区间在 `(0,1)` 内的 shrinkage。V12 不扩大 backbone，不改变 V11 point loss，不加入 method ID，不更换 residual distribution family，也不访问 external outcome。

## 2. 证据、假设与反证

`LOCATED_EVIDENCE`：V11 在完整 20-puzzle screen 上相对 feature41 改善 signed-delta MAE 9.8041%、point absolute-delta MAE 4.5681%、task CRPS 3.6615%，但没有通过冻结的顶刊 Gate，也没有胜过完整任务上的 unanchored matched null。

`LOCATED_EVIDENCE`：V11 末段训练下降的中位数只有 0.2225%，仅 2/20 folds 达到 1%；继续增加 epoch 没有诊断支持。

`LOCATED_EVIDENCE`：只有 3/20 folds 出现至少 5 个百分点的 train-to-held gain drop，中位数为 0.437 个百分点；一般性的全局 overfit shrinkage 没有诊断支持。

`LOCATED_EVIDENCE`：V11 residual correction 在编辑位点使 signed MAE 稳定恶化 1.428%，在 feature41 绝对值小于 0.05 的区域没有收益；但在距离 6–20、距离大于 20 的区域分别改善 10.966% 和 11.372%，在 feature41 绝对值至少 0.05 的三个区间改善 8.51%–14.03%。

`HYPOTHESIS`：V11 residual 含有真实的非局部 mutation-response 信号，但当距离或 feature41 幅度较小时发生 over-correction。只用合法 inner-OOF prediction 拟合的低容量单调 gate，可以保留中远距离收益并抑制近编辑位点和近零效应误差。

`PREDICTION`：V12 同时胜过 feature41、冻结 V11 parent 和 V10 distribution；point、CRPS 和 magnitude 指标全部通过冻结 Gate。

`FALSIFIER`：任何 point、CRPS、distribution-absolute、matched-parent、coverage、integrity 或 crossfit Gate 失败即终止 V12，不增加第二种 gate、阈值、输入或模型。

## 3. 唯一候选与科学 null

候选：`v12_v11_monotone_regime_shrinkage`。

设冻结 V11 point 为 `mu_v11`，outer-train feature41 point 为 `f41`，则：

\[
\mu_{v12}=f_{41}+g(d,|f_{41}|)(\mu_{v11}-f_{41}).
\]

gate 是两个单调 logistic factor 的乘积：

\[
g=\sigma(b_d+\operatorname{softplus}(w_d)\log(1+|d|))
\times
\sigma\left(b_m+\operatorname{softplus}(w_m)
\log\left(1+\frac{|f_{41}|}{0.05}\right)\right).
\]

它只有四个参数，随距离和 feature41 幅度分别单调不减。乘积结构表达“两个条件中任一不足时均允许 shrink”，不使用 hard bins。0.05 只作为预登记诊断中的数值归一化常数，不是 inference threshold。

科学 null 为 `g=1`，即 authoritative V11 anchored parent。V12 必须逐 key 重放 null；任何重训差异都不能被当成 gate 增益。

## 4. Gate 拟合与数据边界

每个 outer fold 内使用 split_v4 的四个 inner puzzle groups。每个 outer-train puzzle 恰好作为 inner-held 一次。inner V11 模型不得见 inner-held outcome；只有 prediction 完成后，inner-held target 才能进入 method-balanced signed L1 gate objective。

禁止用 outer-final V11 在训练集上的 in-sample prediction 拟合 gate。禁止使用 method、puzzle ID、target magnitude、error、mask 或 external outcome。禁止搜索 gate family、feature、threshold、optimizer、step 数或 initialization。

seed-0 screen 直接复用 authoritative V11 outer point 和 parent distribution；只为 gate 生成 inner-OOF prediction，并在 gate point 冻结后拟合同一个 V10 MedianAsymmetricResidual family。这样 candidate 与 parent 的唯一区别是预登记 gate 及其必要的同族 recalibration。

## 5. 训练、评分与停止条件

V12M2 只运行 folds0/1、seed0、3 epoch inner model、20 gate steps、3 calibration epochs，不计算科学分数。

V12M3 固定 seed0、20 outer folds、四折 inner crossfit、40 epoch inner model、500 gate steps、40 calibration epochs。在 20/20 prediction artifacts 完成前禁止读取任何 partial score。完整后只允许 merge 一次、score 一次、qualifier 一次。

顶刊 screen 至少要求：signed-delta 相对 feature41 改善 10%、相对 V11 改善 1%；point absolute-delta 相对 feature41 改善 5%、相对 V11 改善 1%；task CRPS 相对 feature41 改善 5%、相对 V11 和 V10 均改善 1.5%；distribution-derived absolute-delta 相对 feature41 改善 15%、相对 V10 改善 1%。所有对应 puzzle-level CI lower 必须大于 0，并满足预冻结 positive-puzzle、LOO、influence、coverage 和 integrity Gate。

只有 exact `V12M3_TOP_JOURNAL_SCREEN_PASS` 才允许 V12M4 固定 seeds0–4 formal confirmation。任何失败立即回到 benchmark route；不得追加 quantile residual、第二种 gate、method routing、更大 backbone 或降低阈值。

## 6. 主张边界

V12 即使 formal PASS 也只能成为 `POST_HOC_DEVELOPMENT_PASS`。由于 OpenKnot 20 puzzles 已被反复开发使用，external replication、SOTA、mechanism 和 publication readiness 均保持未建立。任何确认性 external 测试必须另立 sealed amendment。
