# ReactFlow-Delta Model Rescue v7 Amendment

**状态：V7M0 合同与 authority 冻结中**  
**日期：2026-08-24**  
**范围：RiNALMo-Giga exact source-to-receiver nucleotide dependency only**

## 1. 合同地位

本 amendment 是 v6 终止后的独立、窄范围模型救援。它不修改 v1–v6 的结果，尤其不修改：

- v4：`MODEL_RESCUE_V4_FAIL`；
- v5：`MODEL_RESCUE_V5_FAIL`；
- v6：`MODEL_RESCUE_V6_FAIL`；
- v3：继续在独立 worktree 和 authority 下运行，v7 不覆盖、不终止、不读取其未完成分数。

v7 的任务不是再增加一组 embedding 或 ViennaRNA descriptor，而是检验一个不同的信息源：预训练 RNA masked language model 在真实 SNV 干预前后产生的、从 mutation source 到每个 receiver 的有向 nucleotide log-odds 变化。

## 2. 为什么继续做、为什么不是重复实验

已完成证据形成一条清楚的排除链：

1. v4 的 paired RNA-FM dual tower 使用约 35M–45M trainable 参数，但相对 corrected B1 的 signed-delta 改善仅 0.227%，CRPS 还略有下降。更大容量和 foundation embedding 本身不足。
2. v5 的 exact-mutant minus WT thermodynamic ensemble features 在 20/20 puzzles 上方向为正，但 signed-delta 改善只有 0.634%。传统 nearest-neighbor ensemble 确有信号，但不够强。
3. v6 在 v5 上加入同一 WT 2A3 constraint field，16/20 puzzles 为正且 CI 下界大于零，但增量仅 0.116%。继续扩展同类热力学特征不值得。

因此 v7 不搜索 backbone、layer、模型大小或 feature subset。它使用 [Nature Genetics 的 nucleotide dependency 定义](https://www.nature.com/articles/s41588-025-02347-3) 和 [官方实现](https://github.com/gagneurlab/dependencies_DNALM)，但把 query 固定为项目中实际注册的 SNV，把 target 固定为 full construct 的每个 receiver。Foundation 采用 [RiNALMo 官方实现](https://github.com/lbcb-sci/RiNALMo) 的 `giga-v1`；其论文发表于 Nature Communications，模型约 650M 参数并在约 36M 条 ncRNA 序列上预训练。

这个设计与 v4 的本质差异是：v4 给小样本监督网络两个上下文 embedding，让它自行学习突变干预关系；v7 直接从冻结的 MLM logits 计算干预后的 target nucleotide odds 变化，使 source→receiver relation 在训练前已经显式存在。

## 3. 科学假设与证伪标准

### 核心假设

对于 WT sequence `x`、注册突变 `(s, ref→alt)` 和 receiver `j`，RiNALMo 对 exact-mutant `x'` 与 WT `x` 的未遮罩输出满足：

\[
d_{s\to j,k}=\log_2\frac{p(k_j\mid x')}{1-p(k_j\mid x')}
-\log_2\frac{p(k_j\mid x)}{1-p(k_j\mid x)},
\]

其中 `k∈{A,C,G,U}`。这个向量应包含传统 ensemble feature 和普通 embedding 未表达的、可跨 puzzle 泛化的 directed dependency signal。

### 最小预测

在完全相同的 20-fold LOPO、method-balanced signed-delta estimand 下，将固定 dependency features 加到 v6 candidate feature universe 后，应使 signed-delta MAE 相对改善至少 1%，paired CI 下界大于 0，并在至少 14/20 puzzles 上改善。

### 证伪

只要完整 V7M2 eligibility probe 未通过任何冻结 Gate，立即终止 v7，不训练 dependency operator，不调整阈值，不搜索 layer、模型大小、feature summary、alpha 或其他 foundation model。

## 4. 冻结的 foundation 与 dependency 定义

- 模型：RiNALMo `giga-v1`，约 650M 参数；
- 官方代码 commit：`2c2c5c14a5ae609d8c560a5d9ca32e51e0288955`；
- 官方权重：`rinalmo_giga_pretrained.pt`，Zenodo record 15043668；
- foundation 全程 `eval()`、`no_grad()`，不 fine-tune；
- 输入是完整、未遮罩 WT 与 exact mutant sequence；
- A/C/G/U 概率由四个对应 MLM logits 的 softmax 得到；
- 概率以 `1e-10` 稳定后计算 base-2 log odds；
- receiver 等于 mutation source 时 dependency 六维全部置零，以排除 trivial self-reconstruction；
- 不读取 mutant reactivity、target error、qualified target mask 或任何 external outcome；
- 同一 biological WT/mutation across methods 复用完全相同的 cache 行。

固定六维 basis：

1. A 的 signed log-odds shift；
2. C 的 signed log-odds shift；
3. G 的 signed log-odds shift；
4. U 的 signed log-odds shift；
5. receiver WT nucleotide 对应的 signed log-odds shift；
6. 四个 shift 的 maximum absolute value。

不增加 attention map、hidden embedding、gradient dependency、masked dependency、模型 ensemble 或多层输出。

RiNALMo 的预训练数据是否包含 OpenKnot/Eterna exact sequence 当前无法确认，登记为 `UNKNOWN_NOT_ASSERTED`。因此任何内部 PASS 仍不能自动成为 external、SOTA 或 publication PASS。

## 5. 阶段与 authority

### V7M0：合同冻结

仅允许写入 human contract、machine contract、decision ledger、设计、计划和 contract tests。Focused commit 之前不下载权重、不安装环境、不运行 inference。

### V7M1：outcome-blind dependency cache

V7M0 exact PASS 后才允许：

- 安装官方 RiNALMo commit；
- 下载官方 Giga 权重；
- 枚举真实 M2 中的唯一 WT 与注册 exact mutant sequence；
- 生成完整六维 source→receiver cache；
- 机械检查 finite、shape、self-zero、method reuse 和 registered coverage。

V7M1 只产生 `ENGINEERING_OUTCOME_BLIND_CACHE_PASS/FAIL`，不读取任何 score。

### V7M2：固定线性 eligibility probe

Baseline：`direct18 + v5 unconstrained12 + v6 constrained11`。  
Candidate：baseline 加 `v7 dependency6`。  
Learner：train-only weighted standardized ridge，`alpha=1.0`。  
权重：puzzle→method→mutant→qualified position 等权。  
Baseline 必须逐 key 重放 v6 candidate，`atol=1e-12`。

所有 20 folds 先输出 prediction-only artifact。完整 universe 前禁止读取 prediction value、loss、MAE、CRPS 或 Gate 方向。20/20 后先 merge，再用一次 focused authority join target、score、qualify。

Gate：

- signed-delta relative MAE gain `>=1%`；
- signed-delta paired 95% CI lower `>0`；
- signed-delta positive puzzles `>=14/20`；
- absolute-delta relative gain `>=-0.5%`；
- coverage `=100%`；failure `=0`；unexpected keys `=0`。

只有 exact `V7M2_RINALMO_DEPENDENCY_SIGNAL_ELIGIBLE` 才开放 V7M3。

### V7M3–V7M4：高容量 dependency operator 与单 seed screen

这一阶段只在 V7M2 PASS 后实施。Corrected B1 checkpoint 冻结；RiNALMo 仍冻结。Operator 使用：

- B1 source hidden 96；
- B1 receiver hidden 96；
- dependency 6；
- mutation ref/alt one-hot 8；
- fixed signed-distance encoding 32。

Dependency 先投影到 128 维，context 投影到 256 维；融合后使用约 0.75M trainable parameters 的四层 residual MLP，最后一层零初始化。Mean 用精确 method-balanced signed-delta L1 训练 80 epochs；mean 冻结后用 zero-mean two-Gaussian CRPS 校准 40 epochs。

Equal-capacity controls：

- dependency 全零；
- receiver dependency 固定循环平移半个 construct 长度，保留边际分布但破坏 source→receiver 对齐。

Top-journal development Gate 同时要求：

- 相对 corrected B1，CRPS 和 signed-delta MAE 均改善至少 5%；
- 两指标 paired CI lower 均大于 0；
- 两指标各至少 16/20 puzzles 改善；
- leave-one-puzzle effect 保持正；
- 单 puzzle 贡献不超过总 effect 的 20%；
- primary 在两指标上均显著优于两个 equal-capacity controls；
- coverage 100%、failure 0、unexpected 0；
- 68%/95% coverage error 相对 baseline 恶化不超过 1 percentage point。

### V7M5：五 seed formal confirmation

固定 seeds 0–4、20 folds、无 family/layer/epoch/seed selection。唯一五-seed mixture 必须重新通过 V7M4 全部 Gate。内部通过也只标记 `HIGH_EFFECT_POST_HOC_DEVELOPMENT_PASS`。

### V7M6：冻结和 M6 handoff

无论 PASS/FAIL，关闭 training 与 held score authority，保存全部 negative artifact，并返回主合同 M6。外部验证必须另立 sealed amendment。

## 6. 资源与执行边界

- GPU0–7 均可使用，只选择有足够空闲显存的卡；
- 可与其他任务共卡，但不得抢占、终止、发信号或修改其他进程；
- OOM 只允许换卡或降低 cache inference batch size，不改变模型、feature、seed、epoch 或 Gate；
- 不访问新的 external outcome；
- 不干预 v3；
- 不降低任何 Gate；
- 不进行 layer/model/feature/alpha search；
- 不因计算预算停止已冻结 universe，但“无预算上限”不等于允许无限候选搜索。

## 7. 允许的论文主张

只有 V7M5 PASS 时，可主张：在 development-consumed OpenKnot LOPO 中，冻结 RiNALMo 的 exact nucleotide dependency operator 对 point 和 probabilistic metrics 有大效应且优于 equal-capacity controls。

以下仍禁止：task-matched SOTA、external replication、biophysical mechanism、practical utility、publication readiness，以及“预训练未暴露 OpenKnot sequence”。
