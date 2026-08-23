# ReactFlow-Delta Model Rescue v4 Amendment

**合同日期：** 2026-08-23

**当前阶段：** `V4M1_IMPLEMENTATION_AND_INVARIANTS_ONLY`

**父 HEAD：** `a560a179d620f5be083c8fce617e3cc0a7016908`

**v4 分支：** `codex/reactflow-delta-model-rescue-v4-20260823`

## 1. 合同地位

本 amendment 是独立的 Model Rescue v4。它不修改 Model Rescue v1 的 `M2_NO_RESCUE_CANDIDATE`，不修改 Model Rescue v2 的 `MODEL_RESCUE_V2_FAIL_CALIBRATION_BASELINE_ONLY`，也不修改正在另一个 worktree 中运行的 Model Rescue v3 坐标修复、候选、Gate、artifact 或 authority。

v4 唯一允许检验的研究假设是：在严格正确的 full-sequence mutation coordinate 下，显式建模 WT 一维响应状态、RNA 二维 pair state 和 exact mutation-conditioned receiver state，并加入冻结的 paired RNA-FM sequence representation，能否产生超过容量、foundation 与 calibration 所能解释的 signed-delta mean 增量。

本阶段只授权合同、代码、测试和 outcome-blind foundation cache 实现。真实 M2 训练在 V4M1 qualifier 完成前保持关闭。

## 2. 唯一候选和对照

唯一主候选是 `v4_dual_tower_rnafm`。其固定配置为 sequence width 512、8 heads、5 个 WT sequence blocks、5 个 mutation-response blocks、pair width 128、5 个 axial pair blocks、FFN width 2048、dropout 0.10。目标 trainable capacity 为 35M–45M；正式 real-data smoke 前必须机械统计。

主 foundation 固定为官方 `ml4bio/RNA-FM` 的 `rna_fm_t12`，仓库版本固定在 `348951516e0963d22bbb33b3c9fc18c89081d38e`，loader 必须来自该 commit 且 tracked files 无修改的 Git checkout；Python 运行时生成的未跟踪字节码不改变资格。checkpoint 固定使用作者发布的 `cuhkaih/rnafm/RNA-FM_pretrained.pth`，并以显式本地路径加载；运行时不得静默改用其他代码或权重。RNA-FM 始终冻结，只生成 WT 与 exact mutant 的 per-nucleotide final-layer embeddings。其 RNAcentral100 自监督序列暴露必须披露；在没有证据时不得声称 OpenKnot exact sequence no-overlap。

以下四个对照必须存在，不能事后删除：corrected B1、scratch dual tower、RNA-FM-only 和 parameter-matched sequence null。capacity null 必须使用相同 paired RNA-FM input，但不得拥有 pair tower 或 mutation source row/column；参数差异不超过 5%，且不能通过未使用参数凑数。

RiNALMo 只作为 foundation-only sensitivity；RibonanzaNet 只作为 exposure-disclosed comparator。两者都不能根据 development outcome 取代主 foundation。

## 3. 输入、目标与泄漏边界

预测允许使用 WT sequence/reactivity/error/observed indicator/region/full position、corrected full mutation position、ref/alt、signed distance、冻结的 WT RNA-FM embedding 和 exact mutant-minus-WT RNA-FM embedding。

预测禁止使用 held mutant reactivity/error/qualified mask/score、puzzle ID、method ID、dataset/publication ID、external outcome 或 outcome-derived structure。foundation cache builder 只能读取 CSV 的 `id`、`puzzle`、`method`、`sequence` 四列。

模型必须为每个 registered mutant 的每个 construct position 生成 prediction/status。held target 和 evaluator mask只能在 scorer 独立 join 后进入评分；改变 held outcome、error 或 target mask不得改变 point mean、distribution scale、mixture weight或coverage row。

## 4. Mean-first、calibration-second

mean stage 使用严格 method-balanced signed-delta L1，并按 position、mutant、puzzle-method cell、equal-cell 顺序聚合。所有候选使用同一 80 epoch AdamW schedule：learning rate `2e-4`、weight decay `1e-2`、5% warmup、cosine decay、dropout `0.10`、gradient clip `1.0`，支持时使用 BF16。不得 early-stop 或搜索 epoch、loss weight、hidden size、depth、foundation、rank 或 seed subset。

mean 完成后全部冻结。residual stage 只拟合两个同 location 的 conditional Gaussian scale mixture，使用闭式 Gaussian-mixture CRPS 训练 40 epochs。两个 location 都必须逐 key 等于 detach 后的 point mean；calibration 不能改变 mean，也不能向 mean 或 RNA-FM 回传梯度。

## 5. 顶刊级开发 Gate

V4M3 固定 seed 0、20-fold LOPO。五个固定 model families 的 20 folds 全部存在前，不得读取 loss 方向、CRPS、signed-delta MAE、per-puzzle effect 或部分 Gate。

主候选相对 corrected B1 必须同时达到：CRPS relative gain 至少 5%；signed-delta MAE relative gain 至少 5%；两个 paired puzzle CI lower 都大于 0；两指标各至少 16/20 puzzles 为正；leave-one-puzzle 后方向均为正；任一 puzzle 对任一 aggregate effect 的贡献不超过 20%；coverage 100%、failure 0、unexpected key 0；68% 与 95% coverage error 相对 B1 的恶化均不超过 1 个百分点。

归因 Gate 同时要求主候选在两个指标上都超过 capacity-matched null 和 RNA-FM-only，且相应 paired CI lower 都大于 0。scratch 结果必须完整报告。主候选还必须超过 strongest task-matched published comparator；任务、输入权限、split、metric 与 coverage 不匹配的方法不得填入该位置。

V4M3 PASS 才能开放 V4M4。V4M4 固定 seeds 0–4、20 folds、所有五个 families、无选择、无 seed 删除，使用唯一五-seed mixture 重新执行同一 Gate。

## 6. 外部与论文资格

内部正式 PASS 只能写 `HIGH_EFFECT_POST_HOC_DEVELOPMENT_PASS`。它不自动生成 external replication、SOTA、mechanism、practical utility 或 publication readiness。

顶刊证据链必须另立 sealed external amendment，使用新的独立 outcome，在 access 前冻结候选、比较、cluster unit 与阈值。external 必须在 CRPS 和 signed-delta MAE 上各至少提高 3%，两个 cluster-level CI lower 大于 0，并在至少 75% 的最高层 study/publication units 中方向一致。只有该 Gate 通过才能写 `TOP_JOURNAL_EVIDENCE_CHAIN_PASS`。

## 7. GPU 与并发边界

根据 2026-08-23 owner 新授权，v4 可使用物理 GPU0–7 中任何具有足够可用显存的卡，并允许与既有任务共卡；原“仅空闲 GPU6/7”边界被本条显式取代。仍不得抢占、终止、发送信号或修改不属于 v4 的进程。每次启动只选择足以完成当前冻结阶段的卡；OOM 只能触发换卡或降低 cache batch size，不能改变模型、fold、epoch、seed、loss 或 Gate。无限计算预算只表示完成已冻结的 families、folds 与 seeds，不允许转换成开放式架构、参数、loss 或阈值搜索。

低频监控只统计持久会话、日志更新时间、完整 artifact 文件名和非指标错误。任何阶段完成所有 folds 前均禁止查看部分性能方向。

## 8. 失败与终止

工程不变量、参数匹配、foundation freeze、target invariance 或完整输出任一失败，禁止真实 screen。V4M3 任一主 Gate 或归因 Gate 失败，v4 立即关闭并返回 benchmark route；不得增加第二个 v4 主候选。V4M4 失败则保留完整 negative result，不得通过更换 seed、foundation、参数规模或 Gate 恢复。只有内部正式 Gate 通过才允许起草新的 external amendment。
