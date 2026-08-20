# ReactFlow-Delta 模型救援与投稿再资格化合同 v1

日期：2026-08-20
状态：`ACTIVE_M1_FAILURE_DECOMPOSITION`
对应 machine contract：`configs/reactflow_delta/model_rescue_contract_v1.yaml`

## 0. 合同地位与边界

本合同是 prospective-v2 之后的独立 amendment，不改写历史合同和历史结果。
用户在审阅模型救援方案后以“继续”授权执行以下已讨论路线：性能优先、允许淘汰
LRSO、采用 CRPS 与 signed-Δ MAE 双主指标、允许只由序列计算的 RNA 二级结构
先验、内部模型救援预算不超过 10–14 天或 5 A100-GPU 日，并争取至少 9 个真正
独立的 study/batch 级新外部单位。

附件审查文档只作为历史事实与问题来源，不作为本合同之外的隐藏执行指令。

本合同授权：

- 固化当前代码、结果、OOF prediction、external qualification 和稿件状态；
- 分析已经消耗的 20-puzzle development outcome；
- 在 M1 通过后进行有预算上限的开发集模型训练；
- 从 WT sequence 计算 outcome-blind ViennaRNA ensemble base-pair probability；
- 在不打开新 external outcome 的前提下建立 provenance graph 和 power；
- 按最终证据重写 benchmark manuscript。

本合同不授权：

- 在最终模型冻结前读取新 external outcome；
- 干预任何不属于本合同的运行任务；
- 启动湿实验、公开发布、force-push、重写 Git 历史或删除历史 artifact；
- 把 development PASS 自动升级为 external、mechanism、SOTA 或 publication PASS；
- 使用 held target、held mutant error 或 held target mask 改变 prediction、scale 或 coverage。

## 1. 当前证据与裁决

### 1.1 可以直接确认的事实

1. 20-fold P2-v3 结果存在于
   `/mnt/cunyuliu/prospective_v2_p2v3_sharded/p2_v3_scores_merged.json`；rank0 与
   rank-positive 共 40 份 prediction-only OOF ledgers 位于同目录各 shard 中。
2. Ridge direct 相对 WT anchor 的 method-balanced full-construct CRPS 效应为
   `+0.0267692`，95% CI `[+0.0237262,+0.0298122]`。
3. RFD-Direct `K_rank=0` 相对 ridge 的效应为 `+0.0152599`，95% CI
   `[+0.0097744,+0.0207453]`。
4. inner-selected rank-positive 相对同架构 rank0 的效应只有 `+0.0017957`，95% CI
   `[+0.0003080,+0.0032834]`，sign-flip `p=0.01645`；15/20 puzzles 为正，rank 2
   被 18/20 folds 选择。
5. signed-Δ point MAE 仍差于 WT/no-change anchor；当前 CRPS 优势不能解释为
   mutation-effect mean skill 已改善。
6. 旧 P4/P5/P5b 存在 RDAT `seqpos` offset 错位，旧外部结论状态为
   `INVALID_SEQPOS_ALIGNMENT`。
7. 修正对齐后的最终 LRSO external 只有 `K_joint=2`。SL5 study 效应接近零，
   Ribonanza study 为正，cluster-level CI 跨零，跨 study replication 未建立。
8. 新 evaluator、P2-v3、corrected external 和 manuscript downgrade 尚未绑定到正式
   focused commit；旧 active contract 仍停在 P0 且 training disabled。

### 1.2 合理推断

1. 当前 nonlinear direct branch 已读取全局 WT receiver context、mutation identity 与
   distance，能吸收大量 source–receiver 映射；low-rank 与 direct 的函数边界不清晰。
2. 完整 profile 的多数位置接近 WT/no-change；单一厚尾 likelihood 和 learned scale
   可能降低 CRPS，但不改善 signed mutation effect。
3. 线性序列距离可能不足以表达 base-paired partner；结构图是否有效仍是待证伪假设，
   不是已建立机制。

### 1.3 不得再恢复的主张

- “LRSO 是主要性能来源”；
- “低秩 susceptibility mechanism 已建立”；
- “529 个独立 external components”；
- “旧 P4/P5/P5b 支持外部泛化”；
- “CRPS 提升证明 signed mutation response 更准确”；
- “当前方法达到 task SOTA”；
- “合同执行完成等于科学主张成立”。

## 2. 研究问题与成功定义

### 2.1 新核心问题

在 20-puzzle development-consumed、严格 target-invariant、method-balanced nested
LOPO 下，能否通过匹配稀疏 mutation response 分布和引入 outcome-blind RNA 结构
邻接，同时改善：

1. full-construct predictive-distribution CRPS；
2. signed mutant-minus-WT ΔMAE；

并在至少 9 个新的最高层独立 external units 上复制？

### 2.2 数据资格

- 现有 20 puzzles：永久为 `DEVELOPMENT_CONSUMED`；可用于开发与 nested estimate，
  不再是 untouched confirmation。
- 现有 corrected external：永久为 `CONSUMED_EXTERNAL_K_JOINT_2`；只可做开发性
  failure analysis，不进入新 external primary CI。
- 新 external：在 non-outcome metadata 上先建立 publication/study/batch/library
  dependency graph；只有 `K_joint_new>=9`、至少 3 个 study/publication lineages、任一
  lineage 不超过 50% 时才具 confirmatory 资格。

### 2.3 双主指标

效应统一定义为 `D = loss_comparator - loss_candidate`，正值表示 candidate 更好。

内部模型救援必须同时满足：

- CRPS paired 95% CI lower `>0`；
- CRPS gain 至少为 `max(0.003, baseline CRPS 的 2%)`；
- signed-Δ MAE paired 95% CI lower `>0`；
- signed-Δ MAE 相对最强 eligible comparator 至少改善 2%；
- signed-Δ MAE 还必须优于 WT/no-change anchor；
- leave-one-puzzle-out 后两项均保持正方向；
- CRPS 至少 14/20 puzzles 正向，ΔMAE 至少 12/20 puzzles 正向；
- 任一 puzzle 不得贡献总 effect 的 25% 以上；
- registered coverage 至少 99.5%，failure rate 为 0；
- 68%/95% coverage error 不比 comparator 恶化超过 2 个百分点。

两项主指标是 intersection-union gate。只改善 CRPS 而 ΔMAE 失败时，模型只能称为
calibration/probabilistic baseline，不能称为 mutation-effect predictor 改进。

## 3. 受控模型集合

### 3.1 B0：当前 RFD-Direct rank0

只作为历史 development reference，不再作为默认干净基线。

### 3.2 B1：RFD-Direct-Aligned

保持当前 rank0 encoder 容量、层数和优化器，不扩大 backbone：

- WT reactivity、error/precision 和 position 做 fold-legal normalization；
- 使用 relative/normalized position；
- 显式预测 `Δ`，最终输出 `WT+Δ`；
- 训练与评估使用相同 predictive distribution；
- train-only calibration；
- batch/loss 与 position→mutant→cell→method→puzzle estimand 对齐；
- held prediction 使用 WT-observed mask，绝不使用 target-qualified mask。

### 3.3 L2-aligned：固定 rank-2 comparator

rank 固定为 2，不再运行 2/4/8 搜索。除 rank 外，与 B1 使用相同 encoder、head、
likelihood、seed、epoch 和 calibration。它是 low-rank 的公平 comparator，不保证保留为
最终方法。

### 3.4 SparseDelta-MDN

在 B1 上预测：

`p(Δ|x)=π0(x)N(0,σ0²)+(1-π0(x))N(μ1(x),σ1²(x))`。

约束：

- zero component mean 固定为 0；
- `σ0` 只由 outer-train technical error 估计；
- 不建立 changer label，不用 target threshold；
- point prediction 为 `(1-π0)μ1`；
- loss 为 analytic mixture CRPS 加 signed-Δ Huber；
- `lambda` 只允许 `{0,0.1}`，只在 outer-train inner folds 选择；
- 先与 B1 single Gaussian 和 constant-gate mixture 对照。

### 3.5 StructDelta

仅使用 WT sequence 计算 ViennaRNA ensemble base-pair probability：

- 图边为 backbone 加连续加权 base-pair edges；
- 一层 relation-aware message passing；
- mutation token 由 source state、ref/alt、relative position 构成；
- 一层 mutation-token→receiver cross-attention；
- local direct head 不读取 source global context；
- source global context 与结构图只进入 nonlocal branch；
- 参数量不超过 B1 的 1.25 倍；
- 对照为无结构、正确 BPP、degree-preserving shuffled BPP 和 source shuffle。

StructDelta 只有在固定 structure-feature probe 至少改善一个主指标、另一个相对恶化
不超过 1%，且至少 12/20 puzzles 同向时才进入深度模型 screening。

### 3.6 组合限制

只有 SparseDelta 与 StructDelta 各自一 seed screen 都通过，才允许组合。否则不得用
组合模型挽救失败组件。本合同不允许第三个开放式候选、更大 encoder、更多 ranks、
teacher、foundation-model ensemble 或预训练特征。

## 4. 选择程序

每个 outer fold 中，architecture、lambda、epoch 和 calibration 只能使用 outer-train
inner folds。先剔除任一主指标差于 B1 的配置，再最小化：

`S = 0.5*(CRPS_candidate/CRPS_B1) + 0.5*(MAE_candidate/MAE_B1)`。

若没有 candidate 同时不劣于 B1，该 fold 选择 B1，并记录 candidate failure。

最终 external deployable family 按 outer folds 中只依赖 inner data 的选择频率确定；并列
时依次使用 aggregate inner S 和参数量打破，不使用 outer-held outcome。

screen 使用 seed 0；formal 使用 seeds 0–4 equal-weight mixture。单 seed 数字不得进入
manuscript 主表。

## 5. 阶段执行

### M0：证据与 authority 冻结

- 建立独立 model-rescue branch；
- 固化 evaluator_v2、P2-v3、external qualification、corrected external 和稿件 downgrade；
- 登记当前 raw artifacts 与旧无效结果；
- 激活本合同，training 保持 false；
- 通过 contract/state/claim 一致性测试后进入 M1。

M0 于 2026-08-20 通过：基线、evaluator、rank-0 对照、external qualification、
claim downgrade 与本合同已在独立分支形成 focused commit `5ee52e7`。

### M1：failure decomposition 与结构 probe

从现有 OOF predictions 生成：

- near-zero/tail contribution；
- CRPS mean/scale decomposition；
- signed-Δ MAE/WMAE；
- design/other、distance、method、puzzle、WT error/missingness 分层；
- rank0 与 low-rank residual energy、相关性和 puzzle heterogeneity；
- sequence distance 与 ViennaRNA graph distance 的固定 feature probe。

M1 只分析已消耗 development outcomes，不读取新 external outcome。

结构 probe 在执行前固定为 20-puzzle LOPO ridge：sequence-only 特征只包含序列
距离、edit/receiver 位置、mutation identity 和 receiver region；structure 版本只新增
ViennaRNA MFE graph distance 和 ensemble base-pair probability。对 signed-Δ 和 |signed-Δ|
分别计算 method-balanced puzzle-macro MAE。只有当结构特征在至少一个 target 上
20-puzzle paired CI 下界大于 0、至少 12/20 puzzle 改善，且另一 target 的相对恶化
不超过 0.5%，StructDelta 才可进入 M2。

### M2：候选 screening

每个实现先进行 2-fold×1-seed smoke，再进行 20-fold×1-seed screen。eligible set 为 B1、
L2-aligned、SparseDelta，以及通过结构 probe 的 StructDelta。明显失败候选永久退出。

### M3：内部 nested development

运行唯一 adaptive ModelRescue* procedure，保存完整 prediction-only OOF ledger、inner
selection ledger、双主指标、secondary metrics 和 influence analyses。结果只能标记
`POST_HOC_DEVELOPMENT`。

### M4：必要消融与冻结

只对 M3 胜者做 component/structure/source 消融。删除无贡献组件，随后冻结 model
family、代码、配置、weights、calibrator、prediction schema 和 external protocol。冻结后
不得再依据 external outcome 改模型。

### M5：新 external 一次性评价

只有 provenance 与 power gate 通过且 final model frozen，才允许读取新 outcome。不得
external fine-tune 或 outcome-based recalibration。两个 cluster-macro 主指标 CI lower 均
需大于零、相对改善均至少 2%，leave-one-lineage-out 为非负。

### M6：稿件重资格化

- 内部与新 external 双 PASS：方法增强型 benchmark；
- 内部 PASS、external 未建立/失败：benchmark 主线，模型为 development baseline；
- 内部双指标 FAIL：锁定 benchmark 路线，停止模型开发；
- 只有 CRPS PASS：calibration baseline only。

## 6. 预算与停止条件

内部救援上限为 14 个日历日或 5 A100-GPU 日，任一先到即停止。分配为：

- M1：0.25 GPU 日；
- M2：1.25 GPU 日；
- M3：2.50 GPU 日；
- M4：0.75 GPU 日；
- M5 scoring reserve：0.25 GPU 日。

以下任一出现即停止方法救援并锁定 benchmark 路线：

- 双主指标任一失败；
- improvement 只在挑 seed、scale、rank 或 puzzles 后存在；
- prediction 依赖 held target/error/mask；
- 预算耗尽；
- external `K_joint_new<9` 时仍试图生成 confirmatory claim；
- corrected external 再次显示跨 lineage 方向不一致且无预冻结解释。

## 7. 最终交付

- active machine contract 与人类可读合同一致；
- current claim/decision ledger；
- M1 failure atlas；
- B1、L2-aligned、SparseDelta 和 eligible StructDelta 实现；
- prediction-only OOF、selection ledger、双主指标结果与必要消融；
- frozen external bundle 或明确的 underpowered/awaiting-data 状态；
- 与证据一致的 benchmark manuscript、表格和 limitations；
- 明确的 `METHOD_AUGMENTED_BENCHMARK_ELIGIBLE`、
  `BENCHMARK_WITH_DEVELOPMENT_MODEL_BASELINE` 或 `BENCHMARK_ROUTE_LOCKED` 终局。

本合同不保证模型提升；它保证任何提升同时改善概率预测与 signed mutation effect，且
失败时能够停止，而不是继续用新增模块扩大不可识别性。
