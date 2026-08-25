# ReactFlow-Delta Model Rescue v11：point representation amendment

**当前状态：V11M0 合同已冻结；仅允许实现与聚焦测试，训练保持关闭。**

## 1. 合同地位

V11 是独立 amendment，不修改 V1–V10 的任何终局。V9 仍因 4.2264% CRPS 改善未达到 5% 门槛而失败；V10 仍因相对公平 feature41-asymmetric comparator 的 task CRPS 改善 3.2776% 未达到 5% 而失败，V10M4 永久关闭。

V11 只执行 score 出现前已登记的 post-V10 branch 5：残差非对称性已经被 matched symmetric null 识别，但固定 V8 point representation 是剩余瓶颈。V11 不读取新的 external outcome，不恢复任何旧 amendment，也不允许通过降低 Gate 把接近门槛改写为 PASS。

## 2. 已确认事实与科学假设

- `LOCATED_EVIDENCE`：corrected feature41 的 signed-delta MAE 为 0.19115973，absolute-delta MAE 为 0.15302427。
- `LOCATED_EVIDENCE`：V8 MeanAligned point 的 signed-delta MAE 为 0.17517273，相对 feature41 改善 8.3632%，20/20 puzzles 正向；但 point magnitude 明显收缩。
- `LOCATED_EVIDENCE`：V10 的 distribution absolute-delta MAE 为 0.13114777，相对 feature41 改善 14.2961%；task CRPS 为 0.12629969，相对公平 feature41-asymmetric 的 0.13057956 改善 3.2776%。
- `LOCATED_EVIDENCE`：V10 asymmetric head 相对 parameter-matched symmetric null 改善 CRPS 4.7296%，说明非对称残差建模不是虚假容量增益。
- `HYPOTHESIS`：强 feature41 已提供跨 puzzle 稳定的一阶点估计；让更有容量的上下文模型只学习它的 signed residual，会比从零重学完整 delta 更容易泛化，并能把 V10 剩余约 1.78% 的相对 CRPS 缺口补上。
- `FALSIFIER`：point、matched-null attribution、task CRPS、absolute、coverage 或完整性 Gate 任一失败。

## 3. 为什么不是 foundation、rank 或更自由的 calibration

V4 的 RNA-FM 双塔同时改变 foundation、pair tower、容量和 response tower，其旧科学分数后来又被 target-identity 缺陷失效；它不能作为可靠正证据，也不值得原样重复。V7 在精确修正 identity 后，RiNALMo dependency6 对 signed-delta 的改善只有 0.0378%，CI 跨零。旧 low-rank operator 没有建立 signed mean 增量。再次更换 foundation、搜索 rank 或释放 V10 location median，分别重复了已经失败的路线，或重新引入 mean/calibration 混淆。

V11 的新增能力非常窄：当前 B1 的两个 `RelativeAttentionBlock` 没有 position-wise FFN，mutation direct head 只有一层 hidden-64 ReLU。V11 用 4 个 d=192、8-head、FFN=768 的 pre-norm relative-attention blocks 建立 WT context，再用两层 hidden-256 mutation-conditioned head 预测 feature41 residual。模型约为数百万参数，而不是再建一个 35–45M foundation/pair tower。

## 4. Primary 与精确 matched null

两个模型具有完全相同的输入、张量形状、可训练参数、初始化、训练 cell、cell 顺序、optimizer 和 epochs：

- `v11_feature41_anchored_context_residual`：
  \[
  \hat\Delta=\hat\Delta_{\mathrm{feature41}}+r_\theta(x,\hat\Delta_{\mathrm{feature41}}).
  \]
- `v11_unanchored_context_null`：
  \[
  \hat\Delta=r_\theta(x,\hat\Delta_{\mathrm{feature41}}).
  \]

Null 仍在 head input 中看到完全相同的 feature41 scalar；唯一差异是不可训练的 skip multiplier 从 1 变为 0。因此 candidate 超过 null 才能把增量归因于“强基线锚定后学习 residual”的归纳偏置，而不是参数量或额外信息权限。

两者都从头训练，不 warm-start V8 mean。feature41 必须来自对应 outer fold 的 train-only TIC2A model。held target、error 和 qualified target mask 不得进入 prediction path。

## 5. Point 训练

输入保留 WT sequence、construct-standardized WT reactivity、WT error precision、observed token、normalized position、region、ref/alt、signed distance，并加入 outer-train feature41 point。正式设置固定为：

- d=192，heads=8，blocks=4，FFN=768，dropout=0.1；
- mutation residual head hidden=256、两层 GELU，final linear 零初始化；
- exact method-balanced signed-delta L1；
- Adam，lr=1e-3，weight decay=0，clip=5；
- 40 epochs，无 early stopping、无 epoch/loss/width/depth 搜索；
- 每个 epoch 每个 outer-train puzzle×method cell 恰好一次，position→mutant→cell 聚合。

## 6. Calibration 保持 V10 不变

Point 训练后冻结。Candidate、null 和 feature41 comparator 分别拟合同一个 V10 `MedianAsymmetricResidual` family：input width 244，包含 feature41 basis-41、对应 point、abs(point) 和冻结的 V8 direct features-201；hidden=256，两个 Gaussian components，location allocation 257 outputs，mixture CDF 在 point 处严格等于 0.5。使用 method-balanced closed-form Gaussian-mixture CRPS、Adam 1e-3、40 epochs。Calibration gradient 不得进入 point 模型。

Seed-0 feature41 comparator 必须在 1e-7 内 replay V10 feature41-asymmetric；否则不允许科学评分。

## 7. V11M2 real-data smoke

只运行 folds 0/1、seed 0、3+3 epochs，artifact 为 prediction-only，禁止科学评分。必须机械验证：candidate/null 参数与初始 state 完全相同、skip 是唯一差异、有限 loss/gradient、point freeze、CDF(point)=0.5、target/error/mask invariance、full registered output、coverage 100%、failure 0、unexpected keys 0。

## 8. V11M3 单 seed 顶刊 screen

固定 seed 0、20 folds、40+40 epochs；20/20 完整前不得读取 loss 方向、CRPS、signed-delta、per-puzzle effect 或部分 Gate。完整后只允许合并一次、评分一次、qualify 一次。

Primary 必须全部满足：

- signed-delta MAE 相对 feature41 改善至少 10%，相对 V8 再改善至少 2%，相对 matched null 至少 1%；对应 CI lower 均大于 0，positive puzzles 分别至少 16/20、14/20、14/20；
- point absolute-delta MAE 相对 feature41 改善至少 1%，CI lower>0，至少 14/20 正向；
- task CRPS 相对 fresh fair feature41-asymmetric 改善至少 5%，相对 terminal V10 至少 1.5%，相对 matched null 至少 1%；对应 CI lower 均大于 0，positive puzzles 分别至少 16/20、14/20、14/20；
- distribution absolute-delta 相对 feature41 改善至少 12%，且相对 V10 不劣于 0.5%；
- headline effect 全部 LOO 保持正向，任一 puzzle 贡献不超过 20%；
- 68%/95% coverage error 相对 feature41 不恶化超过 2 个百分点；
- prediction coverage=100%、failure=0、unexpected keys=0。

只有全部 Gate exact PASS 才开放 V11M4。

## 9. V11M4 五 seed formal confirmation

固定 seeds 0–4、20 folds、40+40 epochs，无 selection。Candidate、matched null 和 feature41-asymmetric 均在相同 fold×seed universe 中重建；最终是等 seed mixture，不删失败 seed、不报告 best seed。正式 mixture 重复 V11M3 中相对 feature41、V8、matched null 的 point、absolute、task-CRPS、CI、puzzle、LOO、influence、coverage 和完整性 Gate；至少 4/5 seeds 的 signed 与 task CRPS 方向必须为正。V11M3 相对 terminal V10 的 1.5% CRPS 及 absolute non-inferiority 必须已经通过，但 terminal V10 seed-0 在 formal 阶段只作为历史上下文，不冒充五-seed comparator，也不与五-seed mixture 生成新的显著性 Gate。

## 10. 证据资格和停止条件

V11M3 或 V11M4 任一 Gate 失败都保持为完整 negative/development result，不降低阈值、不改 architecture、不追加候选。即使 V11M4 PASS，资格也只到 `POST_HOC_DEVELOPMENT_PASS`；external replication、SOTA、mechanism 和 publication readiness 均不自动建立，必须另立 sealed external amendment。
