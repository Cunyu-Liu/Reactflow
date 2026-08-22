# ReactFlow-Delta Model Rescue v3 Full-Sequence Coordinate Correction Amendment

## 1. 合同地位

本 amendment 只纠正 OpenKnot M2 的 mutation coordinate frame，不修改 Model Rescue v1/v2 的历史终局，不修改 v3 的模型、loss、epoch、seed、候选、Gate 或 R3M4 门槛。

旧 R3M3 在任何完整 fold 产生前即被中止，登记为 `R3M3_INVALID_BEFORE_FIRST_FOLD_COORDINATE_FRAME`，不是 PASS 或 FAIL。旧 waiter、runner 和监控均已停止；旧 artifact 保留但不得继续训练、合并、评分或作为修复后 expert reuse。

## 2. 已确认错误

官方 OpenKnot 数据定义中，`mutA` 是 designed sequence 内的 1-based 位置；`sub_start` 是 designed sequence 在 full padded sequence 中的起始位置。仓库 `M2Universe` 从 mutant row ID 解析出的 `_mm_<n>_` 是 designed sequence 内的 0-based位置，但原实现直接把它当作 full sequence index。

冻结的正确映射为：

\[
\text{full\_pos}=\text{sub\_start}-1+\text{design\_pos}.
\]

真实 v4.5.2 数据的 13,976 条 registered mutants 全部满足：

- raw WT→mutant sequence Hamming distance = 1；
- 唯一变化位置等于上述 `full_pos`；
- WT full sequence 在 `full_pos` 等于 ref；
- mutant full sequence 在 `full_pos` 等于 alt；
- `mutA = design_pos + 1`。

旧实现只有 3,726/13,976 条记录在错误的直接索引位置偶然匹配 ref，因此旧模型的大多数 edit-site hidden、distance、WT edit anchor 和 edit region 都对应错误位置。

## 3. 双坐标契约

- `design_pos`：designed sequence 内的 0-based位置，只用于 raw mutant ID、mutation key 和 biological scoring key。
- `full_pos`：full padded construct sequence 内的 0-based位置，只用于 encoder edit index、signed distance、WT edit anchor、region 和 graph/contact indexing。
- receiver `position`：继续是 full construct position。
- target full-profile lookup：继续用 `design_pos` join raw mutant row。

不得保留含义不清的单一 mutation coordinate 继续跨这两种用途传播。

## 4. 证据资格影响

- V1/V2 历史结论文本保持不可变；本 amendment 不把历史 FAIL 改写为未运行或 PASS。
- 但所有 pre-correction B1、MeanAligned 和 v3 smoke artifact 都标记为 `INVALID_COORDINATE_FRAME / NOT_REUSABLE_FOR_CORRECTED_RUN`。
- 旧分数不能用于评估修复收益，因为 comparator 和 candidate 都必须在修复后的同一坐标协议下从头重建。
- 修复本身只能先获得 engineering qualification；模型改善仍必须重新通过完整 R3M3 与原始 R3M4 Gate。

## 5. 执行阶段

1. `R3C0`：冻结真实数据证据；在 0/20 folds 时停止旧 waiters、runner 和自动监控。
2. `R3C1`：实现 `design_pos/full_pos` 分离，修复 active B1/MeanAligned/V3 和 evaluator-adjacent 路径，运行坐标、key、target-invariance 回归测试。
3. `R3C2`：在真实 P01/P02 上运行 corrected engineering smoke，不使用其分数选模。
4. `R3C3`：用修复后的坐标从头重建 seed-0、20-fold B1 与 MeanAligned experts；禁止复用旧 checkpoint/prediction。
5. 只有 R3C3 完整后，才用同一 v3 candidate、同一 frozen Gate 重新运行 R3M3。
6. 只有修复后的 R3M3 Mean Gate 与 Calibration Gate 同时 PASS，才允许原始 R3M4。

R3C3 使用唯一的 expert-only 执行路径：

- runner：`scripts/reactflow_delta/run_model_rescue_v3_expert_rebuild.py`；
- qualifier：`scripts/reactflow_delta/qualify_model_rescue_v3_expert_rebuild.py`；
- 每个 outer fold 从头训练同一 B1 与 MeanAligned，固定 seed 0、40 epochs、Adam、learning rate `1e-3`、weight decay `0`；
- 不运行 gate、residual calibration、rank、architecture 或 loss search；
- 每 fold 只输出两个 checkpoint、训练历史和 prediction-only expert ledger，不计算 held score；
- 20 folds 可以在 GPU0–5 上按不重叠 fold shards 并行，但不得抢占无关任务；
- 只有 qualifier 确认 folds 0–19 完整、loss finite、checkpoint 完整且每个 held puzzle key universe 精确一致，才开放 corrected R3M3；
- R3M3 加载 R3C3 checkpoint 后才生成 baseline/candidate held predictions 和分数；任何 partial fold score 仍禁止查看。

## 6. 不可协商边界

- 不改变模型容量、结构、loss、epoch、seed、threshold、Gate 或统计门槛；
- 不读取旧或新 R3M3 的部分分数；
- 不访问新的 external outcome；
- 不覆盖旧 artifact；corrected artifact 必须使用新目录；
- 不干预无关 GPU 任务；
- 不用坐标 bug 修复自动生成 SOTA、mechanism 或 publication PASS。

本 amendment 的目的不是用数据修复替代模型创新，而是让模型第一次接收到正确的 mutation location；在此之前，任何 backbone 或 gate 比较都无法识别真实架构上限。
