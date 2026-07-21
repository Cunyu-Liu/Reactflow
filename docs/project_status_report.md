# ReactFlow 项目阶段汇报

本文档记录 ReactFlow 当前已经完成的工程、模型、数据、实验和审计状态。它是面向项目推进与论文规划的阶段性汇报，详细实验账本见 [`docs/ablation_experiment_filled.md`](ablation_experiment_filled.md)，cross-family 改进路线见 [`docs/cross_family_improvement_plan.md`](cross_family_improvement_plan.md)，SOTA 追赶目标、same-split 指标对齐和 scale-up 执行清单见 [`docs/sota_catchup_goals_and_todo.md`](sota_catchup_goals_and_todo.md)。

## 1. 总体阶段判断

ReactFlow 已经从概念原型推进到 **真实公开数据 + 远端 GPU + full-scale 队列 + 可审计实验闭环** 阶段。

当前项目已经具备：

- 真实 eFold/RNAndria Dryad 数据接入；
- Rfam clan 与 MMseqs family-disjoint split；
- RibonanzaNet2 Kaggle checkpoint 的真实 frozen feature export；
- torch full-data 训练后端；
- warm-start / contact auxiliary / MMseqs final 队列；
- profiling、resource audit、queue progress audit、final result contract；
- cross-family 指标审计与后续自动接力 watcher；
- README、数据治理、实验 checklist、消融账本和 reproducibility manifest。

但项目还 **不能标记为完整达成顶刊级目标**。三份 final-result 文件已经生成并通过 7-tier 内容契约；当前唯一 hard gate 已从“缺文件”转为更本质的科学问题：`cross_family_claim_ready=false`，即 MMseqs / novel-clan 条件下的准确生成能力仍未达到我们设定的 claim gate。

## 2. 科学问题与模型定位

ReactFlow 的核心科学问题不是再做一个普通的 `sequence -> dot-bracket` 预测器，而是：

> 在 RNA 结构标签稀缺、chemical probing 只提供结构集合的一阶观测、并且训练/测试家族严格隔离的条件下，能否学习一个物理约束的 RNA 二级结构分布 `p_theta(S | x)`，使其既能解释 DMS/SHAPE reactivity，又能在 Rfam/MMseqs-disjoint 未知 family 上泛化？

因此，ReactFlow 的特色是：

- 预测结构分布，而不是只预测单一结构；
- 显式建模 chemical probing forward operator；
- 在训练、采样、评估中保留 RNA 二级结构合法性；
- 把 family-disjoint / cross-family 泛化作为主战场；
- 以 `novel_clan` 和 MMseqs split 作为最终主结果依据。

## 3. 已完成的算法与工程模块

当前已实现并通过测试/审计的核心模块包括：

| 模块 | 当前状态 |
|---|---|
| DFM partner-class 建模 | 已实现，每个位置预测 unpaired 或 paired-to-j 的类别分布 |
| `PairwiseDenoiser` | 已实现手写 forward/backward，并有 finite-difference 测试 |
| reactivity forward operator | 已实现 `rhat_i = a_i q_i + c_i` |
| thermodynamic prior / guidance | 已实现 pilot 与 guidance scan |
| contact-map denoising auxiliary | 已实现 `P_ij = 0.5(pi_i[j+1] + pi_j[i+1])` |
| heteroscedastic ensemble calibration | 已实现 mean/variance 双通道 logit gradient |
| RibonanzaNet2 frozen adapter | 已实现 `a_i = W h_i + b` 与 split-gradient SGD |
| long-sequence windowing / bucketing | 已实现并进入 cache / split / training 路径 |
| distance-bin evaluation | 已实现 short / medium / long-range pair metrics |
| family-balanced sampler | 已实现 `--family-balanced-batches`，用于 RF-CF3 |
| OOM/NaN retry guard | 已实现 batch ladder `16 -> 8 -> 4 -> 2 -> 1` |

算法审计最新状态：

- `public_nodes=282`
- `strict_ready=true`
- `placeholder_bodies=0`
- `missing_docstrings=0`
- `missing_math_markers=0`
- `missing_complexity=0`

## 4. 数据与 split 状态

已接入的公开数据源：

- RNAndria / eFold Dryad: DOI `10.5061/dryad.79cnp5j95`
- RibonanzaNet2 Kaggle model: `shujun717/ribonanzanet2/PyTorch/alpha/1`
- Rfam official clan metadata
- MMseqs2 sequence-identity cluster

当前 full-scale cache 证据：

| Cache | Rows |
|---|---:|
| `efold_train.jsonl` | 307,641 |
| `archiveII.jsonl` | 2,052 |
| `PDB.jsonl` | 333 |
| `viral.jsonl` | 97 |
| `lncRNA.jsonl` | 289 |
| `human_mRNA.jsonl` | 6,627 |

MMseqs final split 已完成：

| Split | Count |
|---|---:|
| train | 228,282 |
| val | 16,606 |
| test | 16,606 |
| novel | 46,147 |

该 split 使用 `cluster_method="mmseqs"`，目标是作为论文主表的最终无泄漏 split。Exact split 只作为工程诊断和 warm-up。

## 5. 已完成实验结果

已经完成并通过 final-result 内容契约的结果：

| Result file | 状态 | 说明 |
|---|---|---|
| `warm_rfam_current_exact_results.json` | ready | 21 行，覆盖 7 个 eval tiers |
| `contact_rfam_current_exact_results.json` | ready | 7 行，覆盖 7 个 eval tiers |
| `mmseqs_final_results.json` | ready | 14 行，覆盖 7 个 eval tiers，包含 RF-M0-base 与 RF-M1-warm |

当前已完成 exact split 的 cross-family 结果仍不够好：

| Run | novel_clan mean F1 | 结论 |
|---|---:|---|
| `RF-A1-warm` | 0.0624 | 当前 best，但未达 claim gate |
| `RF-A2-adapter16` | 0.0515 | 未超过 RF-A1 |
| `RF-A2-adapter4` | 0.0344 | 未带来 OOD 增益 |
| `RF-A3-contact` | 0.0428 | contact auxiliary 默认配置未达标 |

MMseqs final split 的主结果已经落盘，但同样没有达到 cross-family claim gate：

| Run | in_clan mean F1 | novel_clan mean F1 | novel_clan mean MCC | 结论 |
|---|---:|---:|---:|---|
| `RF-M0-base` | 0.0271 | 0.0267 | 0.0248 | MMseqs base 极弱，只能作为工程 baseline |
| `RF-M1-warm` | 0.0456 | 0.0447 | 0.0444 | warm-start 有提升，但仍远低于 claim gate |

当前 cross-family audit：

- `best_novel_mean_f1=0.0624`
- `best_generalization_gap=-0.023`
- `cross_family_healthy=true`
- `cross_family_claim_ready=false`

这意味着：评估链路已经正式把 cross-family 纳入主指标，但当前模型还不能声称已经做到 cross-family 准确生成。

## 6. 为什么当前结果离 SOTA 很远

当前结果和 SOTA 差距非常大，这不是单个 bug 或单个超参数能解释的，而是由 **模型阶段、训练目标、结构解码、数据切分和 cross-family 难度** 共同造成的。当前结果应该被定位为 full-scale 工程 baseline，而不是论文级主模型。

### 6.1 差距量级

当前完成的最好 exact-split `novel_clan` mean F1 只有 `0.0624`；MMseqs split 上 `RF-M0-base` 的 `novel_clan` mean F1 为 `0.0267`，`RF-M1-warm` 为 `0.0447`。公开文献中的强模型在常见 benchmark 上通常报告远高得多的 F1，例如 RNADiffFold source table 中 ArchiveII F1 约 `0.880`，bpRNA TS0 F1 约 `0.711`；eFold/RNAndria 文献在 viral / lncRNA 等 public tiers 上也报告明显更高的可用结构预测性能。

这些数字不能简单视作同 split 的直接 leaderboard 对比，但它们说明一个事实：**ReactFlow 当前实现还没有进入 SOTA 性能区间**。目前的价值主要是数据、训练、监控、评估和可复现闭环已经搭起来，而不是性能已经接近顶刊主结果。

### 6.2 主要问题诊断

| 问题 | 当前状态 | 对性能的影响 | 优先修复方向 |
|---|---|---|---|
| 模型容量太小 | 当前 full-scale 主 run 多为 `hidden_size=8`、1 epoch、轻量 pairwise denoiser | 难以学习复杂 RNA family motif 和长程 base-pair dependencies | 提升 hidden size、adapter dim、训练 epoch，并做多 seed |
| 训练目标没有充分启用项目特色 | 多数 full run 使用 `lambda_react=0`，thermo/calib 默认关闭；contact 只跑了一个默认强度 | 当前模型更接近 structure-only weak baseline，没有真正用 probe-calibrated ensemble 优势 | 系统 sweep `lambda_contact`、`lambda_calib`、`lambda_thermo`，只在真实 profile subset 上启用 reactivity |
| frozen encoder 接入太弱 | RibonanzaNet2 只作为 frozen single-token features，adapter 是线性小头 | 大模型表征没有充分转化为 pair/contact 结构先验 | 增大 adapter、加入 pair-aware adapter、做 contact distillation 或 multi-encoder adapter |
| 解码/投影过于保守 | 当前输出经过 legality projection，且模型本身 pair scores 很弱 | 可能偏向低 pair count 或局部 pair，导致 F1 极低 | 诊断 pair count、distance-bin long-range F1，加入 long-range reweighting 和 thermo-guided decoding |
| 数据多样性/复杂度尚未通过训练 gate | 已生成 `data_diversity_audit.json/md` 与 `source_family_length_manifest.json`，覆盖远端 exact/MMseqs/public tiers 共 `915,715` records；但 public/eFold train tiers 缺少 clan/family metadata，MMseqs split 的 fallback pseudo-clan fraction 为 `1.0000` | 按 eFold/RNAndria 结论，单纯扩大模型或样本数不足以跨 family 泛化；metadata 缺口会让 balanced sampling 退化成 source_id/cluster 近似 | 先做 metadata join 与 pseudo-clan 清洗，再按 source/family/length/complexity 做 curriculum 和 balanced sampling |
| cross-family split 更严格 | MMseqs split 去除了 family/sequence leakage | 真实 OOD 难度暴露，随机 split 或 family-overlap 的虚高性能消失 | 以 MMseqs `novel_clan` 为主指标，做 family-balanced sampling |
| 长序列/windowing 仍有信息割裂 | 长 RNA 用 local windows 处理，跨窗口 pair 被省略 | lncRNA/human_mRNA 和 long-range pair recovery 受损 | divide-and-stitch、跨窗口 context pooling、long-range auxiliary |
| 训练还只是 1 epoch 工程闭环 | 当前 full-scale run 主要验证可跑、可监控、可落盘 | 远未达到充分收敛，更不能支撑 SOTA claim | 多 epoch、多 seed、validation selection、bootstrap CI |
| baseline 尚未同协议完成 | eFold/RNAndria same-split rerun 已启动但 `baseline_efold_results.json` 尚未写出完整 rows；RNADiffFold/RibonanzaNet2-derived 仍未同协议复现 | 完成前无法严谨证明差距或优势，只能做方向性判断 | 等 eFold full rerun 完成后纳入 SOTA 表；继续复现 RNADiffFold/RibonanzaNet2-derived，未完成前保留 cited/local 分栏 |

### 6.3 最核心的性能瓶颈

当前最大问题不是“没有跑完最后一个文件”，而是 **模型现在还没有学到强 pair-recovery 能力**。这体现在：

- `archiveII` mean F1 约 `0.03`；
- `lncRNA` / `human_mRNA` mean F1 约 `0.01` 到 `0.03`；
- `novel_clan` mean F1 最高约 `0.0624`；
- MMseqs base `novel_clan` mean F1 约 `0.0267`，warm-start 后也只有 `0.0447`。

这些结果说明当前模型大概率处在“合法但弱结构生成”的阶段：它能跑通结构分布训练和合法性约束，但没有足够的模型容量、pairwise inductive bias、long-range recovery 和训练强度去接近 SOTA。

### 6.4 不是主要问题的部分

以下部分目前不是主要瓶颈：

- 数据来源与 public provenance：已接入 Dryad / Kaggle / Rfam / MMseqs；
- split 泄漏控制：MMseqs split 已构建并验证；
- 运行稳定性：runtime/resource/queue progress 当前 healthy；
- 文档与审计：algorithm doc audit、preflight、manifest 均已闭环；
- final-result 契约：warm/contact/MMseqs 三份结果均已通过 7-tier 内容契约。

换句话说，工程基础已经基本打通；真正的短板是 **模型性能与 cross-family 泛化能力**。

### 6.5 直接改进路线

接下来要优先做能直接提升 `novel_clan` 的实验，而不是继续只堆工程：

1. **复盘 RF-CF3 family-balanced**：该路线已完成但 `novel_clan_mean_f1=0.0286`，说明当前 family-balanced batch 还没有解决 OOD pair recovery。
2. **复盘 MMseqs weak baseline**：把 `RF-M0-base` 与 `RF-M1-warm` 的 `novel_clan`、distance-bin 和 family-macro 指标作为下一轮优化的基准。
3. **复盘 RF-CF1 contact-strong sweep**：`lambda_contact in {0.1, 0.2, 0.4, 0.8}` 已完成，best `lambda=0.8` 为 `novel_clan_mean_f1=0.0435`，仍低于 RF-M1-warm。
4. **跟踪 RF-CF2 long-range diagnostic/reweighting**：official `w=2` 已接力启动；重点看 `distance_bins.novel_clan.long.mean_f1`，若长程 pair 接近 0，则必须升级 pair head/window stitching。
5. **eFold-inspired data diversity curriculum**：`data_diversity_audit` 已证明 metadata gate 未通过；下一步先做 family/clan metadata join、pseudo-clan 清洗，再分阶段加入 pri-miRNA、human_mRNA、viral fragments、lncRNA/domain windows。
6. **提升模型容量**：把 `hidden_size=8` 和小 adapter 扩到更合理配置，记录参数量、显存、速度和 F1；若容量提升无 OOD 增益，优先回到数据多样性而不是继续加大模型。
7. **启用项目特色损失**：在真实 profile subset 上启用 `lambda_react` / `lambda_calib`，避免 ReactFlow 退化成普通 contact predictor。
8. **多 seed 与 baseline**：对 best config 做多 seed，并尽快建立同 split baseline。

结论：当前结果差，是因为模型还处于弱 baseline 阶段；但项目已经具备把这个差距系统性拆解并逐项攻克的实验基础。

## 7. 远端运行状态

当前远端状态已经从 “等待 MMseqs final” 进入 RF-CF 自动接力链路：

- `RF-M0-base_mmseqs_torch_full_data_e1_bs16` 已完成；
- `RF-M1-warm_mmseqs_torch_full_data_e1_bs16` 已完成；
- `mmseqs_final_results.json` 已生成并通过内容契约；
- `RF-CF3-family-balanced_mmseqs_torch_full_data_e1_bs16` 已完成并生成 `cross_family_balanced_results.json` / `cross_family_balanced_metric_audit.json`；
- RF-CF3 未达标：`novel_clan_mean_f1=0.0286`，`novel_clan_mean_mcc=0.0271`，`gap_mean_f1=0.0003`，`cross_family_claim_ready=false`；
- `RF-CF1-contact-lam0p1/lam0p2/lam0p4/lam0p8` 已完成并生成 `cross_family_contact_sweep_results.json`；best `lambda=0.8` 的 `novel_clan_mean_f1=0.0435`、`novel_clan_mean_mcc=0.0430`，仍低于 RF-M1-warm `0.0447` 和 early gate `0.15`。
- RF-CF2 long-range official watcher 已自动接力启动 `w=2`；RF-CF2 early `w=4/w=8` 已完成，`w=8 early` 为 `novel_clan_mean_f1=0.0404`、`novel_clan_mean_mcc=0.0397`，但仍未通过 claim gate。
- eFold/RNAndria same-split baseline 续跑已恢复并切到双 GPU：GPU6 PID `986449` 写 `in_clan`，GPU7 PID `1124346` 写 `novel_clan`。截至 2026-07-17 15:00 CST 已写出 `PDB` predictions `333/333`、`archiveII` predictions `2052/2052`、`human_mRNA` partial predictions `33/6627`、`in_clan` partial predictions `1632/16606` 和 `novel_clan` partial predictions `227/46147`；完整结果落盘前不进入 SOTA 表。

当前 readiness:

- `pass=23`
- `fail=1`
- 唯一 failure: `cross_family_metrics`
- `ready_for_goal_completion=false`

当前健康审计：

| Audit | 状态 |
|---|---|
| runtime health | RF-CF active healthy, `audited_run_count=13`, `pass=68`, `warn=1`, `fail=0` |
| system resource | RF-CF active healthy, `process_count=6`, `pass=8`, `warn=2`, `fail=0` |
| queue progress | healthy；official RF-CF2 `w=2` 正在运行；整体 `pass=77`, `warn=4`, `fail=0` |
| active eval progress | healthy with warning：RF-CF2/RF-CF5 active run 仍在训练/评估阶段；整体 `pass=2`, `warn=1`, `fail=0` |
| queue preflight | `pass=151`, `fail=0` |
| final queue | `final_results_ready=true`, `pass=4`, `fail=0` |
| reproducibility manifest | `file_count=295` |

## 8. 自动接力与后续实验

RF-CF3 自动接力 watcher 已经完成首轮：

- 脚本：`scripts/run_cross_family_after_mmseqs_final.sh`
- 已检测到 `mmseqs_final_results.json`；
- 已审计出 `cross_family_claim_ready=false`；
- 已自动启动 `RF-CF3-family-balanced`；
- 已输出 `cross_family_balanced_results.json` 和 `cross_family_balanced_metric_audit.json`；
- 结果仍未达标，`novel_clan_mean_f1=0.0286`，低于 RF-M1-warm 的 `0.0447`。

这说明自动化闭环已经跑通：MMseqs weak baseline 不达标后，系统没有停在结果汇总，而是自动进入下一轮 cross-family 定向优化。

RF-CF1 contact-strong 自动接力也已接上：

- 脚本：`scripts/run_contact_sweep_after_cross_family_balanced.sh`
- 当前状态：`lambda_contact=0.1/0.2/0.4/0.8` 已完成并落盘，best `lambda=0.8` 仍未改善到 RF-M1-warm 以上；
- 已完成行为：RF-CF3 结果落盘后已审计 `cross_family_claim_ready=false`，并自动启动 `RF-CF1-contact-strong`；
- 默认 sweep: `CONTACT_SWEEP_LAMBDAS="0.1 0.2 0.4 0.8"`；
- 输出 `cross_family_contact_sweep_results.json` 和 `cross_family_contact_sweep_metric_audit.json`。

RF-CF2 long-range 自动接力也已接上：

- 脚本：`scripts/run_long_range_after_contact_sweep.sh`
- 当前状态：已检测到 `cross_family_contact_sweep_results.json`，并于 2026-07-16 09:59 CST 启动 official `RF-CF2-long-range-w2_mmseqs_torch_full_data_e1_bs16`；
- 行为：RF-CF1 结果落盘后先审计 `cross_family_claim_ready`；
- 如果 RF-CF1 仍未达标，自动启动 `RF-CF2-long-range`；
- 默认 sweep: `LONG_RANGE_WEIGHTS="2 4 8"`，`LONG_RANGE_MIN_DISTANCE=24`；
- 输出 `cross_family_long_range_results.json` 和 `cross_family_long_range_metric_audit.json`。

RF-CF5 capacity scale-up 自动接力也已接上：

- 脚本：`scripts/run_capacity_after_long_range.sh`
- 当前状态：等待 `cross_family_long_range_results.json`；
- 行为：RF-CF2 结果落盘后先审计 `cross_family_claim_ready`；
- 如果 RF-CF2 仍未达标，自动启动 `RF-CF5-capacity`；
- 默认 grid: `CAPACITY_GRID="16:16 32:16"`，分别对应 `hidden_size:adapter_dim`；
- 输出 `cross_family_capacity_results.json` 和 `cross_family_capacity_metric_audit.json`。

## 9. 当前缺口

项目距离顶刊级完成还差：

1. `cross_family_claim_ready=true`，至少达到 early gate: `novel_clan_mean_f1 >= 0.15` 且 `gap_mean_f1 <= 0.10`；
2. 至少一个 cross-family 改进配置显著优于 `RF-M0-base` / `RF-M1-warm`；
3. RF-CF2 official long-range 结果落盘并完成 metric audit；若未达标，RF-CF5 capacity 需自动接力；
4. 多 seed 统计，至少 3-seed pilot，最终目标 10-seed + bootstrap 95% CI；
5. 同 split baseline 复现或清晰区分 cited/local；
6. README/论文表格回填最终 MMseqs 主结果与 RF-CF 系列结果；
7. 当前弱 baseline 的根因诊断需要用 distance-bin、pair count、family macro F1 等细粒度指标证明，而不能只看总 F1。
8. `data_diversity_audit` 已生成但未通过数据 gate；必须补齐 family/clan metadata join，降低 fallback pseudo-clan 对 curriculum/balanced sampling 的污染。

## 10. 下一步计划

短期：

1. 等待 official `RF-CF2-long-range-w2` 完成并生成 `cross_family_long_range_results.json`；
2. 立即运行 `audit_cross_family_metrics.py` 与 long-range bin 检查，比较 RF-CF2 与 RF-CF1/RF-M1 的 `novel_clan` F1/MCC、gap、retention 和 long-range recall；
3. 如果 RF-CF2 仍未达标，确认 RF-CF5 capacity watcher 自动启动；
4. 根据 `data_diversity_audit` 先补 metadata join/pseudo-clan 清洗，再启动真正的 source/family/length/complexity curriculum。

中期：

1. 执行/复盘 `RF-CF2-long-range-head` 或 long-range reweighting；
2. 执行 RF-CF5 capacity sweep；
3. 对 best config 做多 seed；
4. 建立同 split baseline 对比。

长期：

1. 扩大模型容量和 adapter dim；
2. 引入 multi-encoder adapter；
3. 将 probe-calibrated ensemble calibration 作为论文特色实验；
4. 完成最终 SOTA 表、消融表、可复现 manifest 与论文级统计。

## 11. 一句话总结

ReactFlow 已经完成真实数据、模型架构、训练系统、评估协议、服务器队列、三份 final-result 内容契约和审计体系的主体闭环。当前处于 **MMseqs weak baseline 已确认 + RF-CF3 未达标 + RF-CF1 contact sweep 已完成但未过 gate + RF-CF2 official long-range 接力中** 的阶段：contact sweep best `lambda=0.8` 把 `novel_clan_mean_f1` 提到 `0.0435`，但仍略低于 RF-M1-warm `0.0447` 和 claim gate `0.15`；RF-CF2 `w=8 early` 为 `0.0404`。eFold same-split baseline 已修复 diagonal-pair artifact 并恢复续跑，仍未完成 full artifact。项目还不能宣称顶刊结果完成，而且当前性能离 SOTA 仍然很远；下一阶段重点必须从工程闭环转向 pair-recovery、long-range recovery、family-balanced training、contact/thermo/calibration losses、eFold-inspired data diversity curriculum 和同 split baseline 对标。
