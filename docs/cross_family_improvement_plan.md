# ReactFlow Cross-Family 准确生成改进计划

本文档把 cross-family / novel-family RNA 二级结构生成能力设为下一阶段主目标。所有结论必须来自 Rfam/MMseqs family-disjoint split，不能用随机 split 或 family-overlap split 替代。

## 1. 当前证据与问题定义

当前已完成 exact-split warm-start 队列显示，ReactFlow 还不能声称已经做到 cross-family 准确生成：

| Run | Split | in_clan mean F1 | novel_clan mean F1 | novel_clan micro F1 | novel_clan mean MCC | 结论 |
|---|---|---:|---:|---:|---:|---|
| `RF-A1-warm_rfam_current_exact_torch_full_data_e1_bs16` | Rfam exact fallback | 0.0394 | 0.0624 | 0.0506 | 0.0591 | 可运行 baseline，但不能支撑准确生成 |
| `RF-A2-adapter4_rfam_current_exact_torch_full_data_e1_bs16` | Rfam exact fallback | 0.0243 | 0.0344 | 0.0274 | 0.0294 | adapter4 未带来 OOD 增益 |

最终论文主表必须转向 MMseqs split：`train=228,282`, `val=16,606`, `test=16,606`, `novel=46,147`，并使用 `cluster_method="mmseqs"`, `min_seq_id=0.9`, `coverage=0.8` 的 metadata manifest 作为无泄漏证据。

### 1.1 eFold/RNAndria 的数据多样性教训

eFold/RNAndria 论文（Science Advances `10.1126/sciadv.adz4967`）直接指出：RNA 结构预测的 cross-family gap 不是单纯靠扩大数据量或模型规模就能解决；短 ncRNA family 在 bpRNA/RNAstrAlign/ArchiveII 等常用集合中占比过高，会让模型在近域集合上表现好，但在 viral mRNA、long ncRNA、human mRNA 等长序列和复杂结构域上掉点。它们通过新增 pri-miRNA、human mRNA、viral fragments、lncRNA 等更复杂来源，并结合去冗余和 out-of-domain benchmark，来缩小泛化差距。Dryad `10.5061/dryad.79cnp5j95` 中的 `efold_train.json` 也强调了去重、低质量过滤、AUROC/F1 过滤、BLAST 去相似测试集，以及 synthetic clan-balanced RNA 来源。

因此 ReactFlow 的 cross-family 改进计划必须加入一个数据 gate：在继续扩大 `hidden_size` / `adapter_dim` 前，先证明训练集在 source、family/clan、length、long-range pair、domain complexity 上不是单一来源或短 ncRNA 主导。否则更大的模型很可能只学到近域 family 记忆。

## 2. 主评估指标

cross-family 不再作为附属 tier，而作为主目标 gate：

| 指标 | 定义 | 用途 |
|---|---|---|
| `novel_clan_mean_f1` | novel-family tier 上逐样本 F1 的 macro mean | 主排序指标，避免大类支配 |
| `novel_clan_micro_f1` | novel-family tier 上合并 TP/FP/FN 后的 F1 | 观察总体 pair-level 准确性 |
| `novel_clan_mean_mcc` | novel-family tier 上 macro MCC | 处理 class imbalance |
| `gap_mean_f1` | `mean_f1(in_clan) - mean_f1(novel_clan)` | 衡量 OOD 泛化掉点 |
| `retention` | `mean_f1(novel_clan) / max(mean_f1(in_clan), eps)` | 衡量 novel-family 保留率 |
| `long_range_recall` | `|i-j| >= 24` pair recall | 专门诊断跨家族长程相互作用 |
| `family_macro_f1` | 先按 family/cluster 聚合再求均值 | 防止样本数大的 family 主导结论 |

新增 `scripts/audit_cross_family_metrics.py` 会在每次状态刷新时读取 `current_queue_status.json`，输出 `cross_family_metric_audit.json/md`。默认 early engineering gate 为 `novel_clan_mean_f1 >= 0.15` 且 `gap_mean_f1 <= 0.10`；低于该阈值先记为 warning，不阻断正在运行的队列。

`reactflow.evaluate.structure_distance_bin_metrics_by_tier` 已实现 distance-bin 诊断，并由 `reactflow evaluate` / `reactflow evaluate-efold` 输出 `distance_bins` 字段。默认分桶为 `short=1..11`, `medium=12..23`, `long>=24`；后续 RF-CF2 必须重点检查 `distance_bins.novel_clan.long.mean_f1` 和 `micro_f1`。

## 3. 提升路线

### P0：评估协议收紧

目标：保证每次模型改动都直接暴露 cross-family 影响。

- 所有 full-run summary 必须同时报告 `in_clan`、`novel_clan` 和 `gap_mean_f1`。
- `mmseqs_final_results.json` 必须作为论文主表，exact split 只作为工程诊断。
- 每个候选改动都需要同时比较 `novel_clan_mean_f1`、`novel_clan_mean_mcc`、runtime 和 checkpoint 完整性。
- 新增 cross-family audit artifact，纳入 reproducibility manifest 和 goal readiness。
- 已新增并运行 `scripts/audit_data_diversity.py`，生成 `data_diversity_audit.json/md` 与 `source_family_length_manifest.json`：统计 source mix、Rfam clan/family/MMseqs cluster coverage、length buckets、long-range pair ratio、pair count/stem/loop complexity、viral/lncRNA/human_mRNA domain-window provenance。远端首轮结果覆盖 exact/MMseqs/public tiers 共 `915,715` records，但显示 public/eFold train tiers 缺少 clan/family metadata，MMseqs split fallback pseudo-clan fraction 为 `1.0000`；因此不允许把“扩大模型”当作唯一主攻路线，且 balanced curriculum 前必须先补 metadata join。

### P1：结构解码器增强

目标：提升 novel-family pair recovery，而不是只提升同家族记忆。

- `RF-CF1-contact-strong`：提高 `lambda_contact` sweep，建议 `{0.1, 0.2, 0.4, 0.8}`，观察 novel F1 与 reactivity consistency 是否同时改善。
- `RF-CF2-long-range-head`：增加 long-range pair reweighting，对 `|i-j| >= 24` 的合法 pair BCE/DFM loss 加权，专门修复跨家族长程相互作用 recall。
- `RF-CF3-family-balanced-sampler`：训练 batch 对 Rfam family/MMseqs cluster 做 balanced sampling，降低大 family 对梯度的支配。
- `RF-CF4-thermo-guided-decode`：在固定 checkpoint 上扫 `eta`，只接受 novel-family F1/MCC 与能量一致性同时改善的解码设置。

### P2：表征与模型容量增强

目标：让 frozen encoder 的信息真正转化成可迁移结构先验。

- `RF-CF5-adapter-capacity`：adapter dim 从 `{4,16}` 扩展到 `{32,64}`，并记录参数量、吞吐、novel F1。
- `RF-CF6-multi-encoder-adapter`：预留 RiNALMo/RNA-FM/HydraRNA frozen features 的 adapter 接口，先做 offline feature provenance，再进入训练。
- `RF-CF7-contact-distillation`：从 RibonanzaNet2/eFold/RNADiffFold 可复现输出蒸馏 soft contact prior，但只作为 auxiliary，不能替代真实结构标签。

### P2.5：eFold-inspired data diversity curriculum

目标：把 eFold 的“多样性与复杂度优先”迁移到 ReactFlow，而不是只扩大模型规模。

- `RF-CF-D1-diversity-audit`：首轮已完成，artifact 为 `data_diversity_audit.json/md` 与 `source_family_length_manifest.json`；已纳入 MMseqs train/test/novel，下一步需要清理 fallback pseudo-clan 并补真实 family/clan metadata join。
- `RF-CF-D2-source-balanced-training`：训练 batch 按 source 与 family/cluster 双重平衡，避免 bpRNA/Rfam 短 ncRNA 或单一 public source 支配梯度。
- `RF-CF-D3-complex-domain-curriculum`：按 `short ncRNA -> pri-miRNA/human_mRNA windows -> viral/lncRNA domain fragments -> long-window stitching` 的顺序扩展训练，阶段间必须报告 public tiers 与 MMseqs novel 是否同向改善。
- `RF-CF-D4-domain-window-stitching-audit`：对 viral/lncRNA/human_mRNA 的窗口保留 parent coordinates、domain closed-loop/coverage 证据和跨窗 pair 统计，避免随机切窗破坏真实结构域。
- `RF-CF-D5-data-vs-scale-ablation`：固定模型容量，对比同样样本数下“随机更多数据”与“更多来源/复杂度分层数据”；只有后者改善 novel-family 和 public tiers 时，才把数据路线作为 paper claim。

### P3：probe-calibrated ensemble 约束

目标：保持 ReactFlow 的特色，不退化成普通 contact-map predictor。

- `RF-CF8-calib`：启用 `lambda_calib`，检查 novel-family 上 calibrated MAE、ECE/MCE 是否改善。
- `RF-CF9-reactivity-real-profile`：只在有真实 DMS/SHAPE 的 subset 上启用 `lambda_react`，避免 structure-only 记录伪造 probing supervision。
- `RF-CF10-second-moment`：若数据允许，引入 co-reactivity / second-moment 约束，验证 ensemble 不确定性是否帮助 novel-family。

## 4. 服务器实验队列建议

优先顺序如下：

1. 等待 `RF-A2-adapter16` 和 `warm_rfam_current_exact_results.json` 完成，作为 exact warm-start 容量诊断。
2. 保持现有 `RF-A3-contact` watcher，先得到 contact auxiliary 的 exact split 证据。
3. 立即让 MMseqs final queue 输出 `RF-M0-base` 与 `RF-M1-warm`，把主指标切到 MMseqs novel。
4. 若 `RF-M1-warm` 的 `novel_clan_mean_f1 < 0.15`，启动 `RF-CF1-contact-strong` 和 `RF-CF3-family-balanced-sampler`；后者通过 `--family-balanced-batches` 开关启用，按 `cluster -> family -> source_id` 优先级在长度 bucket 内做轮转采样。
5. 若 RF-CF1/RF-CF5 只带来容量或 contact 层面的边际改善，先启动 `RF-CF-D1/D2` 数据多样性审计与 source-balanced training，而不是继续单纯加大模型。
6. 若 `novel_clan_mean_f1` 提升但 `gap_mean_f1` 仍大于 0.10，启动 long-range、domain-window stitching 和 thermo-guided decode 诊断。
7. 对任何候选 best config 做 seed `{0,1,2}`，进入论文前扩展到 10 seeds + bootstrap 95% CI。

## 5. 成功判定

阶段性 gate：

- Engineering gate：`cross_family_metric_audit.json` 可解析，`cross_family_healthy=true`。
- Improvement gate：相对当前 best exact baseline，MMseqs `novel_clan_mean_f1` 有绝对提升，且 `novel_clan_mean_mcc` 同向提升。
- Claim gate：`novel_clan_mean_f1 >= 0.15`，`gap_mean_f1 <= 0.10`，并优于 base/warm baseline。
- Paper gate：同 split 下至少胜过一个本地可复现 baseline，3-seed bootstrap CI 不跨 0；主结果扩展到 10 seeds 后再写 SOTA claim。

任何只提升 `in_clan`、但不提升 `novel_clan` 或扩大 `gap_mean_f1` 的改动，只能作为诊断，不进入主模型路线。
