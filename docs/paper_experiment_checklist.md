# ReactFlow 论文级实验闭环检查清单

目标: 从真实数据、无泄漏切分、训练、评估、消融、baseline 到论文表格，形成可复现、可审计、可投稿的完整闭环。

执行原则:

- 全量规模实验由 Agent 负责运行、监控、重试和汇总结果。
- 用户只需要核对关键决策和不可自动获取的外部权限/资源。
- 禁止伪造数据；遇到缺失、403、baseline 无法复现时必须记录 blocker。
- Smoke run -> Pilot run -> Full run 逐级推进，任何一级缺 artifact 不进入下一阶段。

## 0. 实验账本

| 项目 | 当前值 | 状态 |
|---|---|---|
| 代码版本 | TBD commit hash | pending |
| 服务器/设备 | TBD | pending |
| 数据根目录 | `/home/liucunyu/reactflow_smoke/data/raw/efold/dryad_20260129` 或更新路径 | pending |
| 输出根目录 | `artifacts/full_runs/<run_id>` | pending |
| 主 split manifest | `artifacts/.../split_manifest.json` | pending |
| 主 checkpoint | `training_checkpoint.json` | pending |
| 主 profile summary | `*.summary.json` | pending |
| 主结果表 | `paper_results.tsv` | pending |

## 1. 数据与来源

- [ ] Dryad eFold/RNAndria JSON 解压完成，文件列表包含 `efold_train.json`, `archiveII.json`, `PDB.json`, `viral_fragments.json`, `lncRNA_nonFiltered.json`, `human_mRNA.json`。
- [ ] 每个原始文件记录 sha256、文件大小、来源 DOI。
- [ ] Kaggle Ribonanza/RibonanzaNet2 checkpoint 路径存在，记录 `weights_sha256`。
- [ ] 若使用外部 Rfam/MMseqs 标签，记录 metadata TSV schema: `record_id`, `clan`, `cluster`。
- [ ] 明确哪些记录有真实 `shape/dms/reactivity`，哪些是 structure-only。
- [ ] DMS `-1000` 缺失值处理为 `NaN/null` 已验证。
- [ ] `docs/data_governance.md` 已更新，且列出的 public source、cache row count、MMseqs split 和 audit artifacts 与当前 run root 一致。

通过条件:

- `prepare-efold-cache` 能在 sample 子集上成功运行。
- cache 行数、accepted/skipped/windowed 统计写入实验日志。

## 2. Cache 与长序列 windowing

- [ ] 短序列 cache: `efold_train`, `archiveII`, `PDB`, `viral`。
- [ ] 长序列 cache: `lncRNA_nonFiltered`, `human_mRNA`, 必须使用 `--window-size/--window-stride`。
- [ ] length bucket 使用统一边界，建议 `64,128,256,384`。
- [ ] 每个 cache 保存对应命令、stdout JSON、sha256。
- [ ] 抽样检查窗口坐标 `window.start/end/parent_length` 正确。

推荐命令:

```bash
PYTHONPATH=src reactflow prepare-efold-cache data/raw/efold/dryad_20260129/efold_train.json \
  --output artifacts/full_runs/cache/efold_train.jsonl \
  --window-size 256 --window-stride 128 --max-length 256 \
  --bucket-boundaries 64,128,256,384
```

通过条件:

- cache 是严格 JSONL。
- 长序列不再仅因超过 `max_length` 被全部跳过。

## 3. 无泄漏 split

- [ ] 使用 `split-efold-cache` 生成 `train/val/test/novel.jsonl`。
- [ ] `split_manifest.json` 存在。
- [ ] `validate_split_leakage` 通过。
- [ ] `counts_by_split` 和 `counts_by_bucket` 写入实验账本。
- [ ] Rfam metadata manifest 记录 `clan_membership.txt.gz` 来源、cluster method、MMseqs2 是否可用。
- [ ] 若无真实 Rfam clan 标签，只能标为 fallback split，不进入最终 SOTA 主表；若无 MMseqs2，仅可标为 exact-cluster fallback，需在主表脚注说明。
- [ ] `python-identity` 只用于小规模 sensitivity / CI；其 ungapped global identity 不是 MMseqs2 local-alignment 替代，不能进入最终 SOTA 主表。
- [ ] 最终主表 split 必须由 `--cluster-method mmseqs --mmseqs-min-seq-id 0.9 --mmseqs-coverage 0.8` 生成，且 manifest 中 `cluster_method == "mmseqs"`。

推荐命令:

```bash
PYTHONPATH=src python scripts/build_rfam_metadata.py artifacts/full_runs/cache/efold_train.jsonl \
  --output artifacts/full_runs/metadata/rfam_current_metadata.tsv \
  --manifest artifacts/full_runs/metadata/rfam_current_metadata.manifest.json \
  --rfam-download-dir artifacts/full_runs/metadata/rfam_database_files \
  --cluster-method mmseqs --mmseqs-min-seq-id 0.9 --mmseqs-coverage 0.8

PYTHONPATH=src reactflow split-efold-cache artifacts/full_runs/cache/efold_train.jsonl \
  --output-dir artifacts/full_runs/splits/rfam_current_seed0 \
  --metadata-tsv artifacts/full_runs/metadata/rfam_current_metadata.tsv \
  --bucket-boundaries 64,128,256,384 \
  --novel-clan-fraction 0.15 --seed 0
```

通过条件:

- 任意 clan/cluster 不跨 split。
- split manifest sha256 被记录。

## 4. Frozen features

- [ ] RibonanzaNet2 real checkpoint 严格加载，missing/unexpected keys 为 0。
- [ ] single-only shard 已导出并通过 `read_frozen_shard` 校验。
- [ ] 若导出 pair features，记录磁盘占用和导出耗时。
- [ ] frozen shard provenance 包含 `weights_sha256`, `content_sha256`, model version。
- [ ] matched sequences 统计写入每次 warm-start run。

通过条件:

- warm-start 训练 `matched_sequences / samples` 明确。
- 不允许 dry-run shard 进入论文主实验表。

## 5. 训练 run

- [ ] Base run: no warm-start。
- [ ] Warm-start run: RibonanzaNet2 frozen single。
- [ ] 每个 run 写 `training_checkpoint.json`。
- [ ] 每个 run 写 `--profile-path` JSONL 和 summary。
- [ ] 每个 run 记录完整命令、seed、backend、GPU/CPU 信息。
- [ ] loss 曲线没有 NaN/inf。
- [ ] 训练失败必须记录 stderr、最后 checkpoint、重试策略。

推荐命令:

```bash
PYTHONPATH=src reactflow train-efold artifacts/full_runs/splits/rfam_seed0/train.jsonl \
  --epochs 20 --lambda-react 0 \
  --bucket-boundaries 64,128,256,384 \
  --profile-path artifacts/full_runs/base_seed0/train_profile.jsonl \
  --output-dir artifacts/full_runs/base_seed0
```

通过条件:

- `training_checkpoint.json` 可用 `read_training_checkpoint` 读回。
- `slowest_step_phase` 已记录，便于后续优化。

## 6. Evaluation run

- [ ] in-clan/test tier。
- [ ] novel-clan tier。
- [ ] cross-family 主指标必须写入 `cross_family_metric_audit.json/md`。
- [ ] 每个候选模型必须报告 `novel_clan_mean_f1`、`novel_clan_micro_f1`、`novel_clan_mean_mcc`、`gap_mean_f1 = F1(in_clan)-F1(novel_clan)` 和 `retention`。
- [ ] `distance_bins.novel_clan.long.mean_f1` / `micro_f1` 必须记录，用于判断 cross-family 低分是否来自长程配对恢复不足。
- [ ] 最终主表必须优先使用 MMseqs split 的 `novel_clan`；Rfam exact split 只能作为工程诊断或 ablation warm-up。
- [ ] 若只提升 `in_clan` 但不提升 `novel_clan`，该 run 不能作为主模型路线。
- [ ] archiveII/PDB/viral public tiers。
- [ ] 长序列 lncRNA/human_mRNA windowed tiers。
- [ ] 输出 F1/MCC macro + micro。
- [ ] 输出 reactivity Pearson/Spearman/calibrated MAE。
- [ ] 输出 comparison markdown。
- [ ] evaluation 使用训练 checkpoint 对应 config，避免参数漂移。

推荐命令:

```bash
PYTHONPATH=src reactflow evaluate-efold \
  --train-json artifacts/full_runs/splits/rfam_seed0/train.jsonl \
  --eval-json in_clan=artifacts/full_runs/splits/rfam_seed0/test.jsonl \
  --eval-json novel_clan=artifacts/full_runs/splits/rfam_seed0/novel.jsonl \
  --epochs 20 --lambda-react 0 \
  --profile-path artifacts/full_runs/base_seed0/eval_profile.jsonl \
  --output-dir artifacts/full_runs/base_seed0_eval
```

通过条件:

- 每个 tier 样本数非零。
- 结果 JSON、comparison table、checkpoint、profile 全部留存。
- `cross_family_metric_audit.json` 中 `cross_family_healthy=true`；进入 cross-family claim 前，`cross_family_claim_ready=true` 或必须在文稿中标注为尚未达标。

## 7. Baseline 与 SOTA 对标

- [ ] eFold/RNAndria: 同协议本地重算，若不能重算则引用列和本地列分开。
- [ ] RibonanzaNet2-derived baseline: 同 split，同 eval tiers。
- [ ] TVAE-RNA/MERGE-RNA: 先核实可运行代码和输入格式，再决定本地重算或 cited-only。
- [ ] 所有 baseline 记录数据、split、命令、版本。
- [ ] 不同 split 或不同数据集的数字不得直接标 SOTA。

通过条件:

- 至少 eFold/RNAndria 与 ReactFlow 在同一 eval tier 下有可比数字。
- 表格明确 `cited` vs `local rerun`。

## 8. 消融实验

- [ ] 完成 `docs/ablation_experiment_template.md` 中最小可发表集合。
- [ ] 每个消融只改变一个主变量。
- [ ] 每个消融至少记录 seed 0；主结果建议 seed 0/1/2。
- [ ] 每个消融保留 checkpoint/profile/eval JSON。
- [ ] 失败的消融也要入账，不可只保留成功结果。

通过条件:

- 能解释主模型增益来源: warm-start、reactivity、thermo、guidance、windowing、backend。

## 9. 统计与稳健性

- [ ] 主结果至少 3 seeds 或说明资源限制。
- [ ] 报告 mean/std 或 bootstrap CI。
- [ ] 报告 length bucket 分层结果。
- [ ] 报告 short vs long RNA 分层结果。
- [ ] 报告 train time、eval time、GPU hours、峰值显存/内存。

通过条件:

- 主结论不依赖单一 seed 或单一 tier。

## 10. 论文表格与图

- [ ] 主 SOTA 表: ReactFlow vs baselines。
- [ ] Generalization gap 表: in-clan/test vs novel-clan。
- [ ] 消融表: minimum publishable ablation set。
- [ ] Runtime/profile 表: stdlib vs torch, slowest phase。
- [ ] 长序列表: window size/stride 分析。
- [ ] 训练曲线图、pair heatmap、guidance eta scan 图。
- [ ] 所有表格行都有 artifact 链接。

通过条件:

- 论文中每个数字可追溯到 run ID。

## 11. Release 与复现包

- [ ] `README` 包含 full run SOP。
- [ ] `ReactFlow_research_plan.md` 状态与结果同步。
- [ ] 保存 `requirements` / env 信息。
- [ ] 保存 cache/split/checkpoint/profiles 的 sha256 manifest。
- [ ] `algorithm_doc_audit.json` 显示 `placeholder_bodies=0`, `text_markers=0`, `parse_errors=0`, 且最终提交前 `strict_ready=true`。
- [ ] 准备匿名化 artifact bundle。

通过条件:

- 新机器可从 raw data + checkpoint 或 raw data + commands 复现实验结果。

## 12. Agent 执行承诺

- [ ] Full run 由 Agent 发起。
- [ ] Full run 由 Agent 监控。
- [ ] Full run 失败由 Agent 定位并重试。
- [ ] Full run 结果由 Agent 汇总成表格。
- [ ] 只有当外部系统需要用户授权、凭证过期、磁盘/GPU quota 不足或数据无法合法获取时，才请求用户介入。

结果交付格式:

```text
Run ID:
Status:
Command:
Artifacts:
Metrics:
Slowest phase:
Comparison vs previous:
Decision:
Next run:
```
