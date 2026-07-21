# ReactFlow 消融实验表格模板

本文档用于 full run、pilot run 和论文表格的统一记录。所有数字必须来自可复现 artifact，禁止把引用数字和本地重算数字混列。

## 1. 消融实验矩阵

| Run ID | 分组 | 变量 | 实验设置 | 核心假设 | Train split | Eval tier | 关键指标 | 必需 artifact | 完成状态 |
|---|---|---|---|---|---|---|---|---|---|
| RF-A0-base | 主模型 | 基础结构监督 | `lambda_react=0`, no warm-start, no thermo, no guidance | 真实结构标签能训练出最低可用 DFM head | `train.jsonl` | `test/novel` | F1, MCC, loss | checkpoint, profile, eval JSON | pending |
| RF-A1-warm | Warm-start | RibonanzaNet2 frozen single | `adapter_dim=8`, real RibonanzaNet2 shard | frozen encoder 提供跨家族泛化增益 | 同 A0 | 同 A0 | ΔF1, ΔMCC, matched ratio | checkpoint, frozen provenance | pending |
| RF-A2-adapter-dim | Warm-start | adapter 宽度 | `adapter_dim in {4,8,16}` | adapter 容量存在 sweet spot | 同 A0 | 同 A0 | F1/MCC vs params | 3 checkpoints | pending |
| RF-A3-react | Reactivity loss | `lambda_react` | `{0,0.1,0.3,1.0}` on records with real profiles only | probe profile 作为一阶矩监督提升 OOD | real-profile subset | held-out profile tier | F1, Pearson, Spearman, MAE | profile mask report | pending |
| RF-A4-thermo | Thermo prior | `lambda_thermo` | `{0,0.1,0.3}`, `mse/kl` | Turner prior 改善低数据或长序列稳定性 | 同 A0 | long/novel | F1, pair count, loss | thermo config | pending |
| RF-A5-guidance | Inference guidance | `eta` scan | `eta in {0,0.25,0.5,1,2,4}` | 推理期能量引导提高合法低能结构 | fixed checkpoint | all tiers | F1, energy, legality | guidance scan SVG/JSON | pending |
| RF-A6-window | 长序列策略 | window size/stride | `128/64`, `256/128`, `384/192` | 局部窗口能扩展到 lncRNA/human_mRNA | windowed cache | long tiers | F1, runtime, memory | split manifest, cache summary | pending |
| RF-A7-bucket | Bucketing | length bucket | no bucket vs `64,128,256` | bucketing 降低 profile 噪声并改善吞吐 | 同 A6 | long tiers | time/sample, peak RSS | profile summary | pending |
| RF-A8-backend | 加速后端 | stdlib vs torch | `--backend stdlib` vs `--backend torch` | torch 后端减少 pairwise backward 瓶颈 | same subset | same subset | sec/epoch, F1 parity | profile summary | pending |
| RF-A9-frozen-shard | frozen 特征 | single-only vs single+pair | `--d-pair 0` vs pair export | pair 特征是否值得存储/计算成本 | same split | all tiers | ΔF1 per GB | shard provenance | pending |
| RF-A10-normalize | profile 预处理 | normalization | `p90`, `zscore`, `minmax` | profile 形状指标对尺度处理敏感 | profile subset | profile holdout | Pearson/Spearman/MAE | data QC report | pending |
| RF-A11-estimator | 期望估计 | A vs A+B | mean-field only vs sampling correction | 采样纠偏减少一阶边缘偏差 | same split | all tiers | F1, calibration, diversity | estimator config | not implemented |
| RF-A12-second-moment | 二阶约束 | co-reactivity | off vs on | 二阶矩约束缓解仅一阶不可辨识性 | data permitting | profile holdout | F1, pair correlation | data availability report | not implemented |
| RF-CF1-contact-strong | Cross-family | contact auxiliary sweep | `lambda_contact in {0.1,0.2,0.4,0.8}` | 合法 pair 空间 denoising 提升 novel-family pair recovery | MMseqs train | `novel_clan` | novel F1/MCC, gap, reactivity consistency | cross-family audit, checkpoint | planned |
| RF-CF2-long-range-head | Cross-family | long-range pair reweighting | `|i-j| >= 24` loss weight sweep | 跨家族主要错误来自长程相互作用召回不足 | MMseqs train | novel/long tiers | long-range recall, novel F1, runtime | long-range diagnostic JSON | planned |
| RF-CF3-family-balanced | Cross-family | family/cluster balanced sampling | `--family-balanced-batches`，按 cluster/family 在 length bucket 内轮转 | 降低大 family 梯度支配，提升低支持 novel family | MMseqs train | `novel_clan` | family_macro_f1, novel F1/MCC | sampler manifest | planned |
| RF-CF4-thermo-decode | Cross-family | inference energy guidance | fixed checkpoint, `eta` scan | 物理先验改善 novel-family 合法低能结构 | best checkpoint | `novel_clan` | F1/MCC, energy, legality | guidance scan JSON/SVG | planned |
| RF-CF5-adapter-capacity | Cross-family | larger adapter | `adapter_dim in {32,64}` | 更大 adapter 才能把 frozen encoder 转成可迁移结构先验 | MMseqs train | all tiers | novel F1 per parameter, samples/s | profile + parameter count | planned |

## 2. Baseline 对标表模板

| Model | 来源 | 训练数据 | Split manifest | Eval tier | Macro F1 | Micro F1 | Macro MCC | Micro MCC | React Pearson | React Spearman | Calibrated MAE | Runtime | Artifact | 备注 |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| ReactFlow base | local | eFold train | `split_manifest.json` | in_clan/test | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | checkpoint + eval JSON | 本地重算 |
| ReactFlow warm | local | eFold train + RibonanzaNet2 frozen | same | in_clan/test | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | checkpoint + eval JSON | 本地重算 |
| ReactFlow warm | local | same | same | novel_clan | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | checkpoint + eval JSON | 主结果 |
| ReactFlow cross-family best | local | eFold train + best CF config | MMseqs split manifest | novel_clan | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | `cross_family_metric_audit.json` + final result JSON | cross-family 主结果 |
| eFold/RNAndria | cited/local rerun | paper protocol | same or cited split | public tiers | TBD | TBD | TBD | TBD | N/A | N/A | N/A | TBD | baseline report | 引用数字与本地重算分栏 |
| RibonanzaNet2-derived baseline | local | same frozen source | same | same | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | baseline script output | 待实现 |
| TVAE-RNA / MERGE-RNA | cited/local rerun | matched where possible | same | same | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | baseline report | 可用性待核实 |

## 3. Full-run 结果记录 schema

| 字段 | 必填 | 说明 |
|---|---|---|
| `run_id` | yes | 例: `RF-A1-warm-seed0-full` |
| `git_commit` | yes | 运行代码版本 |
| `command` | yes | 完整 CLI 命令 |
| `host` / `device` | yes | GPU 服务器、CUDA/torch 版本 |
| `input_cache_sha256` | yes | 训练和评估 cache hash |
| `split_manifest_sha256` | yes | 无泄漏 split manifest hash |
| `frozen_provenance` | warm-start yes | frozen shard provenance path/hash |
| `checkpoint` | yes | `training_checkpoint.json` |
| `profile_summary` | yes | `*.summary.json` |
| `eval_json` | yes | stdout JSON 或保存的 eval summary |
| `metrics_table_row` | yes | 可直接复制进论文表格的一行 |
| `failure_notes` | if failed | 失败原因、重跑策略 |

## 4. 最小可发表消融集合

必须完成:

- `RF-A0-base`
- `RF-A1-warm`
- `RF-A2-adapter-dim` 至少 3 个 adapter 宽度
- `RF-A3-react` 至少 `lambda_react=0` vs best non-zero
- `RF-A4-thermo` 至少 off vs best on
- `RF-A5-guidance` 完整 eta scan
- `RF-A6-window` 至少 2 个 window 设置
- `RF-A8-backend` stdlib vs torch 吞吐/一致性对比
- `RF-CF1-contact-strong` 或 `RF-CF3-family-balanced` 至少完成一个，并在 MMseqs `novel_clan` 上证明优于 base/warm baseline
- `cross_family_metric_audit.json/md` 必须生成并纳入 reproducibility manifest

可延后但论文更强:

- `RF-A9-frozen-shard`
- `RF-A10-normalize`
- `RF-A11-estimator`
- `RF-A12-second-moment`
- `RF-CF2-long-range-head`
- `RF-CF4-thermo-decode`
- `RF-CF5-adapter-capacity`

## 5. 判定规则

- 主张 SOTA 前，ReactFlow 数字必须来自同一 split manifest 下的本地重算。
- 如果引用 baseline 数字无法同切分复现，表格必须保留 `cited` 与 `local` 两列。
- 任一 full run 缺少 checkpoint、profile summary、split manifest 或命令记录，不进入论文主表。
- 若 `novel_clan` 提升但 `in_clan` 明显下降，需报告 generalization gap 和稳定性分析，不能只挑单点结果。
- cross-family 主张必须同时报告 `novel_clan_mean_f1`、`novel_clan_mean_mcc`、`gap_mean_f1 = F1(in_clan)-F1(novel_clan)` 和 `retention`；只提升 `in_clan` 不提升 `novel_clan` 的改动不得作为主模型改进。
- 论文主表优先使用 MMseqs split；exact split 只能作为工程诊断或 ablation warm-up。
