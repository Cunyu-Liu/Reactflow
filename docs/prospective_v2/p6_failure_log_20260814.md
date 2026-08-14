# ReactFlow-Delta prospective-v2 failure log (P6 deliverable, 2026-08-14)

> 记录本阶段所有实际遇到、根因确定、已修复的工程与数据故障。每条含：症状 / 根因 / 修复 / 影响 / 回归测试。
> 目的：P6 的 failure log（合同 12.8）——证明结果可审计、修复可复现，无静默掩盖。

| # | 阶段 | 症状 | 根因 | 修复 | 影响 | 回归测试 |
|---|---|---|---|---|---|---|
| F1 | P3 | `B*_held_crps[P20]=NaN`，20-puzzle CI 失效 | `_bstar_held_crps`/`_lrso_held_crps` 对空合格集执行 `np.nanmean([])` → NaN 污染 running total | 空 q 时 `continue` 跳过；B* P20 离线重算=0.194358（有限） | 原 P3 结果 CI 无效；修正后仍是 NO_INCREMENTAL，结论不变 | `test_p3_scoring_nan_fix.py` |
| F2 | P4 | 首次执行 IndexError：`index 139 out of bounds for axis 0 with size 139` | Ribonanza M2-style rdat 的 reactivity 数组（如 139）短于 profile_sequence（如 206）；3' pad/barcode 位置无数据（非 NaN 填充），shared-region mask 未边界化 | shared 索引按 reactivity 长度边界化，越界视为非观测（frozen attrition rule 3）；P4 改为直接加载冻结 outcome-blind component graph | 单次执行崩溃，无 outcome 消耗；修复后为唯一有效 locked 执行 | `test_p4_external_v1.py` |
| F3 | P5 | 置换负对照无效（permuted D == real D） | `rng.shuffle(fv)` 只重排 `(i, f_i)` 元组顺序，每个元组仍保留自己的特征 → 特征从未与位置脱钩（no-op） | 改为独立 shuffle 特征数组（`rng.shuffle(features)`），位置 i 接收另一位置的 feature | 若未修复，P5 负对照会虚假通过；修复后负对照真实（permuted D CI upper −0.062） | `test_p5_mechanism_v1.py::test_real_edit_site_skill_but_permuted_no_skill` |
| F4 | P4 复核 | 合同 §12.6 要求 coverage/calibration 合格，但 frozen protocol 未操作化 | P4 frozen protocol 只操作化统计标准（K_eff/CI/FWER/LOO），遗漏校准门 | 新增 `analyze_p4_calibration_v1.py`：固定 scale 0.3 的 68%/95% 经验覆盖率 + 预声明容差 → CALIBRATION_ACCEPTABLE（cov95 0.874∈[0.85,0.99]） | 补齐 P4 验收证据 | `test_p4_calibration_v1.py` |
| F5 | P6 replay | 重放崩溃 `TypeError: float - NoneType` | `p2_held_position_rows.jsonl` 有 4100/975,600 行因 WT 缺失而无预测（pred=None） | `_replay_p2` 跳过 None 行（等价于原 evaluator 的 qualified-position 逻辑） | 重放可完整跑通 | `test_run_replay_v1.py::test_skips_unqualified_none_rows` |
| F6 | P6 replay | 重放崩溃 `TypeError: -: 'str' and 'str'` | `_compare` 对 verdict 字符串做算术比较 | `_compare` 区分 str（相等比较）与数值（相对差） | 重放可完整跑通 | `test_run_replay_v1.py`（compare 用例） |
| F7 | 资源 | 上一会话遗留 4 个 `exact_pair_stats` 进程 100% CPU ×2 核 >45 分钟 | O(n²) 全序列比对（15KLIB 15000 profiles）卡死，已被冻结 component graph 取代 | runtime 重新发现 PID（124792/124793/132727/132728）后终止；记录于本 log | 释放 2 核，服务器 load ~60 | n/a（运维） |

## 未静默项
- P5 `MECHANISM_NOT_ESTABLISHED`：预冻结"编辑位点集中"机制 claim 未复现（诚实 fail-closed，非故障）；距离曲线均匀，异质性 CI lower −0.0199。
- P2 `signed-delta MAE` 为负（−6.5%）但 CRPS 主估计为正：已作为强制 secondary 如实报告（P2 handoff）。
- M3SARS 数据集噪声大（经验残差 SD 1.37，95% 覆盖率 0.781）：作为校准诊断如实报告，不改冻结 scale。
