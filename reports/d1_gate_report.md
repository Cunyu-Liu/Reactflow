# D1 Gate Report — ReactFlow-Δ Phase D1 Cleanup-Only

**合同**: v3.1 §4 D1 Gate (增补), v3 §15 Phase D1 Gate
**分支**: `codex/reactflow-delta-d0r`
**最新 commit**: `55d8c1d` — feat(d1): hand-computed end-to-end fixtures for D1 pipeline (T-D1.11, v3.1 §4)
**测试**: 496 passed (0 failed, 0 errors) — `PYTHONPATH=src python -m pytest tests/reactflow_delta/`
**data.py**: 2087 lines | **schema.py**: 705 lines

---

## 1. D1 Gate 逐 bullet PASS/FAIL (v3.1 §4)

| # | Gate bullet | 状态 | 证据 |
|---|---|---|---|
| 1 | **fixtures 100% 通过**（T-D1.11 手算 fixtures） | **PASS** | `test_d1_handcomputed_fixtures.py` 28/28 pass；7 fixture classes 覆盖 full-pipeline end-to-end |
| 2 | **missing 不作 0**（v3 §6.7、§6.6 step 9） | **PASS** | Fixture 2 `test_delta_raw_propagates_none` / `test_missing_not_treated_as_zero`；`compute_delta_reactivity` 传播 None；`normalize_2_8_percent` 传播 None；`apply_zscore_normalization` 传播 None |
| 3 | **noise 不用 test 估计**（仅 train/validation，§2.2） | **PASS** | `freeze_control_noise_threshold` 只接受 control \|Δr\| 值（无 test 参数）；`estimate_replicate_noise` / `estimate_error_variance` 由 caller 保证 train+validation only（API 合同）；Fixture 6 `test_threshold_uses_only_control_values` |
| 4 | **normalization 不最小化 pair difference**（v3 §6.7） | **PASS** | Fixture 3 `test_scale_factors_independently_determined`：WT scale=4.0 ≠ Mut scale=8.0（per-construct 92-98th pct，非 pair-level min-diff）；`test_normalization_not_pair_minimizing` |
| 5 | **每个 exclusion 有 machine-readable reason** | **PASS** | `test_every_exclusion_reason_in_frozen_vocabulary`：13 reasons 全部 reachable，全部 ∈ `EXCLUSION_REASONS` frozen vocab（schema.py L225-239） |
| 6 | **不自动进入训练**（training_allowed 仍 False） | **PASS** | `test_no_training_flag_in_upgrade_output`：`evaluate_pair_upgrade` 不输出任何 training-authorization flag；D1 输出无 `training_allowed` / `training_enabled` / `auto_train` 字段 |
| 7 | **Tier gate 不被降低** | **PASS** | `test_true_pair_honest_only_when_no_reasons`：`true_pair=True` 当且仅当 `exclusion_reasons=[]`；soft blocker（corroboration-only）保持 `primary_eligible=True` 但 `true_pair=False`；不降阈值 |

**D1 Gate 结论**: **7/7 bullets PASS** — 实现/合同层面全部满足。

---

## 2. T-D1.1~12 任务完成状态

| Task | 描述 | Commit | Tests | 状态 |
|---|---|---|---|---|
| T-D1.1 | 冻结 construct/pair schema | (D0-R 遗产) | 41 (test_construct_pair_schema) + 13 (test_schema) | ✅ |
| T-D1.2 | condition exact matching | (D0-R 遗产) | 106 (test_d1_cleaning) | ✅ |
| T-D1.3 | substitution verification + HIV3PR offset | `1d5e4e5` | (in test_d1_cleaning) | ✅ |
| T-D1.4 | alignment + unchanged mask | `aa62535` | (in test_d1_cleaning) | ✅ |
| T-D1.5 | probe eligibility | `a3e1fc0` | 53 (test_probe_masks) | ✅ |
| T-D1.6 | replicate/no-edit/control identification | `2bec648` | (in test_d1_cleaning) | ✅ |
| T-D1.7 | raw/upstream/project-normalized 三层 + domain | `846672f` | 48 (test_normalization) | ✅ |
| T-D1.8 | study/probe measurement noise | `2d2cbfe` | 26 (test_noise_estimation) | ✅ |
| T-D1.9 | frozen differential caller + Δreactivity | `592d762` | 28 (test_differential_call) | ✅ |
| T-D1.10 | quality weight + exclusion reasons + true_pair | `6d8e927` | 41 (test_pair_upgrade) | ✅ |
| T-D1.11 | hand-computed fixtures | `55d8c1d` | 28 (test_d1_handcomputed_fixtures) | ✅ |
| T-D1.12 | tests + commit + push + Gate report | 本报告 | 496 total | ✅ |

---

## 3. 测试套件总览

```
496 passed in 0.76s
```

| 测试文件 | 测试数 | 覆盖范围 |
|---|---|---|
| test_d1_cleaning.py | 106 | T-D1.2~6 condition/alignment/probe/replicate |
| test_probe_masks.py | 53 | T-D1.5 probe eligibility masks |
| test_pair_upgrade.py | 41 | T-D1.10 exclusion reasons + quality weight |
| test_construct_pair_schema.py | 41 | T-D1.1 construct/pair schema validation |
| test_normalization.py | 48 | T-D1.7 三层 reactivity + domain |
| test_d1_handcomputed_fixtures.py | 28 | T-D1.11 end-to-end fixtures + Gate invariants |
| test_differential_call.py | 28 | T-D1.9 frozen differential caller |
| test_noise_estimation.py | 26 | T-D1.8 measurement noise |
| test_d0r_functional_anchor.py | 26 | D0-R functional anchor |
| test_d0r_reaudit_tierA.py | 22 | D0-R Tier A re-audit |
| test_d0r_rdat_parser.py | 21 | D0-R RDAT parser |
| test_rdat_parser.py | 15 | RDAT parser |
| test_schema.py | 13 | schema validation |
| 其余 (10 files) | 24 | manifests/registry/pairing/matrix/etc. |
| **合计** | **496** | |

---

## 4. data.py 实现的 D1 building blocks (T-D1.1~10)

### 4.1 T-D1.1~6 (schema + condition + substitution + alignment + probe + replicate)
- `validate_construct_record` / `validate_pair_record` (schema.py)
- `verify_substitution` + HIV3PR genome-numbering offset
- `build_alignment_masks` + `compute_comparable_fraction` (COMPARABLE_MIN_FRACTION=0.60)
- `build_probe_eligibility_masks`
- `identify_replicates` / `identify_no_edit_controls` / `identify_controls`

### 4.2 T-D1.7 (三层 reactivity + normalization domain)
- `NORMALIZATION_DOMAIN_FIELDS` = (study_id, probe, probe_protocol, in_vivo_in_vitro)
- `identify_normalization_domain` / `build_normalization_domains` / `check_normalization_domain_compatible`
- `normalize_2_8_percent` (nearest-rank 92-98th percentile)
- `compute_domain_zscore_stats` / `apply_zscore_normalization`
- `build_reactivity_layers` → raw / upstream / project layers

### 4.3 T-D1.8 (measurement noise)
- `estimate_replicate_noise` (per-pos variance across replicates, ddof=1)
- `estimate_error_variance` (mean squared REACTIVITY_ERROR)
- `freeze_control_noise_threshold` (nearest-rank percentile of |Δr| from controls, min 10)
- `estimate_pair_noise` → replicate_noise_estimate + measurement_variance

### 4.4 T-D1.9 (frozen differential caller)
- `compute_delta_reactivity` (Δr = r_m − r_w, None preserved)
- `_normal_cdf` / `_benjamini_hochberg` (BH-FDR)
- `frozen_differential_call` (frozen threshold + significance + FDR + z-scores)
  - `caller_status` ∈ {replicate_aware, no_replicate_continuous_only, no_threshold}
- `build_pair_delta_reactivity` → delta_reactivity_raw/normalized + caller

### 4.5 T-D1.10 (quality weight + exclusion reasons + true_pair)
- `UPGRADE_BLOCKER_EXCLUSION_REASONS` = {sequence_based_no_independent_corroboration}
- `QUALITY_SNR_FULL_FACTOR_AT` = 10.0, `QUALITY_COVERAGE_FULL_FACTOR_AT` = 30.0
- `QUALITY_NO_REPLICATE_FACTOR` = 0.8, `QUALITY_UNKNOWN_SIGNAL_FACTOR` = 0.5
- `collect_exclusion_reasons` (13-reason frozen vocabulary, sorted unique)
- `determine_primary_eligible` (soft blocker keeps primary)
- `determine_true_pair` (requires empty reason list)
- `compute_pair_quality_weight` (clamped product of 5 factors)
- `evaluate_pair_upgrade` (集成 T-D1.1~9 → 4 pair-schema D1 fields + factors)

---

## 5. 禁止操作合规性 (v3.1 §2.2)

| 禁止项 | 合规 | 证据 |
|---|---|---|
| learned training | ✅ | D1 无任何模型训练代码 |
| model forward/backward | ✅ | D1 无模型代码 |
| 超参搜索 / model selection | ✅ | 所有参数 frozen (常量) |
| test set peeking | ✅ | 无 split 逻辑；noise/normalization API 由 caller 保证 |
| 用 test 估计 noise/normalization | ✅ | API 无 test 参数入口 |
| 降低 Tier A/B/C 阈值 | ✅ | 阈值未修改；true_pair 诚实报告 |
| construct 数冒充 pair 数 | ✅ | evaluate_pair_upgrade 输出 pair-level 字段 |
| annotation-only 当序列验证 pair | ✅ | `annotation_only_alt_not_verifiable` reason |
| 序列级 self-consistency 升级 | ✅ | `sequence_based_no_independent_corroboration` soft blocker |
| 修改 raw RDAT | ✅ | D1 无 raw 文件写入 |
| 删除 D0-R 候选/失败记录 | ✅ | forward-only，无删除 |
| D1 末尾自动启动训练 | ✅ | 无 training_allowed flag |

---

## 6. 数据级执行状态 (v3.1 §7 报告要求)

**状态**: D1 pipeline building blocks (T-D1.1~10) 已全部实现并通过手算 fixtures 验证（496 tests pass）。**但尚未在 D0-R v2 的 7,761 个候选上实际执行 pipeline**。

v3.1 §7 要求的数据级报告项（候选总数 / true_pair 升级数 / reason 分布 / Tier B 重判等）需要构建一个 pipeline executor 来编排 T-D1.1~10 的 building blocks，对 7,761 候选逐个运行 `evaluate_pair_upgrade` 并汇总。这是 D1 的下一步执行内容。

**当前可确认的合同层面合规**:
- 候选总数: 7,761 (D0-R v2, 未变)
- §5 解析器扩展 (T-D1.6 commit `616a8b2`): VERSION 别名 / v0.4,0.22,0.24 / GLYCFN 索引 annotation — forward-only 修复
- §3.3 HIV3PR offset 修复 (T-D1.3 commit `1d5e4e5`): substitution verification 中实现
- `training_allowed`: 仍为 `False`（D1 不自动进入训练，v3.1 §7.1）

---

## 7. 下一步

1. **构建 D1 pipeline executor**: 编排 T-D1.1~10 building blocks，对 7,761 D0-R v2 候选逐个运行 → 产出 true_pair registry + exclusion_reasons 分布
2. **数据级 D1 Gate 报告**: 候选总数 / 升级数 / reason 分布 / study-parent-owner 分布 / Tier A/B/C 重判 / 是否达到 Tier B (≥1,000 true_pair)
3. **D1 Gate 完整评审**: 合同层面 7/7 PASS + 数据层面 Tier B 判定 → 项目决策者审核
4. **D2 前置**: D1 Gate 通过后不自动进入 M0；须先完成 D2 (RSIB-v1 + 数据 Gate)

---

*生成时间: 2026-07-30*
*commit: 55d8c1d | branch: codex/reactflow-delta-d0r | push: origin/codex/reactflow-delta-d0r*
