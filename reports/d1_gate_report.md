# D1 Gate Report — ReactFlow-Δ Phase D1 Cleanup-Only

**合同**: v3.1 §4 D1 Gate (增补), v3 §15 Phase D1 Gate
**分支**: `codex/reactflow-delta-d0r`
**最新 commit**: (本提交) — feat(d1): pipeline executor + 数据级 Gate 报告 (T-D1.13, v3.1 §4/§7)
**测试**: 505 passed (0 failed, 0 errors) — `PYTHONPATH=src python -m pytest tests/reactflow_delta/`
**data.py**: 2087 lines | **schema.py**: 705 lines
**数据级产物**: `artifacts/reactflow_delta/d1/d1_true_pair_registry.json` (7,761 条) + `d1_pipeline_summary.json`
**executor**: `scripts/reactflow_delta/d1_pipeline_executor.py`

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
| 8 | **数据级 Tier B 重判 (≥1,000 true_pair)** (v3.1 §7 / v3 §8) | **FAIL (数据层面)** | 7,761 候选执行后 **true_pair = 0**（全部 `parent_lineage_unverified`）；Tier B 阈值未降，诚实报告未达。实现层面 7/7 PASS，数据层面 Tier B 未达 → 见 §6 |

**D1 Gate 结论**:
- **实现/合同层面: 7/7 bullets PASS** — T-D1.1~12 全部满足。
- **数据层面: Tier B FAIL** — 7,761 候选 0 升级为 true_pair（根因：全部 `candidate_only_pending_parent_lineage`，且 7,476 条 `annotation_only_alt_not_verifiable`）。Tier 阈值未被降低；D1 cleanup 诚实揭示了候选池无法在无 parent lineage 验证下升级，须由 D2 (RSIB-v1) 提供 lineage 验证后方可重判。

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
| T-D1.12 | tests + commit + push + Gate report | `315ad03` | 496 total | ✅ |
| T-D1.13 | D1 pipeline executor + 数据级 Gate (7,761 候选) | 本提交 | 9 (test_d1_pipeline_executor) + executor run (0 parse errors) | ✅ |

---

## 3. 测试套件总览

```
505 passed in 0.74s
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
| test_d1_pipeline_executor.py | 9 | T-D1.13 executor wiring + Tier judgment |
| test_rdat_parser.py | 15 | RDAT parser |
| test_schema.py | 13 | schema validation |
| 其余 (10 files) | 24 | manifests/registry/pairing/matrix/etc. |
| **合计** | **505** | |

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

## 6. 数据级执行结果 (v3.1 §7 / v3 §8 Tier 重判)

**状态**: ✅ 已在 D0-R v2 的 7,761 个候选上完整执行 D1 pipeline。

**Executor**: `scripts/reactflow_delta/d1_pipeline_executor.py` (T-D1.13)
- 按 `rdat_path` 分组（48 个唯一 RDAT 文件），每文件仅解析一次（`parse_rdat`）
- 对每条候选 relation 调用 T-D1.1~10 building blocks：`build_reactivity_layers` → `build_pair_delta_reactivity` → `estimate_error_variance` / `estimate_pair_noise` → `evaluate_pair_upgrade`
- 解析错误: 0；profile 查找失败: 0；7,761/7,761 全部评估

**产物**:
- `artifacts/reactflow_delta/d1/d1_true_pair_registry.json` (7,761 条，每条含 `exclusion_reasons` / `primary_eligible` / `true_pair` / `pair_quality_weight` / `quality_factors` / Δreactivity 摘要 / `caller_status`)
- `artifacts/reactflow_delta/d1/d1_pipeline_summary.json` (聚合统计 + Tier 重判)

### 6.1 候选总数与升级数

| 指标 | 值 |
|---|---|
| 候选总数 (D0-R v2) | 7,761 |
| `primary_eligible` 数 | 0 |
| **`true_pair` 升级数** | **0** |
| `caller_status` 分布 | `no_replicate_continuous_only` × 7,761 (v3 §7.3: 无 replicate → 仅连续 Δr，不输出 significant-changer) |

### 6.2 exclusion_reasons 分布

**按 reason (per-reason 计数，一条候选可有多个 reason)**:

| reason | 计数 | 占比 |
|---|---|---|
| `parent_lineage_unverified` | 7,761 | 100.0% |
| `annotation_only_alt_not_verifiable` | 7,476 | 96.3% |
| `comparable_positions_below_60pct` | 101 | 1.3% |

**按 reason 集合 (per-set 计数)**:

| reason 集合 | 计数 |
|---|---|
| `{annotation_only_alt_not_verifiable, parent_lineage_unverified}` | 7,375 |
| `{parent_lineage_unverified}` | 285 |
| `{annotation_only_alt_not_verifiable, comparable_positions_below_60pct, parent_lineage_unverified}` | 101 |

**根因分析**: 全部 7,761 候选的 `lineage_status = "candidate_only_pending_parent_lineage_and_functional_region_validation"` → `parent_lineage_verified=False` → 触发 `parent_lineage_unverified` (非 soft blocker → 同时阻断 `primary_eligible` 与 `true_pair`)。其中 7,476 条 `alt_not_verified=True` → 额外触发 `annotation_only_alt_not_verifiable`；101 条可比位置 < 60% → 额外触发 `comparable_positions_below_60pct`。285 条 `alt_not_verified=False` 仅因 lineage 未验证而未升级。

### 6.3 study / parent / owner 分布 (升级后 = 升级前，0 升级)

候选级分布（升级后无变化，true_pair 分布为空集）:

| study (citation_doi) | 候选数 | | parent_prefix | 候选数 |
|---|---|---|---|---|
| 10.1038/s41592-020-0878-9 | 4,528 | | (31 个唯一 parent_prefix) | — |
| 10.1073/pnas.1619897114 | 1,771 | | | |
| 10.1073/pnas.2320493121 | 640 | | **owner** | **候选数** |
| 10.1038/s41588-021-00830-1 | 394 | | Kalli Kappel | 4,528 |
| 10.1073/pnas.1313039111 | 220 | | Rhiju Das | 2,001 |
| 10.1038/s41594-021-00653-y | 143 | | rui huang | 640 |
| 10.7554/eLife.07600 | 55 | | Gun Woo Byeon | 394 |
| 10.1038/s41591-022-01908-x | 10 | | Ivan Zheludev | 143 |
| | | | Clarence Cheng | 55 |

- 候选级 study 数: 8 | 候选级 parent 数: 31 | 候选级 owner 数: 6
- **true_pair 级 study 数: 0 | true_pair 级 parent 数: 0 | true_pair 级 owner 数: 0**

### 6.4 probe / condition / in-vivo-in-vitro 分布

| modifier (probe) | 候选数 |
|---|---|
| unknown | 5,311 |
| DMS | 2,165 |
| SHAPE | 120 |
| nomod | 110 |
| 1M7 | 55 |

- `condition_match_status`: 全部 `match`（WT 与 mutant 同属一个 RDAT 文件、共享 modifier → v3 §6.5 condition 耦合满足）
- `in_vivo_in_vitro_mixed`: 全部 `False`（D1 pool 为 RMDB in-vitro）
- `probe_eligible_unchanged`: 全部 `True`（同文件内 probe 一致）
- `normalization_domain_compatible`: 全部 `True`（同 rmdb_id / study）

### 6.5 Tier A/B/C 重判 (用 true_pair 数，v3.1 §7: 禁止用候选数冒充)

| Tier | 基数 | 阈值 | 结果 |
|---|---|---|---|
| **Tier A** | true_pair: 0 (study=0, parent=0) | ≥5 study / ≥20 parent / ≥5,000 pair | **FAIL** |
| **Tier B** | true_pair: 0 | ≥1,000 true_pair | **FAIL** |
| **Tier C** | true_pair: 0 | ≥ Tier B | **FAIL** |

> 候选级参考（**非** D1 Gate 基数）: 8 study / 31 parent / 7,761 candidate (D0-R v2 Tier A 候选级达标，但 D1 须以 true_pair 重判 → 全部 FAIL)。

### 6.6 §5 / §3.3 修复尝试结果

| 修复项 | 实现 | 数据级效果 |
|---|---|---|
| **§5 解析器扩展** (T-D1.6, commit `616a8b2`) | VERSION 别名 / v0.4,0.22,0.24 / GLYCFN 索引 annotation — forward-only | 48 个 RDAT 文件全部 `parse_status=ok`，0 parse error；解析器扩展使全部 7,761 候选可被 pipeline 处理 |
| **§3.3 HIV3PR offset 修复** (T-D1.3, commit `1d5e4e5`) | `verify_substitution` 中 genome-numbering offset | `substitution_verified=True` 对 285 条候选生效（`alt_not_verified=False`），但仅靠序列验证无法清除 `parent_lineage_unverified` → 0 升级 |

**结论**: §5/§3.3 修复在解析/验证层面生效（0 parse error，285 条序列验证通过），但**未能使任何候选达到 true_pair**，因为 `parent_lineage_unverified` 是全部候选的阻断 reason，须由 D2 (RSIB-v1) 提供 parent lineage 验证后才能清除。

### 6.7 `training_allowed` 状态

`training_allowed`: 仍为 `False`（v3.1 §7.1）。D1 pipeline executor 不输出任何 training-authorization flag；Tier B 未达 → 不自动进入 M0。

---

## 7. D1 Gate 总结与下一步

### 7.1 D1 Gate 双层判定

| 层面 | 结果 |
|---|---|
| 实现/合同层面 (v3.1 §4 七 bullets) | **7/7 PASS** |
| 数据层面 (v3 §8 Tier 重判) | **Tier A/B/C 全 FAIL** (true_pair=0/7,761) |

**核心发现**: D1 cleanup-only 诚实揭示——D0-R v2 的 7,761 个候选**全部**为 `candidate_only_pending_parent_lineage`，在无 parent lineage 验证下**无法升级为 true_pair**。这不是 D1 的失败，而是 D1 的**正确输出**: 它证明 Tier B (≥1,000 true_pair) 无法用候选数冒充，必须通过 D2 提供 lineage 验证后重判。Tier 阈值未被降低，`training_allowed` 仍为 False。

### 7.2 下一步 (D2 前置)

1. **D2 (RSIB-v1)**: 提供 parent lineage 验证 + functional region validation → 清除 `parent_lineage_unverified` reason → 重跑 D1 pipeline executor → 重新判定 Tier B
2. **不自动进入 M0**: D1 Gate 数据层面 Tier B FAIL → 项目决策者审核是否进入 D2
3. **executor 复用**: `d1_pipeline_executor.py` 已支持重跑——D2 提供 lineage 验证后，更新 `lineage_status` 字段即可重算 true_pair 数

---

*生成时间: 2026-07-31*
*commit: (本提交) | branch: codex/reactflow-delta-d0r | push: origin/codex/reactflow-delta-d0r*
