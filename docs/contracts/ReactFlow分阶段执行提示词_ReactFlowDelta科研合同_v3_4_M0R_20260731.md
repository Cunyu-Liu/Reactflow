# ReactFlow-Δ EPRO v3.4 M0-R 修复授权增量科研合同

> 中文名称：ReactFlow-Δ 平衡态扰动响应算子科研合同 v3.4（M0 修复授权增量）
> 英文名称：ReactFlow-Δ EPRO Research Contract v3.4 (M0 Remediation Authorization Increment)
> 合同性质：v3.0+v3.1+v3.2+v3.3 的**增量合同**，不取代基线条款；仅在 §2 范围内**有限放宽** v3.3 对 M0 重复训练的禁止，**不放宽**任何其他禁止项。
> 授权阶段：Phase M0-R（M0 修复，M0 FAIL 后的单次 Fail-forward 修复循环）
> 授权证据：M0 已完成并 FAIL（commit `d644dee`，`artifacts/reactflow_delta/m0/pilot_v3/failure_record.json` 记录 `val_skill=-0.73`）；D2-R Tier B PASS（1509 true_pairs）。
> 前置合同：v3.3 `ReactFlow分阶段执行提示词_ReactFlowDelta科研合同_v3_3_TrainingAuth_20260731.md`（必须先签署生效）。

> **For Codex/Claude:** 本合同为 v3.3 的增量，必须与 v3.0+v3.1+v3.2+v3.3 一起阅读。本合同**只授权 §2 范围内的 M0-R 修复循环**（mutant thermo 特征 + EPRO-Lite 参数化修复 + 单次 re-pilot），**不授权** v3 §15 Phase M1（EPRO-Core）、**不授权** EPRO-DiffPF/M2、**不授权**任何超参搜索、**不授权** test 解封、**不降** Tier 阈值、**不从名称或邻近样本推断 alt**（仅允许对 3 个 alt 碱基做边际化）。本合同在决策者签署前为草案，`m0r_allowed = False`，`training_allowed` 仍为 `False`。

---

## 0. 文档权威性、继承关系与生效条件（v3.4 增补）

### 0.1 v3.4 性质

本合同是 v3.0+v3.1+v3.2+v3.3 的**增量合同**。v3.3 全部条款继续生效，**除本合同 §2 显式放宽的 M0-R 范围外**，v3.3 §2.3 禁止清单不变。

本合同**不授权**：
- v3 §15 Phase M1（EPRO-Core：6–15M 参数、bounded endpoint correction、node+edge forcing、stable sparse susceptibility、odd nonlinear switch、P1/P2、heteroscedastic output）；
- v3 §15 Phase M2（EPRO-DiffPF）；
- 任何超参搜索、test 解封、Tier 降阈、从构造名/邻近样本推断 alt。

### 0.2 继承关系

v3.4 完整继承：
- v3.0 全部架构（§3–§4）、物理与数学假设（§5）、数据优先级与 Tier 定义（§6.1、§8）、Split/污染/冻结 benchmark（§9）、强制基线/消融/负控制（§10）、测评与统计（§12）、Fail-forward（§18）、最终执行原则（§20）；
- v3.1 §2.2 全部禁止操作、§3 true_pair 升级规则、§4 D1 Gate、§6 Fail-forward 边界；
- v3.2 §5 training_allowed 边界、§6 D2-R 输出、§4.4 Fail-forward 边界；
- v3.3 §0–§10 全部条款（除本合同 §2 显式放宽项）。

v3.3 §2.3 item 1（"EPRO-Core / Phase M1 的任何实现与训练"禁止）**继续生效，本合同不放宽**。M0-R 在 EPRO-Lite 范围内（v3 §4.9：2–6M 参数、fixed thermodynamic prior、learned local forcing correction、sparse stable susceptibility、probe-specific observation、无 nonlinear switch 或仅一个全局 gate），**不构成** v3 §15 Phase M1。

### 0.3 生效条件

- 本合同由**项目决策者审核签署后生效**；在此之前为草案，`m0r_allowed = False`，`training_allowed` 仍为 `False`（v3.2 §5 + v3.3 §0.3 继续生效）。
- v3.3 必须已签署生效（`d3_allowed = True`）；v3.4 在 v3.3 生效基础上才能生效。
- 签署后 `m0r_allowed = True`，且 `training_allowed` **仅在 §2 授权范围内**置 `True`；对 v3.3 §2.3 全部禁止项，`training_allowed` 仍为 `False`。
- 签署**不**授权 Tier A 训练；Tier A 仍是下一 Gate（v3 §8.1）；未达 Tier A 前禁止启动 M1/M2（v3.3 §8 继续生效）。
- 签署**不**降低任何 Tier 阈值，**不**删除/覆盖/回缩任何历史证据（forward-only）。

### 0.4 M0 FAIL 事实记录（授权触发事实）

M0 已完成并 FAIL，记录于：
- `artifacts/reactflow_delta/m0/pilot_v3/failure_record.json`：`status=FAILED`、`gate="M0 Gate: val Skill > 0"`、`best_val_skill=-0.7129`、`pearson_r=-0.0063`、`pred_min=0.0`（架构非负性偏差）；
- commit `d644dee`（branch `codex/reactflow-delta-d0r`，已 push）；
- 诊断脚本：`scripts/reactflow_delta/diagnose_pilot.py`。

**根因（记录于 failure_record.json root_cause_analysis）**：
- 主因：M0 特征集为 WT-only thermo（unpaired_prob / positional_entropy_bits / bpp_paired_prob / norm_pos / edit_dist），**不含 mutation-effect 信息**，无法支持跨 RNA family 泛化；
- 架构表现：`delta = fixed_positive_bump(0.1) + learned_correction(correction_net)`，unseen parent 上 learned_correction 坍缩为近零，delta 被正 bump 主导，经 forcing/susceptibility/observation 后 `delta_r_hat >= 0`，无法预测负 delta（true min=-0.111）；
- Pearson r = -0.006（零相关）；
- 注意：M0 FAIL **非**模型容量不足，**非**优化失败，**非**数据分布偏移（feature_shift_sigma<0.12，delta_true 分布相似）。

**Fail-forward 层定位（v3 §18）**：
- A 数据/特征：M0 未使用 mutation-effect 信息（可修复）；
- D 物理算子：delta 参数化 `fixed_positive_bump + learned_correction` 在 OOD 上坍缩（可修复）；
- 非 G 科学假设失败：尚不能断言"paired response 跨 parent 不可学习"——M0 从未给模型真实的 mutation-effect 信号，无法判定 H1 是否被证伪。

---

## 1. 证据基础与 M0-R 修复可行性（v3.4 增补）

### 1.1 M0-R 修复路径的科学依据

交接诊断发现：mutant thermo **可计算**，无需推断 alt（合规于 v3.3 §2.3 item 7）：

- 全部 1509 true_pairs：单替换、`edit_count=1`、`encoded_ref` 已知（G:467 / A:391 / C:376 / U:275）、`encoded_alt="X"`（未知，M2-seq 约定）、`is_sequence_based=False`；
- `src/reactflow/delta/baselines.py` **已有**基础设施：
  - `alt_candidates(ref_base)` 返回 3 个非 ref 碱基；
  - `build_mutant_sequences(wt_seq, edit_pos_1idx, ref_base)` 构造 3 条可能 mutant 序列；
  - §10.2 thermo baselines 已对 3 个 alt 碱基做边际化（mean）；
- `src/reactflow/delta/thermo_state.py` **已有** `compute_wt_thermo_state(seq, temperature=37)`（ViennaRNA Python API：mfe/pf/bpp），可直接复用于 mutant 序列。

**关键合规性**：边际化（marginalize over 3 alt bases）**≠** 推断 alt（infer which alt）。边际化计算 3 个 alt 的期望特征，不声称知道真实 alt 是哪一个，符合 v3.3 §2.3 item 7。这与 baselines.py §10.2 thermo baselines 的做法完全一致。

### 1.2 M0-R 修复内容（精确界定）

M0-R 在 EPRO-Lite 范围内执行**两项**修复：

1. **特征修复（Fail-forward A 层）**：对每个 pair，构造 3 条 mutant 序列，计算 3 个 thermo state，边际化（mean）得 `mutant_thermo`；计算 `delta_thermo = mutant_thermo - wt_thermo`（逐位置：unpaired_prob / positional_entropy_bits / bpp_paired_prob / mfe_energy_kcal_mol / pf_energy_kcal_mol）。将 `delta_thermo` 加入模型输入特征。
2. **参数化修复（Fail-forward D 层）**：替换/增强 `delta = fixed_positive_bump(0.1) + learned_correction(correction_net)`，使 delta 由 `delta_thermo` 驱动（携带真实 mutation-effect 信息），消除 OOD 上的正 bump 主导与非负性偏差。

**不修复/不改动**：
- 不升级到 EPRO-Core（参数量保持 v3 §4.9 EPRO-Lite 2–6M 范围，目标 ≤5M）；
- 不引入 nonlinear switch（M0 已 `switch=disabled for epro_lite`，M0-R 维持）；
- 不引入 P1/P2、heteroscedastic output；
- 不改 split、不改 test 冻结、不改 Tier 阈值、不改主指标。

### 1.3 诚实记录边界

- M0-R 是 M0 FAIL 后的**单次** Fail-forward 修复循环，不是新阶段、不是 M1；
- M0-R PASS **不**改写 M0 FAIL 的历史记录（forward-only，v3.3 §6.5）；
- M0-R FAIL **不**触发自动再修复——按 §6 Fail-forward 终止或转 benchmark/negative result；
- M0-R 的所有 negative result 同样保留为历史证据。

---

## 2. 授权范围（v3.4 增补，machine-readable）

### 2.1 授权的阶段

| 阶段 | v3 §15 / v3.3 定义 | 是否含 learned training | 本合同授权 |
|---|---|---|---|
| Phase M0-R | M0 FAIL 后单次 Fail-forward 修复（v3 §18 A+D 层） | 是（EPRO-Lite re-pilot） | **是**（本合同唯一授权范围） |
| Phase M1 | EPRO-Core 与 P2（v3 §15） | 是 | **否**（v3.3 §2.3 item 1 继续禁止） |
| Phase M2 | EPRO-DiffPF（v3 §15） | 是 | **否** |

M0-R 前置：M0 已完成并 FAIL（§0.4）✓、v3.3 已签署生效、D2-R Tier B PASS ✓。

### 2.2 允许的操作（machine-readable）

在 §3–§6 约束下，允许：

1. **特征计算**：复用 `baselines.build_mutant_sequences` + `thermo_state.compute_wt_thermo_state`，对每个 pair 的 3 条 mutant 序列计算 thermo state，边际化（mean）得 `mutant_thermo`；计算 `delta_thermo = mutant_thermo - wt_thermo`；产出 `artifacts/reactflow_delta/m0r/mutant_thermo_features.npz`（per-pair, per-position）；
2. **特征审计**：验证 `delta_thermo` 在 train/val 的分布、与 `delta_true` 的相关性、是否高于 noise ceiling（PH0-style）；
3. **模型修改**：在 `src/reactflow/delta/model.py` 中将 `delta_thermo` 接入 EPRO-Lite 输入（augment 特征维度）或替换 delta 参数化（由 `delta_thermo` 驱动）；参数量保持 v3 §4.9 EPRO-Lite 2–6M（目标 ≤5M）；记录参数量变化前后；
4. **单 seed re-pilot**：与 M0 同 split（by-parent，val=Tetrahymena P4-P6）、同 seed、同预算上限（v3.3 §4.1）、同主指标（val Skill）、同 loss（student-t learned_scale=true，v3 §11.1）；
5. **不变量 re-audit**：复用 `invariants.py` 全套（forcing/susceptibility/switch/observation）；
6. **与基线对照**：与 strongest independent（static_reactivity Skill=-0.0068）+ matched generic paired（v3 §10.4）对照；
7. 模型选择**只在 validation（parent holdout）上**；test 集保持冻结（v3.3 §3.3）；
8. 完成后 tests、聚焦 commit、push（v3 §0.1）。

### 2.3 禁止的操作（machine-readable）

下列操作在本合同下**禁止**，`training_allowed` 对其仍为 `False`（v3.3 §2.3 全部继续生效，下列为重申 + M0-R 特有补充）：

1. **EPRO-Core / Phase M1 的任何实现与训练**（v3.3 §2.3 item 1 不放宽）——M0-R 不得引入 6–15M 参数、bounded endpoint correction、node+edge forcing、stable sparse susceptibility（新形态）、odd nonlinear switch、P1/P2、heteroscedastic output 中的任何一项；
2. **EPRO-DiffPF / Phase M2**（v3.3 §2.3 item 2）；
3. **任何超参搜索 / model selection 轮询**（v3.3 §2.3 item 3）；超参须来自冻结的 `configs/reactflow_delta/epro_lite.yaml`；新增超参（如 delta_thermo 特征维度、参数化修复的新系数）须在 `artifacts/reactflow_delta/m0r/preregistration.json` 预注册并 commit，不得事后调整；
4. **追加 seed**：M0-R 严格单 seed，与 M0 同 seed（v3.3 §2.3 item 4）；
5. **test 集解封**（v3.3 §2.3 item 5）；
6. **降低 Tier 阈值**（v3.3 §2.3 item 6）；
7. **从构造名或邻近样本推断 alt**（v3.3 §2.3 item 7）——M0-R **只允许**对 3 个 alt 碱基做边际化（mean/sum 等不变特征），**不得**选择/排序/加权特定 alt；
8. **引入新数据源 / 新 pair** 超出 D2-R 1509 true_pairs（v3.3 §2.3 item 8）；
9. **删除/覆盖/回缩** M0 FAIL 证据、failure_record.json、checkpoint、任何历史 artifact（v3.3 §2.3 item 9，forward-only）；
10. **扩大模型制造 PASS / 延长训练无上限 / 改 test 或主指标 / 删除不利 study / 回到 PCCNG / 恢复静态 SOTA 主叙事**（v3.3 §2.3 item 10）——M0-R 参数量须 ≤5M（EPRO-Lite 上限内），不得借特征修复之名升级到 EPRO-Core；
11. **预训练污染隐瞒**（v3.3 §2.3 item 11）；
12. **M0-R FAIL 后自动再修复**：M0-R 仅单次循环，FAIL 后按 §6 终止或转 benchmark/negative result，不得自动进入 M0-R2 或 M1。

### 2.4 数据源范围（预先固定）

- 训练集**仅限** D2-R 产出的 1509 true_pairs（与 M0 完全相同，v3.3 §2.4）；
- mutant 序列构造**仅限**对每个 pair 的 `encoded_ref` 位置做 3-alt 边际化，不得引入新 pair、不得解封 candidate_only；
- 若 M0-R 过程中发现数据错误，按 v3 §18 A 层处理：修复后**重新**走 D1→D2-R 升级判定，不得直接改训练集。

---

## 3. Split、污染与冻结 benchmark 约束（v3.4 增补，引用 v3.3 §3）

完全继承 v3.3 §3.1–§3.4：
- by-parent split（train=4 parents，val=Tetrahymena P4-P6，与 M0 完全一致），保证 M0-R 与 M0 可对照；
- test 集冻结（`test_study_holdout_*` / `family_ood` / `classic_classsnitch_external` / `pars_external_stress`）；
- `delta_thermo` 特征计算**不得**使用 test 集任何 label 或 structure；
- ViennaRNA 计算属物理先验，不构成 test 泄漏（与 PH0/M0 一致）。

---

## 4. 训练约束（v3.4 增补）

### 4.1 Seed 与预算冻结
- 单 seed，与 M0 pilot_v3 同 seed（`config seed`）；
- GPU：`CUDA_VISIBLE_DEVICES=2`（与 M0 一致）；
- 预算上限：与 M0 相同（`max_epochs=200`，实际可 early-stop）；
- 不得追加 seed、不得延长预算。

### 4.2 超参来源
- 基础超参来自冻结的 `configs/reactflow_delta/epro_lite.yaml`（v3.3 §4.2）；
- M0-R 新增超参（delta_thermo 特征维度、参数化修复系数、特征归一化方式）须在 `artifacts/reactflow_delta/m0r/preregistration.json` 预注册并 commit **前**写入，不得事后修改；
- 不得做 grid/random/Bayesian 搜索。

### 4.3 模型选择
- 只在 validation（parent holdout）上选；
- 主指标：val Skill（v3 §12.1）；
- 不选 test。

### 4.4 GPU 与频率
- `CUDA_VISIBLE_DEVICES=2`，不得占用其他 GPU；
- 训练频率遵循 v3 §13.1。

---

## 5. 阶段 Gate（v3.4 增补）

### 5.1 M0-R Gate（本合同唯一 Gate）

全部 bullet 须 PASS：

- **validation Skill > 0**（M0-R 主 Gate，与 M0 Gate 一致）；
- **EPRO-Lite (M0-R) ≥ strongest independent**（static_reactivity Skill=-0.0068，预注册阈值）；
- **EPRO-Lite (M0-R) > matched generic paired baseline**（v3 §10.4，强制对照）；
- **增益不只在 edit/local**（远端位置 Skill 改善，预注册"远端"定义）；
- **invariants 全 PASS**（forcing/susceptibility/switch/observation，复用 `invariants.py`）；
- **不追加 seed**；
- **参数量 ≤5M**（EPRO-Lite 范围，不得升级到 EPRO-Core）；
- **pred_min < 0**（验证非负性偏差已消除，能预测负 delta）。

**预注册**：上述阈值与"远端"/"edit/local"的度量定义须在 M0-R 启动**前**写入 `artifacts/reactflow_delta/m0r/preregistration.json` 并 commit，不得事后修改（承袭 v3.3 §5.4 预注册规则）。

### 5.2 M0-R FAIL
- 进入数据/机制诊断，**不扩大模型**（不升级 EPRO-Core）；
- 按 §6 Fail-forward 定位；
- **不**自动进入 M0-R2 或 M1；
- 诚实记录 failure_record.json，forward-only。

### 5.3 M0-R PASS
- 记录 training_run.json、mechanism_failure_matrix.json（含 M0 vs M0-R 对照）、model_card_draft.md；
- **不**自动进入 M1；M1 需 Tier A + 新合同（v3.3 §8 继续生效）；
- M0-R PASS **不**改写 M0 FAIL 历史（forward-only）。

---

## 6. Fail-forward 边界（v3.4 增补，引用 v3 §18、v3.3 §6）

### 6.1 冻结流程
M0-R FAIL 时先冻结：run/config/git/data/split/feature hashes、logs、last usable checkpoint、metrics、system metrics、invariant audit、failure evidence。

### 6.2 七层定位（v3 §18）
A 数据 / B 观测 / C 泄漏 / D 物理算子 / E 基线 / F 优化 / G 科学假设。每次最多选三个最高信息增益的最小实验。

### 6.3 允许的补救（v3 §18，但 M0-R 已是单次修复循环）
- 修复数据/评测错误；
- 缩小到 Tier B；
- 删除不能识别的可选模块；
- 转 benchmark/data/negative result；
- 安全停止模型路线。

### 6.4 禁止的补救（v3 §18）
扩大模型制造 PASS / 延长训练无上限 / 追加 seed / 改 test 或主指标 / 删除不利 study / 回到 PCCNG / 恢复静态 SOTA 主叙事。

### 6.5 forward-only
- D0/D0-R/D1/D2/D2-R/PH0/B0/O0/M0/M0-R 全部候选、失败记录、artifact 保留，不删除、不覆盖、不回缩；
- M0 FAIL 与 M0-R 结果（无论 PASS/FAIL）并列保留为历史证据；
- 不降低 v3 §8 Tier A/B/C 阈值。

---

## 7. 输出（v3.4 增补）

### 7.1 M0-R artifact
- `artifacts/reactflow_delta/m0r/preregistration.json`（启动前）；
- `artifacts/reactflow_delta/m0r/mutant_thermo_features.npz`（per-pair, per-position delta_thermo）；
- `artifacts/reactflow_delta/m0r/feature_audit.json`（delta_thermo 分布、与 delta_true 相关性、noise ceiling）；
- `artifacts/reactflow_delta/m0r/training_run.json`（seed/budget/config hash/metrics/参数量前后）；
- `artifacts/reactflow_delta/m0r/mechanism_failure_matrix.json`（M0 vs M0-R 对照：val_skill / pearson_r / pred_min / 远端 Skill）；
- `artifacts/reactflow_delta/m0r/model_card_draft.md`；
- `artifacts/reactflow_delta/m0r/failure_record.json`（仅 FAIL 时）。

### 7.2 v3.4 acceptance
输出 `artifacts/reactflow_delta/v3_4/v3_4_acceptance.json`：
- `m0r_status`：`m0r_pass / m0r_fail`；
- `training_allowed_within_scope`：签署后为 `true`（仅 §2 范围）；
- `tier_a_reached`：`false`（§8）；
- `m1_allowed`：`false`（v3.3 §2.3 item 1 继续生效）；
- 引用 M0 failure_record.json + D2-R artifact 作为证据基础。

### 7.3 Tests、commit、push
M0-R 附 tests（`tests/reactflow_delta/test_*.py`）覆盖：delta_thermo 计算正确性（3-alt 边际化 ≠ 推断 alt）、split 隔离、test 不泄漏、seed/预算冻结、参数量 ≤5M、invariant、预注册阈值、M0 vs M0-R 对照。完成后 commit + push。

---

## 8. 后续 Gate（v3.4 增补）

- Tier A（v3 §8.1）**未达**，缺口见 v3.3 §1.2；
- 本合同**不**授权为达标 Tier A 而扩集；达标需新数据/新证据/新合同；
- 在 Tier A 达标 + 新授权前，**禁止** M1（EPRO-Core）、M2（EPRO-DiffPF）（v3.3 §8 继续生效）；
- **M0-R PASS 不自动进入 M1**；M1 需 M0-R PASS + Tier A + 额外批准（v3 §8.2、§15 M1 前置）；
- **M0-R FAIL 不自动进入 M0-R2**；本合同仅授权单次修复循环。

---

## 9. 决策者签署栏（v3.4 增补）

本合同在决策者签署前为**草案**，`m0r_allowed = False`，`training_allowed = False`（v3.2 §5 + v3.3 §0.3 继续生效）。签署后 `m0r_allowed = True`，`training_allowed` **仅在 §2 授权范围内**置 `True`。

前置：v3.3 已签署生效（`d3_allowed = True`）。

- 决策者：________________  日期：________________
- 授权范围确认（M0-R：mutant thermo 特征 + EPRO-Lite 参数化修复，单 seed，固定预算，无超参搜索，test 冻结，参数量 ≤5M，不升级 EPRO-Core）：________________
- 签署即确认：不降 Tier、不删历史、forward-only、M0-R FAIL 不追加 seed 不自动再修复、M1 仍禁止、Tier A 仍是下一 Gate。

---

## 10. 一句话总结（v3.4 增补）

> M0 已 FAIL（val_skill=-0.73，根因：WT-only 特征不含 mutation-effect 信息）。本合同据此授权**单次** M0-R 修复循环：在 EPRO-Lite 范围内（≤5M 参数）加入 mutant thermo 边际化特征（delta_thermo）并修复 delta 参数化，单 seed、固定预算、无超参搜索、test 冻结、matched generic paired 强制对照。**不授权** EPRO-Core/M1、EPRO-DiffPF/M2、Tier A 训练。M0-R FAIL 不自动再修复、不升级模型，按 v3 §18 fail-forward。**不降阈、不删历史、forward-only。**
