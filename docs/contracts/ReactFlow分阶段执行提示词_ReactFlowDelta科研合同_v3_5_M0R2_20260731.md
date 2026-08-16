# ReactFlow-Δ EPRO v3.5 M0-R2 参数化修复授权增量科研合同

> 中文名称：ReactFlow-Δ 平衡态扰动响应算子科研合同 v3.5（M0-R2 参数化修复授权增量）
> 英文名称：ReactFlow-Δ EPRO Research Contract v3.5 (M0-R2 Parameterization Remediation Authorization Increment)
> 合同性质：v3.0+v3.1+v3.2+v3.3+v3.4 的**增量合同**，不取代基线条款；仅在 §2 范围内**有限放宽** v3.4 对 M0-R FAIL 后终止的约束，授权**单次** M0-R2 参数化修复循环。**不放宽**任何其他禁止项。
> 授权阶段：Phase M0-R2（M0-R FAIL 后的单次参数化修复循环，由决策者签署授权，非自动触发）
> 授权证据：M0-R 已完成并 FAIL（commit `03aa255`，`artifacts/reactflow_delta/m0r/failure_record.json` 记录 `best_val_skill=-0.4083`、`pred_min=0.0`、`n_negative=0/6464`）；M0-R 根因诊断为 delta 参数化非负性偏差（positive bump 主导 + correction OOD 坍缩），特征修复（delta_thermo）已验证不足。
> 前置合同：v3.4 `ReactFlow分阶段执行提示词_ReactFlowDelta科研合同_v3_4_M0R_20260731.md`（必须已签署生效，`m0r_allowed=True`）。

> **For Codex/Claude:** 本合同为 v3.4 的增量，必须与 v3.0+v3.1+v3.2+v3.3+v3.4 一起阅读。本合同**只授权 §2 范围内的 M0-R2 参数化修复循环**（去 positive bump + delta_thermo 驱动 correction_net + 单次 re-pilot），**不授权** v3 §15 Phase M1（EPRO-Core）、**不授权** EPRO-DiffPF/M2、**不授权**任何超参搜索、**不授权** test 解封、**不降** Tier 阈值、**不从名称或邻近样本推断 alt**（仅允许对 3 个 alt 碱基做边际化）。本合同在决策者签署前为草案，`m0r2_allowed = False`，`training_allowed` 仍为 `False`。

---

## 0. 文档权威性、继承关系与生效条件（v3.5 增补）

### 0.1 v3.5 性质

本合同是 v3.0+v3.1+v3.2+v3.3+v3.4 的**增量合同**。v3.4 全部条款继续生效，**除本合同 §2 显式放宽的 M0-R2 范围外**，v3.4 §2.3 禁止清单不变。

本合同**不授权**：
- v3 §15 Phase M1（EPRO-Core：6–15M 参数、bounded endpoint correction、node+edge forcing、stable sparse susceptibility、odd nonlinear switch、P1/P2、heteroscedastic output）；
- v3 §15 Phase M2（EPRO-DiffPF）；
- 任何超参搜索、test 解封、Tier 降阈、从构造名/邻近样本推断 alt。

### 0.2 合规性论证（v3.5 授权的合法性基础）

v3.5 授权 M0-R2 的合规性建立在以下四点之上：

1. **非自动触发**：v3.4 §2.3 item 12 禁止"M0-R FAIL 后**自动**再修复"。M0-R2 由**决策者签署新合同**授权，非自动触发，不违反 item 12 的"自动"禁止。
2. **不升级 EPRO-Core**：v3.4 §2.3 item 1 禁止 EPRO-Core 任何组件。M0-R2 的参数化修复（去 positive bump + delta_thermo 驱动 correction_net）**不引入** 6–15M 参数、bounded endpoint correction、node+edge forcing、stable sparse susceptibility（新形态）、odd nonlinear switch、P1/P2、heteroscedastic output 中任何一项。参数量保持 v3 §4.9 EPRO-Lite 2–6M（目标 ≤5M），仍在 EPRO-Lite 范围内。
3. **v3.4 §1.2 item 2 本意的真正执行**：v3.4 §1.2 item 2 本授权"替换/增强 delta = fixed_positive_bump(0.1) + learned_correction，使 delta 由 delta_thermo 驱动，消除 OOD 上的正 bump 主导与非负性偏差"。但 M0-R 实际执行时在 preregistration 中**保留了 bump 项**（scope 限定为 delta=bump*0.1+correction），导致参数化修复**不彻底**，非负性偏差未消除（pred_min=0.0）。v3.5 授权**彻底的**参数化修复（去 bump），是 v3.4 §1.2 item 2 本意的真正执行，而非新科学方向。
4. **不绕过 Tier A**：M0-R2 仍属 EPRO-Lite 范围（≤5M 参数），不进入 M1/EPRO-Core，因此不触发 v3.3 §8 / v3.4 §8 的"M1 需 Tier A"前置。Tier A 仍是下一 Gate，未达 Tier A 前仍禁止 M1/M2。

### 0.3 继承关系

v3.5 完整继承：
- v3.0 全部架构（§3–§4）、物理与数学假设（§5）、数据优先级与 Tier 定义（§6.1、§8）、Split/污染/冻结 benchmark（§9）、强制基线/消融/负控制（§10）、测评与统计（§12）、Fail-forward（§18）、最终执行原则（§20）；
- v3.1 §2.2 全部禁止操作、§3 true_pair 升级规则、§4 D1 Gate、§6 Fail-forward 边界；
- v3.2 §5 training_allowed 边界、§6 D2-R 输出、§4.4 Fail-forward 边界；
- v3.3 §0–§10 全部条款；
- v3.4 §0–§10 全部条款（除本合同 §2 显式放宽项）。

v3.4 §2.3 item 1（"EPRO-Core / Phase M1 的任何实现与训练"禁止）**继续生效，本合同不放宽**。M0-R2 在 EPRO-Lite 范围内（v3 §4.9：2–6M 参数、fixed thermodynamic prior、learned local forcing correction、sparse stable susceptibility、probe-specific observation、无 nonlinear switch 或仅一个全局 gate），**不构成** v3 §15 Phase M1。

### 0.4 生效条件

- 本合同由**项目决策者审核签署后生效**；在此之前为草案，`m0r2_allowed = False`，`training_allowed` 仍为 `False`（v3.2 §5 + v3.3 §0.3 + v3.4 §0.3 继续生效）。
- v3.4 必须已签署生效（`m0r_allowed = True`）；v3.5 在 v3.4 生效基础上才能生效。
- 签署后 `m0r2_allowed = True`，且 `training_allowed` **仅在 §2 授权范围内**置 `True`；对 v3.4 §2.3 全部禁止项，`training_allowed` 仍为 `False`。
- 签署**不**授权 Tier A 训练；Tier A 仍是下一 Gate（v3 §8.1）；未达 Tier A 前禁止启动 M1/M2（v3.3 §8、v3.4 §8 继续生效）。
- 签署**不**降低任何 Tier 阈值，**不**删除/覆盖/回缩任何历史证据（forward-only）。

### 0.5 M0-R FAIL 事实记录（授权触发事实）

M0-R 已完成并 FAIL，记录于：
- `artifacts/reactflow_delta/m0r/failure_record.json`：`status=FAILED`、`gate="M0-R Gate (v3.4 §5.1): all 8 bullets must pass"`、`best_val_skill=-0.4083`（ep39）、`pred_min=0.0`（n_negative=0/6464）、5/8 bullets FAIL（validation_skill_positive / beats_strongest_independent / beats_matched_generic_paired / gain_not_only_edit_local / pred_min_negative FAIL；invariants_all_pass / single_seed / param_count_within_epro_lite PASS）；
- commit `03aa255`（branch `codex/reactflow-delta-d0r`，已 push，4 files: model.py/train.py/build_m0r_features.py/epro_lite.yaml，+138/-27）。

**根因（记录于 failure_record.json root_cause_analysis + fundamental_limitation）**：
- 主因：delta 参数化 `delta = bump*0.1 + correction(z_w)` 在 OOD parent（Tetrahymena P4-P6）上 correction_net 坍缩为近零，delta 被**positive bump 主导**（bump≥0），经 forcing `b=w_sym*delta`（w_sym=softplus≥0）、susceptibility、monotone observation 后 `delta_r_hat >= 0`，**无法预测负 delta**（true min=-0.111）；
- delta_thermo 特征已加入（5→10 dim）但信号弱（|r|<0.05），+0.30 改善但不足以跨越 Gate；
- `fundamental_limitation` 字段明确："the parameterization itself must change (e.g. signed bump, or correction without positive bias, or removing the bump term)... No amount of input feature augmentation can overcome this"；
- 注意：M0-R FAIL **非**模型容量不足（4.2M ≤5M）、**非**优化失败（train_skill=0.33 正常拟合）、**非**数据分布偏移（feature_shift_sigma<0.12），而是**参数化结构性非负性偏差**。

**Fail-forward 层定位（v3 §18）**：
- A 数据/特征：delta_thermo 已加但信号弱（已尝试，非本轮修复重点）；
- D 物理算子：delta 参数化 `fixed_positive_bump + learned_correction` 在 OOD 上坍缩（**本轮修复重点**——彻底去除 positive bump，由 delta_thermo 驱动 correction）；
- 非 G 科学假设失败：M0-R 证明"特征修复不足"，但尚未测试"彻底参数化修复"——H1（paired response 跨 parent 可学习）尚未被证伪。

---

## 1. 证据基础与 M0-R2 修复可行性（v3.5 增补）

### 1.1 M0-R2 修复路径的科学依据

M0-R 的 failure_record.json `fundamental_limitation` 字段已明确指出修复方向："the parameterization itself must change (e.g. signed bump, or correction without positive bias, or removing the bump term)"。M0-R2 据此执行**彻底的参数化修复**：

- **去 positive bump**：删除 `model.py` ForcingModule L237 的 `delta_window = bump.unsqueeze(-1) * 0.1` 项，使 delta 不再被非负 bump 主导；
- **delta_thermo 驱动 correction_net**：将 correction_net 的输入从仅 `z_w[lo:hi].flatten()` 改为由 `delta_thermo`（携带 mutation-effect 信号）驱动（或 z_w + delta_thermo 拼接），解决"OOD 上 correction 坍缩为近零"的根因——delta_thermo 提供 OOD 上仍可用的物理先验信号（ViennaRNA 计算，非学习特征，不构成 test 泄漏）；
- **delta = correction_net(delta_thermo_context)**：delta 完全由学习项决定，可正可负，打破非负性链。

### 1.2 M0-R2 修复内容（精确界定）

M0-R2 在 EPRO-Lite 范围内执行**一项**核心修复（特征修复已在 M0-R 完成，本轮保留不重做）：

1. **参数化修复（Fail-forward D 层，彻底版）**：
   - 删除 ForcingModule 的 positive bump 项（L234-237 的 `bump` 与 `delta_window = bump * 0.1`）；
   - 将 correction_net 输入改为 delta_thermo 驱动（具体拼接方式在 preregistration 冻结）；
   - delta = correction_net(delta_thermo_context)，无 bump，可正可负；
   - 保留 w_sym = softplus(...) ≥0（不修改，因 delta 可负后 b=w_sym*delta 可负，已打破非负性链的 delta 环）；
   - 参数量保持 ≤5M（EPRO-Lite 范围，记录变化前后）。

**不修复/不改动**：
- 不升级到 EPRO-Core（参数量保持 v3 §4.9 EPRO-Lite 2–6M 范围，目标 ≤5M）；
- 不引入 nonlinear switch（M0/M0-R 已 `switch=disabled for epro_lite`，M0-R2 维持）；
- 不引入 P1/P2、heteroscedastic output、bounded endpoint correction、node+edge forcing、stable sparse susceptibility（新形态）中任何一项；
- 不改 split、不改 test 冻结、不改 Tier 阈值、不改主指标；
- 不重做 delta_thermo 特征计算（M0-R 已产出 `mutant_thermo_features.npz`，M0-R2 直接复用）。

### 1.3 诚实记录边界

- M0-R2 是 M0-R FAIL 后的**单次**参数化修复循环，由决策者签署授权（非自动触发），不是新阶段、不是 M1；
- M0-R2 PASS **不**改写 M0-R FAIL 的历史记录（forward-only，v3.4 §6.5）；
- M0-R2 FAIL **不**触发自动再修复——按 §6 Fail-forward 终止或转 benchmark/negative result，**禁止 M0-R3**；
- M0-R2 的所有 negative result 同样保留为历史证据。

---

## 2. 授权范围（v3.5 增补，machine-readable）

### 2.1 授权的阶段

| 阶段 | v3 §15 / v3.4 定义 | 是否含 learned training | 本合同授权 |
|---|---|---|---|
| Phase M0-R2 | M0-R FAIL 后单次参数化修复（v3 §18 D 层，决策者签署授权） | 是（EPRO-Lite re-pilot） | **是**（本合同唯一授权范围） |
| Phase M1 | EPRO-Core 与 P2（v3 §15） | 是 | **否**（v3.4 §2.3 item 1 继续禁止） |
| Phase M2 | EPRO-DiffPF（v3 §15） | 是 | **否** |

M0-R2 前置：M0-R 已完成并 FAIL（§0.5）✓、v3.4 已签署生效 ✓、D2-R Tier B PASS ✓（1509 true_pairs）。

### 2.2 允许的操作（machine-readable）

在 §3–§6 约束下，允许：

1. **模型参数化修改**：在 `src/reactflow/delta/model.py` ForcingModule 中：
   - 删除 positive bump 项（L234 `bump = torch.exp(...)` 与 L237 `delta_window = bump.unsqueeze(-1) * 0.1`）；
   - 将 correction_net 输入从 `z_w[lo:hi].flatten()`（维度 latent_dim*(2*local_window+1)=448）改为 delta_thermo 驱动（具体拼接方式：z_w + delta_thermo 或纯 delta_thermo，在 preregistration 冻结）；
   - delta = correction_net(delta_thermo_context)，无 bump，可正可负；
   - 保留 w_sym = softplus(...) ≥0（不修改）；
   - 参数量保持 ≤5M，记录变化前后；
2. **特征复用**：直接复用 M0-R 已产出的 `artifacts/reactflow_delta/m0r/mutant_thermo_features.npz`（1509 pairs 的 delta_thermo），**不重算**、**不扩展**；
3. **单 seed re-pilot**：与 M0/M0-R 同 split（by-parent，val=Tetrahymena P4-P6）、同 seed（42）、同预算上限（max_epochs=200）、同主指标（val Skill）、同 loss（student-t learned_scale=true，v3 §11.1）、同 lr（1e-4）、同 batch（8）；
4. **不变量 re-audit**：复用 `invariants.py` 全套（forcing/susceptibility/switch/observation）；
5. **与基线对照**：与 strongest independent（static_reactivity Skill=-0.0068）+ matched generic paired（v3 §10.4）+ M0-R（best_val_skill=-0.4083）三方对照；
6. 模型选择**只在 validation（parent holdout）上**；test 集保持冻结（v3.3 §3.3、v3.4 §3）；
7. 完成后 tests、聚焦 commit、push（v3 §0.1）。

### 2.3 禁止的操作（machine-readable）

下列操作在本合同下**禁止**，`training_allowed` 对其仍为 `False`（v3.4 §2.3 全部继续生效，下列为重申 + M0-R2 特有补充）：

1. **EPRO-Core / Phase M1 的任何实现与训练**（v3.4 §2.3 item 1 不放宽）——M0-R2 不得引入 6–15M 参数、bounded endpoint correction、node+edge forcing、stable sparse susceptibility（新形态）、odd nonlinear switch、P1/P2、heteroscedastic output 中的任何一项；
2. **EPRO-DiffPF / Phase M2**（v3.4 §2.3 item 2）；
3. **任何超参搜索 / model selection 轮询**（v3.4 §2.3 item 3）；超参须来自冻结的 `configs/reactflow_delta/epro_lite.yaml`；新增超参（如 correction_net 输入拼接方式、delta_thermo 输入维度、初始化方式）须在 `artifacts/reactflow_delta/m0r2/preregistration.json` 预注册并 commit，不得事后调整；
4. **追加 seed**：M0-R2 严格单 seed，与 M0/M0-R 同 seed（seed=42）（v3.4 §2.3 item 4）；
5. **test 集解封**（v3.4 §2.3 item 5）；
6. **降低 Tier 阈值**（v3.4 §2.3 item 6）；
7. **从构造名或邻近样本推断 alt**（v3.4 §2.3 item 7）——M0-R2 复用 M0-R 已产出的 delta_thermo（3-alt 边际化），**不重算**、**不选择/排序/加权特定 alt**；
8. **引入新数据源 / 新 pair** 超出 D2-R 1509 true_pairs（v3.4 §2.3 item 8）；
9. **删除/覆盖/回缩** M0/M0-R FAIL 证据、failure_record.json、checkpoint、任何历史 artifact（v3.4 §2.3 item 9，forward-only）；
10. **扩大模型制造 PASS / 延长训练无上限 / 改 test 或主指标 / 删除不利 study / 回到 PCCNG / 恢复静态 SOTA 主叙事**（v3.4 §2.3 item 10）——M0-R2 参数量须 ≤5M（EPRO-Lite 上限内），不得借参数化修复之名升级到 EPRO-Core；
11. **预训练污染隐瞒**（v3.4 §2.3 item 11）；
12. **M0-R2 FAIL 后自动再修复**：M0-R2 仅单次循环，FAIL 后按 §6 终止或转 benchmark/negative result，**不得自动进入 M0-R3 或 M1**（本条为 v3.4 §2.3 item 12 的重申与强化）；
13. **修改 w_sym 的非负约束**：M0-R2 **不**修改 w_sym = softplus(...) ≥0（参数化修复仅作用于 delta 环，不扩展到 w_sym/susceptibility/observation 环，避免触及 EPRO-Core 边界）；若实验发现仅修 delta 仍无法使 pred_min<0，须在 preregistration 中诚实记录并按 §6 Fail-forward 处理，**不**擅自扩大修改范围。

### 2.4 数据源范围（预先固定）

- 训练集**仅限** D2-R 产出的 1509 true_pairs（与 M0/M0-R 完全相同，v3.4 §2.4）；
- delta_thermo 特征**直接复用** M0-R 产出的 `mutant_thermo_features.npz`，不重算、不扩展；
- 若 M0-R2 过程中发现数据错误，按 v3 §18 A 层处理：修复后**重新**走 D1→D2-R 升级判定，不得直接改训练集。

---

## 3. Split、污染与冻结 benchmark 约束（v3.5 增补，引用 v3.4 §3）

完全继承 v3.4 §3（及 v3.3 §3.1–§3.4）：
- by-parent split（train=4 parents，val=Tetrahymena P4-P6，与 M0/M0-R 完全一致），保证 M0-R2 与 M0/M0-R 可对照；
- test 集冻结（`test_study_holdout_*` / `family_ood` / `classic_classsnitch_external` / `pars_external_stress`）；
- delta_thermo 特征为 ViennaRNA 物理先验计算（M0-R 已产出），不构成 test 泄漏（与 PH0/M0/M0-R 一致）；
- correction_net 输入修改**不得**使用 test 集任何 label 或 structure。

---

## 4. 训练约束（v3.5 增补）

### 4.1 Seed 与预算冻结
- 单 seed，与 M0/M0-R 同 seed（seed=42）；
- GPU：`CUDA_VISIBLE_DEVICES=2`（与 M0/M0-R 一致）；
- 预算上限：与 M0/M0-R 相同（`max_epochs=200`，实际可 early-stop）；
- lr=1e-4（pilot_lr，与 M0-R 一致）、batch_size=8（与 M0-R 一致）；
- 不得追加 seed、不得延长预算。

### 4.2 超参来源
- 基础超参来自冻结的 `configs/reactflow_delta/epro_lite.yaml`（v3.4 §4.2）；
- M0-R2 新增超参（correction_net 输入拼接方式、delta_thermo 输入维度、初始化方式、是否保留 z_w 输入）须在 `artifacts/reactflow_delta/m0r2/preregistration.json` 预注册并 commit **前**写入，不得事后修改；
- 不得做 grid/random/Bayesian 搜索。

### 4.3 模型选择
- 只在 validation（parent holdout）上选；
- 主指标：val Skill（v3 §12.1）；
- 不选 test。

### 4.4 GPU 与频率
- `CUDA_VISIBLE_DEVICES=2`，不得占用其他 GPU；
- 训练频率遵循 v3 §13.1。

---

## 5. 阶段 Gate（v3.5 增补）

### 5.1 M0-R2 Gate（本合同唯一 Gate，完全沿用 M0-R Gate v3.4 §5.1）

全部 8 bullets 须 PASS（与 M0-R Gate 一致，便于直接对照）：

- **validation Skill > 0**（M0-R2 主 Gate，与 M0/M0-R Gate 一致）；
- **EPRO-Lite (M0-R2) ≥ strongest independent**（static_reactivity Skill=-0.0068，预注册阈值）；
- **EPRO-Lite (M0-R2) > matched generic paired baseline**（v3 §10.4，强制对照，threshold=-0.1226）；
- **增益不只在 edit/local**（远端位置 Skill 改善，预注册"远端"定义：|seq_pos-edit_pos|/seq_len>0.10）；
- **invariants 全 PASS**（forcing/susceptibility/switch/observation，复用 `invariants.py`）；
- **不追加 seed**（seed=42）；
- **参数量 ≤5M**（EPRO-Lite 范围，不得升级到 EPRO-Core）；
- **pred_min < 0**（验证非负性偏差已消除，能预测负 delta——**M0-R2 的核心验证 bullet**，M0-R 此项 FAIL：pred_min=0.0）。

**预注册**：上述阈值与"远端"/"edit/local"的度量定义须在 M0-R2 启动**前**写入 `artifacts/reactflow_delta/m0r2/preregistration.json` 并 commit，不得事后修改（承袭 v3.4 §5.4 预注册规则）。

### 5.2 M0-R2 FAIL
- 进入数据/机制诊断，**不扩大模型**（不升级 EPRO-Core）；
- 按 §6 Fail-forward 定位；
- **不**自动进入 M0-R3 或 M1（§2.3 item 12）；
- 诚实记录 failure_record.json，forward-only。

### 5.3 M0-R2 PASS
- 记录 training_run.json、mechanism_failure_matrix.json（含 M0 vs M0-R vs M0-R2 三方对照）、model_card_draft.md；
- **不**自动进入 M1；M1 需 Tier A + 新合同（v3.4 §8 继续生效）；
- M0-R2 PASS **不**改写 M0/M0-R FAIL 历史（forward-only）。

---

## 6. Fail-forward 边界（v3.5 增补，引用 v3 §18、v3.4 §6）

### 6.1 冻结流程
M0-R2 FAIL 时先冻结：run/config/git/data/split/feature hashes、logs、last usable checkpoint、metrics、system metrics、invariant audit、failure evidence。

### 6.2 七层定位（v3 §18）
A 数据 / B 观测 / C 泄漏 / D 物理算子 / E 基线 / F 优化 / G 科学假设。每次最多选三个最高信息增益的最小实验。

### 6.3 允许的补救（v3 §18，但 M0-R2 已是单次修复循环）
- 修复数据/评测错误；
- 缩小到 Tier B；
- 删除不能识别的可选模块；
- 转 benchmark/data/negative result；
- 安全停止模型路线。

### 6.4 禁止的补救（v3 §18）
扩大模型制造 PASS / 延长训练无上限 / 追加 seed / 改 test 或主指标 / 删除不利 study / 回到 PCCNG / 恢复静态 SOTA 主叙事。

### 6.5 forward-only
- D0/D0-R/D1/D2/D2-R/PH0/B0/O0/M0/M0-R/M0-R2 全部候选、失败记录、artifact 保留，不删除、不覆盖、不回缩；
- M0 FAIL、M0-R FAIL 与 M0-R2 结果（无论 PASS/FAIL）并列保留为历史证据；
- 不降低 v3 §8 Tier A/B/C 阈值。

---

## 7. 输出（v3.5 增补）

### 7.1 M0-R2 artifact
- `artifacts/reactflow_delta/m0r2/preregistration.json`（启动前：参数化修复方案、correction_net 输入拼接方式、delta_thermo 输入维度、初始化方式、Gate 阈值、远端定义）；
- `artifacts/reactflow_delta/m0r2/training_run.json`（seed/budget/config hash/metrics/参数量前后/pred_min/n_negative）；
- `artifacts/reactflow_delta/m0r2/mechanism_failure_matrix.json`（M0 vs M0-R vs M0-R2 三方对照：val_skill / pearson_r / pred_min / n_negative / 远端 Skill / edit_local Skill）；
- `artifacts/reactflow_delta/m0r2/model_card_draft.md`；
- `artifacts/reactflow_delta/m0r2/failure_record.json`（仅 FAIL 时）。

### 7.2 v3.5 acceptance
输出 `artifacts/reactflow_delta/v3_5/v3_5_acceptance.json`：
- `m0r2_status`：`m0r2_pass / m0r2_fail`；
- `training_allowed_within_scope`：签署后为 `true`（仅 §2 范围）；
- `tier_a_reached`：`false`（§8）；
- `m1_allowed`：`false`（v3.4 §2.3 item 1 继续生效）；
- 引用 M0-R failure_record.json + M0 failure_record.json + D2-R artifact 作为证据基础。

### 7.3 Tests、commit、push
M0-R2 附 tests（`tests/reactflow_delta/test_*.py`）覆盖：
- 参数化修复正确性（delta 无 bump、可正可负、correction_net 输入由 delta_thermo 驱动）；
- 非负性偏差消除验证（pred_min<0、n_negative>0）；
- split 隔离、test 不泄漏；
- seed/预算冻结（seed=42、max_epochs=200）；
- 参数量 ≤5M；
- invariant 全 PASS；
- 预注册阈值冻结；
- M0 vs M0-R vs M0-R2 三方对照。
完成后 commit + push。

---

## 8. 后续 Gate（v3.5 增补）

- Tier A（v3 §8.1）**未达**，缺口见 v3.3 §1.2（study 2/5、parent 6/20、pair 1509/5000、无≥2 完整留出 test study）；
- 本合同**不**授权为达标 Tier A 而扩集；达标需新数据/新证据/新合同；
- 在 Tier A 达标 + 新授权前，**禁止** M1（EPRO-Core）、M2（EPRO-DiffPF）（v3.3 §8、v3.4 §8 继续生效）；
- **M0-R2 PASS 不自动进入 M1**；M1 需 M0-R2 PASS + Tier A + 额外批准（v3 §8.2、§15 M1 前置）；
- **M0-R2 FAIL 不自动进入 M0-R3**；本合同仅授权单次参数化修复循环（§2.3 item 12）。

---

## 9. 决策者签署栏（v3.5 增补）

本合同在决策者签署前为**草案**，`m0r2_allowed = False`，`training_allowed = False`（v3.2 §5 + v3.3 §0.3 + v3.4 §0.3 继续生效）。签署后 `m0r2_allowed = True`，`training_allowed` **仅在 §2 授权范围内**置 `True`。

前置：v3.4 已签署生效（`m0r_allowed = True`）。

- 决策者：________________  日期：________________
- 授权范围确认（M0-R2：去 positive bump + delta_thermo 驱动 correction_net，单 seed=42，固定预算 max_epochs=200，lr=1e-4，batch=8，无超参搜索，test 冻结，参数量 ≤5M，不升级 EPRO-Core，不修 w_sym/observation）：
- 签署即确认：不降 Tier、不删历史、forward-only、M0-R2 FAIL 不追加 seed 不自动再修复（禁止 M0-R3）、M1 仍禁止、Tier A 仍是下一 Gate。

---

## 10. 一句话总结（v3.5 增补）

> M0-R 已 FAIL（val_skill=-0.4083，pred_min=0.0，根因：delta 参数化 positive bump 主导 + correction OOD 坍缩，非负性偏差未消除）。failure_record 明确"parameterization itself must change"。本合同据此由决策者签署授权**单次** M0-R2 参数化修复循环：在 EPRO-Lite 范围内（≤5M 参数）**彻底去除 positive bump**，使 delta = correction_net(delta_thermo_context) 可正可负，打破非负性链。保留 M0-R 的 delta_thermo 特征与全部超参（seed=42、200 epochs、lr=1e-4、batch=8）。**不授权** EPRO-Core/M1、EPRO-DiffPF/M2、Tier A 训练。M0-R2 FAIL 不自动再修复（禁止 M0-R3）、不升级模型，按 v3 §18 fail-forward。**不降阈、不删历史、forward-only。**
