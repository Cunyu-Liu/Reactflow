# O0-X Operator Engineering 执行计划

**项目**: ReactFlow-Δ (RNA structure delta regression)
**阶段**: O0-X Operator Engineering (合同 §20.9)
**证据类别**: `ENGINEERING_ONLY`
**日期**: 2026-08-04
**仓库**: `/home/cunyuliu/reactflow_delta_goal_20260729` (A100 服务器)
**分支**: `codex/reactflow-delta-d0r`
**授权 epoch**: 11
**run_id**: `o0x_operator_engineering_20260804_v1`

---

## 1. 目标与合同定位

证明 exact endpoint-response 实现（EPRO-Small）满足数学与训练工程不变量，且可在不使用 test/mutant profile 下稳定优化。

- **PASS 解锁**: M0-X
- **禁止**: sealed validation/test consumer、把 unit PASS 写成方法成功、EPRO 方法优化、科学模型选择
- **依赖**: B0-X PASS（已闭环，commit `dd6cce3`）

## 2. 前置条件（修正案绑定）

| 前置 | 值 |
|------|-----|
| B0-X terminal PASS | 已确认 |
| B0-X terminal manifest SHA | `2500fc66dd92184f8915497aa94dd11fa69a53170ee974efcdd58f1f28d9516b` |
| B0-X sentinel SHA | `e8bfca70e9282dde6e492af71e653618784da757c7211b149d820c489524647e` |
| B0-X ledger SHA | `4eb641ae6f135ead8e4374e106c12f961897eab0b33132186b3a826b2b8aa077` |
| B0-X closure commit | `dd6cce3` |
| 授权 epoch | 11 |
| 上一 authority state | `B0X_CLOSED_AWAIT_O0X` |

## 3. 现状盘点（已核实）

已存在且全部通过（commit `69b5b12`，v3.3 时期遗留）：
- 5 个 O0 算子：`forcing` / `susceptibility` / `switch` / `observation` / `anchor` + 合成 fixture 不变量套件（`src/reactflow/delta/invariants.py`）
- 74 个 numpy-only 单元测试（`test_forcing`/`test_susceptibility`/`test_switch`/`test_observation`/`test_anchor`/`test_invariants`）— 全部 PASS
- torch `EPROModel`（`src/reactflow/delta/model.py`）forward 链路 + M0/M0-R2 不变量测试 — CPU 全 PASS
- 历史 invariant suite 报告：`/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/o0/o0_invariant_suite.json`（45/45 PASS，v3.3）

**O0-X 完成所需的实质性缺口**：
1. O0-X 权威修正案（epoch 11 授权执行）
2. **真实 CUDA forward/backward**（fallback=0）— 目前只有 CPU 测试
3. **tiny-subset overfit**（8-32 pair，训练误差 < constant baseline 的 1%）
4. 边界情况（NaN/Inf、empty mask、long sequence、all-nonchanger）
5. evaluator 与独立 reference implementation 对拍
6. finalizer/ledger/sentinel 闭环

## 4. 交付物清单

### 4.1 治理文件（3 个，Git 跟踪）
```
docs/contracts/amendments/reactflow_delta_v4_o0x_20260804.yaml   # O0-X 权威修正案
docs/approvals/reactflow_delta_v4_o0x_approval_20260804.yaml     # 批准记录
scripts/reactflow_delta/o0x_validate_authority.py                # 预检 validator
```

### 4.2 工程验证脚本（/mnt 外部存储，Git 跟踪源码）
```
scripts/reactflow_delta/o0x_run.py        # 主 runner：编排全部必测
scripts/reactflow_delta/o0x_audit.py      # 审计脚本（fail-closed）
scripts/reactflow_delta/o0x_finalize.py   # finalizer（写 ledger/sentinel）
```

### 4.3 测试（新增 torch + CUDA 测试）
```
tests/reactflow_delta/test_o0x_cuda.py     # 真实 CUDA forward/backward, fallback=0
tests/reactflow_delta/test_o0x_overfit.py  # tiny-subset overfit
tests/reactflow_delta/test_o0x_edges.py    # 边界情况
tests/reactflow_delta/test_o0x_eval_ref.py # evaluator vs reference 对拍
```

### 4.4 Artifact 输出（/mnt，外部存储）
```
/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/o0x/o0x_<timestamp>/
  ├── run_manifest.json
  ├── invariant_report.json
  ├── cuda_report.json
  ├── overfit_report.json
  ├── edge_cases_report.json
  ├── eval_ref_report.json
  ├── o0x_audit.json
  ├── manual_audit.json
  ├── terminal_manifest.yaml
  ├── SHA256SUMS
  └── O0X_CLOSED.yaml
```

## 5. 执行步骤（按 §15.4 全部必测）

### Step 1 — 权威修正案 + 预检
- 起草 O0-X 修正案/批准记录，authority_epoch=11，`run_id=o0x_operator_engineering_20260804_v1`
- `allowed_actions`：property/unit/tiny-overfit tests、CUDA forward/backward、evaluator 对拍、finalizer/ledger
- `explicit_denials`：sealed test、EPRO 方法优化、科学模型选择、seed retry、cross-project export
- 运行 `o0x_validate_authority.py` 预检（B0-X PASS、CUDA 可用、GPU 非 GPU4）

### Step 2 — 数学不变量（§15.2）
| # | 不变量 | 阈值 | 状态 |
|---|--------|------|------|
| 1 | Identity (`x_m==x_w` + no-edit → 0) | max_abs < 1e-7 | ✅ 已有 |
| 2 | P1 full swap antisymmetry | max_abs < 1e-6 | ✅ 已有 |
| 3 | P2 conditional seq/edit antisymmetry | max_abs < 1e-6 | ⚠️ 需确认构造 |
| 4 | Forcing support（mask 外=0） | max_abs < 1e-7 | ✅ 已有 |
| 5 | Deterministic eval（bitwise equal） | same / 1e-8 | ⚠️ 需新增 |
| 6 | Stability（rho_max） | ≤ 0.98 | ✅ 已有（0.95） |
| 7 | Solver residual/iteration/convergence | rel.residual < 1e-5 | ✅ 已有 |
| 8 | Probe observation monotonicity | tol 1e-7 | ✅ 已有 |
| 9 | No permanent zero-gradient | — | ⚠️ 需新增 CUDA 梯度检查 |

### Step 3 — 真实 CUDA forward/backward（核心缺口）
- 在 GPU（非 GPU4）运行 `EPROModel` forward + backward
- `fallback=0`：CUDA 不可用 → 直接 FAIL
- 验证：每个科学参数 block 梯度有限且非永久零；`all_gradients_finite`、`no_permanent_zero_grad`
- device 一致性：model/input/target/forward/backward 全在 CUDA（§18.1）

### Step 4 — Tiny-subset Overfit（核心缺口）
- 构造 8-32 pair train-only fixture（合成，无 test 数据）
- 训练：训练误差 < constant baseline 的 1%
- **失败即判工程 FAIL**（不得通过超参搜索写成科学困难）

### Step 5 — Evaluator 与独立 reference 对拍
- frozen evaluator 与独立 numpy reference implementation 输出一致
- 覆盖 pooled Skill、WMAE/MAE、mask 对齐

### Step 6 — 边界情况（核心缺口）
- NaN/Inf、empty mask、long sequence、all-nonchanger
- effective batch size、mask、edited-site exclusion
- P2 mutant-profile access audit read count = 0

### Step 7 — 审计 + finalizer + sentinel + 提交
- `o0x_audit.py`：核实 §15.4 每项必测 PASS、CUDA fallback=0、Tiny-overfit 达标
- `o0x_finalize.py`：写 `terminal_manifest.yaml` / `SHA256SUMS` / `O0X_CLOSED.yaml`（存 `/mnt`）
- 更新 active contract：`O0-X → TERMINAL/PASS`，`current_phase → M0-X`
- commit + push（`origin/codex/reactflow-delta-d0r`）
- 停在 **M0-X gate**

## 6. PASS / FAIL 判定

**PASS** = §15.4 每项必测 PASS + 真实 CUDA forward/backward + fallback=0 + tiny-overfit 达标 + finalizer/ledger 闭合。

**FAIL** = 任一 invariant / tiny-overfit / gradient / device / determinism 失败 → 路由 REPORT-X。

## 7. 合同硬约束（§18/§20.9/§24.2）

- **真实 CUDA，fallback=0**：仅 nvidia-smi 不足以证明，须记录 runtime CUDA forward/backward count 与 CPU fallback count
- **GPU 安全**：禁止抢占/杀死/修改无关用户/项目进程；不占用 GPU4
- **clone 前 read-only preflight**：记录 HEAD/branch/upstream、dirty state、GPU、artifact root、contract/data/split/config hashes
- **manifest 强制字段**：完整 run manifest（§19）含 `gpu.uuid`、`forward_calls`/`backward_calls`/`fallback_count`、`max_memory_allocated_bytes`、`preflight_snapshot_sha256`
- **Git**：原始数据/checkpoint/权重/cache/secret/完整日志不进 Git；每个实质阶段 focused commit；失败 run 不删除/不覆盖/不改标签
- **环境 allowlist**：token/key/cookie/credential 不得写 manifest/log

## 8. 风险与假设

- **风险**：现有算子代码是 v3.3 时期遗留，可能不完全符合 v4 §15.2（尤其 P2 swap 构造、deterministic eval）。需在 Step 2 逐项核对。
- **风险**：Tiny-overfit 需在 GPU 上跑一小段训练，需确认预算与超参不触发"超参搜索"禁令。
- **假设**：复用现有 5 算子 + torch EPROModel，不需重写数学核心。
- **环境**：`pc_cng_gpu`（torch 2.6.0+cu124）用于 CUDA 测试；`editflow311`（numpy）用于非依赖测试。

## 9. 待确认项（已由用户确认）

1. `run_id=o0x_operator_engineering_20260804_v1` ✅
2. `authority_epoch=11` ✅
3. 复用现有算子代码 + 只新增测试/编排层 ✅
4. 本计划落成正式 markdown 文档 ✅
