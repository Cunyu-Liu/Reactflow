# Phase O0 Gate Report — EPRO 算子力学与不变量

**Contract**: ReactFlow-Δ EPRO V3.0 §4.3–4.8, §5.3 (Phase O0, T-O0.1~T-O0.14)
**Branch**: `codex/reactflow-delta-d0r`
**Date**: 2026-07-31

## 1. 范围

O0 proves the EPRO operator mechanics are correct **without** scientific test data.
Six modules implement the operator pipeline (forcing → susceptibility → switch →
observation → anchor) plus an invariant suite with three synthetic fixtures.

Files created:
- `src/reactflow/delta/forcing.py` — local mutation forcing operator
- `src/reactflow/delta/susceptibility.py` — stable susceptibility kernel + solver
- `src/reactflow/delta/switch.py` — odd nonlinear switch operator
- `src/reactflow/delta/observation.py` — monotone probe observation operator
- `src/reactflow/delta/anchor.py` — P2 WT-anchored posterior update + access guard
- `src/reactflow/delta/invariants.py` — invariant suite + synthetic fixtures
- `tests/reactflow_delta/test_{forcing,susceptibility,switch,observation,anchor,invariants}.py`

## 2. §5.3 数学验收阈值

| Invariant | Threshold | Result | Pass |
|---|---|---|---|
| identity error (FP32) | `max_abs < 1e-7` | forced by construction `(z_w-z_m)*w` | YES |
| swap error | `max_abs(G(a,b)+G(b,a)) < 1e-6` | forced by antisymmetric decomposition | YES |
| forcing leakage | off-support `== 0` | masked to `forcing_support_mask` | YES |
| stability | `ρ(K) ≤ ρ_max < 1` | rescaled to `ρ_max=0.95` | YES |
| solver relative residual | `< 1e-5` | direct + Neumann both audited | YES |
| probe monotonicity | derivative `≥ 0` | non-neg weights × non-decreasing basis | YES |
| P2 mutant-profile access | `== 0` (static + runtime) | vocab-disjoint API + access log | YES |

## 3. 不变量构造保证 (by construction)

- **Forcing**: `b_i = w_i(z_w,z_m)·(z_w_i − z_m_i)` with symmetric `w_i`. Identity
  (`Δ=0⇒b=0`), swap (`Δ→−Δ⇒b→−b`), and leakage (mask multiply) all hold structurally.
- **Susceptibility**: `z̄=Sym(z_w,z_m)` swap-invariant; edges = union of WT/mutant
  contacts (swap-invariant); `K` rescaled so `ρ(K)≤ρ_max`. Solver residual audited.
- **Switch**: `π=σ(f(z̄,|b|))` swap-invariant; `h_nl=π⊙tanh(S·h_lin)` with symmetric
  bias-free `S` and odd `tanh` ⇒ `h(−b)=−h(b)`. No bias term.
- **Observation**: `f_p=Σ softplus(·)·basis_k(a)` with non-negative weights and
  non-decreasing basis ⇒ derivative `≥ 0`. DMS/SHAPE/2A3 separate heads; no study ID.
- **Anchor**: `P2AnchorGuard` accepts only WT-side inputs (`q`, `σ`, prior, probe);
  forbidden vocabulary (mut_reactivity, etc.) disjoint from allowed; runtime log
  records zero mutant access.

## 4. Synthetic fixtures (T-O0.12)

- `no_change`: `z_w==z_m` ⇒ `b=0`, `h=0`, `Δr̂≈0` (identity response).
- `hairpin_release`: localized edit-window forcing propagates via sparse contact `K`.
- `two_state`: large-amplitude perturbation exercises the odd nonlinear switch.

## 5. Invariant suite (T-O0.13)

`run_invariant_suite()` aggregates all §5.3 checks across the three fixtures plus
the P2 anchor audit. Artifact: `artifacts/reactflow_delta/o0/o0_invariant_suite.json`.

- **n_checks**: 45
- **n_passed**: 45
- **failed_checks**: [] (empty)
- **all_pass**: true

## 6. 测试结果

- O0 tests (6 files): **74 passed**, 0 failed.
- Full regression (`tests/reactflow_delta/`): all passed, no regressions.

## 7. 验收 (Phase O0)

All T-O0.1~T-O0.13 complete. All §5.3 mathematical thresholds PASS by construction
and by test. O0 Gate **PASS**. Proceeding to commit+push (T-O0.14), then M0.

O0 does not touch scientific test data; it is a pure operator-mechanics audit.
