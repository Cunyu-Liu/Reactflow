"""T-M0-R2: Parameterization remediation tests (v3.5 §1.2, §2.2, §5.1).

Tests the M0-R2 parameterization fix (removing positive bump, delta_thermo
driving correction_net):

  * No positive bump: delta = correction_net(concat(z_w, delta_thermo)) only,
    can be negative (breaks non-negativity chain at delta ring).
  * correction_net input includes delta_thermo (context_dim check).
  * ForcingModule.forward returns (b, delta) tuple.
  * pred_min < 0 capability (CORE GATE validation, v3.5 §5.1 bullet 8).
  * param count <= 5,000,000 (EPRO-Lite range, v3.5 §5.1 bullet 7).
  * Key invariants still hold with M0-R2 parameterization.

Pre-registered in configs/reactflow_delta/m0r2_preregistration.json.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from reactflow.delta.model import EPROModel, EPROConfig, ForcingModule  # noqa: E402

# M0-R2 config mirrors epro_lite.yaml (local_window=50, latent_dim=64).
LOCAL_WINDOW = 50
LATENT_DIM = 64
HIDDEN_DIM = 512
N_ENCODER_LAYERS = 3
DELTA_THERMO_DIM = 5
RHO_MAX = 0.95
SUPPORT_TOL = 1e-7
MAX_PARAMS = 5_000_000


def make_m0r2_config(switch_enabled: bool = False) -> EPROConfig:
    return EPROConfig(
        model_type="epro_lite",
        latent_dim=LATENT_DIM,
        hidden_dim=HIDDEN_DIM,
        n_encoder_layers=N_ENCODER_LAYERS,
        local_window=LOCAL_WINDOW,
        rho_max=RHO_MAX,
        neumann_iter=10,
        switch_enabled=switch_enabled,
        delta_thermo_dim=DELTA_THERMO_DIM,
    )


def make_m0r2_batch(n: int = 120, seed: int = 0) -> dict:
    """Batch with delta_thermo for M0-R2 tests (n >= 2*local_window+1 = 101)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    wt = torch.randn(n, 5) * 0.5 + 0.5
    wt[:, 3] = torch.rand(n)
    wt[:, 4] = torch.rand(n)
    delta_thermo = torch.randn(n, 5) * 0.3
    features = torch.cat([wt, delta_thermo], dim=1)  # (n, 10)

    edges_list, edge_feats_list = [], []
    for i in range(n - 1):
        edges_list.append((i, i + 1))
        edges_list.append((i + 1, i))
        edge_feats_list.append([0.0, 1.0, 0.0])
        edge_feats_list.append([0.0, 1.0, 0.0])
    for _ in range(15):
        i, j = sorted(np.random.choice(n, 2, replace=False))
        if abs(i - j) <= 1:
            continue
        edges_list.append((i, j))
        edges_list.append((j, i))
        bpp = float(np.random.rand() * 0.5)
        dist = abs(i - j)
        edge_feats_list.append([bpp, float(dist), bpp])
        edge_feats_list.append([bpp, float(dist), bpp])

    edges = torch.tensor(edges_list, dtype=torch.long).T
    edge_features = torch.tensor(edge_feats_list, dtype=torch.float32)
    mask = torch.ones(n, dtype=torch.bool)
    return {
        "features": features,
        "delta_thermo": delta_thermo,
        "edit_pos": n // 2,
        "edges": edges,
        "edge_features": edge_features,
        "mask": mask,
    }


# ---------------------------------------------------------------------------
# 1. Parameterization fix: no bump, delta_thermo drives correction_net
# ---------------------------------------------------------------------------


def test_forcing_returns_tuple():
    """ForcingModule.forward returns (b, delta) tuple (M0-R2 contract)."""
    model = EPROModel(make_m0r2_config())
    model.eval()
    batch = make_m0r2_batch(n=120, seed=1)
    with torch.no_grad():
        z_w = model.encoder(batch["features"].unsqueeze(0)).squeeze(0)
        n = batch["features"].shape[0]
        result = model.forcing(z_w, batch["edit_pos"], n, batch["mask"],
                               batch["delta_thermo"])
    assert isinstance(result, tuple) and len(result) == 2, (
        "ForcingModule.forward must return (b, delta) tuple (v3.5 §1.2)"
    )
    b, delta = result
    assert b.shape == z_w.shape
    assert delta.shape == z_w.shape


def test_correction_net_input_includes_delta_thermo():
    """correction_net first layer input dim = (latent_dim + delta_thermo_dim) * window."""
    fm = ForcingModule(
        latent_dim=LATENT_DIM,
        local_window=LOCAL_WINDOW,
        hidden_dim=HIDDEN_DIM,
        learned=True,
        delta_thermo_dim=DELTA_THERMO_DIM,
    )
    expected_context = (LATENT_DIM + DELTA_THERMO_DIM) * (2 * LOCAL_WINDOW + 1)
    first_linear = fm.correction_net[0]
    assert first_linear.in_features == expected_context, (
        f"correction_net input dim={first_linear.in_features}, "
        f"expected={expected_context} (concat z_w + delta_thermo, v3.5 §2.2)"
    )


def test_delta_no_positive_bump_at_init():
    """At init (bias=0, small weights), delta in window is ~0, NOT a positive bump.

    M0-R had delta = bump*0.1 + correction where bump = exp(...) > 0 always.
    M0-R2 has delta = correction only. At init (bias=0, std=0.01 weights),
    correction ~ 0, so delta ~ 0 (not positively biased).
    """
    model = EPROModel(make_m0r2_config())
    model.eval()
    batch = make_m0r2_batch(n=120, seed=2)
    with torch.no_grad():
        out = model(batch)
    delta = out["delta"]
    ep = batch["edit_pos"]
    lw = LOCAL_WINDOW
    lo, hi = max(0, ep - lw), min(delta.shape[0], ep + lw + 1)
    delta_window = delta[lo:hi]
    # At init, delta should be near-zero (NOT a fixed positive bump).
    # M0-R would have had delta_window ~ bump*0.1 > 0.05 uniformly.
    max_abs = delta_window.abs().max().item()
    assert max_abs < 0.1, (
        f"delta at init max_abs={max_abs:.4f} suggests a bump term is present "
        f"(M0-R2 should have delta ~ 0 at init, v3.5 §1.2)"
    )


def test_delta_can_be_negative():
    """delta can be negative (core parameterization fix, v3.5 §1.2).

    Set correction_net last layer bias to -1 to force negative output,
    verify delta in window < 0. This proves the non-negativity chain is broken
    at the delta ring (b = w_sym * delta, w_sym >= 0, delta < 0 => b < 0).
    """
    model = EPROModel(make_m0r2_config())
    model.eval()
    # Force correction_net to output negative values.
    last_linear = model.forcing.correction_net[-1]
    with torch.no_grad():
        last_linear.bias.fill_(-1.0)
        last_linear.weight.zero_()
    batch = make_m0r2_batch(n=120, seed=3)
    with torch.no_grad():
        out = model(batch)
    delta = out["delta"]
    ep = batch["edit_pos"]
    lw = LOCAL_WINDOW
    lo, hi = max(0, ep - lw), min(delta.shape[0], ep + lw + 1)
    delta_window = delta[lo:hi]
    # delta should be negative in the window (correction_net outputs -1).
    n_negative = (delta_window[:, 0] < -0.01).sum().item()
    assert n_negative > 0, (
        f"delta has {n_negative} negative values in window; expected > 0 "
        f"(M0-R2 must allow negative delta, v3.5 §1.2)"
    )


def test_pred_min_can_be_negative():
    """CORE GATE: model can produce negative delta_r_hat predictions.

    This is the unit-test form of the gate bullet 'pred_min < 0' (v3.5 §5.1
    bullet 8). M0-R failed because pred_min=0.0 (n_negative=0/6464). M0-R2
    must be ABLE to predict negative values.

    We force correction_net negative and verify delta_r_hat has negative values.
    """
    model = EPROModel(make_m0r2_config())
    model.eval()
    last_linear = model.forcing.correction_net[-1]
    with torch.no_grad():
        last_linear.bias.fill_(-2.0)
        last_linear.weight.zero_()
    batch = make_m0r2_batch(n=120, seed=4)
    with torch.no_grad():
        out = model(batch)
    pred = out["delta_r_hat"]
    pred_min = pred.min().item()
    n_negative = (pred < -1e-4).sum().item()
    assert pred_min < 0.0, (
        f"pred_min={pred_min:.6f} >= 0; M0-R2 must be able to predict negative "
        f"delta_r_hat (gate bullet pred_min<0, v3.5 §5.1 bullet 8). "
        f"n_negative={n_negative}"
    )


def test_delta_thermo_required_when_learned():
    """Forcing raises ValueError when delta_thermo=None and learned=True."""
    model = EPROModel(make_m0r2_config())
    model.eval()
    batch = make_m0r2_batch(n=120, seed=5)
    z_w = model.encoder(batch["features"].unsqueeze(0)).squeeze(0)
    n = batch["features"].shape[0]
    try:
        with torch.no_grad():
            model.forcing(z_w, batch["edit_pos"], n, batch["mask"], delta_thermo=None)
    except ValueError as e:
        assert "delta_thermo" in str(e)
        return
    raise AssertionError("Expected ValueError when delta_thermo=None with learned=True")


# ---------------------------------------------------------------------------
# 2. Param count gate (v3.5 §5.1 bullet 7)
# ---------------------------------------------------------------------------


def test_param_count_within_epro_lite():
    """Trainable param count <= 5,000,000 (EPRO-Lite range, v3.5 §5.1 bullet 7)."""
    model = EPROModel(make_m0r2_config())
    pc = model.param_count()
    assert pc <= MAX_PARAMS, (
        f"param_count={pc:,} > {MAX_PARAMS:,} (EPRO-Lite limit, "
        f"v3.5 §5.1 bullet 7). Preregistration estimate: 4,460,523."
    )
    # Sanity: should be in EPRO-Lite 2-6M range.
    assert pc >= 2_000_000, (
        f"param_count={pc:,} < 2,000,000 (below EPRO-Lite range)"
    )


# ---------------------------------------------------------------------------
# 3. Invariants still hold with M0-R2 parameterization
# ---------------------------------------------------------------------------


def test_forcing_support_leakage_m0r2():
    """Off-support forcing b=0 (support leakage invariant, v3.5 §5.1 bullet 5)."""
    model = EPROModel(make_m0r2_config())
    model.eval()
    batch = make_m0r2_batch(n=120, seed=6)
    n = batch["features"].shape[0]
    ep = batch["edit_pos"]
    lw = LOCAL_WINDOW
    lo, hi = max(0, ep - lw), min(n, ep + lw + 1)
    with torch.no_grad():
        out = model(batch)
    off = out["b"].clone()
    off[lo:hi] = 0.0
    max_off = off.abs().max().item()
    assert max_off < SUPPORT_TOL, (
        f"Support leakage: off-support max|b|={max_off:.2e} > {SUPPORT_TOL}"
    )


def test_forcing_identity_m0r2():
    """h=0 => delta_r_hat=0 (observation identity invariant)."""
    model = EPROModel(make_m0r2_config())
    model.eval()
    batch = make_m0r2_batch(n=120, seed=7)
    with torch.no_grad():
        out = model(batch)
        dr = model.observation(torch.zeros_like(out["h"]), out["z_w"], out["delta"])
    max_dr = dr.abs().max().item()
    assert max_dr < 1e-6, f"Observation identity: |dr(h=0)|={max_dr:.2e}"


def test_susceptibility_stability_m0r2():
    """rho(K) < 1.0 (Neumann convergence, critical invariant)."""
    model = EPROModel(make_m0r2_config())
    model.eval()
    batch = make_m0r2_batch(n=80, seed=8)
    with torch.no_grad():
        out = model(batch)
    K = out["K"]
    n = K.shape[0]
    v = torch.ones(n) / (n ** 0.5)
    for _ in range(100):
        v = K @ v
        v = v / (v.norm() + 1e-12)
    rho = (K @ v).norm().item() / (v.norm().item() + 1e-12)
    assert rho < 1.0, (
        f"Stability CRITICAL: rho(K)={rho:.4f} >= 1.0 (Neumann diverges)"
    )


def test_observation_swap_antisymmetry_m0r2():
    """Negate h => negate delta_r_hat (swap antisymmetry invariant)."""
    model = EPROModel(make_m0r2_config())
    model.eval()
    batch = make_m0r2_batch(n=80, seed=9)
    with torch.no_grad():
        out = model(batch)
        dr_pos = model.observation(out["h"], out["z_w"], out["delta"])
        dr_neg = model.observation(-out["h"], out["z_w"], out["delta"])
    err = (dr_pos + dr_neg).abs().max().item()
    assert err < 1e-6, f"Swap antisymmetry: err={err:.2e}"


if __name__ == "__main__":
    import json
    tests = [
        ("forcing_returns_tuple", test_forcing_returns_tuple),
        ("correction_net_input_includes_delta_thermo", test_correction_net_input_includes_delta_thermo),
        ("delta_no_positive_bump_at_init", test_delta_no_positive_bump_at_init),
        ("delta_can_be_negative", test_delta_can_be_negative),
        ("pred_min_can_be_negative", test_pred_min_can_be_negative),
        ("delta_thermo_required_when_learned", test_delta_thermo_required_when_learned),
        ("param_count_within_epro_lite", test_param_count_within_epro_lite),
        ("forcing_support_leakage_m0r2", test_forcing_support_leakage_m0r2),
        ("forcing_identity_m0r2", test_forcing_identity_m0r2),
        ("susceptibility_stability_m0r2", test_susceptibility_stability_m0r2),
        ("observation_swap_antisymmetry_m0r2", test_observation_swap_antisymmetry_m0r2),
    ]
    results = {}
    all_pass = True
    for name, fn in tests:
        try:
            fn()
            results[name] = {"pass": True}
            print(f"  PASS  {name}")
        except AssertionError as e:
            results[name] = {"pass": False, "error": str(e)}
            print(f"  FAIL  {name}: {e}")
            all_pass = False
    report = {
        "schema_version": "reactflow-delta-m0r2-invariants-v1",
        "stage": "M0-R2",
        "all_pass": all_pass,
        "n_tests": len(tests),
        "n_pass": sum(1 for r in results.values() if r["pass"]),
        "results": results,
    }
    print(f"\n{'PASS' if all_pass else 'FAIL'}: {report['n_pass']}/{report['n_tests']}")
    out = "/mnt/cunyuliu/reactflow_delta_artifacts_20260729/m0r2/invariant_audit.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report: {out}")
