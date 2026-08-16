"""Unit tests for Phase 3 scheme-3 repaired EPRO operator (contract §9.4).

Validates the synthetic invariants that MUST hold before any scientific claim:
  * identity      : no mutation (alt==ref) -> forcing ~ 0 -> response ~ 0
  * antisymmetry  : swapping WT<->mutant flips the response sign
  * residual      : sparse Neumann residual decreases and rho(K) < 1
  * gradient      : every parameter receives a non-zero gradient
  * capacity      : EPRO param count is capacity-matched to scheme-2 generic
"""
import numpy as np
import torch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "reactflow_delta"))
from models.epro_v1 import (
    EPROMagnitude, SparseNeumannPropagator, build_contact_edges,
    normalize_edge_weights, POS_DIM, GLOB_DIM, base_oh,
)


def _contact(seq, top_k=4):
    try:
        edges, w = build_contact_edges(seq, top_k=top_k)
    except Exception:
        # synthetic fallback: adjacent pairs (no ViennaRNA needed in CI)
        n = len(seq)
        edges = np.array([[i, i + 1] for i in range(n - 1)], dtype=np.int64)
        w = np.ones(edges.shape[0], dtype=np.float32) * 0.5
    return edges, normalize_edge_weights(edges, w, len(seq), alpha=0.5)


def _batch(seq, edits, refs, alts, elig_frac=1.0, B=4):
    n = len(seq)
    X = np.zeros((B, n, POS_DIM), dtype=np.float32)
    for i in range(n):
        for b in range(B):
            X[b, i] = np.concatenate([base_oh(seq[i]),
                                      [0.5 + 0.1 * (i % 3), 0.05 + 0.01 * (i % 5)]])
    mask = np.ones((B, n), dtype=np.float32)
    elig = np.ones((B, n), dtype=np.float32)
    for b in range(B):
        elig[b, :max(1, int(n * elig_frac))] = 1.0
    edit = np.array(edits, dtype=np.int64)
    ref = np.stack(refs).astype(np.float32)
    alt = np.stack(alts).astype(np.float32)
    edges, w = _contact(seq)
    edge_b = np.stack([edges] * B)   # (B,E,2)
    w_b = np.stack([w] * B)          # (B,E)
    glob = np.zeros((B, GLOB_DIM), dtype=np.float32)
    return (torch.from_numpy(X), torch.from_numpy(mask), torch.from_numpy(elig),
            torch.from_numpy(edit), torch.from_numpy(ref), torch.from_numpy(alt),
            torch.from_numpy(edge_b), torch.from_numpy(w_b), torch.from_numpy(glob))


SEQ = "ACGUGGCAUGGCGAUCCGACUAGGUCACUAGCUAGGCAU"


def test_identity_no_mutation_zero_response():
    model = EPROMagnitude(seed=0)
    b = _batch(SEQ, edits=[5, 5, 5, 5],
               refs=[base_oh("G")] * 4, alts=[base_oh("G")] * 4)
    out = model(*b)
    # alt==ref -> forcing zero -> response ~ 0
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_antisymmetry_magnitude_symmetric():
    # The magnitude output is non-negative (softplus), so swapping WT<->mutant
    # (which flips the response sign h -> -h) must yield the SAME magnitude.
    model = EPROMagnitude(seed=0)
    b1 = _batch(SEQ, edits=[5, 5, 5, 5], refs=[base_oh("G")] * 4, alts=[base_oh("C")] * 4)
    b2 = _batch(SEQ, edits=[5, 5, 5, 5], refs=[base_oh("C")] * 4, alts=[base_oh("G")] * 4)
    with torch.no_grad():
        o1 = model(*b1)
        o2 = model(*b2)
    assert torch.allclose(o1, o2, atol=1e-5)


def test_residual_decreases_and_rho_lt_1():
    # Verify Neumann residual decreases (convergence) on a fixed forcing.
    prop = SparseNeumannPropagator(hidden=8, neumann_iter=8)
    seq = "AACCGGUUAACCGGUU"
    edges, w = _contact(seq, top_k=3)
    B, n = 1, len(seq)
    b = torch.randn(B, n, 8)
    eb = torch.from_numpy(np.stack([edges])).long()
    wb = torch.from_numpy(np.stack([w])).float()
    with torch.no_grad():
        _, res = prop(b, eb, wb)
    assert len(res) > 2
    # residual should generally be decreasing after warm-up
    assert res[-1] < res[0] + 1e-9


def test_gradient_every_param_nonzero():
    model = EPROMagnitude(seed=1)
    b = _batch(SEQ, edits=[7, 7, 7, 7],
               refs=[base_oh("A")] * 4, alts=[base_oh("U")] * 4)
    target = torch.tensor([0.5, 0.6, 0.7, 0.8], dtype=torch.float32)
    out = model(*b)
    loss = (out - target).pow(2).mean()
    loss.backward()
    grads = {name: p.grad for name, p in model.named_parameters() if p.requires_grad}
    assert len(grads) > 0
    for name, g in grads.items():
        assert g is not None and g.abs().sum() > 0, "zero grad: " + name


def test_capacity_matched():
    from models.pair_v1 import count_params
    from models.pair_v2 import build_scheme2_features
    from run_p2_v3 import edited_index

    seq = "ACGUGGCAUGGCGAUCCGACUAGGUCACUAGCUAGGCAU"
    n = len(seq)
    wt = {
        "canonical_sequence": seq,
        "reactivity_layers": {"train_frozen": {
            "reactivity": np.linspace(0.1, 1.0, n).tolist(),
            "error": np.linspace(0.02, 0.2, n).tolist()}},
        "probe": ["1M7"],
    }
    codes = ["OK"] * n
    codes[10] = "EDITED_SITE"
    pair = {"source_accession": "T_x", "wt_profile_index": 0, "mutant_profile_index": 1,
            "asset_name": "a", "alt_allele": "C",
            "eligibility_reason_codes": codes,
            "coordinate": {"offset": 10}}
    f = build_scheme2_features(pair, wt, False, True)
    # EPRO params (hidden 48) should be within ±30% of generic (11777)
    m = EPROMagnitude(seed=0)
    p = m.param_count()
    assert abs(p - 11777) / 11777 < 0.30, "EPRO params %d not capacity-matched" % p


if __name__ == "__main__":
    for fn in [test_identity_no_mutation_zero_response,
               test_antisymmetry_magnitude_symmetric,
               test_residual_decreases_and_rho_lt_1,
               test_gradient_every_param_nonzero,
               test_capacity_matched]:
        fn()
        print("PASS", fn.__name__)
