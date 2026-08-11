import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "reactflow_delta"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_deviation_v1 as ev

W = 21
NPAIR = 4
NPUB = 5
SEEDS = [0, 1, 2, 3, 4]


def _mk_rows():
    rng = np.random.default_rng(0)
    rows = []
    # baseline wmed_spectrum: prior = per-position median (here just a fixed pattern)
    prior = np.linspace(-0.3, 0.3, W)
    for p in range(NPUB):
        for pid in range(NPAIR):
            y = prior + rng.normal(0, 0.1, W)
            b = {
                "task": "magnitude_spectrum", "coverage_status": "CALLED",
                "model_variant": "wmed_spectrum", "seed": 0,
                "pair_id": f"p{pid}", "fold_id": f"pub{p}",
                "raw_prediction": prior.tolist(), "y": y.tolist(),
                "weight": [1.0] * W,
            }
            rows.append(b)
            for s in SEEDS:
                # model deviation correlates with true deviation (real signal)
                dev_true = y - prior
                pred = prior + 0.6 * dev_true + rng.normal(0, 0.2, W)
                m = {
                    "task": "magnitude_spectrum", "coverage_status": "CALLED",
                    "model_variant": "wmae_resid_deepsets_seq", "seed": s,
                    "pair_id": f"p{pid}", "fold_id": f"pub{p}",
                    "raw_prediction": pred.tolist(), "y": y.tolist(),
                    "weight": [1.0] * W,
                }
                rows.append(m)
    return rows


def test_spearman_ordering():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    assert ev._spearman(a, b) == -1.0
    assert ev._spearman(a, a) == 1.0
    c = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    assert np.isnan(ev._spearman(a, c))


def test_auroc_perfect_and_chance():
    label = np.array([1, 1, 1, 0, 0, 0])
    score_perfect = np.array([3, 2, 1, 0, -1, -2])
    score_chance = np.array([1, 2, 3, 4, 5, 6])
    assert ev._auroc(label, score_perfect) == 1.0
    assert ev._auroc(label, score_chance) == 0.0


def test_unroll_separates_variants():
    rows = _mk_rows()
    base, model = ev._unroll(rows)
    assert len(base) == NPUB * NPAIR
    for s in SEEDS:
        assert len(model[s]) == NPUB * NPAIR


def test_pub_blocks_group_by_publication():
    rows = _mk_rows()
    base, model = ev._unroll(rows)
    blocks = ev.pub_blocks(base, model, 0)
    assert len(blocks) == NPUB
    # each publication block has NPAIR*W positions
    for p in blocks.values():
        assert len(p["dt"]) == NPAIR * W


def test_pooled_metrics_positive_signal():
    rows = _mk_rows()
    base, model = ev._unroll(rows)
    m = ev.pooled_metrics(ev.pub_blocks(base, model, 0))
    assert m["spearman_signed"] > 0.3
    assert m["auroc_abs"] > 0.5


def test_perm_p_significant_under_signal():
    rows = _mk_rows()
    base, model = ev._unroll(rows)
    blocks = ev.pub_blocks(base, model, 0)
    rho, p = ev.perm_test(blocks, n_perm=100, seed=1)
    assert rho > 0.3
    assert p < 0.05


def test_perm_null_when_no_signal():
    """Under null (model deviation = noise, uncorrelated with truth) perm p should be
    high and real spearman near 0."""
    rng = np.random.default_rng(7)
    blocks = {}
    for pub in range(NPUB):
        dt = rng.normal(0, 1, NPAIR * W)
        dp = rng.normal(0, 1, NPAIR * W)  # independent noise
        blocks[f"pub{pub}"] = {"dt": dt.tolist(), "dp": dp.tolist(),
                               "adt": np.abs(dt).tolist(), "adp": np.abs(dp).tolist()}
    rho, p = ev.perm_test(blocks, n_perm=100, seed=3)
    assert abs(rho) < 0.2
    assert p > 0.05


def test_exclude_dominant_pub_reduces_n():
    rows = _mk_rows()
    base, model = ev._unroll(rows)
    blocks = ev.pub_blocks(base, model, 0, exclude_pub="pub2")
    assert len(blocks) == NPUB - 1
