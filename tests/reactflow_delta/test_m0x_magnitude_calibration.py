"""Unit tests for M0-X EPRO_DEV_12 magnitude-calibration core (pure, no remote).

Covers the generic, model-agnostic burden proxies introduced to fix the
negative cross-pair burden correlation (dev12 diagnosis, 20260807):
  - _within_pair_z_max  : the proxy that recovered a POSITIVE Spearman (+0.408)
  - _within_pair_z_topk_mean, _within_rank_concentration
  - pair_magnitude_proxies : unified suite reused by every model in the
    horizontal comparison (m0x_unified_zmax_compare.py).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.reactflow_delta.m0x_dev12_magnitude_calibration import (  # noqa: E402
    _within_pair_z_max, _within_pair_z_topk_mean, _within_rank_concentration,
    pair_magnitude_proxies,
)


# ---------------------------------------------------------------------------
# _within_pair_z_max: the scale-invariant burden proxy
# ---------------------------------------------------------------------------
def test_z_max_high_for_strong_hot_spot():
    # one clearly dominant position -> large max z-score
    m = np.array([0.1, 0.1, 0.1, 0.1, 5.0])
    z = _within_pair_z_max(m)
    assert z > 1.0


def test_z_max_flat_distribution_is_zero():
    # constant magnitudes -> zero std -> proxy defined as 0.0 (no hot spot)
    assert _within_pair_z_max(np.array([1.0, 1.0, 1.0])) == 0.0


def test_z_max_empty_returns_nan():
    assert np.isnan(_within_pair_z_max(np.array([])))


def test_z_max_two_point_distinguishes_relative_extremes():
    # [1,3] -> z = [-1,1] -> max = 1 (max position is one std above mean)
    assert _within_pair_z_max(np.array([1.0, 3.0])) == pytest.approx(1.0)


def test_z_max_is_scale_invariant():
    # scaling magnitudes does NOT change the within-pair z-score (the property
    # that removes cross-pair absolute-magnitude drift).
    a = np.array([0.2, 0.4, 3.0])
    b = a * 100.0
    assert _within_pair_z_max(a) == pytest.approx(_within_pair_z_max(b))


# ---------------------------------------------------------------------------
# Other proxies
# ---------------------------------------------------------------------------
def test_z_topk_mean_strong_hotspot():
    m = np.array([0.1, 0.1, 0.1, 0.1, 5.0])
    top = _within_pair_z_topk_mean(m, frac=0.2)   # top 1 of 5
    assert top > _within_pair_z_topk_mean(m, frac=0.8)


def test_rank_concentration_half_below_median():
    # even count: exactly half at/above median
    c = _within_rank_concentration(np.array([1.0, 2.0, 3.0, 4.0]))
    assert c == pytest.approx(0.5)


def test_rank_concentration_single_element_nan():
    assert np.isnan(_within_rank_concentration(np.array([2.0])))


# ---------------------------------------------------------------------------
# pair_magnitude_proxies: unified model-agnostic suite
# ---------------------------------------------------------------------------
def test_proxies_all_computed_for_every_pair():
    mags = {"p1": np.array([0.1, 0.2, 3.0]), "p2": np.array([1.0, 1.0, 1.0])}
    prox = pair_magnitude_proxies(mags)
    expected = {"raw", "log", "max", "global_rank", "within_pair_z_max",
                "within_pair_z_top20", "within_rank_concentration"}
    assert set(prox.keys()) == expected
    for name in expected:
        assert set(prox[name].keys()) == {"p1", "p2"}


def test_raw_proxy_is_mean_magnitude():
    mags = {"p1": np.array([1.0, 2.0, 3.0])}
    prox = pair_magnitude_proxies(mags)
    assert prox["raw"]["p1"] == pytest.approx(2.0)


def test_max_proxy_is_max_magnitude():
    mags = {"p1": np.array([1.0, 2.0, 5.0])}
    prox = pair_magnitude_proxies(mags)
    assert prox["max"]["p1"] == pytest.approx(5.0)


def test_zmax_proxy_recovers_cross_pair_order_for_hotspot_design():
    # p1 has a genuine dominant hot spot; p2 is flat.  under the z_max proxy p1
    # must rank above p2 (this is the exact discriminator that fixed dev12).
    mags = {"p1": np.array([0.2, 0.2, 0.2, 4.0]),
            "p2": np.array([1.0, 1.0, 1.0, 1.0])}
    prox = pair_magnitude_proxies(mags)
    assert prox["within_pair_z_max"]["p1"] > \
        prox["within_pair_z_max"]["p2"]
    # and the flat pair gets ~0
    assert prox["within_pair_z_max"]["p2"] == pytest.approx(0.0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))