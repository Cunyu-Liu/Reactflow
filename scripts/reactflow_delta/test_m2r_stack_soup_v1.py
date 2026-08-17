#!/usr/bin/env python3
"""Tests for m2r_stack_soup_v1 — config-soup bonus lever."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2r_stack_soup_v1 as soup


def test_metrics_consistent():
    y = np.array([0.1, 0.4, 0.9, 0.2, 0.7, 0.3])
    p = np.array([0.15, 0.35, 0.85, 0.25, 0.75, 0.35])
    mae = soup._mae(y, p)
    bl = soup._mae(y, np.full_like(y, np.median(y)))
    assert soup._skill(mae, bl) == pytest.approx(1.0 - mae / bl)


def test_seeds_and_cfg():
    assert len(soup.SEEDS) == 5
    assert soup.CFG_B["n_estimators"] == 500
    assert soup.CFG_B["max_depth"] == 8


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
