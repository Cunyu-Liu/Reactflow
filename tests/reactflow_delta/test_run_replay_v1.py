#!/usr/bin/env python3
"""Unit fixtures for run_replay_v1 (P6 clean-checkout replay driver).

Covers:
  - _ci_from_effects: one-sided 95% t-CI from per-puzzle effects.
  - _compare: match/abs-diff logic with tolerance.
  - _replay_p2: recompute per-puzzle D and 20-puzzle CI from held-position rows
    using Gaussian CRPS at scale 0.3 (deterministic, no retrain).
  - _replay_p3: recompute rank CIs from per-puzzle rank D and the
    NO_INCREMENTAL verdict (CI upper < 0).
"""

from __future__ import annotations

import json

import numpy as np

import scripts.reactflow_delta.run_replay_v1 as R


def _crps(mu: float, y: float) -> float:
    return R.crps_gaussian(mu, R.SCALE, y)


class TestCiFromEffects:
    def test_positive_ci(self):
        ci = R._ci_from_effects([0.01 + 0.001 * i for i in range(20)])
        assert ci["n"] == 20 and ci["ci_low"] > 0.0

    def test_negative_ci_upper(self):
        ci = R._ci_from_effects([-0.01 - 0.001 * i for i in range(20)])
        assert ci["ci_high"] < 0.0


class TestCompare:
    def test_match_within_tol(self):
        assert R._compare("k", 0.5, 0.5000000005)["match"]

    def test_mismatch(self):
        assert not R._compare("k", 0.5, 0.6)["match"]

    def test_none_is_mismatch(self):
        assert not R._compare("k", None, 0.5)["match"]


class TestReplayP2:
    def _row(self, puzzle: str, target: float, pred_zero: float | None,
             pred_direct: float | None, k: int = 0) -> dict:
        return {"puzzle": puzzle, "construct": f"{puzzle}_C{k}", "edit_pos": 5,
                "ref": "A", "alt": "G", "target": target,
                "pred_zero": pred_zero, "pred_direct": pred_direct}

    def _held_rows(self, tmp_path, n_puzzles: int = 20, rows_per_puzzle: int = 50):
        p = tmp_path / "held.jsonl"
        with p.open("w") as f:
            for pi in range(n_puzzles):
                puzzle = f"P{pi + 1:02d}"
                for k in range(rows_per_puzzle):
                    f.write(json.dumps(self._row(puzzle, 0.5, 0.9, 0.5, k=k)) + "\n")
        return p

    def test_replays_positive_ci_when_direct_exact(self, tmp_path):
        out = R._replay_p2(self._held_rows(tmp_path))
        assert out["n_puzzles"] == 20
        assert out["ci20"]["ci_low"] > 0.0
        assert out["verdict"] == "PROSPECTIVE_SIGNAL_ESTABLISHED_FOR_DEVELOPMENT"

    def test_matches_hand_computed_per_puzzle(self, tmp_path):
        p = tmp_path / "held.jsonl"
        with p.open("w") as f:
            # one puzzle, one row (one record, one position)
            f.write(json.dumps(self._row("P01", 0.5, 0.9, 0.5)) + "\n")
        out = R._replay_p2(p)
        expected = _crps(0.9, 0.5) - _crps(0.5, 0.5)
        assert abs(out["per_puzzle_D"]["P01"] - expected) < 1e-12

    def test_skips_unqualified_none_rows(self, tmp_path):
        # rows with null WT reactivity -> null predictions must be skipped,
        # not crash (regression of the 4100-null-row replay failure)
        p = tmp_path / "held.jsonl"
        with p.open("w") as f:
            f.write(json.dumps(self._row("P01", 0.5, None, None)) + "\n")
            for puzzle in ("P01", "P02"):
                f.write(json.dumps(self._row(puzzle, 0.5, 0.9, 0.5)) + "\n")
        out = R._replay_p2(p)
        assert out["n_records_skipped_unqualified"] == 1
        assert out["n_puzzles"] == 2
        assert np.isfinite(out["ci20"]["ci_low"])


class TestReplayP3:
    def _p3_doc(self) -> dict:
        return {"rank_d_p3": {
            r: {f"P{i + 1:02d}": -0.02 - 0.001 * i for i in range(20)}
            for r in ("2", "4", "8")}}

    def test_replays_no_incremental_verdict(self, tmp_path):
        p = tmp_path / "p3.json"
        p.write_text(json.dumps(self._p3_doc()))
        out = R._replay_p3(p)
        for r in ("2", "4", "8"):
            assert out[r]["verdict"] == "NO_INCREMENTAL_LRSO_SKILL"
            assert out[r]["ci"]["ci_high"] < 0.0

    def test_replays_exceeds_direct_verdict(self, tmp_path):
        # v3 spec-compliant re-run: positive per-puzzle D -> ci_low > 0 ->
        # LRSO_EXCEEDS_DIRECT_FOR_DEVELOPMENT (regression for the v3 verdict path)
        doc = {"rank_d_p3": {
            r: {f"P{i + 1:02d}": 0.012 + 0.0003 * i for i in range(20)}
            for r in ("2", "4", "8")}}
        p = tmp_path / "p3.json"
        p.write_text(json.dumps(doc))
        out = R._replay_p3(p)
        for r in ("2", "4", "8"):
            assert out[r]["verdict"] == "LRSO_EXCEEDS_DIRECT_FOR_DEVELOPMENT"
            assert out[r]["ci"]["ci_low"] > 0.0
