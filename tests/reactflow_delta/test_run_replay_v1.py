#!/usr/bin/env python3
"""Unit fixtures for run_replay_v1 (P6 clean-checkout replay driver).

Covers:
  - _ci_from_effects: one-sided 95% t-CI from per-puzzle effects.
  - _compare: match/abs-diff logic with tolerance.
  - _replay_p2: recompute per-puzzle D and 20-puzzle CI from held-position rows
    using Gaussian CRPS at scale 0.3 (deterministic, no retrain).
  - _replay_p3: recompute rank CIs from per-puzzle rank D and the
    NO_INCREMENTAL verdict (CI upper < 0).
  - route split: default P2/P3 only; P4/P5/P5b/P5_COMBINED require --external.
  - external authority: exact dual permission + runnable-phase fail-closed gate.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import scripts.reactflow_delta.run_replay_v1 as R


def _authority(*, top=True, nested=True, phase=R._EXTERNAL_REPLAY_PHASE) -> dict:
    return {
        "new_external_outcome_access_allowed": top,
        "authorization": {"new_external_outcome_access_allowed": nested},
        "authority": {"current_runnable_phase": phase},
    }


def _install_internal_replay_fixtures(tmp_path, monkeypatch):
    p2_replayed = {
        "per_puzzle_D": {"P01": 0.2},
        "ci20": {"n": 1, "mean": 0.2, "sd": 0.0,
                 "ci_low": 0.1, "ci_high": 0.3},
        "verdict": "PROSPECTIVE_SIGNAL_ESTABLISHED_FOR_DEVELOPMENT",
    }
    p3_replayed = {
        rank: {
            "ci": {"n": 20, "mean": 0.2, "sd": 0.01,
                   "ci_low": 0.1, "ci_high": 0.3},
            "verdict": "LRSO_EXCEEDS_DIRECT_FOR_DEVELOPMENT",
        }
        for rank in ("2", "4", "8")
    }
    locked_p2 = tmp_path / "locked_p2.json"
    locked_p2.write_text(json.dumps({
        "per_puzzle_D_p2": {"P01": 0.2},
        "p2_ci20": {"mean": 0.2, "ci_low": 0.1},
        "verdict": "PROSPECTIVE_SIGNAL_ESTABLISHED_FOR_DEVELOPMENT",
    }))
    locked_p3 = tmp_path / "locked_p3.json"
    locked_p3.write_text(json.dumps({
        **{
            f"ci_rank_{rank}": {"ci_low": 0.1, "ci_high": 0.3}
            for rank in ("2", "4", "8")
        },
        "verdict": {
            rank: "LRSO_EXCEEDS_DIRECT_FOR_DEVELOPMENT"
            for rank in ("2", "4", "8")
        },
    }))
    monkeypatch.setattr(R, "_replay_p2", lambda _path: p2_replayed)
    monkeypatch.setattr(R, "_replay_p3", lambda _path: p3_replayed)
    return locked_p2, locked_p3


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


class TestExternalReplayAuthority:
    @pytest.mark.parametrize("contract", [
        {},
        {
            "authorization": {"new_external_outcome_access_allowed": True},
            "authority": {"current_runnable_phase": R._EXTERNAL_REPLAY_PHASE},
        },
        {
            "new_external_outcome_access_allowed": True,
            "authority": {"current_runnable_phase": R._EXTERNAL_REPLAY_PHASE},
        },
        _authority(top=False, nested=False),
        _authority(top=True, nested=False),
        _authority(top=False, nested=True),
        _authority(phase="V14M3"),
    ])
    def test_missing_false_inconsistent_or_wrong_phase_is_denied(self, contract):
        with pytest.raises(PermissionError, match="P6 external replay denied"):
            R._require_external_replay_authority(contract)

    def test_only_dual_true_and_exact_phase_is_accepted(self):
        R._require_external_replay_authority(_authority())

    def test_denied_external_route_creates_nothing_and_reads_nothing(
            self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            R, "_load_active_contract", lambda: _authority(top=False, nested=False)
        )
        reader_calls = []

        def forbidden_read_text(path, *args, **kwargs):
            reader_calls.append(path)
            raise AssertionError("replay input read before authority acceptance")

        monkeypatch.setattr(Path, "read_text", forbidden_read_text)
        monkeypatch.setattr(
            R, "_replay_p2",
            lambda _path: (_ for _ in ()).throw(
                AssertionError("P2 reader called before authority acceptance")
            ),
        )
        external_calls = []
        monkeypatch.setattr(R, "run_p4", lambda *a, **k: external_calls.append("P4"))
        monkeypatch.setattr(R, "run_p5", lambda *a, **k: external_calls.append("P5"))
        monkeypatch.setattr(R, "run_p5b", lambda *a, **k: external_calls.append("P5b"))

        replay_out = tmp_path / "new_replay_dir"
        out = tmp_path / "new_report_dir" / "report.json"
        with pytest.raises(PermissionError, match="P6 external replay denied"):
            R.run_replay(
                tmp_path / "locked_p2.json",
                tmp_path / "locked_p3.json",
                tmp_path / "held_rows.jsonl",
                out,
                external=True,
                dev_csv=tmp_path / "dev.csv",
                rdat_dir=tmp_path / "rdat",
                components=tmp_path / "components.json",
                locked_p4=tmp_path / "locked_p4.json",
                locked_p5=tmp_path / "locked_p5.json",
                replay_out=replay_out,
            )

        assert reader_calls == []
        assert external_calls == []
        assert not replay_out.exists()
        assert not out.parent.exists()

    def test_external_required_args_are_validated_before_replay_dispatch(
            self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "_load_active_contract", lambda: _authority())
        monkeypatch.setattr(
            R, "_replay_p2",
            lambda _path: (_ for _ in ()).throw(
                AssertionError("internal replay dispatched before argument validation")
            ),
        )
        out = tmp_path / "new_report_dir" / "report.json"
        with pytest.raises(ValueError, match="missing required arguments"):
            R.run_replay(
                tmp_path / "locked_p2.json",
                tmp_path / "locked_p3.json",
                tmp_path / "held_rows.jsonl",
                out,
                external=True,
            )
        assert not out.parent.exists()


class TestReplayRouting:
    def test_default_run_is_p2_p3_only(self, tmp_path, monkeypatch):
        locked_p2, locked_p3 = _install_internal_replay_fixtures(tmp_path, monkeypatch)
        monkeypatch.setattr(
            R, "_load_active_contract", lambda: _authority(top=False, nested=False,
                                                            phase="V14M3")
        )

        def forbidden(*args, **kwargs):
            raise AssertionError("external replay function called by default route")

        monkeypatch.setattr(R, "run_p4", forbidden)
        monkeypatch.setattr(R, "run_p5", forbidden)
        monkeypatch.setattr(R, "run_p5b", forbidden)
        monkeypatch.setattr(R, "evaluate_combined", forbidden)

        out = tmp_path / "report.json"
        report = R.run_replay(
            locked_p2,
            locked_p3,
            tmp_path / "held_rows.jsonl",
            out,
        )
        assert set(report["replay"]) == {"P2", "P3"}
        assert report["replay_mode"] == "internal_artifact_only"
        assert report["replay_output_dir"] is None
        assert out.exists()

    def test_default_cli_does_not_require_external_arguments(self, tmp_path, monkeypatch):
        captured = {}

        def capture(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return {}

        monkeypatch.setattr(R, "run_replay", capture)
        rc = R.main([
            "--locked-p2", str(tmp_path / "p2.json"),
            "--locked-p3", str(tmp_path / "p3.json"),
            "--p2-held-rows", str(tmp_path / "held.jsonl"),
            "--out", str(tmp_path / "report.json"),
        ])
        assert rc == 0
        assert captured["kwargs"]["external"] is False
        assert captured["kwargs"]["dev_csv"] is None
        assert captured["kwargs"]["locked_p4"] is None
        assert captured["kwargs"]["locked_p5"] is None
        assert captured["kwargs"]["replay_out"] is None

    def test_authorized_external_flag_routes_p4_p5_p5b_and_combined(
            self, tmp_path, monkeypatch):
        locked_p2, locked_p3 = _install_internal_replay_fixtures(tmp_path, monkeypatch)
        monkeypatch.setattr(R, "_load_active_contract", lambda: _authority())

        p4_doc = {
            "verdict": "P4_PASS",
            "ci_zero": {"ci_low": 0.1, "mean": 0.2},
            "K_eff_realized": 3,
        }
        p5_doc = {
            "verdict": "P5_PASS",
            "band_stats": {"edit_site": {"mean": 0.2}},
        }
        p5b_doc = {
            "verdict": "P5B_PASS",
            "primary_very_far": {"ci_low": 0.1},
            "K_eff_realized": 4,
        }
        combined_doc = {
            "verdict": "P5_COMBINED_PASS",
            "inputs": {
                "total_components_across_both_sets": 7,
                "p5_set_a_verdict": "P5_PASS",
                "p5b_set_b_verdict": "P5B_PASS",
            },
            "primary_spatial_extension": {"replicated_across_both": True},
            "feature_dependence_negative_control": {
                "conceptual_overall_pass": True,
                "set_b_literal_pass": True,
            },
            "caveats": ["fixed"],
            "claim_evidence_map": [{"pass": True}],
        }

        locked_p4 = tmp_path / "locked_p4.json"
        locked_p5 = tmp_path / "locked_p5.json"
        locked_p5b = tmp_path / "locked_p5b.json"
        locked_combined = tmp_path / "locked_combined.json"
        for path, doc in (
            (locked_p4, p4_doc),
            (locked_p5, p5_doc),
            (locked_p5b, p5b_doc),
            (locked_combined, combined_doc),
        ):
            path.write_text(json.dumps(doc))

        calls = []

        def fake_p4(_rdat, _dev, _components, out):
            calls.append("P4")
            out.write_text(json.dumps(p4_doc))

        def fake_p5(_rdat, _dev, _components, _locked_p4, out):
            calls.append("P5")
            out.write_text(json.dumps(p5_doc))

        def fake_p5b(_rdat, _dev, _components, _locked_p4, out):
            calls.append("P5b")
            out.write_text(json.dumps(p5b_doc))

        def fake_combined(_p5, _p5b):
            calls.append("P5_COMBINED")
            return combined_doc

        monkeypatch.setattr(R, "run_p4", fake_p4)
        monkeypatch.setattr(R, "run_p5", fake_p5)
        monkeypatch.setattr(R, "run_p5b", fake_p5b)
        monkeypatch.setattr(R, "evaluate_combined", fake_combined)

        p5b_components = tmp_path / "p5b_components.json"
        p5b_components.write_text("{}")
        replay_out = tmp_path / "replay"
        report = R.run_replay(
            locked_p2,
            locked_p3,
            tmp_path / "held_rows.jsonl",
            tmp_path / "report.json",
            external=True,
            dev_csv=tmp_path / "dev.csv",
            rdat_dir=tmp_path / "rdat",
            components=tmp_path / "components.json",
            locked_p4=locked_p4,
            locked_p5=locked_p5,
            replay_out=replay_out,
            locked_p5b=locked_p5b,
            p5b_components=p5b_components,
            locked_p5_combined=locked_combined,
        )
        assert calls == ["P4", "P5", "P5b", "P5_COMBINED"]
        assert set(report["replay"]) == {
            "P2", "P3", "P4", "P5", "P5b", "P5_COMBINED"
        }
        assert report["replay_mode"] == "external_authorized"
