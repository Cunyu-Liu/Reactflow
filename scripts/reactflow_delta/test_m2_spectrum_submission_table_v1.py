#!/usr/bin/env python3
"""test_m2_spectrum_submission_table_v1.py — tests for the M2 response-spectrum
submission horizontal table generator."""
import json
from pathlib import Path

import m2_spectrum_submission_table_v1 as st


def _write(tmp_path, name, obj):
    p = Path(tmp_path) / name
    p.write_text(json.dumps(obj), encoding="utf-8")
    return str(p)


def _fixture_reports(tmp_path):
    attn = {
        "wmae_skill": {
            "plain_residual_mlp": {"skill": 0.0888, "wmae_model": 0.6867,
                                   "wmae_baseline": 0.7537, "ci_low": 0.0832,
                                   "ci_high": 0.0954, "permutation_p": 0.0033,
                                   "pct_positive": 0.9937},
            "position_aware": {"skill": 0.1041, "wmae_model": 0.6752,
                               "wmae_baseline": 0.7537, "ci_low": 0.0977,
                               "ci_high": 0.1114, "permutation_p": 0.0033,
                               "pct_positive": 1.0},
            "position_aware_attention": {"skill": 0.1232, "wmae_model": 0.6608,
                                         "wmae_baseline": 0.7537, "ci_low": 0.1158,
                                         "ci_high": 0.1308, "permutation_p": 0.0033,
                                         "pct_positive": 0.9937},
        }
    }
    cross = {"ensemble": {"skill": 0.1275, "wmae_model": 0.6576, "ci_low": 0.1190,
                          "ci_high": 0.1363, "permutation_p": 0.0033}}
    threeway = {
        "components": {
            "v5_attn_2layer": {"skill": 0.1263, "wmae_model": 0.6585,
                               "ci_low": 0.1190, "ci_high": 0.1345,
                               "permutation_p": 0.0033},
        },
        "ensemble": {"skill": 0.1284, "wmae_model": 0.6569, "ci_low": 0.1215,
                     "ci_high": 0.1357, "permutation_p": 0.0033,
                     "n_positions": 272988, "n_perm": 300, "n_boot": 300},
        "n_designs": 158,
    }
    fourway = {
        "components": {"v6_studentt": {"skill": 0.1064, "wmae_model": 0.6735,
                                       "ci_low": 0.1005, "ci_high": 0.1128,
                                       "permutation_p": 0.0033}},
        "grid": {"ens4_attn_heavy": {"skill": 0.1278, "wmae_model": 0.6574,
                                     "permutation_p": 0.0033}},
    }
    gbdt = {
        "n_rows_matched": 272988, "n_rows_total": 277451,
        "results": {
            "blend": {"mae": 0.6550, "skill": 0.1400,
                      "sig": {"ci_low": 0.1320, "ci_high": 0.1480,
                              "permutation_p": 0.0033}},
        },
        "blend_vs_deep": {
            "pooled_gain_pp": 1.60, "per_design_mean_pp": 1.25,
            "loo_exclusion": {"gain_mean_pp": 1.6, "gain_min_pp": 1.3,
                              "gain_max_pp": 1.7, "pct_positive": 1.0,
                              "n_folds": 159},
        },
    }
    puzzle = {
        "n_rows_matched": 272988, "n_puzzles": 20,
        "deep_component": "attn_1layer_puzzle_oof",
        "results": {
            "gbdt_puzzle": {"mae": 0.66, "skill": 0.10},
            "deep_attn_puzzle": {"mae": 0.67, "skill": 0.09},
            "blend": {"mae": 0.64, "skill": 0.15,
                      "sig": {"ci_low": 0.14, "ci_high": 0.16,
                              "permutation_p": 0.0033}},
        },
        "blend_vs_deep": {
            "pooled_gain_pp": 1.0, "per_puzzle_mean_pp": 0.8,
            "loo_exclusion": {"gain_mean_pp": 1.0, "gain_min_pp": 0.7,
                              "gain_max_pp": 1.3, "pct_positive": 1.0,
                              "n_folds": 20},
        },
    }
    return (_write(tmp_path, "attn.json", attn),
            _write(tmp_path, "cross.json", cross),
            _write(tmp_path, "threeway.json", threeway),
            _write(tmp_path, "fourway.json", fourway),
            _write(tmp_path, "gbdt.json", gbdt),
            _write(tmp_path, "puzzle.json", puzzle))


def test_build_table_structure(tmp_path):
    a, c, t, f, g, pz = _fixture_reports(tmp_path)
    rep = st.build_table(a, c, t, f, g, str(tmp_path / "out"))
    rows = rep["rows"]
    assert rows[0]["model"].startswith("baseline")
    assert rows[0]["mae"] == 0.7537
    # 1 baseline + 6 deep rows + 1 headline + 2 closed rows
    assert len(rows) == 10
    # headline flagged and is the GBDT cross-arch row
    hl = [r for r in rows if r.get("headline")][0]
    assert hl["model"].startswith("GBDT cross-arch")
    assert hl["skill"] == 0.1400
    assert hl["perm_p"] == 0.0033
    # deep 3-way row present with +12.84%
    assert any(r.get("skill") is not None and abs(r["skill"] - 0.1284) < 1e-9
               for r in rows)
    # fail-closed rows present
    assert sum(1 for r in rows if r.get("closed")) == 2


def test_build_table_puzzle_rows(tmp_path):
    a, c, t, f, g, pz = _fixture_reports(tmp_path)
    rep = st.build_table(a, c, t, f, g, str(tmp_path / "out"), puzzle_report=pz)
    rows = rep["rows"]
    # 10 design rows + 3 puzzle rows
    assert len(rows) == 13
    pz_rows = [r for r in rows if r.get("puzzle_level")]
    assert len(pz_rows) == 3
    ph = [r for r in pz_rows if r.get("puzzle_headline")][0]
    assert ph["skill"] == 0.15
    assert ph["perm_p"] == 0.0033
    assert rep["significance"]["puzzle_level"]["n_puzzles"] == 20
    assert rep["significance"]["puzzle_level"]["headline"]["blend_vs_deep_pp"] == 1.0


def test_outputs_written(tmp_path):
    a, c, t, f, g, pz = _fixture_reports(tmp_path)
    st.build_table(a, c, t, f, g, str(tmp_path / "out"), puzzle_report=pz)
    jp = Path(tmp_path) / "out" / "submission_horizontal_table_m2.json"
    mp = Path(tmp_path) / "out" / "submission_horizontal_table_m2.md"
    assert jp.exists() and mp.exists()
    j = json.loads(jp.read_text(encoding="utf-8"))
    assert j["schema"].endswith("m2_submission_horizontal_table.v1")
    md = mp.read_text(encoding="utf-8")
    assert "headline" in md and "perm_p" in md and "LOO-exclusion" in md
    assert "Puzzle-level LOPO" in md and "puzzle headline" in md
