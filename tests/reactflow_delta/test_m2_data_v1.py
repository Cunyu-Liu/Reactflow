#!/usr/bin/env python3
"""Tests for m2_data_v1 — OpenKnot M2 loader + per-mutant sample builder."""
from __future__ import annotations

import sys
import math
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "reactflow_delta"))
import m2_data_v1 as m2d


def _row(seq, puzzle, method, mutA, sub_start, sub_end, react_vals, err_vals):
    """Build one CSV row dict with reactivity/error columns from lists (len<=177)."""
    row = {
        "id": f"{puzzle}_{method}_{mutA or 'wt'}",
        "sequence": seq, "experiment_type": "2A3_MaP",
        "SN_filter": "1", "puzzle": puzzle, "method": method,
        "sub_start": str(sub_start), "sub_end": str(sub_end), "mutA": str(mutA),
        "M2_F1": "0.5",
    }
    for i in range(1, m2d.REACT_COLS + 1):
        row[f"reactivity_{i:04d}"] = "" if i > len(react_vals) else str(react_vals[i - 1])
        row[f"reactivity_error_{i:04d}"] = "" if i > len(err_vals) else str(err_vals[i - 1])
    return row


def _write_csv(tmp_path, rows):
    path = tmp_path / "m2.csv"
    import csv
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return path


def _make_design(seq_len=30, sub_start=11, sub_end=20, n_mut=3):
    """A WT + n_mut single-nt mutants; design region seq[sub_start-1:sub_end]."""
    wt_seq = "A" * seq_len
    rows = [_row(wt_seq, "P01", "M1", "", sub_start, sub_end,
                 [1.0] * seq_len, [0.1] * seq_len)]
    for p in range(1, n_mut + 1):
        pos = sub_start - 1 + (p - 1)  # 0-indexed
        mut_seq = list(wt_seq)
        mut_seq[pos] = "C" if wt_seq[pos] == "A" else "A"
        rows.append(_row("".join(mut_seq), "P01", "M1", p, sub_start, sub_end,
                         [1.2] * seq_len, [0.1] * seq_len))
    return rows


def test_parse_m2_csv_counts_and_alignment(tmp_path):
    path = _write_csv(tmp_path, _make_design())
    designs, meta = m2d.parse_m2_csv(path)
    assert meta["n_rows"] == 4
    assert meta["n_designs"] == 1
    assert meta["n_mutants"] == 3
    d = designs[0]
    assert d["source_accession"] == "OK7a_M2_P01_M1"
    assert d["sub_start"] == 11 and d["sub_end"] == 20
    assert len(d["mutants"]) == 3
    # alignment: mutant p=1 at design pos 1 -> edit_seq_pos = 11-1+0 = 10
    assert d["mutants"][0]["edit_seq_pos"] == 10
    assert d["mutants"][2]["edit_seq_pos"] == 12
    assert d["usable"] is True


def test_parse_mark_empty_wt_unusable(tmp_path):
    wt_len = 10
    rows = [_row("A" * wt_len, "P02", "M2", "", 3, 7, [""] * wt_len, [""] * wt_len)]
    rows.append(_row("A" * wt_len, "P02", "M2", 1, 3, 7, [1.0] * wt_len, [0.1] * wt_len))
    path = _write_csv(tmp_path, rows)
    designs, meta = m2d.parse_m2_csv(path)
    assert designs[0]["usable"] is False


def test_build_samples_skip_edit_site_without_reactivity(tmp_path):
    wt_len = 30
    rows = [_row("A" * wt_len, "P01", "M1", "", 11, 20, [1.0] * wt_len, [0.1] * wt_len)]
    # mutant whose edit site reactivity is empty -> skipped
    mut_seq = list("A" * wt_len)
    mut_seq[10] = "C"
    r = _row("".join(mut_seq), "P01", "M1", 1, 11, 20, [1.0] * wt_len, [0.1] * wt_len)
    # blank reactivity at the edit site (0-indexed 10) -> col 11
    r["reactivity_0011"] = ""
    rows.append(r)
    path = _write_csv(tmp_path, rows)
    designs, _ = m2d.parse_m2_csv(path)
    samples = m2d.build_samples(designs[0])
    assert len(samples) == 0  # the only mutant is skipped


def test_build_samples_eligible_mask_and_alleles(tmp_path):
    wt_len = 30
    wt_seq = "A" * wt_len
    rows = [_row(wt_seq, "P01", "M1", "", 11, 20, [1.0] * wt_len, [0.1] * wt_len)]
    mut_seq = list(wt_seq)
    mut_seq[10] = "C"
    rows.append(_row("".join(mut_seq), "P01", "M1", 1, 11, 20,
                     [1.2] * wt_len, [0.1] * wt_len))
    path = _write_csv(tmp_path, rows)
    designs, _ = m2d.parse_m2_csv(path)
    samples = m2d.build_samples(designs[0])
    assert len(samples) == 1
    s = samples[0]
    assert s.edit_seq_pos == 10
    assert s.pair["ref_allele"] == "A"
    assert s.pair["alt_allele"] == "C"
    assert s.pair["coordinate"]["offset"] == 10
    assert sum(s.eligibility_mask) == wt_len  # all positions finite
    # build_feature-compatible record shape
    assert s.wt_rec["canonical_sequence"] == wt_seq
    assert s.wt_rec["reactivity_layers"]["train_frozen"]["reactivity"][0] == 1.0


def test_build_all_samples_skips_unusable():
    # directly test that unusable designs are excluded by build_all_samples
    d_good = {"source_accession": "OK7a_M2_P01_M1", "puzzle": "P01", "method": "M1",
              "usable": True,
              "sequence": "A" * 10, "sub_start": 3, "sub_end": 8,
              "wt_reactivity": [1.0] * 10, "wt_error": [0.1] * 10,
              "mutants": [{"mutA": 1, "edit_seq_pos": 2, "sequence": "A" * 10,
                           "reactivity": [1.0] * 10, "error": [0.1] * 10}]}
    d_bad = dict(d_good, usable=False)
    out = m2d.build_all_samples([d_good, d_bad])
    assert len(out) == 1
    assert out[0].design_id.startswith("OK7a_M2_")
