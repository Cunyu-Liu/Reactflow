#!/usr/bin/env python3
"""m2r_data_v1.py — OpenKnot M2R (Mutate-Map-Rescue) dataset loader.

Purpose
-------
M2R-seq data directly probe RNA secondary-structure pairing mechanisms:
for each design (puzzle x method) and each base pair (i,j) in the target
pseudoknotted structure, there are:
  * a single mutant at i (disrupts the pair)
  * a single mutant at j (disrupts the pair)
  * a double mutant (i,j) which should restore (rescue) the pair
  * rescue_factor = fraction of SHAPE-profile RMSD of the single mutants
    (added in quadrature) that the double mutant restores.
    0 => no rescue (no pair evidence); 1 => full rescue (pair confirmed).

This module parses the M2R CSV into per-pair samples with:
  * the WT sequence + reactivity + error
  * the single-mutant sequences + reactivity + error (mutA and mutB rows)
  * the double-mutant sequence + reactivity + error
  * rescue_factor (the prediction target)
  * the full target_structure (dot-bracket) for the design
  * per-position features for the pair sites

Exchangeable unit = (puzzle, method) design (same as M2).

Everything is pure/stdlib + numpy and unit-testable.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

REACT_COLS = 177
M2R_SOURCE_URL = ("https://media.githubusercontent.com/media/eternagame/"
                  "OpenKnotAIDesignData/main/Data/OK7a_M2R_data.v4.5.1.csv")
M2R_SCHEMA = "reactflow_delta.m2r_data.v1"


def _parse_arrays(row: dict, prefix: str) -> list[Optional[float]]:
    out: list[Optional[float]] = []
    for i in range(1, REACT_COLS + 1):
        v = row.get(f"{prefix}{i:04d}")
        out.append(None if v in (None, "") else float(v))
    return out


def attach_m2_structure(designs: list[dict], m2_csv_path: str) -> None:
    """Attach M2_structure / M2_F1 / M2_F1_crossed_pair to each M2R design.

    The M2 experiment (OK7a_M2_data) is an INDEPENDENT data source from the
    M2R rescue_factor target: it measures single-mutant SHAPE reactivity and
    infers the experimentally observed secondary structure via ShapeKnots.
    M2 and M2R share the same (puzzle, method) designs.

    Modifies designs in place: adds keys
      "m2_structure", "m2_f1", "m2_f1_crossed_pair", "m2_sub_start".
    If no matching M2 design is found, the keys default to empty/None.
    """
    with open(m2_csv_path, newline="", encoding="utf-8") as fh:
        m2_by_key = {}
        for row in csv.DictReader(fh):
            if not row.get("mutA") and not row.get("mutB"):
                key = (row["puzzle"], row["method"])
                m2_by_key[key] = {
                    "m2_structure": row.get("M2_structure") or "",
                    "m2_f1": float(row["M2_F1"]) if row.get("M2_F1") else None,
                    "m2_f1_crossed_pair": float(row["M2_F1_crossed_pair"])
                    if row.get("M2_F1_crossed_pair") else None,
                    "m2_sub_start": int(row["sub_start"]) if row.get("sub_start") else None,
                }
    for d in designs:
        m2 = m2_by_key.get((d["puzzle"], d["method"]), {})
        d["m2_structure"] = m2.get("m2_structure", "")
        d["m2_f1"] = m2.get("m2_f1")
        d["m2_f1_crossed_pair"] = m2.get("m2_f1_crossed_pair")
        d["m2_sub_start"] = m2.get("m2_sub_start")


def _has(x) -> bool:
    return x not in (None, "")


def parse_m2r_csv(path) -> tuple[list[dict], dict]:
    """Parse M2R CSV into design records.

    Each design dict:
      {
        "puzzle", "method", "source_accession",
        "sequence", "sub_start", "sub_end", "target_structure",
        "wt_reactivity", "wt_error",
        "pairs": [ {
            "mutA", "mutB",            # 1-indexed design positions of the pair
            "mutA_seq", "mutB_seq",    # mutated full sequences
            "double_seq",
            "rescue_factor",
            "singleA_reactivity", "singleA_error",
            "singleB_reactivity", "singleB_error",
            "double_reactivity", "double_error",
        }, ... ],
        "usable": bool,
      }
    """
    designs: dict[tuple[str, str], dict] = {}
    meta = {"n_rows": 0, "n_designs": 0, "n_pairs": 0}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            meta["n_rows"] += 1
            puzzle, method = row["puzzle"], row["method"]
            key = (puzzle, method)
            d = designs.get(key)
            if d is None:
                d = {
                    "puzzle": puzzle, "method": method,
                    "source_accession": f"OK7a_M2R_{puzzle}_{method}",
                    "sequence": row["sequence"],
                    "sub_start": int(row["sub_start"]) if _has(row["sub_start"]) else None,
                    "sub_end": int(row["sub_end"]) if _has(row["sub_end"]) else None,
                    "target_structure": row.get("target_structure") or "",
                    "wt_reactivity": _parse_arrays(row, "reactivity_"),
                    "wt_error": _parse_arrays(row, "reactivity_error_"),
                    "pairs": [], "usable": True,
                }
                designs[key] = d
                meta["n_designs"] += 1
            else:
                if not _has(row["mutA"]) and not _has(row["mutB"]):
                    d["wt_reactivity"] = _parse_arrays(row, "reactivity_")
                    d["wt_error"] = _parse_arrays(row, "reactivity_error_")
                    continue

            mutA = row.get("mutA")
            mutB = row.get("mutB")
            # ---- pair row: the double mutant row defines the pair ----
            if _has(mutA) and _has(mutB):
                d["pairs"].append({
                    "mutA": int(mutA), "mutB": int(mutB),
                    "mutA_seq": row["sequence"],
                    "mutB_seq": None,   # filled below from single rows
                    "double_seq": row["sequence"],
                    "rescue_factor": float(row["rescue_factor"]) if _has(row["rescue_factor"]) else None,
                    "singleA_reactivity": _parse_arrays(row, "reactivity_"),
                    "singleA_error": _parse_arrays(row, "reactivity_error_"),
                    "singleB_reactivity": None, "singleB_error": None,
                    "double_reactivity": _parse_arrays(row, "reactivity_"),
                    "double_error": _parse_arrays(row, "reactivity_error_"),
                })
                meta["n_pairs"] += 1

    # second pass: attach single-mutant rows (mutA only / mutB only) to their pair
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            mutA = row.get("mutA"); mutB = row.get("mutB")
            if _has(mutA) and not _has(mutB):
                key = (row["puzzle"], row["method"])
                d = designs.get(key)
                if d is None:
                    continue
                ma = int(mutA)
                for p in d["pairs"]:
                    if p["mutA"] == ma:
                        p["mutA_seq"] = row["sequence"]
                        p["singleA_reactivity"] = _parse_arrays(row, "reactivity_")
                        p["singleA_error"] = _parse_arrays(row, "reactivity_error_")
            elif _has(mutB) and not _has(mutA):
                key = (row["puzzle"], row["method"])
                d = designs.get(key)
                if d is None:
                    continue
                mb = int(mutB)
                for p in d["pairs"]:
                    if p["mutB"] == mb:
                        p["mutB_seq"] = row["sequence"]
                        p["singleB_reactivity"] = _parse_arrays(row, "reactivity_")
                        p["singleB_error"] = _parse_arrays(row, "reactivity_error_")

    # mark designs with no pairs as unusable
    for d in designs.values():
        if not d["pairs"]:
            d["usable"] = False
    return list(designs.values()), meta


@dataclass
class M2RPair:
    """A single (i,j) base-pair rescue experiment sample."""
    design_id: str
    puzzle: str
    method: str
    mutA: int          # 1-indexed design position
    mutB: int
    editA_seq_pos: int  # 0-indexed full-sequence index of mutA
    editB_seq_pos: int
    sequence: str
    wt_reactivity: list
    wt_error: list
    singleA_reactivity: list
    singleA_error: list
    singleB_reactivity: list
    singleB_error: list
    double_reactivity: list
    double_error: list
    rescue_factor: Optional[float]
    eligibility_mask: list
    target_structure: str
    sub_start: int
    sub_end: int
    mutA_seq: Optional[str] = None   # full sequence of the single-A mutant
    mutB_seq: Optional[str] = None   # full sequence of the single-B mutant
    m2_structure: str = ""
    m2_f1: Optional[float] = None
    m2_f1_crossed_pair: Optional[float] = None


def build_pair_samples(design: dict) -> list[M2RPair]:
    """Build one M2RPair per double-mutant pair of a design."""
    seq = design["sequence"]
    sub_start = design["sub_start"] if design["sub_start"] is not None else 1
    out = []
    for p in design["pairs"]:
        editA = sub_start - 1 + (p["mutA"] - 1)
        editB = sub_start - 1 + (p["mutB"] - 1)
        mask = [1 if (a is not None and b is not None
                      and np.isfinite(a) and np.isfinite(b))
                else 0
                for a, b in zip(design["wt_reactivity"], p["double_reactivity"])]
        out.append(M2RPair(
            design_id=design["source_accession"],
            puzzle=design["puzzle"], method=design["method"],
            mutA=p["mutA"], mutB=p["mutB"],
            editA_seq_pos=editA, editB_seq_pos=editB,
            sequence=seq,
            wt_reactivity=design["wt_reactivity"],
            wt_error=design["wt_error"],
            singleA_reactivity=p["singleA_reactivity"],
            singleA_error=p["singleA_error"],
            singleB_reactivity=p["singleB_reactivity"],
            singleB_error=p["singleB_error"],
            double_reactivity=p["double_reactivity"],
            double_error=p["double_error"],
            rescue_factor=p["rescue_factor"],
            eligibility_mask=mask,
            target_structure=design["target_structure"],
            m2_structure=design.get("m2_structure", ""),
            m2_f1=design.get("m2_f1"),
            m2_f1_crossed_pair=design.get("m2_f1_crossed_pair"),
            sub_start=sub_start, sub_end=design["sub_end"] or sub_start,
            mutA_seq=p.get("mutA_seq"),
            mutB_seq=p.get("mutB_seq"),
        ))
    return out


def build_all_pair_samples(designs: list[dict]) -> list[M2RPair]:
    out = []
    for d in designs:
        if not d.get("usable", True):
            continue
        out.extend(build_pair_samples(d))
    return out


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else (
        "/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/openknot_m2/OK7a_M2R_data.v4.5.1.csv")
    designs, meta = parse_m2r_csv(path)
    print(f"rows={meta['n_rows']} designs={meta['n_designs']} pairs={meta['n_pairs']}")
    print(f"usable designs={sum(1 for d in designs if d['usable'])}")
    samples = build_all_pair_samples(designs)
    print(f"pair samples={len(samples)}")
    rfs = [s.rescue_factor for s in samples if s.rescue_factor is not None]
    import numpy as np
    rf = np.array(rfs)
    print(f"rescue n={len(rf)} mean={rf.mean():.4f} median={np.median(rf):.4f} "
          f"min={rf.min():.4f} max={rf.max():.4f}")
    print(f"hist={np.histogram(rf, bins=[-1,-0.1,0.1,0.3,0.5,0.7,0.9,1.1])[0].tolist()}")
