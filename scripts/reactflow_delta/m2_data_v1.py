#!/usr/bin/env python3
"""m2_data_v1 — OpenKnot M2 dataset loader -> per-mutant response-spectrum samples.

Purpose
-------
Expand the reactflow_delta learnability experiment's exchangeable-unit count (N)
beyond 100 by adding the OpenKnot M2 design dataset.  The original P2 development
pool had only ~13 resolved publications as exchangeable units.  M2 provides
20 target puzzles x 8 design methods = 160 (target x method) DESIGNS, each with a
WT construct and 52-100 single-nucleotide mutants, all with per-position 2A3 SHAPE
reactivity + per-position error.  Treating each (puzzle, method) design as the
statistical exchangeable unit raises N to >= 100 (159 usable designs here).

M2 coordinate / alignment facts (verified empirically from the raw CSV):
  * ``sequence`` is the FULL construct (177 nt).  ``sub_start``..``sub_end``
    (1-indexed) delimit the DESIGN region within the full sequence.
  * reactivity column j (1-indexed) aligns to sequence position j (1-indexed);
    positions outside the measured block are EMPTY in the CSV (treated as NaN).
  * a single-nt mutant at design position p (1-indexed) differs from its WT at
    full-sequence index  edit_seq_pos = sub_start - 1 + (p - 1)   (0-indexed).
    Verified: all 13,976 mutants have exactly one diff at that predicted index.
  * No WT replicates exist in M2 -> changer calling must use the per-position
    reactivity_error (see m2_caller_v1).

This module is STRICT-legal: it only exposes WT sequence + exact single-nt
mutation + the per-position reactivity / error needed for the response target.
No held-out target leaks into features here (the caller/model split is handled by
the runner).

Everything is pure/stdlib + numpy and unit-testable without the run_* dependency
chain.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

REACT_COLS = 177          # number of per-position reactivity columns in the CSV
M2_PROBE = "2A3"          # all M2 rows are 2A3_MaP
M2_EXPTYPE = "MutateAndMap"

# canonical source of the OpenKnot M2 dataset (media URL bypasses the LFS pointer)
M2_SOURCE_URL = ("https://media.githubusercontent.com/media/eternagame/"
                 "OpenKnotAIDesignData/main/Data/OK7a_M2_data.v4.5.2.csv")
M2_SCHEMA = "reactflow_delta.m2_data.v1"


def _parse_arrays(row: dict, prefix: str) -> list[Optional[float]]:
    """Parse reactivity_/reactivity_error_ columns; '' -> None, else float."""
    out: list[Optional[float]] = []
    for i in range(1, REACT_COLS + 1):
        v = row.get(f"{prefix}{i:04d}")
        out.append(None if v in (None, "") else float(v))
    return out


def parse_m2_csv(path) -> tuple[list[dict], dict]:
    """Parse the OpenKnot M2 CSV into design records.

    Returns (designs, meta).  Each design dict:
        {
          "puzzle", "method", "source_accession",
          "sequence", "sub_start", "sub_end",
          "wt_reactivity": list[float|None] (len 177),
          "wt_error":      list[float|None] (len 177),
          "mutants": [ {mutA, edit_seq_pos, sequence, reactivity, error}, ... ],
        }

    Designs whose WT has no finite reactivity at all are kept in ``designs`` but
    flagged with ``"usable": False``; the runner filters them out (so the
    exchangeable-unit count uses only usable designs).
    """
    designs: dict[tuple[str, str], dict] = {}
    meta = {"n_rows": 0, "n_designs": 0, "n_mutants": 0}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            meta["n_rows"] += 1
            puzzle = row["puzzle"]
            method = row["method"]
            key = (puzzle, method)

            if not row["mutA"]:
                # --- WT row: create the design or refresh its WT arrays ---
                d = designs.get(key)
                if d is None:
                    d = {
                        "puzzle": puzzle,
                        "method": method,
                        "source_accession": f"OK7a_M2_{puzzle}_{method}",
                        "sequence": row["sequence"],
                        "sub_start": int(row["sub_start"]),
                        "sub_end": int(row["sub_end"]),
                        "wt_reactivity": _parse_arrays(row, "reactivity_"),
                        "wt_error": _parse_arrays(row, "reactivity_error_"),
                        "mutants": [],
                        "usable": True,
                    }
                    designs[key] = d
                    meta["n_designs"] += 1
                else:
                    d["wt_reactivity"] = _parse_arrays(row, "reactivity_")
                    d["wt_error"] = _parse_arrays(row, "reactivity_error_")
                continue

            # --- mutant row ---
            d = designs.get(key)
            if d is None:
                # a design whose WT row has not appeared yet: create a placeholder
                # (the WT row will fill the WT arrays when it is seen).
                d = {
                    "puzzle": puzzle,
                    "method": method,
                    "source_accession": f"OK7a_M2_{puzzle}_{method}",
                    "sequence": row["sequence"],
                    "sub_start": int(row["sub_start"]),
                    "sub_end": int(row["sub_end"]),
                    "wt_reactivity": [],
                    "wt_error": [],
                    "mutants": [],
                    "usable": True,
                }
                designs[key] = d
                meta["n_designs"] += 1
            mutA = int(row["mutA"])
            sub_start = d["sub_start"]
            edit_seq_pos = sub_start - 1 + (mutA - 1)   # 0-indexed seq index
            d["mutants"].append({
                "mutA": mutA,
                "edit_seq_pos": edit_seq_pos,
                "sequence": row["sequence"],
                "reactivity": _parse_arrays(row, "reactivity_"),
                "error": _parse_arrays(row, "reactivity_error_"),
            })
            meta["n_mutants"] += 1

    # mark designs whose WT reactivity is entirely empty as unusable
    for key, d in designs.items():
        if not any(v is not None and np.isfinite(v) for v in d["wt_reactivity"]):
            d["usable"] = False
    return list(designs.values()), meta


def _eligibility(wt_react, mut_react) -> list[int]:
    """1 where both WT and mutant reactivity are finite, else 0."""
    out = []
    for a, b in zip(wt_react, mut_react):
        ok = (a is not None and b is not None
              and np.isfinite(a) and np.isfinite(b))
        out.append(1 if ok else 0)
    return out


def _nan(v: Optional[float]) -> float:
    return float("nan") if v is None else float(v)


@dataclass
class M2Sample:
    """A single-mutant response-spectrum sample with build_feature-ready records.

    ``wt_rec``/``pair`` mirror the canonical record shape consumed by
    run_p2_v3.build_feature and build_pair_features_aligned_robust, so the M2
    samples drop into the existing residual-spectrum pipeline unchanged.
    """
    design_id: str            # exchangeable unit id (puzzle x method)
    puzzle: str
    method: str
    mutA: int
    edit_seq_pos: int         # 0-indexed full-sequence index of the edit
    sequence: str
    wt_reactivity: list       # len 177, NaN for missing
    mut_reactivity: list
    wt_error: list
    mut_error: list
    eligibility_mask: list    # 0/1 len 177
    wt_rec: dict = field(default_factory=dict)
    pair: dict = field(default_factory=dict)


def build_samples(design: dict) -> list[M2Sample]:
    """Build one M2Sample per mutant of a design (skips mutants with no usable
    reactivity at the edit site)."""
    seq = design["sequence"]
    sub_start = design["sub_start"]
    samples = []
    for m in design["mutants"]:
        ep = m["edit_seq_pos"]
        wt_react = [_nan(v) for v in design["wt_reactivity"]]
        mut_react = [_nan(v) for v in m["reactivity"]]
        wt_err = [_nan(v) for v in design["wt_error"]]
        mut_err = [_nan(v) for v in m["error"]]
        # skip mutant if the edit site has no measurable reactivity in WT or mutant
        if not (np.isfinite(wt_react[ep]) and np.isfinite(mut_react[ep])):
            continue
        mask = _eligibility(design["wt_reactivity"], m["reactivity"])
        ref_allele = seq[ep]
        alt_allele = m["sequence"][ep]

        wt_rec = {
            "source_accession": design["source_accession"],
            "canonical_sequence": seq,
            "is_wt": True,
            "probe": [M2_PROBE],
            "temperature": [],
            "reactivity_layers": {
                "train_frozen": {
                    "reactivity": wt_react,
                    "error": wt_err,
                }
            },
        }
        pair = {
            "source_accession": design["source_accession"],
            "wt_profile_index": 0,
            "mutant_profile_index": m["mutA"],
            "asset_name": "M2_2A3",
            "eligibility_reason_codes": [
                "ELIGIBLE" if x else "EXCLUDED" for x in mask
            ],
            "coordinate": {"offset": ep},
            "ref_allele": ref_allele,
            "alt_allele": alt_allele,
            "condition": {
                "modifier": [M2_PROBE],
                "experimentType": [M2_EXPTYPE],
                "temperature": [],
            },
        }
        samples.append(M2Sample(
            design_id=design["source_accession"],
            puzzle=design["puzzle"], method=design["method"],
            mutA=m["mutA"], edit_seq_pos=ep, sequence=seq,
            wt_reactivity=wt_react, mut_reactivity=mut_react,
            wt_error=wt_err, mut_error=mut_err,
            eligibility_mask=mask, wt_rec=wt_rec, pair=pair,
        ))
    return samples


def build_all_samples(designs: list[dict]) -> list[M2Sample]:
    """Build samples from all usable designs."""
    out = []
    for d in designs:
        if not d.get("usable", True):
            continue
        out.extend(build_samples(d))
    return out
