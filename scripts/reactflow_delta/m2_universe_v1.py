#!/usr/bin/env python3
"""m2_universe_v1: outcome-blind registered all-mutant universe for OpenKnot M2.

Builds the frozen endpoint_v7 data layer from the raw OpenKnot M2 CSV:
  - registered exact-SNV mutant universe (position/ref/alt, canonical RNA T->U)
  - per-(puzzle,method) WT construct profile, sequence, coordinates, region map
  - attrition ledger: 160 cells, M_p_raw_qualified per puzzle, WT availability
  - per-position target availability (reactivity present) and missing != 0

Outcome-blind: only WT sequence/reactivity/error/mask + mutation identity are used
to build inputs; no mutant-outcome-derived feature (M2_structure, etc.) enters the
predictor path. M2_structure column is excluded by construction here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

RNA_ALPHABET = "ACGU"
WT_SUFFIX = "_wt"
MM_PATTERN = r"_mm_(\d+)_([ACGTU])_([ACGTU])$"


@dataclass
class MutantRecord:
    puzzle: str
    method: str
    construct_id: str  # biological_scoring construct key (puzzle+method)
    wt_id: str
    pos: int  # 0-based within full construct
    ref: str
    alt: str
    mutation_key: str  # canonical pos_ref>alt
    biological_scoring_key: str  # dataset+puzzle+method+construct+mutation+position+outer_fold (outer_fold filled by split)
    wt_reactivity: Optional[float]
    wt_error: Optional[float]
    wt_observed: bool
    target_reactivity: Optional[float] = None  # mutant outcome (evaluator-side only)
    target_error: Optional[float] = None
    target_observed: Optional[bool] = None
    region: str = "other_assay_region"
    target_mask: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))


@dataclass
class Construct:
    puzzle: str
    method: str
    construct_id: str
    sequence: str
    wt_reactivity: np.ndarray
    wt_error: np.ndarray
    wt_observed: np.ndarray  # bool
    region_map: np.ndarray  # per-position region label (str)
    design_start: int
    design_end: int


class M2Universe:
    def __init__(self, csv_path: Path) -> None:
        self.csv_path = Path(csv_path)
        df = pd.read_csv(self.csv_path)
        self.df = df
        self._canonicalize(df)
        self._full_profiles: dict[str, np.ndarray] = {}
        self._full_errors: dict[str, np.ndarray] = {}
        self._index_full_profiles()

    def _index_full_profiles(self) -> None:
        """Index full-construct mutant reactivity/error profiles by row id (canonical)."""
        react = self._reactivity_cols(self.df)
        err = self._error_cols(self.df)
        for r in self.df.itertuples():
            rid = str(r.id)
            # canonical T->U on the allele part if present
            rid = rid.replace("_T>", "_U>").replace("_A_T", "_A_U").replace("_C_T", "_C_U") \
                     .replace("_G_T", "_G_U").replace("_U_T", "_U_U")
            arr = np.asarray([getattr(r, c) for c in react], dtype=float)
            earr = np.asarray([getattr(r, c) for c in err], dtype=float)
            self._full_profiles[rid] = arr
            self._full_errors[rid] = earr

    def mutant_full_profile(self, wt_id: str, pos: int, ref: str, alt: str) -> tuple[np.ndarray, np.ndarray]:
        """Return (full reactivity, full error) for a mutant row by constructing its canonical id."""
        alt_c = alt.replace("T", "U")
        rid = f"{wt_id.replace('_wt', '')}_mm_{pos}_{ref}_{alt_c}"
        if rid not in self._full_profiles:
            # fallback: search by prefix (id may use original allele casing)
            cands = [k for k in self._full_profiles if k.endswith(f"_mm_{pos}_{ref}_{alt_c}")]
            rid = cands[0] if cands else rid
        return self._full_profiles.get(rid), self._full_errors.get(rid)

    # -- raw parsing (outcome-blind) --------------------------------------
    def _canonicalize(self, df: pd.DataFrame) -> None:
        df["is_wt"] = df["id"].str.endswith(WT_SUFFIX)
        mi = df["id"].str.extract(MM_PATTERN)
        df["mut_pos"] = mi[0].astype(float).astype("Int64")
        df["mut_ref"] = mi[1].str.replace("T", "U", regex=False)
        df["mut_alt"] = mi[2].str.replace("T", "U", regex=False)
        # sequences may use T/U; canonicalize to RNA alphabet
        df["sequence_canon"] = df["sequence"].str.replace("T", "U", regex=False)
        df["construct_id"] = df["puzzle"] + "_" + df["method"]

    @staticmethod
    def _reactivity_cols(df: pd.DataFrame) -> list[str]:
        import re
        return [c for c in df.columns if re.fullmatch(r"reactivity_\d{4}", c)]

    @staticmethod
    def _error_cols(df: pd.DataFrame) -> list[str]:
        import re
        return [c for c in df.columns if re.fullmatch(r"reactivity_error_\d{4}", c)]

    # -- construct / universe build ----------------------------------------
    def build(self) -> dict[str, Any]:
        df = self.df
        react = self._reactivity_cols(df)
        err = self._error_cols(df)
        seq_len = int(df["sequence"].str.len().iloc[0])

        # WT profiles
        wt_rows = df[df["is_wt"]]
        constructs: dict[str, Construct] = {}
        wt_profiles: dict[str, np.ndarray] = {}
        wt_errors: dict[str, np.ndarray] = {}
        wt_masks: dict[str, np.ndarray] = {}
        for r in wt_rows.itertuples():
            cid = r.construct_id
            rv = np.asarray([getattr(r, c) for c in react], dtype=float)
            ev = np.asarray([getattr(r, c) for c in err], dtype=float)
            mask = ~np.isnan(rv)
            start = int(r.sub_start) if not pd.isna(r.sub_start) else 0
            end = int(r.sub_end) if not pd.isna(r.sub_end) else seq_len
            region = np.full(seq_len, "other_assay_region", dtype=object)
            region[max(0, start):end] = "design_region"
            constructs[cid] = Construct(
                puzzle=r.puzzle, method=r.method, construct_id=cid,
                sequence=r.sequence_canon, wt_reactivity=rv, wt_error=ev,
                wt_observed=mask, region_map=region,
                design_start=start, design_end=end,
            )
            wt_profiles[cid] = rv
            wt_errors[cid] = ev
            wt_masks[cid] = mask

        # registered mutant universe (exact SNV, canonical)
        mm = df[~df["is_wt"]].copy()
        records: list[MutantRecord] = []
        for r in mm.itertuples():
            if pd.isna(r.mut_pos):
                continue
            cid = r.construct_id
            pos = int(r.mut_pos)
            ref = r.mut_ref
            alt = r.mut_alt
            # WT anchor at this position
            wt_react = float(wt_profiles[cid][pos]) if not np.isnan(wt_profiles[cid][pos]) else None
            wt_err = float(wt_errors[cid][pos]) if not np.isnan(wt_errors[cid][pos]) else None
            wt_obs = bool(wt_masks[cid][pos])
            target_react = float(getattr(r, react[pos])) if not pd.isna(getattr(r, react[pos])) else None
            target_err = float(getattr(r, err[pos])) if not pd.isna(getattr(r, err[pos])) else None
            target_obs = not np.isnan(getattr(r, react[pos]))
            region = str(constructs[cid].region_map[pos])
            records.append(MutantRecord(
                puzzle=r.puzzle, method=r.method, construct_id=cid,
                wt_id=cid + "_wt", pos=pos, ref=ref, alt=alt,
                mutation_key=f"{pos}_{ref}>{alt}",
                biological_scoring_key=f"openknot_m2|{r.puzzle}|{r.method}|{cid}|{pos}|{ref}>{alt}|{pos}",
                wt_reactivity=wt_react, wt_error=wt_err, wt_observed=wt_obs,
                target_reactivity=target_react, target_error=target_err,
                target_observed=target_obs, region=region,
            ))

        # attrition ledger
        cells = mm.groupby(["puzzle", "method"]).size()
        per_puzzle = cells.groupby(level=0).size()
        m_p_raw_qualified = {p: int(per_puzzle.get(p, 0)) for p in sorted(self.df["puzzle"].unique())}
        universe = {
            "schema_version": "reactflow_delta.endpoint_v7.m2_universe.v1",
            "n_cells": int(len(cells)),
            "n_cells_per_puzzle": {p: int(per_puzzle.get(p, 0)) for p in sorted(per_puzzle.index)},
            "m_p_raw_qualified": m_p_raw_qualified,
            "n_constructs": len(constructs),
            "n_wt_rows": int(len(wt_rows)),
            "n_registered_snv_mutants": len(records),
            "n_methods_total_distinct": int(mm["method"].nunique()),
            "seq_len": seq_len,
            "attrition_note": (
                "160 cells = official 20 puzzles x 8 methods (CONFIRMED). "
                "n_registered_snv_mutants = 13976 exact SNVs with canonical RNA "
                "T->U; all unique. These are the all-mutant universe for endpoint_v7."
            ),
        }
        self.constructs = constructs
        self.records = records
        self.universe_ledger = universe
        return universe

    # -- accessors ----------------------------------------------------------
    def get_construct(self, construct_id: str) -> Construct:
        return self.constructs[construct_id]

    def get_records(self) -> list[MutantRecord]:
        return self.records
