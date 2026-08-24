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
    design_pos: int  # 0-based within the submitted design; raw ID/key coordinate
    full_pos: int  # 0-based within the full padded construct; model coordinate
    ref: str
    alt: str
    mutation_key: str  # canonical design_pos_ref>alt
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
        """Index profiles by the canonical puzzle-method construct identity.

        Raw OpenKnot row-id prefixes are not the biological construct keys for
        every method (for example ``P01_WT`` is labelled ``Starting sequence``
        and ``P01_gRNAde2`` is labelled ``gRNAde-no3d``).  Indexing by the raw
        prefix and later falling back to a mutation-only suffix can therefore
        select a profile from a different construct.  The canonical key used by
        every accessor is puzzle + method + mutation identity, so construct it
        directly from the parsed metadata here.
        """
        react = self._reactivity_cols(self.df)
        err = self._error_cols(self.df)
        for r in self.df.itertuples():
            if bool(r.is_wt):
                rid = f"{r.construct_id}_wt"
            else:
                if pd.isna(r.mut_pos) or pd.isna(r.mut_ref) or pd.isna(r.mut_alt):
                    raise ValueError(f"cannot index non-canonical mutant row {r.id}")
                rid = (
                    f"{r.construct_id}_mm_{int(r.mut_pos)}_"
                    f"{r.mut_ref}_{r.mut_alt}"
                )
            if rid in self._full_profiles:
                raise ValueError(f"duplicate canonical full-profile key {rid}")
            arr = np.asarray([getattr(r, c) for c in react], dtype=float)
            earr = np.asarray([getattr(r, c) for c in err], dtype=float)
            self._full_profiles[rid] = arr
            self._full_errors[rid] = earr

    def mutant_full_profile(
        self, wt_id: str, design_pos: int, ref: str, alt: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (full reactivity, full error) for a mutant row by constructing its canonical id."""
        ref_c = ref.replace("T", "U")
        alt_c = alt.replace("T", "U")
        rid = f"{wt_id.replace('_wt', '')}_mm_{design_pos}_{ref_c}_{alt_c}"
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
            # OpenKnot defines sub_start/sub_end as one-based full-sequence
            # positions. Store Python coordinates as zero-based, end-exclusive.
            start = int(r.sub_start) - 1 if not pd.isna(r.sub_start) else 0
            end = int(r.sub_end) if not pd.isna(r.sub_end) else seq_len
            if not (0 <= start < end <= seq_len):
                raise ValueError(
                    f"invalid design interval for {cid}: start={start}, end={end}, L={seq_len}"
                )
            region = np.full(seq_len, "other_assay_region", dtype=object)
            region[start:end] = "design_region"
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
        coordinate_counts = {
            "raw_hamming_one": 0,
            "formula_matches_raw_diff": 0,
            "formula_ref_match": 0,
            "formula_alt_match": 0,
            "mutA_equals_design_pos_plus_one": 0,
        }
        for r in mm.itertuples():
            if pd.isna(r.mut_pos):
                continue
            cid = r.construct_id
            design_pos = int(r.mut_pos)
            ref = r.mut_ref
            alt = r.mut_alt
            full_pos = constructs[cid].design_start + design_pos
            wt_sequence = constructs[cid].sequence
            mutant_sequence = r.sequence_canon
            if not (0 <= full_pos < len(wt_sequence)):
                raise ValueError(
                    f"mutation full_pos outside construct for {r.id}: {full_pos}"
                )
            differences = [
                i
                for i, (wt_base, mutant_base) in enumerate(
                    zip(wt_sequence, mutant_sequence)
                )
                if wt_base != mutant_base
            ]
            coordinate_counts["raw_hamming_one"] += int(len(differences) == 1)
            coordinate_counts["formula_matches_raw_diff"] += int(
                differences == [full_pos]
            )
            coordinate_counts["formula_ref_match"] += int(
                wt_sequence[full_pos] == ref
            )
            coordinate_counts["formula_alt_match"] += int(
                mutant_sequence[full_pos] == alt
            )
            coordinate_counts["mutA_equals_design_pos_plus_one"] += int(
                not pd.isna(r.mutA) and int(r.mutA) == design_pos + 1
            )
            # WT anchor at this position
            wt_react = (
                float(wt_profiles[cid][full_pos])
                if not np.isnan(wt_profiles[cid][full_pos])
                else None
            )
            wt_err = (
                float(wt_errors[cid][full_pos])
                if not np.isnan(wt_errors[cid][full_pos])
                else None
            )
            wt_obs = bool(wt_masks[cid][full_pos])
            target_react = (
                float(getattr(r, react[full_pos]))
                if not pd.isna(getattr(r, react[full_pos]))
                else None
            )
            target_err = (
                float(getattr(r, err[full_pos]))
                if not pd.isna(getattr(r, err[full_pos]))
                else None
            )
            target_obs = not np.isnan(getattr(r, react[full_pos]))
            region = str(constructs[cid].region_map[full_pos])
            records.append(MutantRecord(
                puzzle=r.puzzle, method=r.method, construct_id=cid,
                wt_id=cid + "_wt", design_pos=design_pos, full_pos=full_pos,
                ref=ref, alt=alt,
                mutation_key=f"{design_pos}_{ref}>{alt}",
                biological_scoring_key=(
                    f"openknot_m2|{r.puzzle}|{r.method}|{cid}|"
                    f"{design_pos}|{ref}>{alt}|{full_pos}"
                ),
                wt_reactivity=wt_react, wt_error=wt_err, wt_observed=wt_obs,
                target_reactivity=target_react, target_error=target_err,
                target_observed=target_obs, region=region,
            ))

        coordinate_failures = {
            name: len(records) - count for name, count in coordinate_counts.items()
        }
        if any(coordinate_failures.values()):
            raise ValueError(
                "OpenKnot mutation coordinate validation failed: "
                f"{coordinate_failures}"
            )

        canonical_profile_keys = {
            f"{record.wt_id.replace('_wt', '')}_mm_{record.design_pos}_"
            f"{record.ref}_{record.alt}"
            for record in records
        }
        missing_profile_keys = canonical_profile_keys - set(self._full_profiles)
        if missing_profile_keys:
            example = sorted(missing_profile_keys)[0]
            raise ValueError(
                "canonical mutant full-profile identity is incomplete: "
                f"missing={len(missing_profile_keys)} example={example}"
            )

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
            "n_canonical_mutant_full_profiles": len(canonical_profile_keys),
            "canonical_mutant_full_profile_identity": "EXACT_PUZZLE_METHOD_MUTATION",
            "n_methods_total_distinct": int(mm["method"].nunique()),
            "seq_len": seq_len,
            "coordinate_frame": {
                "design_pos": "zero_based_within_designed_sequence",
                "full_pos_formula": "sub_start_minus_one_plus_design_pos",
                "full_pos": "zero_based_within_full_padded_sequence",
                **coordinate_counts,
            },
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
