#!/usr/bin/env python3
"""Build the outcome-blind WT-2A3-constrained ensemble-delta cache for v6."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path
import re
import sys
from typing import Any

import h5py
import numpy as np
import pandas as pd
import RNA
import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.reactflow_delta.build_model_rescue_v5_ensemble_cache import (
    build_construct_groups,
    ensemble_delta_features,
    fold_ensemble,
)
from scripts.reactflow_delta.model_rescue_v6_schema import (
    CACHE_SCHEMA as SCHEMA,
    DEIGAN_INTERCEPT,
    DEIGAN_SLOPE,
    FEATURE_NAMES,
    METADATA_COLUMNS,
    MISSING_REACTIVITY,
)


REACTIVITY_COLUMN = re.compile(r"reactivity_\d{4}")


def assert_cache_authority(repo_root: Path) -> None:
    active = yaml.safe_load(
        (repo_root / "configs/reactflow_delta/active_contract.yaml").read_text(
            encoding="utf-8"
        )
    )
    if active["authority"]["current_phase"] != "V6M1":
        raise RuntimeError("v6 constrained cache is closed outside active V6M1")
    if active.get("outcome_blind_cache_allowed") is not True:
        raise RuntimeError("v6 outcome-blind cache authority is absent")
    if active.get("training_allowed") is not False:
        raise RuntimeError("v6 cache construction requires training closed")
    if active.get("held_score_read_allowed") is not False:
        raise RuntimeError("v6 cache construction requires held scores closed")
    if active.get("new_external_outcome_access_allowed") is not False:
        raise RuntimeError("v6 cache construction requires external outcomes locked")


def load_outcome_blind_inputs(
    m2_csv: Path,
) -> tuple[pd.DataFrame, tuple[str, ...], dict[str, np.ndarray]]:
    """Load all-row metadata and WT-only 2A3 values without materializing mutant outcomes."""

    header = tuple(pd.read_csv(m2_csv, nrows=0).columns)
    reactivity_columns = tuple(column for column in header if REACTIVITY_COLUMN.fullmatch(column))
    if not reactivity_columns:
        raise ValueError("v6 source contains no full-construct reactivity columns")
    missing_metadata = set(METADATA_COLUMNS) - set(header)
    if missing_metadata:
        raise ValueError(f"v6 source is missing metadata columns {sorted(missing_metadata)}")

    metadata = pd.read_csv(m2_csv, usecols=list(METADATA_COLUMNS))
    metadata = metadata.loc[:, list(METADATA_COLUMNS)]
    wt_row_indices = frozenset(
        int(index)
        for index, row_id in enumerate(metadata["id"].astype(str))
        if row_id.endswith("_wt")
    )
    if not wt_row_indices:
        raise ValueError("v6 source contains no WT anchors")

    def skip_non_wt(csv_line_number: int) -> bool:
        return csv_line_number > 0 and (csv_line_number - 1) not in wt_row_indices

    wt_frame = pd.read_csv(
        m2_csv,
        usecols=["id", *reactivity_columns],
        skiprows=skip_non_wt,
    )
    if len(wt_frame) != len(wt_row_indices):
        raise RuntimeError("WT-only reactivity pass did not recover the registered WT universe")
    if not wt_frame["id"].astype(str).str.endswith("_wt").all():
        raise RuntimeError("mutant outcome row entered the WT-only reactivity pass")
    if wt_frame["id"].duplicated().any():
        raise ValueError("v6 source contains duplicate WT ids")

    wt_reactivity: dict[str, np.ndarray] = {}
    for row in wt_frame.itertuples(index=False):
        row_id = str(row[0])
        values = pd.to_numeric(pd.Series(row[1:]), errors="coerce").to_numpy(dtype=np.float64)
        wt_reactivity[row_id] = values
    return metadata, reactivity_columns, wt_reactivity


def shape_constraint_vector(wt_reactivity: np.ndarray) -> tuple[list[float], dict[str, int | bool]]:
    values = np.asarray(wt_reactivity, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("WT reactivity must be a one-dimensional full-construct profile")
    observed = np.isfinite(values)
    negative = observed & (values < 0.0)
    transformed = np.where(observed, np.maximum(values, 0.0), MISSING_REACTIVITY)
    vector = [MISSING_REACTIVITY, *transformed.tolist()]
    return vector, {
        "observed_positions": int(observed.sum()),
        "missing_positions": int((~observed).sum()),
        "negative_positions_clamped": int(negative.sum()),
        "all_missing": bool(not observed.any()),
    }


def _extract_ensemble(
    compound: Any, sequence: str
) -> tuple[np.ndarray, np.ndarray, float]:
    pf_result = compound.pf()
    ensemble_energy = float(pf_result[1])
    raw = compound.bpp()
    length = len(sequence)
    bpp = np.zeros((length, length), dtype=np.float32)
    for i in range(length):
        for j in range(i + 1, length):
            probability = float(raw[i + 1][j + 1])
            bpp[i, j] = probability
            bpp[j, i] = probability
    paired = np.clip(bpp.sum(axis=1, dtype=np.float64), 0.0, 1.0)
    unpaired = 1.0 - paired
    eps = np.finfo(np.float64).tiny
    entropy = -unpaired * np.log(np.maximum(unpaired, eps))
    entropy -= np.sum(
        bpp.astype(np.float64) * np.log(np.maximum(bpp.astype(np.float64), eps)),
        axis=1,
    )
    if not np.isfinite(ensemble_energy):
        raise RuntimeError("ViennaRNA returned non-finite constrained ensemble energy")
    if not np.isfinite(bpp).all() or not np.isfinite(entropy).all():
        raise RuntimeError("ViennaRNA returned non-finite constrained ensemble features")
    return bpp, entropy.astype(np.float32), ensemble_energy


def fold_constrained_ensemble(
    sequence: str,
    wt_reactivity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    vector, policy = shape_constraint_vector(wt_reactivity)
    if len(vector) != len(sequence) + 1:
        raise ValueError("WT reactivity and sequence lengths disagree")
    if policy["all_missing"]:
        return fold_ensemble(sequence)
    compound = RNA.fold_compound(sequence, None, RNA.OPTION_PF)
    accepted = compound.sc_add_SHAPE_deigan(
        vector,
        DEIGAN_SLOPE,
        DEIGAN_INTERCEPT,
        RNA.OPTION_PF,
    )
    if accepted != 1:
        raise RuntimeError("ViennaRNA rejected the frozen Deigan constraint vector")
    return _extract_ensemble(compound, sequence)


def attach_wt_constraints(
    metadata: pd.DataFrame,
    wt_reactivity: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    groups = build_construct_groups(metadata)
    for group in groups:
        wt_id = str(group["wt_id"])
        if wt_id not in wt_reactivity:
            raise ValueError(f"construct {group['construct_id']} has no WT 2A3 profile")
        values = np.asarray(wt_reactivity[wt_id], dtype=np.float64)
        if len(values) != len(group["wt_sequence"]):
            raise ValueError(f"construct {group['construct_id']} has mismatched WT profile length")
        _, policy = shape_constraint_vector(values)
        group["wt_reactivity"] = values
        group["constraint_policy"] = policy
    return groups


def process_construct(group: dict[str, Any]) -> list[dict[str, Any]]:
    reactivity = np.asarray(group["wt_reactivity"], dtype=np.float64)
    wt_bpp, wt_entropy, wt_energy = fold_constrained_ensemble(
        group["wt_sequence"], reactivity
    )
    output: list[dict[str, Any]] = []
    for mutant in group["mutants"]:
        mutant_bpp, mutant_entropy, mutant_energy = fold_constrained_ensemble(
            mutant["sequence"], reactivity
        )
        output.append(
            {
                "row_id": mutant["row_id"],
                "construct_id": group["construct_id"],
                "puzzle": group["puzzle"],
                "method": group["method"],
                "design_pos": mutant["design_pos"],
                "full_pos": mutant["full_pos"],
                "ref": mutant["ref"],
                "alt": mutant["alt"],
                "constraint_observed_positions": group["constraint_policy"][
                    "observed_positions"
                ],
                "constraint_all_missing": group["constraint_policy"]["all_missing"],
                "features": ensemble_delta_features(
                    wt_bpp,
                    wt_entropy,
                    wt_energy,
                    mutant_bpp,
                    mutant_entropy,
                    mutant_energy,
                    mutant["full_pos"],
                ),
            }
        )
    return output


def build_cache(
    m2_csv: Path,
    out_h5: Path,
    out_manifest: Path,
    *,
    workers: int,
    max_constructs: int | None,
) -> dict[str, Any]:
    metadata, reactivity_columns, wt_reactivity = load_outcome_blind_inputs(m2_csv)
    groups = attach_wt_constraints(metadata, wt_reactivity)
    if max_constructs is not None:
        groups = groups[:max_constructs]
    if not groups:
        raise ValueError("v6 cache has no construct groups to process")

    if workers <= 1:
        nested = [process_construct(group) for group in groups]
    else:
        context = mp.get_context("spawn")
        with context.Pool(processes=workers) as pool:
            nested = list(pool.imap(process_construct, groups))
    rows = sorted(
        (row for group_rows in nested for row in group_rows), key=lambda row: row["row_id"]
    )
    length = rows[0]["features"].shape[0]
    if any(row["features"].shape != (length, len(FEATURE_NAMES)) for row in rows):
        raise RuntimeError("v6 cache rows have inconsistent feature shapes")

    string_dtype = h5py.string_dtype(encoding="utf-8")
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_h5, "w") as handle:
        handle.attrs["schema_version"] = SCHEMA
        handle.attrs["metadata_columns"] = json.dumps(METADATA_COLUMNS)
        handle.attrs["feature_names"] = json.dumps(FEATURE_NAMES)
        for name in ("row_id", "construct_id", "puzzle", "method", "ref", "alt"):
            handle.create_dataset(
                name,
                data=np.asarray([row[name] for row in rows], dtype=object),
                dtype=string_dtype,
            )
        for name in ("design_pos", "full_pos", "constraint_observed_positions"):
            handle.create_dataset(
                name, data=np.asarray([row[name] for row in rows], dtype=np.int16)
            )
        handle.create_dataset(
            "constraint_all_missing",
            data=np.asarray([row["constraint_all_missing"] for row in rows], dtype=np.bool_),
        )
        handle.create_dataset(
            "features",
            data=np.stack([row["features"] for row in rows]),
            dtype=np.float32,
            chunks=(1, length, len(FEATURE_NAMES)),
            compression="gzip",
            compression_opts=4,
        )

    policies = [group["constraint_policy"] for group in groups]
    manifest = {
        "schema_version": SCHEMA,
        "status": "OUTCOME_BLIND_WT_2A3_CONSTRAINED_CACHE_COMPLETE",
        "source_csv": str(m2_csv),
        "metadata_columns": list(METADATA_COLUMNS),
        "reactivity_columns": list(reactivity_columns),
        "feature_names": list(FEATURE_NAMES),
        "n_constructs": len(groups),
        "n_registered_mutants": len(rows),
        "sequence_length": length,
        "feature_width": len(FEATURE_NAMES),
        "viennarna_version": RNA.__version__,
        "constraint_algorithm": "DEIGAN_SOFT_CONSTRAINT_PF",
        "deigan_slope": DEIGAN_SLOPE,
        "deigan_intercept": DEIGAN_INTERCEPT,
        "wt_and_mutant_share_identical_constraint_vector": True,
        "remove_mutation_site_constraint": False,
        "negative_reactivity_policy": "CLAMP_TO_ZERO",
        "missing_reactivity_policy": "MINUS_999_UNCONSTRAINED",
        "wt_reactivity_rows_read": len(wt_reactivity),
        "mutant_reactivity_rows_read": 0,
        "negative_positions_clamped": int(
            sum(int(policy["negative_positions_clamped"]) for policy in policies)
        ),
        "missing_positions": int(sum(int(policy["missing_positions"]) for policy in policies)),
        "all_missing_construct_ids": [
            group["construct_id"]
            for group in groups
            if bool(group["constraint_policy"]["all_missing"])
        ],
        "coordinate_invariants_checked": True,
        "mutant_outcome_columns_used": False,
        "external_outcome_accessed": False,
        "cache": str(out_h5),
    }
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--m2-csv", type=Path, required=True)
    parser.add_argument("--out-h5", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=max(mp.cpu_count() // 2, 1))
    parser.add_argument("--max-constructs", type=int)
    args = parser.parse_args(argv)
    assert_cache_authority(args.repo_root.resolve())
    result = build_cache(
        args.m2_csv,
        args.out_h5,
        args.out_manifest,
        workers=args.workers,
        max_constructs=args.max_constructs,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
