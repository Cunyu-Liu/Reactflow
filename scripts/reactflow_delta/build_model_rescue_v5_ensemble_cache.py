#!/usr/bin/env python3
"""Build the outcome-blind exact-mutant ensemble-delta cache for v5."""

from __future__ import annotations

import argparse
import json
import math
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

from scripts.reactflow_delta.model_rescue_v5_schema import (
    CACHE_SCHEMA as SCHEMA,
    FEATURE_NAMES,
    SOURCE_COLUMNS,
)

MUTANT_ID = re.compile(r"^(?P<prefix>.+)_mm_(?P<design_pos>\d+)_(?P<ref>[ACGTU])_(?P<alt>[ACGTU])$")


def assert_cache_authority(repo_root: Path) -> None:
    active = yaml.safe_load(
        (repo_root / "configs/reactflow_delta/active_contract.yaml").read_text(
            encoding="utf-8"
        )
    )
    if active["authority"]["current_phase"] != "V5M1":
        raise RuntimeError("v5 ensemble cache is closed outside active V5M1")
    if active.get("outcome_blind_cache_allowed") is not True:
        raise RuntimeError("v5 outcome-blind cache authority is absent")
    if active.get("held_score_read_allowed") is not False:
        raise RuntimeError("v5 cache construction requires held scores closed")
    if active.get("new_external_outcome_access_allowed") is not False:
        raise RuntimeError("v5 cache construction requires external outcomes locked")


def _normalize_sequence(sequence: str) -> str:
    value = str(sequence).upper().replace("T", "U")
    if not value or set(value) - set("ACGU"):
        raise ValueError("v5 cache only accepts non-empty A/C/G/U sequences")
    return value


def _parse_mutant_id(row_id: str) -> dict[str, Any]:
    match = MUTANT_ID.fullmatch(str(row_id))
    if match is None:
        raise ValueError(f"invalid registered mutant id {row_id}")
    out: dict[str, Any] = match.groupdict()
    out["design_pos"] = int(out["design_pos"])
    out["ref"] = out["ref"].replace("T", "U")
    out["alt"] = out["alt"].replace("T", "U")
    return out


def build_construct_groups(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if tuple(frame.columns) != SOURCE_COLUMNS:
        raise ValueError(f"v5 cache requires exactly the columns {SOURCE_COLUMNS}")
    if frame["id"].duplicated().any():
        raise ValueError("v5 cache input contains duplicate row ids")
    rows = {str(row.id): row for row in frame.itertuples(index=False)}
    groups: dict[str, dict[str, Any]] = {}
    for row in frame.itertuples(index=False):
        row_id = str(row.id)
        if row_id.endswith("_wt"):
            continue
        parsed = _parse_mutant_id(row_id)
        prefix = str(parsed["prefix"])
        wt_id = f"{prefix}_wt"
        if wt_id not in rows:
            raise ValueError(f"mutant {row_id} has no matching WT row")
        wt_row = rows[wt_id]
        wt_sequence = _normalize_sequence(wt_row.sequence)
        mutant_sequence = _normalize_sequence(row.sequence)
        if len(wt_sequence) != len(mutant_sequence):
            raise ValueError(f"mutant {row_id} changed sequence length")
        design_pos = int(parsed["design_pos"])
        mut_a = int(row.mutA)
        if mut_a != design_pos + 1:
            raise ValueError(f"mutA and row id disagree for {row_id}")
        if int(row.sub_start) != int(wt_row.sub_start):
            raise ValueError(f"WT and mutant sub_start disagree for {row_id}")
        full_pos = int(row.sub_start) - 1 + design_pos
        if full_pos < 0 or full_pos >= len(wt_sequence):
            raise ValueError(f"corrected full coordinate is outside {row_id}")
        difference = np.flatnonzero(
            np.frombuffer(wt_sequence.encode("ascii"), dtype="S1")
            != np.frombuffer(mutant_sequence.encode("ascii"), dtype="S1")
        )
        if not np.array_equal(difference, np.asarray([full_pos])):
            raise ValueError(f"{row_id} is not an exact one-base mutant at full_pos")
        if wt_sequence[full_pos] != parsed["ref"]:
            raise ValueError(f"WT reference base mismatch for {row_id}")
        if mutant_sequence[full_pos] != parsed["alt"]:
            raise ValueError(f"mutant alternate base mismatch for {row_id}")
        group = groups.setdefault(
            prefix,
            {
                "construct_id": prefix,
                "wt_id": wt_id,
                "wt_sequence": wt_sequence,
                "puzzle": str(row.puzzle),
                "method": str(row.method),
                "mutants": [],
            },
        )
        if group["wt_sequence"] != wt_sequence:
            raise ValueError(f"construct {prefix} has inconsistent WT sequences")
        group["mutants"].append(
            {
                "row_id": row_id,
                "sequence": mutant_sequence,
                "design_pos": design_pos,
                "full_pos": full_pos,
                "ref": parsed["ref"],
                "alt": parsed["alt"],
            }
        )
    output = []
    for prefix in sorted(groups):
        group = groups[prefix]
        group["mutants"].sort(key=lambda row: row["row_id"])
        output.append(group)
    return output


def fold_ensemble(sequence: str) -> tuple[np.ndarray, np.ndarray, float]:
    sequence = _normalize_sequence(sequence)
    compound = RNA.fold_compound(sequence)
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
        raise RuntimeError("ViennaRNA returned non-finite ensemble energy")
    if not np.isfinite(bpp).all() or not np.isfinite(entropy).all():
        raise RuntimeError("ViennaRNA returned non-finite ensemble features")
    return bpp, entropy.astype(np.float32), ensemble_energy


def ensemble_delta_features(
    wt_bpp: np.ndarray,
    wt_entropy: np.ndarray,
    wt_energy: float,
    mutant_bpp: np.ndarray,
    mutant_entropy: np.ndarray,
    mutant_energy: float,
    full_pos: int,
) -> np.ndarray:
    if wt_bpp.shape != mutant_bpp.shape or wt_bpp.ndim != 2:
        raise ValueError("WT and mutant BPP matrices must have equal square shape")
    length = wt_bpp.shape[0]
    if wt_bpp.shape[1] != length or not 0 <= full_pos < length:
        raise ValueError("invalid BPP shape or source coordinate")
    delta = mutant_bpp.astype(np.float64) - wt_bpp.astype(np.float64)
    wt_paired = wt_bpp.sum(axis=1, dtype=np.float64)
    mutant_paired = mutant_bpp.sum(axis=1, dtype=np.float64)
    delta_unpaired = wt_paired - mutant_paired
    delta_entropy = mutant_entropy.astype(np.float64) - wt_entropy.astype(np.float64)
    wt_source = wt_bpp[full_pos].astype(np.float64)
    mutant_source = mutant_bpp[full_pos].astype(np.float64)
    source_delta = mutant_source - wt_source
    row_l1 = np.abs(delta).sum(axis=1)
    row_l2 = np.sqrt(np.square(delta).sum(axis=1))
    lower = np.tril(np.ones((length, length), dtype=bool), k=-1)
    upper = np.triu(np.ones((length, length), dtype=bool), k=1)
    upstream_delta = np.where(lower, delta, 0.0).sum(axis=1)
    downstream_delta = np.where(upper, delta, 0.0).sum(axis=1)
    global_l2 = float(np.sqrt(np.square(delta).sum()))
    energy_delta = float(mutant_energy - wt_energy)
    source_unpaired_delta = float(delta_unpaired[full_pos])
    result = np.column_stack(
        [
            delta_unpaired,
            delta_entropy,
            source_delta,
            wt_source,
            mutant_source,
            row_l1,
            row_l2,
            upstream_delta,
            downstream_delta,
            np.full(length, global_l2),
            np.full(length, energy_delta),
            np.full(length, source_unpaired_delta),
        ]
    ).astype(np.float32)
    if result.shape != (length, len(FEATURE_NAMES)) or not np.isfinite(result).all():
        raise RuntimeError("invalid v5 ensemble-delta feature matrix")
    return result


def process_construct(group: dict[str, Any]) -> list[dict[str, Any]]:
    wt_bpp, wt_entropy, wt_energy = fold_ensemble(group["wt_sequence"])
    output = []
    for mutant in group["mutants"]:
        mutant_bpp, mutant_entropy, mutant_energy = fold_ensemble(mutant["sequence"])
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
    frame = pd.read_csv(m2_csv, usecols=list(SOURCE_COLUMNS))
    frame = frame.loc[:, list(SOURCE_COLUMNS)]
    groups = build_construct_groups(frame)
    if max_constructs is not None:
        groups = groups[:max_constructs]
    if not groups:
        raise ValueError("v5 cache has no construct groups to process")
    if workers <= 1:
        nested = [process_construct(group) for group in groups]
    else:
        context = mp.get_context("spawn")
        with context.Pool(processes=workers) as pool:
            nested = list(pool.imap(process_construct, groups))
    rows = sorted((row for group_rows in nested for row in group_rows), key=lambda x: x["row_id"])
    length = rows[0]["features"].shape[0]
    if any(row["features"].shape != (length, len(FEATURE_NAMES)) for row in rows):
        raise RuntimeError("v5 cache rows have inconsistent feature shapes")
    string_dtype = h5py.string_dtype(encoding="utf-8")
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_h5, "w") as handle:
        handle.attrs["schema_version"] = SCHEMA
        handle.attrs["source_columns"] = json.dumps(SOURCE_COLUMNS)
        handle.attrs["feature_names"] = json.dumps(FEATURE_NAMES)
        for name in ("row_id", "construct_id", "puzzle", "method", "ref", "alt"):
            handle.create_dataset(
                name,
                data=np.asarray([row[name] for row in rows], dtype=object),
                dtype=string_dtype,
            )
        for name in ("design_pos", "full_pos"):
            handle.create_dataset(
                name, data=np.asarray([row[name] for row in rows], dtype=np.int16)
            )
        handle.create_dataset(
            "features",
            data=np.stack([row["features"] for row in rows]),
            dtype=np.float32,
            chunks=(1, length, len(FEATURE_NAMES)),
            compression="gzip",
            compression_opts=4,
        )
    manifest = {
        "schema_version": SCHEMA,
        "status": "OUTCOME_BLIND_ENSEMBLE_CACHE_COMPLETE",
        "source_csv": str(m2_csv),
        "source_columns": list(SOURCE_COLUMNS),
        "feature_names": list(FEATURE_NAMES),
        "n_constructs": len(groups),
        "n_registered_mutants": len(rows),
        "sequence_length": length,
        "feature_width": len(FEATURE_NAMES),
        "viennarna_version": RNA.__version__,
        "coordinate_invariants_checked": True,
        "outcome_columns_read": False,
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
