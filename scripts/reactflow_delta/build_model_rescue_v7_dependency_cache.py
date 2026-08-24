#!/usr/bin/env python3
"""Build the outcome-blind RiNALMo exact-SNV dependency cache for v7."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import h5py
import numpy as np
import pandas as pd
import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.reactflow_delta.model_rescue_v5_schema import SOURCE_COLUMNS
from scripts.reactflow_delta.model_rescue_v7_dependency import (
    RiNALMoGigaLogitInferer,
    batched_infer,
    dependency_features_from_acgu_logits,
    exact_mutant_sequence,
    normalize_rna_sequence,
)
from scripts.reactflow_delta.model_rescue_v7_schema import (
    CACHE_SCHEMA,
    DEPENDENCY_CODE_COMMIT,
    FEATURE_NAMES,
    RINALMO_CODE_COMMIT,
    RINALMO_MODEL_NAME,
    RINALMO_PARAMETER_COUNT,
)


MUTANT_ID = re.compile(
    r"^(?P<prefix>.+)_mm_(?P<design_pos>\d+)_(?P<ref>[ACGTU])_(?P<alt>[ACGTU])$"
)


def assert_cache_authority(repo_root: Path) -> None:
    active = yaml.safe_load(
        (repo_root / "configs/reactflow_delta/active_contract.yaml").read_text(
            encoding="utf-8"
        )
    )
    if active["authority"]["current_phase"] != "V7M1":
        raise RuntimeError("v7 dependency cache is closed outside active V7M1")
    authorization = active.get("authorization", {})
    if authorization.get("outcome_blind_foundation_preparation_allowed") is not True:
        raise RuntimeError("v7 foundation preparation authority is absent")
    if authorization.get("outcome_blind_cache_preparation_allowed") is not True:
        raise RuntimeError("v7 cache preparation authority is absent")
    if active.get("outcome_blind_cache_allowed") is not True:
        raise RuntimeError("v7 outcome-blind cache authority is absent")
    if active.get("training_allowed") is not False:
        raise RuntimeError("v7 cache construction requires training closed")
    if active.get("held_score_read_allowed") is not False:
        raise RuntimeError("v7 cache construction requires held scores closed")
    if active.get("new_external_outcome_access_allowed") is not False:
        raise RuntimeError("v7 cache construction requires external outcomes locked")


def build_registered_construct_groups(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Build exact-SNV groups from metadata columns without importing outcomes."""

    if tuple(frame.columns) != SOURCE_COLUMNS:
        raise ValueError(f"v7 cache requires exactly the columns {SOURCE_COLUMNS}")
    if frame["id"].duplicated().any():
        raise ValueError("v7 cache input contains duplicate row ids")
    rows = {str(row.id): row for row in frame.itertuples(index=False)}
    groups: dict[str, dict[str, Any]] = {}
    for row in frame.itertuples(index=False):
        row_id = str(row.id)
        if row_id.endswith("_wt"):
            continue
        match = MUTANT_ID.fullmatch(row_id)
        if match is None:
            raise ValueError(f"invalid registered mutant id {row_id}")
        parsed = match.groupdict()
        prefix = str(parsed["prefix"])
        wt_id = f"{prefix}_wt"
        if wt_id not in rows:
            raise ValueError(f"mutant {row_id} has no matching WT row")
        wt_row = rows[wt_id]
        wt_sequence = normalize_rna_sequence(wt_row.sequence)
        mutant_sequence = normalize_rna_sequence(row.sequence)
        design_pos = int(parsed["design_pos"])
        ref = normalize_rna_sequence(parsed["ref"])
        alt = normalize_rna_sequence(parsed["alt"])
        if int(row.mutA) != design_pos + 1:
            raise ValueError(f"mutA and row id disagree for {row_id}")
        if int(row.sub_start) != int(wt_row.sub_start):
            raise ValueError(f"WT and mutant sub_start disagree for {row_id}")
        full_pos = int(row.sub_start) - 1 + design_pos
        reconstructed = exact_mutant_sequence(wt_sequence, full_pos, ref, alt)
        if reconstructed != mutant_sequence:
            raise ValueError(f"{row_id} is not the registered exact one-base mutant")
        if str(row.puzzle) != str(wt_row.puzzle) or str(row.method) != str(
            wt_row.method
        ):
            raise ValueError(f"WT and mutant identity disagree for {row_id}")
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
                "ref": ref,
                "alt": alt,
            }
        )
    output = []
    for prefix in sorted(groups):
        group = groups[prefix]
        group["mutants"].sort(key=lambda mutant: mutant["row_id"])
        output.append(group)
    return output


def load_outcome_blind_groups(
    m2_csv: Path, *, max_constructs: int | None
) -> list[dict[str, Any]]:
    frame = pd.read_csv(m2_csv, usecols=list(SOURCE_COLUMNS))
    frame = frame.loc[:, list(SOURCE_COLUMNS)]
    groups = build_registered_construct_groups(frame)
    if max_constructs is not None:
        groups = groups[:max_constructs]
    if not groups:
        raise ValueError("v7 cache has no registered construct groups")
    return groups


def _git_head(code_root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(code_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _registered_sequences(groups: list[dict[str, Any]]) -> set[str]:
    sequences: set[str] = set()
    for group in groups:
        wt_sequence = str(group["wt_sequence"])
        sequences.add(wt_sequence)
        for mutant in group["mutants"]:
            reconstructed = exact_mutant_sequence(
                wt_sequence,
                int(mutant["full_pos"]),
                str(mutant["ref"]),
                str(mutant["alt"]),
            )
            if reconstructed != str(mutant["sequence"]):
                raise ValueError(
                    f"registered mutant sequence disagrees for {mutant['row_id']}"
                )
            sequences.add(reconstructed)
    return sequences


def build_cache(
    m2_csv: Path,
    out_h5: Path,
    out_manifest: Path,
    *,
    inferer: Any,
    batch_size: int,
    max_constructs: int | None,
    model_code_root: Path | None,
    weights_path: Path | None,
    attention_backend: str,
) -> dict[str, Any]:
    groups = load_outcome_blind_groups(m2_csv, max_constructs=max_constructs)
    sequence_universe = _registered_sequences(groups)
    logits_by_sequence = batched_infer(
        inferer, sequence_universe, batch_size=batch_size
    )
    if set(logits_by_sequence) != sequence_universe:
        raise RuntimeError("v7 inference sequence universe is incomplete")

    feature_cache: dict[tuple[str, str, int], np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for group in groups:
        wt_sequence = str(group["wt_sequence"])
        wt_logits = logits_by_sequence[wt_sequence]
        for mutant in group["mutants"]:
            mutant_sequence = str(mutant["sequence"])
            full_pos = int(mutant["full_pos"])
            dependency_key = (wt_sequence, mutant_sequence, full_pos)
            if dependency_key not in feature_cache:
                feature_cache[dependency_key] = dependency_features_from_acgu_logits(
                    wt_logits,
                    logits_by_sequence[mutant_sequence],
                    wt_sequence,
                    full_pos,
                )
            rows.append(
                {
                    "row_id": str(mutant["row_id"]),
                    "construct_id": str(group["construct_id"]),
                    "puzzle": str(group["puzzle"]),
                    "method": str(group["method"]),
                    "wt_sequence": wt_sequence,
                    "mutant_sequence": mutant_sequence,
                    "design_pos": int(mutant["design_pos"]),
                    "full_pos": full_pos,
                    "ref": str(mutant["ref"]),
                    "alt": str(mutant["alt"]),
                    "features": feature_cache[dependency_key],
                }
            )
    rows.sort(key=lambda row: row["row_id"])
    if not rows:
        raise RuntimeError("v7 cache contains no registered mutants")
    length = len(rows[0]["wt_sequence"])
    expected_shape = (length, len(FEATURE_NAMES))
    if any(row["features"].shape != expected_shape for row in rows):
        raise RuntimeError("v7 cache rows have inconsistent feature shapes")

    string_dtype = h5py.string_dtype(encoding="utf-8")
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_h5, "w") as handle:
        handle.attrs["schema_version"] = CACHE_SCHEMA
        handle.attrs["feature_names"] = json.dumps(FEATURE_NAMES)
        handle.attrs["model_name"] = RINALMO_MODEL_NAME
        handle.attrs["attention_backend"] = attention_backend
        for name in (
            "row_id",
            "construct_id",
            "puzzle",
            "method",
            "wt_sequence",
            "mutant_sequence",
            "ref",
            "alt",
        ):
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
        "schema_version": CACHE_SCHEMA,
        "status": "OUTCOME_BLIND_RINALMO_DEPENDENCY_CACHE_COMPLETE",
        "source_csv": str(m2_csv),
        "source_columns": list(SOURCE_COLUMNS),
        "feature_names": list(FEATURE_NAMES),
        "n_constructs": len(groups),
        "n_registered_mutants": len(rows),
        "n_unique_wt_sequences": len({row["wt_sequence"] for row in rows}),
        "n_unique_mutant_sequences": len(
            {row["mutant_sequence"] for row in rows}
        ),
        "n_unique_dependency_edges": len(feature_cache),
        "n_dependency_edge_reuse_rows": len(rows) - len(feature_cache),
        "n_unique_inference_sequences": len(sequence_universe),
        "sequence_length": length,
        "feature_width": len(FEATURE_NAMES),
        "model_name": RINALMO_MODEL_NAME,
        "model_parameter_count": RINALMO_PARAMETER_COUNT,
        "model_code_commit": RINALMO_CODE_COMMIT,
        "dependency_code_commit_reviewed": DEPENDENCY_CODE_COMMIT,
        "model_code_root": str(model_code_root) if model_code_root else "TEST_INFERER",
        "weights_path": str(weights_path) if weights_path else "TEST_INFERER",
        "attention_backend": attention_backend,
        "full_unmasked_wt_and_exact_mutant": True,
        "self_dependency_zero": True,
        "unique_sequence_deduplication": "EXACT_SEQUENCE_IDENTITY_ONLY",
        "registered_mutations_only": True,
        "mutant_reactivity_rows_read": 0,
        "target_error_or_mask_read": False,
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
    parser.add_argument("--model-code-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--attention-backend", choices=("flash",), default="flash")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-constructs", type=int)
    parser.add_argument("--out-h5", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    assert_cache_authority(args.repo_root.resolve())
    if args.batch_size < 1:
        raise ValueError("v7 batch size must be positive")
    actual_commit = _git_head(args.model_code_root.resolve())
    if actual_commit != RINALMO_CODE_COMMIT:
        raise RuntimeError(
            f"RiNALMo code commit {actual_commit} differs from frozen {RINALMO_CODE_COMMIT}"
        )
    inferer = RiNALMoGigaLogitInferer(
        code_root=args.model_code_root,
        weights_path=args.weights,
        device=args.device,
        attention_backend=args.attention_backend,
    )
    result = build_cache(
        args.m2_csv,
        args.out_h5,
        args.out_manifest,
        inferer=inferer,
        batch_size=args.batch_size,
        max_constructs=args.max_constructs,
        model_code_root=args.model_code_root,
        weights_path=args.weights,
        attention_backend=args.attention_backend,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
