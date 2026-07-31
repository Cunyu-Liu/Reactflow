#!/usr/bin/env python3
"""Build M0 per-position WT thermo features + sparse contact edges.

Extends PH0 (edit-position-only scalars) to FULL-sequence per-position thermo
features required by the M0 susceptibility kernel K (§4.4) and forcing support
(§4.3). For each of 6 parents:

  * ViennaRNA WT thermo state (MFE/PF/BPP) -- reused from PH0 logic
  * per-position: unpaired_prob, positional_entropy_bits, bpp_paired_prob
  * sparse contact edges: (i, j) pairs with BPP > threshold (0-indexed seq coords)

For each of 1509 true_pairs:

  * pair_id, parent, split, aligned_length, edit_arr_idx, edit_pos_1indexed
  * seq_positions: array index -> 1-indexed SEQUENCE position (from RDAT SEQPOS)

Storage (on /mnt, gitignored):
  m0/
    parent_thermo/{parent_sha256}.npz   -- full-sequence arrays + sparse edges
    m0_pair_manifest.json               -- per-pair metadata + provenance
    m0_feature_build_report.json        -- build summary

No mutant thermo (encoded_alt="X" blocks mutant sequence construction).
No test labels used (split membership only).

Usage (editflow311 env, has ViennaRNA):
    PYTHONPATH=src python scripts/reactflow_delta/build_m0_features.py \
        --registry artifacts/reactflow_delta/d2r/d1_true_pair_registry.json \
        --split artifacts/reactflow_delta/ph0/split_members.json \
        --output-dir /mnt/cunyuliu/reactflow_delta_artifacts_20260729/m0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from reactflow.delta.rdat import parse_rdat  # noqa: E402
from reactflow.delta.manifests import sha256_file  # noqa: E402
from reactflow.delta.thermo_state import (  # noqa: E402
    compute_wt_thermo_state,
    get_tool_version,
    seqpos_to_sequence_positions,
)

MANIFEST_SCHEMA_VERSION = "reactflow-delta-m0-features-manifest-v1"
CONTACT_BPP_THRESHOLD = 0.05  # sparse contact edge threshold (§4.4)


def pair_id_from_entry(entry: dict) -> str:
    """Deterministic pair identifier (matches evaluate._pair_id_from_entry)."""

    rdat_name = os.path.basename(entry["rdat_path"])
    mut = entry["matched_mutation"]
    return "{}:{}:{}:{}".format(
        rdat_name,
        entry["wt_profile_index"],
        entry["mutant_profile_index"],
        mut["encoded_position_1indexed"],
    )


def build_sparse_contacts(bpp: list[list[float]], n: int, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Extract sparse contact edges from BPP matrix.

    Returns (edges (2, n_edges) int64, weights (n_edges,) float32) in 0-indexed
    sequence coordinates. Only upper triangle (i < j) to avoid duplicates.
    """

    edges_i: list[int] = []
    edges_j: list[int] = []
    weights: list[float] = []
    for i in range(n):
        row = bpp[i]
        for j in range(i + 1, n):
            p = float(row[j])
            if p > threshold:
                edges_i.append(i)
                edges_j.append(j)
                weights.append(p)
    if not edges_i:
        return np.zeros((2, 0), dtype=np.int64), np.zeros(0, dtype=np.float32)
    edges = np.stack([np.array(edges_i, dtype=np.int64), np.array(edges_j, dtype=np.int64)], axis=0)
    return edges, np.array(weights, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True, help="Path to d1_true_pair_registry.json")
    parser.add_argument("--split", required=True, help="Path to split_members.json")
    parser.add_argument("--output-dir", required=True, help="Output directory (on /mnt)")
    parser.add_argument("--temperature", type=float, default=37.0)
    parser.add_argument("--contact-threshold", type=float, default=CONTACT_BPP_THRESHOLD)
    args = parser.parse_args()

    registry_path = Path(args.registry)
    split_path = Path(args.split)
    output_dir = Path(args.output_dir)
    thermo_dir = output_dir / "parent_thermo"
    thermo_dir.mkdir(parents=True, exist_ok=True)

    # Load registry
    print("Loading registry...", flush=True)
    t0 = time.time()
    with registry_path.open() as f:
        registry = json.load(f)
    true_pairs = [e for e in registry["registry"] if e.get("true_pair")]
    print(f"  true_pairs={len(true_pairs)} ({time.time()-t0:.1f}s)", flush=True)

    # Load split members
    with split_path.open() as f:
        splits_doc = json.load(f)
    split_map: dict[str, str] = {}  # pair_id -> split name
    for split_name in ("train", "validation", "test"):
        for pid in splits_doc[split_name]["pair_ids"]:
            split_map[pid] = split_name

    # Group true_pairs by parent
    by_parent: dict[str, list[dict]] = defaultdict(list)
    for e in true_pairs:
        by_parent[e["parent_prefix"]].append(e)

    # ------------------------------------------------------------------
    # Step 1: compute per-parent WT thermo state, store .npz
    # ------------------------------------------------------------------
    print("Computing per-parent WT thermo states...", flush=True)
    parent_info: dict[str, dict] = {}  # parent -> {sha256, npz_path, seq_length, ...}
    rdat_cache: dict[str, dict] = {}  # rdat_path -> parsed RDAT doc

    for parent, entries in sorted(by_parent.items()):
        # All entries for a parent share the same WT sequence; use the first RDAT.
        rdat_path = entries[0]["rdat_path"]
        if rdat_path not in rdat_cache:
            rdat_cache[rdat_path] = parse_rdat(rdat_path)
        doc = rdat_cache[rdat_path]
        wt_seq = doc["headers"]["SEQUENCE"]
        offset = int(doc["headers"]["OFFSET"])

        print(f"  {parent[:50]}: folding len={len(wt_seq)}...", flush=True, end=" ")
        state = compute_wt_thermo_state(wt_seq, temperature=args.temperature)
        n = state["length"]
        sha = state["seq_sha256"]

        # Per-position features (full sequence, 0-indexed)
        unpaired_prob = np.array(state["unpaired_prob"], dtype=np.float32)
        positional_entropy = np.array(state["positional_entropy_bits"], dtype=np.float32)
        bpp_paired_prob = np.ones(n, dtype=np.float32) - unpaired_prob

        # Sparse contact edges from BPP
        edges, edge_weights = build_sparse_contacts(state["bpp"], n, args.contact_threshold)

        npz_path = thermo_dir / f"{sha}.npz"
        np.savez_compressed(
            str(npz_path),
            unpaired_prob=unpaired_prob,
            positional_entropy_bits=positional_entropy,
            bpp_paired_prob=bpp_paired_prob,
            contact_edges=edges,
            contact_weights=edge_weights,
            seq_length=np.int64(n),
            mfe_energy=np.float32(state["mfe_energy_kcal_mol"]),
            pf_energy=np.float32(state["pf_energy_kcal_mol"]),
        )
        print(f"done (n_contacts={edges.shape[1]}, npz={npz_path.name})", flush=True)

        parent_info[parent] = {
            "parent_sha256": sha,
            "npz_path": str(npz_path.relative_to(output_dir)),
            "seq_length": n,
            "mfe_energy_kcal_mol": state["mfe_energy_kcal_mol"],
            "pf_energy_kcal_mol": state["pf_energy_kcal_mol"],
            "n_contacts": int(edges.shape[1]),
            "offset": offset,
            "wt_sequence": wt_seq,
        }

    # ------------------------------------------------------------------
    # Step 2: build per-pair manifest
    # ------------------------------------------------------------------
    print("Building per-pair manifest...", flush=True)
    per_pair: list[dict] = []
    n_rdat_parsed = 0

    for idx, entry in enumerate(true_pairs):
        parent = entry["parent_prefix"]
        rdat_path = entry["rdat_path"]
        pid = pair_id_from_entry(entry)

        # Parse RDAT for SEQPOS mapping (cache per file)
        if rdat_path not in rdat_cache:
            rdat_cache[rdat_path] = parse_rdat(rdat_path)
            n_rdat_parsed += 1
        doc = rdat_cache[rdat_path]
        offset = int(doc["headers"]["OFFSET"])
        seqpos_tokens = list(doc["seqpos"])

        # Map array index -> 1-indexed SEQUENCE position
        seq_positions_raw = seqpos_to_sequence_positions(seqpos_tokens, offset)
        aligned_length = int(entry["aligned_length"])

        # Truncate/pad to aligned_length (SEQPOS may be longer)
        seq_positions_list: list[int | None] = []
        for i in range(aligned_length):
            if i < len(seq_positions_raw) and seq_positions_raw[i] is not None:
                seq_positions_list.append(int(seq_positions_raw[i]))
            else:
                seq_positions_list.append(None)

        mut = entry["matched_mutation"]
        edit_pos_1idx = int(mut["encoded_position_1indexed"])

        # Find edit_arr_idx (0-indexed in delta array)
        edit_arr_idx = None
        for i, sp in enumerate(seq_positions_list):
            if sp == edit_pos_1idx:
                edit_arr_idx = i
                break

        split_name = split_map.get(pid, "unknown")

        per_pair.append({
            "pair_id": pid,
            "parent": parent,
            "parent_sha256": parent_info[parent]["parent_sha256"],
            "split": split_name,
            "rdat_path": os.path.basename(rdat_path),
            "aligned_length": aligned_length,
            "edit_arr_idx": edit_arr_idx,
            "edit_pos_1indexed": edit_pos_1idx,
            "encoded_ref": str(mut["encoded_ref"]),
            "seq_positions": seq_positions_list,
            "citation_doi": entry["citation_doi"],
            "owner": entry.get("owner"),
            "modifier": entry.get("modifier"),
            "pair_quality_weight": float(entry.get("pair_quality_weight", 1.0)),
            "measurement_variance": entry.get("measurement_variance"),
            "noise_wt_variance": entry.get("noise_wt_variance"),
            "noise_mut_variance": entry.get("noise_mut_variance"),
        })

        if (idx + 1) % 300 == 0:
            print(f"  processed {idx + 1}/{len(true_pairs)}", flush=True)

    # ------------------------------------------------------------------
    # Step 3: write manifest + report
    # ------------------------------------------------------------------
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "stage": "M0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "tool": get_tool_version(),
            "params": {
                "temperature_celsius": args.temperature,
                "contact_bpp_threshold": args.contact_threshold,
            },
            "input_registry": {
                "path": str(registry_path),
                "sha256": sha256_file(registry_path),
            },
            "input_split": {
                "path": str(split_path),
                "sha256": sha256_file(split_path),
            },
            "git_commit": _git_commit(),
        },
        "n_pairs": len(per_pair),
        "n_parents": len(parent_info),
        "splits": {
            "train": sum(1 for p in per_pair if p["split"] == "train"),
            "validation": sum(1 for p in per_pair if p["split"] == "validation"),
            "test": sum(1 for p in per_pair if p["split"] == "test"),
        },
        "per_parent": {
            p: {
                "parent_sha256": info["parent_sha256"],
                "npz_path": info["npz_path"],
                "seq_length": info["seq_length"],
                "n_contacts": info["n_contacts"],
                "mfe_energy_kcal_mol": info["mfe_energy_kcal_mol"],
                "pf_energy_kcal_mol": info["pf_energy_kcal_mol"],
            }
            for p, info in parent_info.items()
        },
        "per_pair": per_pair,
    }

    manifest_path = output_dir / "m0_pair_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    # Build report
    edit_arr_idx_found = sum(1 for p in per_pair if p["edit_arr_idx"] is not None)
    report = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "stage": "M0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_pairs": len(per_pair),
        "n_parents": len(parent_info),
        "n_rdat_files_parsed": len(rdat_cache),
        "edit_arr_idx_found": edit_arr_idx_found,
        "edit_arr_idx_missing": len(per_pair) - edit_arr_idx_found,
        "splits": manifest["splits"],
        "per_parent_contact_counts": {
            p: info["n_contacts"] for p, info in parent_info.items()
        },
        "output_files": {
            "manifest": str(manifest_path),
            "parent_thermo_dir": str(thermo_dir),
        },
    }
    report_path = output_dir / "m0_feature_build_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(f"\nManifest: {manifest_path}", flush=True)
    print(f"Report: {report_path}", flush=True)
    print(f"  pairs: {len(per_pair)} (edit_arr_idx found: {edit_arr_idx_found})", flush=True)
    print(f"  parents: {len(parent_info)}", flush=True)
    print(f"  splits: {manifest['splits']}", flush=True)
    print(f"  elapsed: {time.time()-t0:.1f}s", flush=True)


def _git_commit() -> str:
    """Return current git HEAD commit hash, or 'unknown'."""

    try:
        import subprocess
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=str(Path(__file__).resolve().parents[2]),
        )
        return out.decode("ascii").strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
