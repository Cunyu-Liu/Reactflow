#!/usr/bin/env python3
"""Build expanded M0 manifest + mutant thermo features (true + annotation-only pairs).

Extends the original M0 (1509 true pairs, 6 parents) to include 6151
annotation-only pairs for a total of 7660 pairs across 31 parents (rmdb_id).

For each unique parent WT sequence:
  * ViennaRNA WT thermo state (MFE/PF/BPP)
  * Per-position: unpaired_prob, positional_entropy_bits, bpp_paired_prob
  * Sparse contact edges (BPP > 0.05)
  * Cached to parent_thermo/{parent_sha256}.npz (shared with existing M0)

For each pair:
  * 3 mutant sequences (substituting edit pos with 3 non-ref alts)
  * Marginalized (mean) mutant thermo
  * delta_thermo = mutant_mean - wt_thermo (5 features per position)
  * Cached by (parent_sha256, resolved_edit_pos, encoded_ref)

Outputs (to /mnt/.../m0/expanded/):
  * m0_pair_manifest_expanded.json  -- per-pair + per-parent metadata
  * mutant_thermo_features_expanded.npz -- pair_ids + delta_thermo arrays

Usage (editflow env, has ViennaRNA):
    PYTHONPATH=src python scripts/reactflow_delta/build_expanded_features.py \
        --registry artifacts/reactflow_delta/d2r/d1_true_pair_registry.json \
        --split /mnt/cunyuliu/reactflow_delta_artifacts_20260729/m0/expanded_split_members.json \
        --output-dir /mnt/cunyuliu/reactflow_delta_artifacts_20260729/m0/expanded
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from reactflow.delta.rdat import parse_rdat  # noqa: E402
from reactflow.delta.manifests import sha256_file  # noqa: E402
from reactflow.delta.baselines import build_mutant_sequences  # noqa: E402
from reactflow.delta.thermo_state import (  # noqa: E402
    compute_wt_thermo_state,
    get_tool_version,
    seqpos_to_sequence_positions,
)

MANIFEST_SCHEMA_VERSION = "reactflow-delta-m0-features-manifest-v1"
CONTACT_BPP_THRESHOLD = 0.05  # sparse contact edge threshold


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


def build_sparse_contacts(
    bpp: list[list[float]], n: int, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
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
    edges = np.stack(
        [np.array(edges_i, dtype=np.int64), np.array(edges_j, dtype=np.int64)], axis=0
    )
    return edges, np.array(weights, dtype=np.float32)


def resolve_edit_sequence_position(
    wt_sequence: str,
    edit_pos_1indexed: int,
    encoded_ref: str,
    offset: int,
) -> tuple[int, str]:
    """Resolve the actual 1-indexed SEQUENCE position of the edit.

    Tries construct-local first (pos - 1), then offset-adjusted (pos - 1 + offset).
    Both sequence and ref are T->U normalized before comparison.

    Returns (seq_pos_1indexed, ref_match_index). Raises ValueError if ref cannot
    be verified at either position.
    """

    wt = wt_sequence.upper().replace("T", "U")
    ref = encoded_ref.upper().replace("T", "U")
    pos0 = edit_pos_1indexed - 1

    # 1. Construct-local 1-indexed (common case).
    if 0 <= pos0 < len(wt) and wt[pos0] == ref:
        return edit_pos_1indexed, "construct_local_1indexed"

    # 2. Offset-adjusted (annotation uses numbering coords; add OFFSET).
    if offset:
        pos0_off = edit_pos_1indexed - 1 + offset
        if 0 <= pos0_off < len(wt) and wt[pos0_off] == ref:
            return edit_pos_1indexed + offset, "offset_adjusted"

    actual = wt[pos0] if 0 <= pos0 < len(wt) else "?"
    raise ValueError(
        f"Could not verify encoded_ref={encoded_ref!r} for edit_pos={edit_pos_1indexed} "
        f"(construct_local base={actual!r}, offset={offset}). RDAT SEQUENCE does "
        f"not match the annotation at either position."
    )


def compute_mutant_thermo_marginal(
    wt_seq: str,
    edit_pos_1idx: int,
    encoded_ref: str,
    temperature: float,
) -> dict[str, np.ndarray]:
    """Fold 3 mutant sequences, marginalize (mean) over alts.

    Returns dict with per-position arrays (length = len(wt_seq)):
      unpaired_prob, positional_entropy_bits, bpp_paired_prob
    and scalars: mfe_energy_kcal_mol, pf_energy_kcal_mol
    """

    mut_seqs = build_mutant_sequences(wt_seq, edit_pos_1idx, encoded_ref)
    n = len(wt_seq)
    n_alts = len(mut_seqs)

    sum_unpaired = np.zeros(n, dtype=np.float64)
    sum_entropy = np.zeros(n, dtype=np.float64)
    sum_bpp_paired = np.zeros(n, dtype=np.float64)
    sum_mfe = 0.0
    sum_pf = 0.0

    for ms in mut_seqs:
        state = compute_wt_thermo_state(ms, temperature=temperature)
        sum_unpaired += np.asarray(state["unpaired_prob"], dtype=np.float64)
        sum_entropy += np.asarray(state["positional_entropy_bits"], dtype=np.float64)
        ms_unpaired = np.asarray(state["unpaired_prob"], dtype=np.float64)
        sum_bpp_paired += (1.0 - ms_unpaired)
        sum_mfe += float(state["mfe_energy_kcal_mol"])
        sum_pf += float(state["pf_energy_kcal_mol"])

    return {
        "unpaired_prob": (sum_unpaired / n_alts).astype(np.float32),
        "positional_entropy_bits": (sum_entropy / n_alts).astype(np.float32),
        "bpp_paired_prob": (sum_bpp_paired / n_alts).astype(np.float32),
        "mfe_energy_kcal_mol": np.float32(sum_mfe / n_alts),
        "pf_energy_kcal_mol": np.float32(sum_pf / n_alts),
    }


def map_seq_array_to_delta(
    seq_array: np.ndarray,
    seq_positions: list,
    aligned_length: int,
    seq_length: int,
) -> np.ndarray:
    """Map a per-sequence-position array to per-array-index alignment.

    seq_array[i] is the value at 1-indexed SEQUENCE position i+1.
    Output out[arr_idx] = seq_array[seq_positions[arr_idx] - 1].
    Missing/out-of-range positions get 0.0.
    """

    out = np.zeros(aligned_length, dtype=np.float32)
    for i in range(aligned_length):
        sp = seq_positions[i]
        if sp is not None and 1 <= sp <= seq_length:
            out[i] = float(seq_array[sp - 1])
    return out


def _git_commit() -> str:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True, help="Path to d1_true_pair_registry.json")
    parser.add_argument("--split", required=True, help="Path to expanded_split_members.json")
    parser.add_argument("--output-dir", required=True, help="Output directory (on /mnt)")
    parser.add_argument("--temperature", type=float, default=37.0)
    parser.add_argument("--contact-threshold", type=float, default=CONTACT_BPP_THRESHOLD)
    args = parser.parse_args()

    registry_path = Path(args.registry)
    split_path = Path(args.split)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # parent_thermo NPZs go to the shared m0/parent_thermo/ directory
    m0_root = output_dir.parent  # /mnt/.../m0
    thermo_dir = m0_root / "parent_thermo"
    thermo_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # ------------------------------------------------------------------
    # Load registry and build pair_id -> entry map
    # ------------------------------------------------------------------
    print("Loading registry...", flush=True)
    with registry_path.open() as f:
        registry = json.load(f)
    pid_to_entry: dict[str, dict] = {}
    for e in registry["registry"]:
        pid = pair_id_from_entry(e)
        if pid not in pid_to_entry:
            pid_to_entry[pid] = e
    print(f"  registry entries={len(registry['registry'])}, unique pair_ids={len(pid_to_entry)}", flush=True)

    # ------------------------------------------------------------------
    # Load expanded split and build pair_id -> split map
    # ------------------------------------------------------------------
    print("Loading expanded split...", flush=True)
    with split_path.open() as f:
        splits_doc = json.load(f)
    split_map: dict[str, str] = {}  # pair_id -> split name
    all_pair_ids: list[str] = []
    for split_name in ("train", "validation", "test"):
        pids = splits_doc[split_name]["pair_ids"]
        for pid in pids:
            split_map[pid] = split_name
        all_pair_ids.extend(pids)
    print(
        f"  train={len(splits_doc['train']['pair_ids'])}, "
        f"validation={len(splits_doc['validation']['pair_ids'])}, "
        f"test={len(splits_doc['test']['pair_ids'])}, "
        f"total={len(all_pair_ids)}",
        flush=True,
    )

    # Verify all split pair_ids exist in registry
    missing = [p for p in all_pair_ids if p not in pid_to_entry]
    if missing:
        raise RuntimeError(f"{len(missing)} split pair_ids missing from registry; e.g. {missing[:3]}")

    # ------------------------------------------------------------------
    # Step 1: Parse RDAT per pair, compute parent_sha256, group unique seqs
    # ------------------------------------------------------------------
    print("Parsing RDAT files and computing parent_sha256 per pair...", flush=True)
    rdat_cache: dict[str, dict] = {}  # rdat_path -> parsed doc
    pair_rdat_info: dict[str, dict] = {}  # pair_id -> {wt_seq, parent_sha256, offset, seqpos, ...}
    sha_to_wt: dict[str, str] = {}  # parent_sha256 -> wt_seq (for thermo computation)
    sha_to_rdat: dict[str, str] = {}  # parent_sha256 -> representative rdat_path
    sha_to_offset: dict[str, int] = {}  # parent_sha256 -> RDAT OFFSET

    for idx, pid in enumerate(all_pair_ids):
        entry = pid_to_entry[pid]
        rdat_path = entry["rdat_path"]
        if rdat_path not in rdat_cache:
            rdat_cache[rdat_path] = parse_rdat(rdat_path)
        doc = rdat_cache[rdat_path]
        wt_seq = doc["headers"]["SEQUENCE"]
        try:
            offset = int(doc["headers"].get("OFFSET", "0") or "0")
        except (ValueError, TypeError):
            offset = 0
        seqpos_tokens = list(doc["seqpos"])
        sha = hashlib.sha256(wt_seq.upper().replace("T", "U").encode("ascii")).hexdigest()

        pair_rdat_info[pid] = {
            "wt_seq": wt_seq,
            "parent_sha256": sha,
            "offset": offset,
            "seqpos_tokens": seqpos_tokens,
            "rdat_path": rdat_path,
        }
        sha_to_wt[sha] = wt_seq
        if sha not in sha_to_rdat:
            sha_to_rdat[sha] = rdat_path
            sha_to_offset[sha] = offset

        if (idx + 1) % 1000 == 0:
            print(f"  parsed {idx + 1}/{len(all_pair_ids)} pairs ({len(rdat_cache)} RDAT files, {len(sha_to_wt)} unique seqs)", flush=True)

    print(
        f"  done: {len(all_pair_ids)} pairs, {len(rdat_cache)} RDAT files, "
        f"{len(sha_to_wt)} unique WT sequences",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Step 2: Compute per-parent WT thermo state (skip cached)
    # ------------------------------------------------------------------
    print("Computing per-parent WT thermo states...", flush=True)
    parent_thermo: dict[str, dict] = {}  # parent_sha256 -> thermo arrays + metadata
    n_cached = 0
    n_computed = 0

    for sha, wt_seq in sorted(sha_to_wt.items()):
        npz_path = thermo_dir / f"{sha}.npz"
        if npz_path.exists():
            data = dict(np.load(str(npz_path)))
            parent_thermo[sha] = {
                "unpaired_prob": data["unpaired_prob"],
                "positional_entropy_bits": data["positional_entropy_bits"],
                "bpp_paired_prob": data["bpp_paired_prob"],
                "mfe_energy_kcal_mol": float(data["mfe_energy"]),
                "pf_energy_kcal_mol": float(data["pf_energy"]),
                "seq_length": int(data["seq_length"]),
                "n_contacts": int(data["contact_edges"].shape[1]) if data["contact_edges"].ndim == 2 else 0,
            }
            n_cached += 1
            print(f"  CACHED sha={sha[:12]}... len={parent_thermo[sha]['seq_length']}", flush=True)
            continue

        print(f"  folding sha={sha[:12]}... len={len(wt_seq)}", flush=True, end=" ")
        state = compute_wt_thermo_state(wt_seq, temperature=args.temperature)
        n = state["length"]

        unpaired_prob = np.array(state["unpaired_prob"], dtype=np.float32)
        positional_entropy = np.array(state["positional_entropy_bits"], dtype=np.float32)
        bpp_paired_prob = np.ones(n, dtype=np.float32) - unpaired_prob

        edges, edge_weights = build_sparse_contacts(state["bpp"], n, args.contact_threshold)

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
        n_computed += 1
        print(f"done (n_contacts={edges.shape[1]})", flush=True)

        parent_thermo[sha] = {
            "unpaired_prob": unpaired_prob,
            "positional_entropy_bits": positional_entropy,
            "bpp_paired_prob": bpp_paired_prob,
            "mfe_energy_kcal_mol": state["mfe_energy_kcal_mol"],
            "pf_energy_kcal_mol": state["pf_energy_kcal_mol"],
            "seq_length": n,
            "n_contacts": int(edges.shape[1]),
        }

    print(f"  parent thermo: {n_cached} cached, {n_computed} computed ({time.time()-t0:.1f}s)", flush=True)

    # ------------------------------------------------------------------
    # Step 3: Compute mutant thermo + delta_thermo per pair (cached)
    # ------------------------------------------------------------------
    print("Computing mutant thermo + delta_thermo per pair...", flush=True)
    mut_thermo_cache: dict[str, dict] = {}  # "sha:pos:ref" -> mutant thermo marginal
    pair_ids_out: list[str] = []
    delta_thermo_list: list[np.ndarray] = []
    per_pair: list[dict] = []
    n_cache_hits = 0
    n_cache_misses = 0
    n_errors = 0
    ref_match_index_counts: dict[str, int] = {}

    for idx, pid in enumerate(all_pair_ids):
        entry = pid_to_entry[pid]
        rdat_info = pair_rdat_info[pid]
        wt_seq = rdat_info["wt_seq"]
        parent_sha = rdat_info["parent_sha256"]
        offset = rdat_info["offset"]
        seqpos_tokens = rdat_info["seqpos_tokens"]
        rmdb_id = entry["rmdb_id"]

        mut = entry["matched_mutation"]
        edit_pos_1idx = int(mut["encoded_position_1indexed"])
        encoded_ref = str(mut["encoded_ref"])
        aligned_length = int(entry["aligned_length"])
        split_name = split_map.get(pid, "unknown")

        # Map seqpos to sequence positions
        seq_positions_raw = seqpos_to_sequence_positions(seqpos_tokens, offset)
        seq_positions_list: list[int | None] = []
        for i in range(aligned_length):
            if i < len(seq_positions_raw) and seq_positions_raw[i] is not None:
                seq_positions_list.append(int(seq_positions_raw[i]))
            else:
                seq_positions_list.append(None)

        # Find edit_arr_idx (0-indexed in delta array)
        edit_arr_idx = None
        for i, sp in enumerate(seq_positions_list):
            if sp == edit_pos_1idx:
                edit_arr_idx = i
                break
        if edit_arr_idx is None:
            edit_arr_idx = edit_pos_1idx - 1  # fallback per task spec

        wt = parent_thermo[parent_sha]
        seq_length = wt["seq_length"]

        # Compute delta_thermo (with error handling for ref verification)
        try:
            mut_pos_1idx, ref_match_index = resolve_edit_sequence_position(
                wt_seq, edit_pos_1idx, encoded_ref, offset
            )
            ref_match_index_counts[ref_match_index] = ref_match_index_counts.get(ref_match_index, 0) + 1

            cache_key = f"{parent_sha}:{mut_pos_1idx}:{encoded_ref}"
            if cache_key in mut_thermo_cache:
                mut_thermo = mut_thermo_cache[cache_key]
                n_cache_hits += 1
            else:
                mut_thermo = compute_mutant_thermo_marginal(
                    wt_seq, mut_pos_1idx, encoded_ref, args.temperature
                )
                mut_thermo_cache[cache_key] = mut_thermo
                n_cache_misses += 1

            # delta_thermo on sequence coordinates
            delta_unpaired_seq = mut_thermo["unpaired_prob"] - wt["unpaired_prob"]
            delta_entropy_seq = mut_thermo["positional_entropy_bits"] - wt["positional_entropy_bits"]
            delta_bpp_seq = mut_thermo["bpp_paired_prob"] - wt["bpp_paired_prob"]
            delta_mfe = float(mut_thermo["mfe_energy_kcal_mol"] - wt["mfe_energy_kcal_mol"])
            delta_pf = float(mut_thermo["pf_energy_kcal_mol"] - wt["pf_energy_kcal_mol"])

            # Map to array coordinates
            delta_unpaired_arr = map_seq_array_to_delta(delta_unpaired_seq, seq_positions_list, aligned_length, seq_length)
            delta_entropy_arr = map_seq_array_to_delta(delta_entropy_seq, seq_positions_list, aligned_length, seq_length)
            delta_bpp_arr = map_seq_array_to_delta(delta_bpp_seq, seq_positions_list, aligned_length, seq_length)

            delta_mfe_arr = np.full(aligned_length, delta_mfe, dtype=np.float32)
            delta_pf_arr = np.full(aligned_length, delta_pf, dtype=np.float32)

            delta_thermo = np.stack([
                delta_unpaired_arr,
                delta_entropy_arr,
                delta_bpp_arr,
                delta_mfe_arr,
                delta_pf_arr,
            ], axis=1).astype(np.float32)
        except Exception as e:
            n_errors += 1
            if n_errors <= 10:
                print(f"  ERROR pair {pid}: {e}", flush=True)
            # Zero delta_thermo for failed pairs
            delta_thermo = np.zeros((aligned_length, 5), dtype=np.float32)
            ref_match_index = "error"

        pair_ids_out.append(pid)
        delta_thermo_list.append(delta_thermo)

        per_pair.append({
            "pair_id": pid,
            "parent": rmdb_id,
            "parent_sha256": parent_sha,
            "split": split_name,
            "rdat_path": os.path.basename(rdat_info["rdat_path"]),
            "aligned_length": aligned_length,
            "edit_arr_idx": edit_arr_idx,
            "edit_pos_1indexed": edit_pos_1idx,
            "encoded_ref": encoded_ref,
            "seq_positions": seq_positions_list,
            "citation_doi": entry.get("citation_doi"),
            "owner": entry.get("owner"),
            "modifier": entry.get("modifier"),
            "pair_quality_weight": float(entry.get("pair_quality_weight", 1.0)),
            "true_pair": bool(entry.get("true_pair", False)),
            "is_annotation_only": bool(entry.get("is_annotation_only", False)),
            "measurement_variance": entry.get("measurement_variance"),
            "noise_wt_variance": entry.get("noise_wt_variance"),
            "noise_mut_variance": entry.get("noise_mut_variance"),
        })

        if (idx + 1) % 500 == 0:
            print(
                f"  processed {idx + 1}/{len(all_pair_ids)} "
                f"(cache: {n_cache_hits} hits, {n_cache_misses} misses, "
                f"errors: {n_errors}, {time.time()-t0:.0f}s)",
                flush=True,
            )

    print(
        f"  done: {len(all_pair_ids)} pairs (cache: {n_cache_hits} hits, "
        f"{n_cache_misses} misses, errors: {n_errors}, {time.time()-t0:.0f}s)",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Step 4: Build per_parent metadata
    # ------------------------------------------------------------------
    print("Building per_parent metadata...", flush=True)
    # Group sha256 -> rmdb_id (a sha256 may appear under one rmdb_id)
    sha_to_rmdb: dict[str, str] = {}
    rmdb_to_shas: dict[str, set] = defaultdict(set)
    for pid in all_pair_ids:
        info = pair_rdat_info[pid]
        rmdb_id = pid_to_entry[pid]["rmdb_id"]
        sha = info["parent_sha256"]
        sha_to_rmdb[sha] = rmdb_id
        rmdb_to_shas[rmdb_id].add(sha)

    per_parent: dict[str, dict] = {}
    for rmdb_id, shas in sorted(rmdb_to_shas.items()):
        # Use the first sha as primary
        primary_sha = sorted(shas)[0]
        wt = parent_thermo[primary_sha]
        npz_path = thermo_dir / f"{primary_sha}.npz"
        entry_data = {
            "parent_sha256": primary_sha,
            "npz_path": str(npz_path.relative_to(m0_root)),
            "seq_length": wt["seq_length"],
            "n_contacts": wt["n_contacts"],
            "mfe_energy_kcal_mol": wt["mfe_energy_kcal_mol"],
            "pf_energy_kcal_mol": wt["pf_energy_kcal_mol"],
        }
        if len(shas) > 1:
            entry_data["all_parent_sha256"] = sorted(shas)
            entry_data["n_unique_sequences"] = len(shas)
        per_parent[rmdb_id] = entry_data

    # ------------------------------------------------------------------
    # Step 5: Write manifest
    # ------------------------------------------------------------------
    print("Writing manifest...", flush=True)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "stage": "M0-expanded",
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
        "n_parents": len(per_parent),
        "n_unique_sequences": len(sha_to_wt),
        "splits": {
            "train": sum(1 for p in per_pair if p["split"] == "train"),
            "validation": sum(1 for p in per_pair if p["split"] == "validation"),
            "test": sum(1 for p in per_pair if p["split"] == "test"),
        },
        "pair_type_counts": {
            "true_pair": sum(1 for p in per_pair if p["true_pair"]),
            "annotation_only": sum(1 for p in per_pair if p["is_annotation_only"] and not p["true_pair"]),
        },
        "ref_match_index_counts": ref_match_index_counts,
        "per_parent": per_parent,
        "per_pair": per_pair,
    }

    manifest_path = output_dir / "m0_pair_manifest_expanded.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    # ------------------------------------------------------------------
    # Step 6: Save mutant thermo features NPZ
    # ------------------------------------------------------------------
    print("Saving mutant_thermo_features_expanded.npz...", flush=True)
    features_path = output_dir / "mutant_thermo_features_expanded.npz"
    np.savez_compressed(
        str(features_path),
        pair_ids=np.array(pair_ids_out, dtype=object),
        delta_thermo=np.array(delta_thermo_list, dtype=object),
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.time() - t0
    print(f"\n=== DONE ({elapsed:.1f}s) ===", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)
    print(f"Features: {features_path}", flush=True)
    print(f"  pairs: {len(per_pair)}", flush=True)
    print(f"  parents (rmdb_id): {len(per_parent)}", flush=True)
    print(f"  unique WT sequences: {len(sha_to_wt)}", flush=True)
    print(f"  parent thermo NPZ: {n_cached} cached + {n_computed} computed", flush=True)
    print(f"  mutant thermo cache: {n_cache_hits} hits, {n_cache_misses} misses", flush=True)
    print(f"  errors: {n_errors}", flush=True)
    print(f"  splits: {manifest['splits']}", flush=True)
    print(f"  pair types: {manifest['pair_type_counts']}", flush=True)
    print(f"  ref_match_index: {ref_match_index_counts}", flush=True)


if __name__ == "__main__":
    main()
