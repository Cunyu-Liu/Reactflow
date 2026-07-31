#!/usr/bin/env python3
"""Build M0-R per-pair mutant thermo features (delta_thermo).

M0-R remediation (v3.4 §2.2): for each of 1509 true_pairs, construct 3 mutant
sequences by substituting the edit base with the 3 non-ref alt bases, fold each
with ViennaRNA, marginalize (mean) over the 3 alts, and compute the per-position
delta_thermo = mutant_thermo_mean - wt_thermo.

This is **3-alt marginalization**, NOT alt inference (compliant with v3.3 §2.3
item 7). The same approach is already used by baselines.py §10.2 thermo baselines.

Output features (per-position, 5 dims, in array coordinates):
  0. delta_unpaired_prob
  1. delta_positional_entropy_bits
  2. delta_bpp_paired_prob
  3. delta_mfe_energy_kcal_mol  (scalar broadcast to all positions)
  4. delta_pf_energy_kcal_mol   (scalar broadcast to all positions)

Reuses:
  * baselines.build_mutant_sequences(wt_seq, edit_pos_1idx, ref_base) -> 3 seqs
  * thermo_state.compute_wt_thermo_state(seq, temperature) -> thermo dict

Caches mutant thermo by (parent_sha256, edit_pos_1indexed, encoded_ref) since
many pairs within a parent may share the same edit position.

Usage (editflow env, has ViennaRNA):
    PYTHONPATH=src python scripts/reactflow_delta/build_m0r_features.py \\
        --m0-manifest /mnt/cunyuliu/reactflow_delta_artifacts_20260729/m0/m0_pair_manifest.json \\
        --output-dir /mnt/cunyuliu/reactflow_delta_artifacts_20260729/m0r
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from reactflow.delta.rdat import parse_rdat  # noqa: E402
from reactflow.delta.manifests import sha256_file  # noqa: E402
from reactflow.delta.baselines import build_mutant_sequences  # noqa: E402
from reactflow.delta.thermo_state import compute_wt_thermo_state, get_tool_version  # noqa: E402

MANIFEST_SCHEMA_VERSION = "reactflow-delta-m0r-features-v1"


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
        # bpp_paired_prob = 1 - unpaired_prob (consistent with M0 parent_thermo)
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

    ``seq_array[i]`` is the value at 1-indexed SEQUENCE position ``i+1``.
    Output ``out[arr_idx]`` = seq_array[seq_positions[arr_idx] - 1].
    Missing/out-of-range positions get 0.0 (no delta signal).
    """

    out = np.zeros(aligned_length, dtype=np.float32)
    for i in range(aligned_length):
        sp = seq_positions[i]
        if sp is not None and 1 <= sp <= seq_length:
            out[i] = float(seq_array[sp - 1])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m0-manifest", required=True, help="Path to m0_pair_manifest.json")
    parser.add_argument("--output-dir", required=True, help="Output directory (on /mnt)")
    parser.add_argument("--temperature", type=float, default=37.0)
    args = parser.parse_args()

    m0_manifest_path = Path(args.m0_manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading M0 manifest...", flush=True)
    t0 = time.time()
    with m0_manifest_path.open() as f:
        m0_manifest = json.load(f)
    per_pair = m0_manifest["per_pair"]
    per_parent = m0_manifest["per_parent"]
    print(f"  pairs={len(per_pair)} parents={len(per_parent)} ({time.time()-t0:.1f}s)", flush=True)

    # ------------------------------------------------------------------
    # Step 1: load WT sequences per parent (from RDAT)
    # ------------------------------------------------------------------
    print("Loading WT sequences per parent...", flush=True)
    # parent -> wt_sequence. We need the RDAT path; reconstruct from m0 manifest.
    # per_parent has parent_sha256 but not wt_sequence. We get wt_seq from the
    # first pair of each parent (rdat_path is in per_pair).
    parent_to_rdat: dict[str, str] = {}
    for p in per_pair:
        if p["parent"] not in parent_to_rdat:
            parent_to_rdat[p["parent"]] = p["rdat_path"]

    # Map rdat_path (basename) -> full path. We need to find the RDAT file.
    # The registry has the full path; the manifest has basename only.
    # Reconstruct: RDAT files are in data_registry/ or artifacts. Use the d2r registry.
    # Actually, build_m0_features.py used entry["rdat_path"] (full path) and stored
    # basename in manifest. We need to find full paths. Let's search common dirs.
    rdat_search_dirs = [
        Path("/mnt/cunyuliu/reactflow_delta_artifacts_20260729/data"),
        Path("data_registry"),
        Path("artifacts/reactflow_delta"),
    ]
    rdat_full_paths: dict[str, Path] = {}

    # First, try to get full paths from the d2r registry.
    d2r_registry_path = Path("artifacts/reactflow_delta/d2r/d1_true_pair_registry.json")
    if d2r_registry_path.exists():
        with d2r_registry_path.open() as f:
            d2r_reg = json.load(f)
        for e in d2r_reg["registry"]:
            if e.get("true_pair"):
                bn = os.path.basename(e["rdat_path"])
                if bn not in rdat_full_paths:
                    rdat_full_paths[bn] = Path(e["rdat_path"])

    print(f"  resolved {len(rdat_full_paths)} RDAT full paths", flush=True)

    # Load WT sequences
    wt_sequences: dict[str, str] = {}  # parent -> wt_seq
    rdat_cache: dict[str, dict] = {}  # rdat_path -> parsed doc
    for parent, rdat_bn in parent_to_rdat.items():
        rdat_path = rdat_full_paths.get(rdat_bn)
        if rdat_path is None:
            # Fallback: search dirs
            for d in rdat_search_dirs:
                candidate = d / rdat_bn
                if candidate.exists():
                    rdat_path = candidate
                    break
        if rdat_path is None or not Path(rdat_path).exists():
            raise FileNotFoundError(f"RDAT file not found for parent {parent}: {rdat_bn}")
        if str(rdat_path) not in rdat_cache:
            rdat_cache[str(rdat_path)] = parse_rdat(str(rdat_path))
        doc = rdat_cache[str(rdat_path)]
        wt_sequences[parent] = doc["headers"]["SEQUENCE"]
        print(f"  {parent[:40]}: wt_seq len={len(wt_sequences[parent])}", flush=True)

    # ------------------------------------------------------------------
    # Step 2: load WT thermo per parent (from M0 parent_thermo npz)
    # ------------------------------------------------------------------
    print("Loading WT thermo per parent...", flush=True)
    m0_thermo_dir = m0_manifest_path.parent / "parent_thermo"
    wt_thermo: dict[str, dict] = {}  # parent -> {unpaired_prob, ...}
    for parent, info in per_parent.items():
        sha = info["parent_sha256"]
        npz_path = m0_thermo_dir / f"{sha}.npz"
        if not npz_path.exists():
            # Try relative path from manifest
            npz_path = m0_manifest_path.parent / info["npz_path"]
        data = dict(np.load(str(npz_path)))
        wt_thermo[parent] = {
            "unpaired_prob": data["unpaired_prob"],
            "positional_entropy_bits": data["positional_entropy_bits"],
            "bpp_paired_prob": data["bpp_paired_prob"],
            "mfe_energy_kcal_mol": float(data["mfe_energy"]),
            "pf_energy_kcal_mol": float(data["pf_energy"]),
            "seq_length": int(data["seq_length"]),
        }
        print(f"  {parent[:40]}: loaded WT thermo (n={wt_thermo[parent]['seq_length']})", flush=True)

    # ------------------------------------------------------------------
    # Step 3: compute mutant thermo + delta_thermo per pair (cached)
    # ------------------------------------------------------------------
    print("Computing mutant thermo + delta_thermo per pair...", flush=True)
    mut_thermo_cache: dict[str, dict] = {}  # "sha:pos:ref" -> mutant thermo marginal
    pair_ids: list[str] = []
    delta_thermo_list: list[np.ndarray] = []
    n_cache_hits = 0
    n_cache_misses = 0

    for idx, pm in enumerate(per_pair):
        parent = pm["parent"]
        pid = pm["pair_id"]
        edit_pos_1idx = int(pm["edit_pos_1indexed"])
        encoded_ref = str(pm["encoded_ref"])
        aligned_length = int(pm["aligned_length"])
        seq_positions = pm["seq_positions"]
        parent_sha = per_parent[parent]["parent_sha256"]
        seq_length = wt_thermo[parent]["seq_length"]

        cache_key = f"{parent_sha}:{edit_pos_1idx}:{encoded_ref}"
        if cache_key in mut_thermo_cache:
            mut_thermo = mut_thermo_cache[cache_key]
            n_cache_hits += 1
        else:
            wt_seq = wt_sequences[parent]
            mut_thermo = compute_mutant_thermo_marginal(
                wt_seq, edit_pos_1idx, encoded_ref, args.temperature
            )
            mut_thermo_cache[cache_key] = mut_thermo
            n_cache_misses += 1

        # delta_thermo on sequence coordinates
        wt = wt_thermo[parent]
        delta_unpaired_seq = mut_thermo["unpaired_prob"] - wt["unpaired_prob"]
        delta_entropy_seq = mut_thermo["positional_entropy_bits"] - wt["positional_entropy_bits"]
        delta_bpp_seq = mut_thermo["bpp_paired_prob"] - wt["bpp_paired_prob"]
        delta_mfe = float(mut_thermo["mfe_energy_kcal_mol"] - wt["mfe_energy_kcal_mol"])
        delta_pf = float(mut_thermo["pf_energy_kcal_mol"] - wt["pf_energy_kcal_mol"])

        # Map to array coordinates
        delta_unpaired_arr = map_seq_array_to_delta(delta_unpaired_seq, seq_positions, aligned_length, seq_length)
        delta_entropy_arr = map_seq_array_to_delta(delta_entropy_seq, seq_positions, aligned_length, seq_length)
        delta_bpp_arr = map_seq_array_to_delta(delta_bpp_seq, seq_positions, aligned_length, seq_length)

        # Scalar features broadcast to all positions
        delta_mfe_arr = np.full(aligned_length, delta_mfe, dtype=np.float32)
        delta_pf_arr = np.full(aligned_length, delta_pf, dtype=np.float32)

        # Stack (n, 5)
        delta_thermo = np.stack([
            delta_unpaired_arr,
            delta_entropy_arr,
            delta_bpp_arr,
            delta_mfe_arr,
            delta_pf_arr,
        ], axis=1).astype(np.float32)

        pair_ids.append(pid)
        delta_thermo_list.append(delta_thermo)

        if (idx + 1) % 200 == 0:
            print(f"  processed {idx + 1}/{len(per_pair)} (cache: {n_cache_hits} hits, {n_cache_misses} misses, {time.time()-t0:.0f}s)", flush=True)

    print(f"  done: {len(per_pair)} pairs (cache: {n_cache_hits} hits, {n_cache_misses} misses, {time.time()-t0:.0f}s)", flush=True)

    # ------------------------------------------------------------------
    # Step 4: save features
    # ------------------------------------------------------------------
    print("Saving delta_thermo features...", flush=True)
    features_path = output_dir / "mutant_thermo_features.npz"
    np.savez_compressed(
        str(features_path),
        pair_ids=np.array(pair_ids, dtype=object),
        delta_thermo=np.array(delta_thermo_list, dtype=object),
    )

    # ------------------------------------------------------------------
    # Step 5: feature audit (distribution + correlation with delta_true)
    # ------------------------------------------------------------------
    print("Building feature audit...", flush=True)
    # Load delta_true via evaluate.load_split_pairs for correlation analysis
    try:
        from reactflow.delta.evaluate import load_split_pairs
        registry_path = "artifacts/reactflow_delta/d2r/d1_true_pair_registry.json"
        split_path = "artifacts/reactflow_delta/ph0/split_members.json"
        all_records = []
        for split_name in ("train", "validation"):
            recs = load_split_pairs(split_name, registry_path=registry_path, split_members_path=split_path)
            all_records.extend(recs)
        record_by_pid = {r.pair_id: r for r in all_records}
    except Exception as e:
        print(f"  WARNING: could not load delta_true for audit: {e}", flush=True)
        record_by_pid = {}

    # Compute per-feature stats and correlation with delta_true
    feat_names = ["delta_unpaired_prob", "delta_positional_entropy_bits",
                  "delta_bpp_paired_prob", "delta_mfe_energy", "delta_pf_energy"]
    feat_stats = {name: {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0} for name in feat_names}
    correlations = {name: {"pearson_r": 0.0, "n_positions": 0} for name in feat_names}

    all_feat_vals = {i: [] for i in range(5)}
    feat_true_pairs = {i: ([], []) for i in range(5)}  # (feat_vals, delta_true_vals)

    for pid, dt in zip(pair_ids, delta_thermo_list):
        for i in range(5):
            vals = dt[:, i]
            all_feat_vals[i].extend(vals.tolist())
        if pid in record_by_pid:
            rec = record_by_pid[pid]
            mask = rec.endpoint_mask
            delta_true = rec.delta_true
            for i in range(5):
                vals = dt[:, i]
                valid = mask & ~np.isnan(delta_true) & np.isfinite(vals)
                if valid.sum() > 0:
                    f_vals = vals[valid]
                    d_vals = delta_true[valid]
                    feat_true_pairs[i][0].extend(f_vals.tolist())
                    feat_true_pairs[i][1].extend(d_vals.tolist())

    for i, name in enumerate(feat_names):
        vals = np.array(all_feat_vals[i])
        feat_stats[name] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }
        fv, dv = feat_true_pairs[i]
        if len(fv) > 10 and np.std(fv) > 1e-12:
            r = float(np.corrcoef(fv, dv)[0, 1])
            correlations[name] = {"pearson_r": r, "n_positions": len(fv)}
        else:
            correlations[name] = {"pearson_r": None, "n_positions": len(fv)}

    audit = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "stage": "M0-R",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_pairs": len(pair_ids),
        "feature_dim": 5,
        "feature_names": feat_names,
        "feature_stats": feat_stats,
        "correlation_with_delta_true": correlations,
        "notes": [
            "delta_thermo = mutant_thermo_mean(3 alts) - wt_thermo, per-position",
            "mfe/pf are whole-molecule scalars broadcast to all positions",
            "correlation_with_delta_true: per-position Pearson r between feature and delta_true (masked)",
            "positive |r| > 0.05 indicates the feature carries mutation-effect signal",
        ],
        "provenance": {
            "tool": get_tool_version(),
            "m0_manifest_sha256": sha256_file(m0_manifest_path),
            "git_commit": _git_commit(),
            "temperature_celsius": args.temperature,
        },
    }
    audit_path = output_dir / "feature_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")

    print(f"\nFeatures: {features_path}", flush=True)
    print(f"Audit: {audit_path}", flush=True)
    print(f"  pairs: {len(pair_ids)}", flush=True)
    print(f"  cache: {n_cache_hits} hits / {n_cache_misses} misses", flush=True)
    print(f"  feature correlations with delta_true:", flush=True)
    for name in feat_names:
        r = correlations[name]["pearson_r"]
        n = correlations[name]["n_positions"]
        print(f"    {name}: r={r} (n={n})", flush=True)
    print(f"  elapsed: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
