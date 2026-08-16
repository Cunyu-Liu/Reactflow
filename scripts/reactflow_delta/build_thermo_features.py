#!/usr/bin/env python3
"""Build PH0 WT thermodynamic features and manifest.

Computes WT-only thermo states (ViennaRNA MFE/PF/BPP) for 6 parents, extracts
edit-position features for 1509 true_pairs, and writes a provenance-tracked
manifest. No mutant states (alt=X blocked). No test labels used.

Usage:
    python scripts/reactflow_delta/build_thermo_features.py \
        --registry artifacts/reactflow_delta/d2r/d1_true_pair_registry.json \
        --output artifacts/reactflow_delta/ph0/thermo_features_manifest.json
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
    extract_position_features,
    get_tool_version,
    seqpos_to_sequence_positions,
)

MANIFEST_SCHEMA_VERSION = "reactflow-delta-ph0-thermo-features-manifest-v1"

# Pre-registered definitions (forward-only, no hyperparameter search)
LOCAL_WINDOW = 10  # |seq_pos - edit_pos| <= 10
REMOTE_THRESHOLD = 20  # |seq_pos - edit_pos| > 20
CONTACT_BPP_THRESHOLD = 0.05
SWITCH_DISTANCE_THRESHOLD = 20  # max_abs_delta_distance_from_edit > 20 -> switch


def pair_id(entry: dict) -> str:
    """Deterministic pair identifier."""

    raw = "{}:{}:{}:{}".format(
        os.path.basename(entry["rdat_path"]),
        entry["wt_profile_index"],
        entry["mutant_profile_index"],
        entry["matched_mutation"]["encoded_position_1indexed"],
    )
    return raw


def compute_delta_stats(
    delta: list,
    seq_positions: list[int | None],
    edit_pos_1indexed: int,
    edit_arr_idx: int,
) -> dict:
    """Compute Δreactivity statistics relative to the edit position.

    ``seq_positions`` maps array index → 1-indexed SEQUENCE position.
    """

    arr = np.array(delta, dtype=float)
    n = len(arr)
    n_none = sum(1 for v in delta if v is None)
    abs_arr = np.abs(arr)

    # Edit position
    delta_at_edit = float(arr[edit_arr_idx]) if edit_arr_idx is not None and edit_arr_idx < n else None

    # Max |delta| (excluding None)
    max_idx = int(np.nanargmax(abs_arr))
    max_seq_pos = seq_positions[max_idx] if max_idx < len(seq_positions) else None
    max_distance = abs(max_seq_pos - edit_pos_1indexed) if max_seq_pos is not None else None

    # Local / remote windows (by sequence distance)
    local_mask = np.zeros(n, dtype=bool)
    remote_mask = np.zeros(n, dtype=bool)
    for i in range(n):
        sp = seq_positions[i] if i < len(seq_positions) else None
        if sp is None:
            continue
        dist = abs(sp - edit_pos_1indexed)
        if dist <= LOCAL_WINDOW:
            local_mask[i] = True
        elif dist > REMOTE_THRESHOLD:
            remote_mask[i] = True

    local_vals = abs_arr[local_mask]
    remote_vals = abs_arr[remote_mask]

    return {
        "delta_at_edit": delta_at_edit,
        "abs_delta_at_edit": abs(delta_at_edit) if delta_at_edit is not None else None,
        "max_abs_delta": float(abs_arr[max_idx]),
        "max_abs_delta_seq_pos": max_seq_pos,
        "max_abs_delta_arr_idx": max_idx,
        "max_abs_delta_distance_from_edit": max_distance,
        "local_mean_abs_delta": float(np.mean(local_vals)) if len(local_vals) > 0 else None,
        "local_n": int(local_mask.sum()),
        "remote_mean_abs_delta": float(np.mean(remote_vals)) if len(remote_vals) > 0 else None,
        "remote_median_abs_delta": float(np.median(remote_vals)) if len(remote_vals) > 0 else None,
        "remote_95pct_abs_delta": float(np.percentile(remote_vals, 95)) if len(remote_vals) > 0 else None,
        "remote_n": int(remote_mask.sum()),
        "n_nonnone_delta": n - n_none,
        "n_none_delta": n_none,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True, help="Path to d1_true_pair_registry.json")
    parser.add_argument("--output", required=True, help="Output manifest path")
    parser.add_argument("--temperature", type=float, default=37.0)
    args = parser.parse_args()

    registry_path = Path(args.registry)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load registry
    print("Loading registry...", flush=True)
    with registry_path.open() as f:
        registry = json.load(f)
    all_entries = registry["registry"]
    true_pairs = [e for e in all_entries if e.get("true_pair")]
    excluded = [e for e in all_entries if not e.get("true_pair")]
    print(f"  total={len(all_entries)} true_pair={len(true_pairs)} excluded={len(excluded)}", flush=True)

    # Group by parent
    by_parent: dict[str, list[dict]] = defaultdict(list)
    for e in true_pairs:
        by_parent[e["parent_prefix"]].append(e)

    # Cache: rdat_path → parsed RDAT (SEQPOS, OFFSET, SEQUENCE)
    rdat_cache: dict[str, dict] = {}

    # Cache: parent → WT thermo state
    wt_state_cache: dict[str, dict] = {}

    # Compute WT states per parent
    print("Computing WT thermo states per parent...", flush=True)
    per_parent: dict[str, dict] = {}
    for parent, entries in sorted(by_parent.items()):
        rdat_path = entries[0]["rdat_path"]
        doc = parse_rdat(rdat_path)
        rdat_cache[rdat_path] = doc
        wt_seq = doc["headers"]["SEQUENCE"]
        offset = int(doc["headers"]["OFFSET"])

        state = compute_wt_thermo_state(wt_seq, temperature=args.temperature)
        wt_state_cache[parent] = state

        rdat_files = sorted(set(e["rdat_path"] for e in entries))
        per_parent[parent] = {
            "seq_sha256": state["seq_sha256"],
            "seq_length": state["length"],
            "mfe_energy_kcal_mol": state["mfe_energy_kcal_mol"],
            "pf_energy_kcal_mol": state["pf_energy_kcal_mol"],
            "mfe_structure": state["mfe_structure"],
            "n_pairs": len(entries),
            "rdat_files": [os.path.basename(p) for p in rdat_files],
            "offset": offset,
        }
        print(f"  {parent[:40]}: len={state['length']} mfe={state['mfe_energy_kcal_mol']:.1f} n={len(entries)}", flush=True)

    # Extract per-pair features
    print("Extracting per-pair features...", flush=True)
    per_pair: list[dict] = []
    for idx, entry in enumerate(true_pairs):
        parent = entry["parent_prefix"]
        rdat_path = entry["rdat_path"]

        # Get cached or parse RDAT
        if rdat_path not in rdat_cache:
            rdat_cache[rdat_path] = parse_rdat(rdat_path)
        doc = rdat_cache[rdat_path]

        offset = int(doc["headers"]["OFFSET"])
        seqpos_tokens = doc["seqpos"]
        seq_positions = seqpos_to_sequence_positions(seqpos_tokens, offset)

        edit_pos = entry["matched_mutation"]["encoded_position_1indexed"]
        encoded_ref = entry["matched_mutation"]["encoded_ref"]

        # Find array index for edit position
        edit_arr_idx = None
        for i, sp in enumerate(seq_positions):
            if sp == edit_pos:
                edit_arr_idx = i
                break

        # WT features at edit position
        state = wt_state_cache[parent]
        wt_features = extract_position_features(
            state, edit_pos, contact_bpp_threshold=CONTACT_BPP_THRESHOLD
        )

        # Delta stats
        delta = entry.get("delta_reactivity_normalized") or entry.get("delta_reactivity_raw")
        delta_field = "delta_reactivity_normalized" if entry.get("delta_reactivity_normalized") else "delta_reactivity_raw"
        delta_stats = compute_delta_stats(delta, seq_positions, edit_pos, edit_arr_idx or 0)

        # Fragility proxy (pre-registered)
        fragility_value = wt_features["bpp_paired_prob"]
        switch_enriched = (
            delta_stats["max_abs_delta_distance_from_edit"] is not None
            and delta_stats["max_abs_delta_distance_from_edit"] > SWITCH_DISTANCE_THRESHOLD
        )

        pair_record = {
            "pair_id": pair_id(entry),
            "parent_prefix": parent,
            "citation_doi": entry["citation_doi"],
            "owner": entry.get("owner"),
            "rdat_file": os.path.basename(rdat_path),
            "wt_profile_index": entry["wt_profile_index"],
            "mutant_profile_index": entry["mutant_profile_index"],
            "encoded_position_1indexed": edit_pos,
            "encoded_ref": encoded_ref,
            "aligned_length": entry["aligned_length"],
            "edit_arr_idx": edit_arr_idx,
            "delta_field_used": delta_field,
            **delta_stats,
            "wt_features": wt_features,
            "fragility_proxy": "bpp_paired_prob",
            "fragility_proxy_value": fragility_value,
            "switch_enriched": switch_enriched,
        }
        per_pair.append(pair_record)

        if (idx + 1) % 200 == 0:
            print(f"  processed {idx + 1}/{len(true_pairs)}", flush=True)

    # Build manifest
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "stage": "PH0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "tool": get_tool_version(),
            "params": {
                "temperature_celsius": args.temperature,
                "local_window": LOCAL_WINDOW,
                "remote_threshold": REMOTE_THRESHOLD,
                "contact_bpp_threshold": CONTACT_BPP_THRESHOLD,
                "switch_distance_threshold": SWITCH_DISTANCE_THRESHOLD,
            },
            "input_registry": {
                "path": str(registry_path),
                "sha256": sha256_file(registry_path),
            },
        },
        "true_pairs_used": len(true_pairs),
        "excluded_count": len(excluded),
        "exclusion_reasons": "see d1_pipeline_summary.json for full exclusion distribution",
        "per_parent": per_parent,
        "per_pair": per_pair,
        "fragility_proxy_definition": {
            "name": "bpp_paired_prob",
            "description": (
                "BPP-derived paired probability at the edit position in the WT "
                "structure. Higher = more paired in WT = more fragile to mutation."
            ),
            "hypothesis": (
                "Edits at well-paired (high BPP) WT positions cause larger "
                "structural disruption, producing larger |Δreactivity| at the "
                "edit position and/or more remote propagation."
            ),
            "switch_enriched_definition": (
                f"max_abs_delta_distance_from_edit > {SWITCH_DISTANCE_THRESHOLD} "
                "(the largest |Δreactivity| occurs at a remote position, "
                "suggesting a structural switch rather than local disruption)"
            ),
        },
    }

    # Write manifest
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(manifest, indent=2, sort_keys=True)
    output_path.write_text(content, encoding="utf-8")
    output_sha256 = sha256_file(output_path)

    print(f"\nManifest written: {output_path}", flush=True)
    print(f"  pairs: {len(per_pair)}", flush=True)
    print(f"  parents: {len(per_parent)}", flush=True)
    print(f"  output_sha256: {output_sha256}", flush=True)
    print(f"  switch_enriched: {sum(1 for p in per_pair if p['switch_enriched'])}", flush=True)


if __name__ == "__main__":
    main()
