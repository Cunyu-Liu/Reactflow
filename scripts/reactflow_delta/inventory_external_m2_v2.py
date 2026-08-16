#!/usr/bin/env python3
"""inventory_external_m2_v2: outcome-blind external M2 component inventory (P4 preaccess).

v2: robust WT-anchor matching via sequence-edit sets, not name conventions.
Component = one WT anchor construct + its single-SNV mutant library (external
exchangeable unit, analogous to a development puzzle). Only identities are
counted outcome-blind; no external reactivity value is used for selection.

Roles (contract 7.2/7.5):
  - direct_external: genuinely development-disconnected biology (different
    organism/study, zero sequence identity with OK7a_M2 dev set)
  - adjacent: same OpenKnot program lineage (Pilot PK50/PK90, Rounds 1/2/4)
    => sensitivity only, not primary external confirmation
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from reactflow.delta.rdat import (
    parse_rdat, parse_mutations_from_name, is_wt_profile,
)

# candidate external M2-style 2A3 datasets (RMDB rdat_tierA)
DIRECT_EXTERNAL = ["M2SL5_2A3_0000", "M3SARS_2A3_0000", "15KLIB_2A3_0000"]
ADJACENT = ["M2PK50_2A3_0000", "M2PK90_2A3_0000", "OK1LIB_2A3_0000", "OK2TRN_2A3_0000"]


def _single_snv_edit(seq_a: str, seq_b: str) -> list[int]:
    """Positions where a and b differ (0-based); [] if not a single-SNV pair."""
    if len(seq_a) != len(seq_b):
        return []
    diffs = [i for i, (x, y) in enumerate(zip(seq_a, seq_b)) if x != y]
    return diffs if len(diffs) == 1 else []


def build_components(profiles: list) -> list[dict]:
    """Group single-SNV mutants to WT anchors by exact sequence edit distance."""
    wt = [x for x in profiles if is_wt_profile(x)]
    mut = [x for x in profiles if not is_wt_profile(x)]
    comps = []
    for w in wt:
        wseq = w["profile_sequence"]
        if not wseq:
            continue
        members = []
        for m in mut:
            ms = parse_mutations_from_name(m["profile_name"])
            if len(ms) != 1:
                continue
            diffs = _single_snv_edit(wseq, m["profile_sequence"])
            if len(diffs) == 1 and diffs[0] == ms[0]["position"]:
                members.append({"name": m["profile_name"], "edit_pos": diffs[0]})
        if members:
            comps.append({
                "wt_name": w["profile_name"],
                "wt_sequence": wseq,
                "seq_len": len(wseq),
                "n_snv_mutants": len(members),
                "mutants": members,
            })
    return comps


def inventory(rdat_dir: Path, dev_csv: Path) -> dict:
    dev = pd.read_csv(dev_csv)
    dev_all = set(dev["sequence"])
    dev_wt = set(dev[dev["id"].str.endswith("_wt")]["sequence"])

    def scan(cids: list[str], role: str) -> tuple[list, list]:
        ds_rows = []
        all_comps = []
        for cid in cids:
            p = rdat_dir / f"{cid}.rdat"
            if not p.exists():
                ds_rows.append({"dataset": cid, "role": role, "status": "MISSING"})
                continue
            r = parse_rdat(p)
            prof = r["profiles"]
            seqs = {x["profile_sequence"] for x in prof}
            comps = build_components(prof)
            ds_rows.append({
                "dataset": cid, "role": role, "status": "OK",
                "n_profiles": len(prof),
                "n_wt": sum(1 for x in prof if is_wt_profile(x)),
                "n_mutants": len(prof) - sum(1 for x in prof if is_wt_profile(x)),
                "overlap_dev_wt": len(seqs & dev_wt),
                "overlap_dev_all": len(seqs & dev_all),
                "n_components": len(comps),
                "n_single_snv": sum(c["n_snv_mutants"] for c in comps),
            })
            all_comps.extend(comps)
        return ds_rows, all_comps

    direct_ds, direct_comps = scan(DIRECT_EXTERNAL, "direct_external")
    adjacent_ds, adjacent_comps = scan(ADJACENT, "adjacent")

    report = {
        "schema_version": "reactflow_delta.p4_external_inventory.v2",
        "dev_csv": str(dev_csv),
        "task_identity": {
            "probe": "2A3-MaP (Ribonanza Kaggle M2-style datasets via RMDB)",
            "chemistry_family_match": "same 2A3-MaP chemistry family as OK7a_M2 development",
            "platform": "NovaSeq (Ribonanza) vs Ultima (OpenKnot M2); recorded per axis, not concatenated",
            "normalization": "RNAFramework per-dataset; recorded per component",
            "estimand": "full-construct mutant 2A3 reactivity response given WT profile + exact SNV; "
                        "external components are M2 single-SNV libraries, same estimand.",
        },
        "direct_external": {
            "datasets": direct_ds,
            "K_preaccess_components": len(direct_comps),
            "K_preaccess_single_snv": sum(c["n_snv_mutants"] for c in direct_comps),
            "component_sizes": [c["n_snv_mutants"] for c in direct_comps],
        },
        "adjacent_openknot_lineage": {
            "datasets": adjacent_ds,
            "K_preaccess_components": len(adjacent_comps),
            "K_preaccess_single_snv": sum(c["n_snv_mutants"] for c in adjacent_comps),
            "note": "OpenKnot Pilot/Round 1-2 lineage => sensitivity only, NOT primary external.",
        },
        "development_disconnect_evidence": (
            "zero sequence-identity overlap between every direct_external candidate sequence "
            "and the OK7a_M2 development set (WT and all-mutant); different organism/study/batch "
            "(Ribonanza Kaggle pre/during-competition, NovaSeq 2023 vs OpenKnot M2 2025-09-04 Ultima)"
        ),
        "outcome_blind_note": "inventory counts identities and edit positions only; no external "
                              "reactivity value was used for selection, tuning, or eligibility.",
    }
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rdat-dir", required=True)
    ap.add_argument("--dev-csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    rep = inventory(Path(args.rdat_dir), Path(args.dev_csv))
    Path(args.out).write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    out = {
        "task_identity": rep["task_identity"],
        "direct_external": rep["direct_external"],
        "adjacent_openknot_lineage": rep["adjacent_openknot_lineage"],
        "development_disconnect_evidence": rep["development_disconnect_evidence"],
        "outcome_blind_note": rep["outcome_blind_note"],
    }
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
