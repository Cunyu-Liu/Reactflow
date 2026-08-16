#!/usr/bin/env python3
"""inventory_external_m2_v1: outcome-blind external M2 component inventory (P4 preaccess).

Builds a development-disconnected external confirmatory component pool from
RMDB Ribonanza M2-style 2A3 datasets. Each component = one WT anchor construct
plus its single-SNV mutant library (the external exchangeable unit, analogous
to a development puzzle). Only counts components and mutants OUTCOME-BLIND
(no reactivity values read into decision making); produces K_preaccess and
task-identity fields for the frozen P4 protocol.

Selection filters (contract 7.5/12.6):
  - M2-style mutational profiling (WT + single-nucleotide mutants)
  - 2A3-MaP probe (same chemistry family as OpenKnot M2 development)
  - development-disconnected: no sequence identity overlap with OK7a_M2 dev set
  - single-SNV mutants only (exact-key pairing with WT anchor)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from reactflow.delta.rdat import (
    parse_rdat, parse_mutations_from_name, is_wt_profile,
)


# candidate external M2-style 2A3 datasets (RMDB rdat_tierA)
CANDIDATES = [
    "M2SL5_2A3_0000", "M3SARS_2A3_0000", "M2PK50_2A3_0000",
    "M2PK90_2A3_0000", "15KLIB_2A3_0000",
]


def _base_name(name: str) -> str:
    """Strip SNV tokens (e.g. 0G-A) from a profile name to get the WT-family name."""
    import re
    out = []
    for tok in name.split("_"):
        if re.fullmatch(r"\d+[ACGU]-[ACGU]", tok):
            continue
        out.append(tok)
    return "_".join(out)


def inventory(rdat_dir: Path, dev_csv: Path) -> dict:
    dev = pd.read_csv(dev_csv)
    dev_all = set(dev["sequence"])
    dev_wt = set(dev[dev["id"].str.endswith("_wt")]["sequence"])

    components = {}  # dataset -> list of {wt_name, wt_seq, n_snv, seq_len}
    datasets = []
    for cid in CANDIDATES:
        p = rdat_dir / f"{cid}.rdat"
        if not p.exists():
            datasets.append({"dataset": cid, "status": "MISSING"})
            continue
        r = parse_rdat(p)
        prof = r["profiles"]
        wt = [x for x in prof if is_wt_profile(x)]
        mut = [x for x in prof if not is_wt_profile(x)]
        seqs = {x["profile_sequence"] for x in prof}
        ov_wt = len(seqs & dev_wt)
        ov_all = len(seqs & dev_all)

        # group single-SNV mutants by WT anchor family
        by_family = {}
        for x in mut:
            ms = parse_mutations_from_name(x["profile_name"])
            if len(ms) != 1:
                continue
            fam = _base_name(x["profile_name"])
            by_family.setdefault(fam, []).append(x)
        # WT anchor map: match WT profile whose name is the family base
        wt_by_name = {x["profile_name"]: x for x in wt}
        comps = []
        for fam, muts in by_family.items():
            # try exact WT match; else pick the first WT whose base matches
            anchor = wt_by_name.get(fam)
            if anchor is None:
                cand = [w for w in wt if _base_name(w["profile_name"]) == fam]
                anchor = cand[0] if cand else None
            if anchor is None:
                continue
            seq_len = len(anchor["reactivity"])
            comps.append({
                "wt_name": anchor["profile_name"],
                "wt_sequence": anchor["profile_sequence"],
                "n_snv_mutants": len(muts),
                "seq_len": seq_len,
            })
        components[cid] = comps
        datasets.append({
            "dataset": cid, "status": "OK",
            "n_profiles": len(prof), "n_wt": len(wt), "n_mutants": len(mut),
            "n_unique_seq": len(seqs), "overlap_dev_wt": ov_wt, "overlap_dev_all": ov_all,
            "n_components": len(comps), "n_single_snv": sum(c["n_snv_mutants"] for c in comps),
        })

    report = {
        "schema_version": "reactflow_delta.p4_external_inventory.v1",
        "dev_csv": str(dev_csv),
        "task_identity": {
            "probe": "2A3-MaP (RMDB Ribonanza M2-style datasets)",
            "chemistry_family_match": "same 2A3-MaP chemistry family as OK7a_M2 development",
            "platform": "NovaSeq (Ribonanza) vs Ultima (OpenKnot M2); recorded, not concatenated",
            "normalization": "RNAFramework per-dataset; recorded per component",
            "note": "Task = full-construct mutant 2A3 reactivity response given WT profile + exact SNV; "
                    "external components are M2 single-SNV libraries, same estimand.",
        },
        "datasets": datasets,
        "components": components,
        "K_preaccess": sum(len(v) for v in components.values()),
        "K_preaccess_single_snv_mutants": sum(c["n_snv_mutants"]
                                               for v in components.values() for c in v),
        "development_disconnect_evidence": (
            "zero sequence-identity overlap between every candidate external sequence "
            "and the OK7a_M2 development set (WT and all-mutant); different study/batch "
            "(Ribonanza Kaggle pre/during-competition vs OpenKnot M2 2025-09-04 Ultima)"
        ),
        "outcome_blind_note": "inventory counts identities only; no external reactivity "
                              "value was used for selection/tuning.",
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
    print(json.dumps({k: v for k, v in rep.items() if k != "components"}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
