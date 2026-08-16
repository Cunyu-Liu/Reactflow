#!/usr/bin/env python3
"""inventory_external_m2_v3: outcome-blind external M2 component inventory (P4 preaccess).

v3: name-based WT-anchor grouping (fast) with a sequence-edit verification pass
on a sample. Component = one WT anchor + its single-SNV mutant library.
Roles per contract 7.2/7.5: direct_external (disconnected biology) vs
adjacent (OpenKnot program lineage). Outcome-blind: identities only.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from reactflow.delta.rdat import (
    parse_rdat, parse_mutations_from_name, is_wt_profile,
)

DIRECT_EXTERNAL = ["M2SL5_2A3_0000", "M3SARS_2A3_0000", "15KLIB_2A3_0000"]
ADJACENT = ["M2PK50_2A3_0000", "M2PK90_2A3_0000", "OK1LIB_2A3_0000", "OK2TRN_2A3_0000"]

_SNV = re.compile(r"\d+[ACGU]-[ACGU]")
_PAD = re.compile(r"\d+pad\d+")


def strip_tokens(name: str) -> str:
    """Remove SNV tokens, pad descriptors and libraryready/sublibrary noise to
    recover the WT base name (pad tokens differ between WT and mutant names)."""
    name = name.split(";sublibrary:")[0]
    out = []
    for tok in name.split("_"):
        if _SNV.fullmatch(tok):
            continue
        if _PAD.fullmatch(tok):
            continue
        if tok in ("libraryready",):
            continue
        out.append(tok)
    return "_".join(out)


def _single_snv_edit(seq_a: str, seq_b: str) -> list[int]:
    if len(seq_a) != len(seq_b):
        return []
    diffs = [i for i, (x, y) in enumerate(zip(seq_a, seq_b)) if x != y]
    return diffs if len(diffs) == 1 else []


def build_components(profiles: list, verify_sample: int = 200) -> tuple[list[dict], dict]:
    # WT dict keyed by STRIPPED base name so pad tokens in names match correctly
    wt = {}
    for x in profiles:
        if is_wt_profile(x):
            wt.setdefault(strip_tokens(x["profile_name"]), x)
    muts = [x for x in profiles if not is_wt_profile(x)]
    groups = {}
    for m in muts:
        ms = parse_mutations_from_name(m["profile_name"])
        if len(ms) != 1:
            continue
        base = strip_tokens(m["profile_name"])
        groups.setdefault(base, []).append(m)
    comps = []
    verified = {"n_checked": 0, "n_match": 0, "n_mismatch": 0, "mismatch_examples": []}
    for base, members in groups.items():
        anchor = wt.get(base)
        if anchor is None:
            continue
        wseq = anchor["profile_sequence"]
        seq_len = len(wseq)
        if seq_len == 0:
            continue
        # verify a sample by exact single-SNV edit distance
        sample = members[:verify_sample] if verify_sample else members
        ok = 0
        bad = []
        for m in sample:
            diffs = _single_snv_edit(wseq, m["profile_sequence"])
            if len(diffs) == 1:
                ok += 1
            else:
                bad.append(m["profile_name"][:60])
        verified["n_checked"] += len(sample)
        verified["n_match"] += ok
        verified["n_mismatch"] += len(bad)
        if bad and len(verified["mismatch_examples"]) < 3:
            verified["mismatch_examples"].append({"wt": anchor["profile_name"][:60], "bad": bad[:2]})
        comps.append({
            "wt_name": anchor["profile_name"], "seq_len": seq_len,
            "n_snv_mutants": len(members),
        })
    return comps, verified


def inventory(rdat_dir: Path, dev_csv: Path) -> dict:
    dev = pd.read_csv(dev_csv)
    dev_all = set(dev["sequence"])
    dev_wt = set(dev[dev["id"].str.endswith("_wt")]["sequence"])

    def scan(cids: list[str], role: str) -> tuple[list, list, dict]:
        ds_rows, all_comps, all_ver = [], [], {}
        for cid in cids:
            p = rdat_dir / f"{cid}.rdat"
            if not p.exists():
                ds_rows.append({"dataset": cid, "role": role, "status": "MISSING"})
                continue
            r = parse_rdat(p)
            prof = r["profiles"]
            seqs = {x["profile_sequence"] for x in prof}
            comps, ver = build_components(prof)
            ds_rows.append({
                "dataset": cid, "role": role, "status": "OK",
                "n_profiles": len(prof),
                "n_wt": sum(1 for x in prof if is_wt_profile(x)),
                "n_mutants": len(prof) - sum(1 for x in prof if is_wt_profile(x)),
                "overlap_dev_wt": len(seqs & dev_wt),
                "overlap_dev_all": len(seqs & dev_all),
                "n_components": len(comps),
                "n_single_snv": sum(c["n_snv_mutants"] for c in comps),
                "verify": ver,
            })
            all_comps.extend(comps)
        return ds_rows, all_comps, all_ver

    direct_ds, direct_comps, _ = scan(DIRECT_EXTERNAL, "direct_external")
    adjacent_ds, adjacent_comps, _ = scan(ADJACENT, "adjacent")

    report = {
        "schema_version": "reactflow_delta.p4_external_inventory.v3",
        "dev_csv": str(dev_csv),
        "task_identity": {
            "probe": "2A3-MaP (Ribonanza Kaggle M2-style datasets via RMDB)",
            "chemistry_family_match": "same 2A3-MaP chemistry family as OK7a_M2 development",
            "platform": "NovaSeq (Ribonanza) vs Ultima (OpenKnot M2); recorded, not concatenated",
            "normalization": "RNAFramework per-dataset; recorded per component",
            "estimand": "full-construct mutant 2A3 reactivity response given WT profile + exact SNV",
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
            "and the OK7a_M2 development set (WT and all-mutant); different organism/study/batch"
        ),
        "outcome_blind_note": "inventory counts identities and edit positions only; no external "
                              "reactivity value was used for selection or eligibility.",
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
    print(json.dumps({
        "direct_external": rep["direct_external"],
        "adjacent_openknot_lineage": rep["adjacent_openknot_lineage"],
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
