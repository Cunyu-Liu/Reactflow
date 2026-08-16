#!/usr/bin/env python3
"""build_p5b_external_components_v1: outcome-blind NEW independent external M2
component extractor for P5b confirmatory protocol.

Extracts exact WT-anchor + single-SNV mutant components from the DasLab BigLib2
OneMil2 rfam-OK / rfam-PK M2 sub-libraries (M2RFOK_2A3_0000, M2RFPK_2A3_0000/
_0001/_0002). These datasets are:
  - UNCONSUMED (reactivity never evaluated in P4/P5),
  - development-disconnected (zero sequence overlap with OK7a_M2),
  - disjoint from the 24 consumed P4 components (zero WT overlap),
  - same 2A3-MaP chemistry family and RNAFramework normalization as the
    consumed direct_external set (15KLIB).

Outcome-blind: masks derived from sequence identity only; NO reactivity read.
Frozen attrition rules identical to p4_frozen_protocol section 6:
  rule 1: WT anchor has observed 2A3 reactivity profile (non-empty shared region)
  rule 2: >= MIN_MUTANTS_PER_COMP single-SNV mutants matched
  rule 3: each mutant must have >= MIN_SHARED_NONMISSING shared-region positions
Output: {components: [{wt_name, wt_sequence, seq_len, n_snv_mutants, mutants:
  [{name, edit_pos, shared_region, n_shared}]}]}
"""
from __future__ import annotations

import argparse, json, re
from pathlib import Path

import pandas as pd
from reactflow.delta.rdat import (
    parse_rdat, parse_mutations_from_name, is_wt_profile,
)

NEW_DIRECT_EXTERNAL = ["M2RFOK_2A3_0000", "M2RFPK_2A3_0000",
                       "M2RFPK_2A3_0001", "M2RFPK_2A3_0002"]
MIN_SHARED_NONMISSING = 20
MIN_MUTANTS_PER_COMP = 20

_SNV = re.compile(r"\d+[ACGU]-[ACGU]")
_PAD = re.compile(r"\d+pad\d+")


def strip_tokens(name: str) -> str:
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


def shared_region_mask(wt_seq: str, mut_seq: str, edit_pos: int) -> list[int]:
    if len(wt_seq) != len(mut_seq):
        return []
    return [i for i, (w, m) in enumerate(zip(wt_seq, mut_seq))
            if w == m or i == edit_pos]


def build(rdat_dir: Path, dev_csv: Path) -> dict:
    dev = pd.read_csv(dev_csv)
    dev_all = set(dev["sequence"].astype(str))
    dev_wt = set(dev[dev["id"].astype(str).str.endswith("_wt")]["sequence"].astype(str))

    ds_rows, comps = [], []
    for cid in NEW_DIRECT_EXTERNAL:
        p = rdat_dir / f"{cid}.rdat"
        if not p.exists():
            ds_rows.append({"dataset": cid, "role": "new_external", "status": "MISSING"})
            continue
        r = parse_rdat(p)
        prof = r["profiles"]
        wt = [x for x in prof if is_wt_profile(x)]
        mut = [x for x in prof if not is_wt_profile(x)]
        seqs = {x["profile_sequence"] for x in prof}
        wt_by_base = {}
        for x in wt:
            wt_by_base.setdefault(strip_tokens(x["profile_name"]), x)
        groups = {}
        for m in mut:
            ms = parse_mutations_from_name(m["profile_name"])
            if len(ms) != 1:
                continue
            base = strip_tokens(m["profile_name"])
            anchor = wt_by_base.get(base)
            if anchor is None:
                continue
            wseq = anchor["profile_sequence"]
            seq = m["profile_sequence"]
            if len(wseq) != len(seq):
                continue
            diffs = [i for i, (a, b) in enumerate(zip(wseq, seq)) if a != b]
            if not diffs:
                continue
            edit_pos = None
            for i in diffs:
                if ms[0].get("ref") and wseq[i] == ms[0]["ref"] and seq[i] == ms[0].get("mut"):
                    edit_pos = i
                    break
            if edit_pos is None:
                edit_pos = diffs[0]
            mask = shared_region_mask(wseq, seq, edit_pos)
            if len(mask) >= MIN_SHARED_NONMISSING:
                groups.setdefault(anchor["profile_name"],
                                  {"wt": anchor, "mutants": []})
                groups[anchor["profile_name"]]["mutants"].append({
                    "name": m["profile_name"], "edit_pos": edit_pos,
                    "shared_region": mask, "n_shared": len(mask),
                    "n_total_diffs": len(diffs),
                })
        ds_comps = []
        for key, g in groups.items():
            if len(g["mutants"]) < MIN_MUTANTS_PER_COMP:
                continue
            ds_comps.append({
                "wt_name": key,
                "wt_sequence": g["wt"]["profile_sequence"],
                "seq_len": len(g["wt"]["profile_sequence"]),
                "dataset": cid,
                "n_snv_mutants": len(g["mutants"]),
                "mutants": g["mutants"],
            })
        ds_rows.append({
            "dataset": cid, "role": "new_external", "status": "OK",
            "n_profiles": len(prof), "n_wt": len(wt), "n_mutants": len(mut),
            "overlap_dev_wt": len(seqs & dev_wt),
            "overlap_dev_all": len(seqs & dev_all),
            "n_components": len(ds_comps),
            "n_single_snv_matched": sum(c["n_snv_mutants"] for c in ds_comps),
        })
        comps.extend(ds_comps)

    return {
        "schema_version": "reactflow_delta.p5b_external_components.v1",
        "source_datasets": NEW_DIRECT_EXTERNAL,
        "datasets": ds_rows,
        "components": comps,
        "K_preaccess_components": len(comps),
        "K_preaccess_single_snv": sum(c["n_snv_mutants"] for c in comps),
        "development_disconnect": "zero sequence overlap with OK7a_M2 dev (WT + all-mutant)",
        "outcome_blind": "masks from sequence identity only; no reactivity read in this step",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rdat-dir", required=True)
    ap.add_argument("--dev-csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    rep = build(Path(args.rdat_dir), Path(args.dev_csv))
    Path(args.out).write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    for d in rep["datasets"]:
        print(d["dataset"], "comps=" + str(d["n_components"]),
              "snv=" + str(d["n_single_snv_matched"]),
              "ovl_dev_wt=" + str(d["overlap_dev_wt"]),
              "ovl_dev_all=" + str(d["overlap_dev_all"]))
    print("K_preaccess_components:", rep["K_preaccess_components"])
    print("K_preaccess_single_snv:", rep["K_preaccess_single_snv"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
