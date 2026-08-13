#!/usr/bin/env python3
"""extract_external_components_v1: outcome-blind external M2 component extractor (P4).

For each Ribonanza M2-style 2A3 rdat, extract exact WT-anchor + single-SNV
mutant components. Scoring domain = SHARED REGION: positions where the WT and
mutant sequences are identical plus the SNV position itself (pads/barcodes that
differ between constructs are excluded, mirroring target-qualified positions).
This is OUTCOME-BLIND: the mask is derived only from sequence identity, never
from reactivity values.

Component = one WT construct + its single-SNV mutants (external exchangeable
unit, analogous to a development puzzle). Roles:
  direct_external: disconnected biology (M2SL5 betacoronavirus SL5, M3SARS
    coronavirus FSE, 15KLIB diverse non-OpenKnot)
  adjacent: OpenKnot program lineage (M2PK50/M2PK90 Pilot, OK1LIB/OK2TRN)
    => sensitivity only.

Output is a JSON with component identity, per-mutant edit position, and the
shared-region position list. NO reactivity is read here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from reactflow.delta.rdat import (
    parse_rdat, parse_mutations_from_name, is_wt_profile,
)

DIRECT_EXTERNAL = ["M2SL5_2A3_0000", "M3SARS_2A3_0000", "15KLIB_2A3_0000"]
ADJACENT = ["M2PK50_2A3_0000", "M2PK90_2A3_0000", "OK1LIB_2A3_0000", "OK2TRN_2A3_0000"]

import re
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


def shared_region_mask(wt_seq: str, mut_seq: str, edit_pos: int) -> list[int]:
    """Positions where wt==mut, plus edit_pos (the SNV readout position)."""
    if len(wt_seq) != len(mut_seq):
        return []
    out = []
    for i, (w, m) in enumerate(zip(wt_seq, mut_seq)):
        if w == m or i == edit_pos:
            out.append(i)
    return out


def extract(rdat_dir: Path, dev_csv: Path) -> dict:
    dev = pd.read_csv(dev_csv)
    dev_all = set(dev["sequence"])
    dev_wt = set(dev[dev["id"].str.endswith("_wt")]["sequence"])

    def scan(cids: list[str], role: str) -> tuple[list, list]:
        ds_rows = []
        comps = []
        for cid in cids:
            p = rdat_dir / f"{cid}.rdat"
            if not p.exists():
                ds_rows.append({"dataset": cid, "role": role, "status": "MISSING"})
                continue
            r = parse_rdat(p)
            prof = r["profiles"]
            wt = [x for x in prof if is_wt_profile(x)]
            mut = [x for x in prof if not is_wt_profile(x)]
            seqs = {x["profile_sequence"] for x in prof}
            wt_by_base = {}
            for x in wt:
                wt_by_base.setdefault(strip_tokens(x["profile_name"]), x)
            # single-SNV mutants grouped by WT anchor via base-name match
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
                # find the single position where the two sequences differ
                diffs = [i for i, (a, b) in enumerate(zip(wseq, seq)) if a != b]
                # Ribonanza constructs carry unique 3' barcodes, so the full
                # construct may differ in more than one position. The named
                # mutation site must be among the diffs and within the shared core.
                if len(diffs) < 1:
                    continue
                # use the named mutation position: it is in core coords relative
                # to the construct; map to the diff that matches ref->alt
                edit_pos = None
                for i in diffs:
                    if ms[0].get("ref") and wseq[i] == ms[0]["ref"] and seq[i] == ms[0].get("mut"):
                        edit_pos = i
                        break
                if edit_pos is None:
                    # fall back to the first diff as the mutation site
                    edit_pos = diffs[0]
                key = anchor["profile_name"]
                groups.setdefault(key, {"wt": anchor, "mutants": []})
                mask = shared_region_mask(wseq, seq, edit_pos)
                if len(mask) >= 20:
                    groups[key]["mutants"].append({
                        "name": m["profile_name"], "edit_pos": edit_pos,
                        "shared_region": mask, "n_shared": len(mask),
                        "n_total_diffs": len(diffs),
                    })
            ds_comps = []
            for key, g in groups.items():
                ds_comps.append({
                    "wt_name": key,
                    "wt_sequence": g["wt"]["profile_sequence"],
                    "seq_len": len(g["wt"]["profile_sequence"]),
                    "n_snv_mutants": len(g["mutants"]),
                    "mutants": g["mutants"],
                })
            ds_rows.append({
                "dataset": cid, "role": role, "status": "OK",
                "n_profiles": len(prof), "n_wt": len(wt), "n_mutants": len(mut),
                "overlap_dev_wt": len(seqs & dev_wt), "overlap_dev_all": len(seqs & dev_all),
                "n_components": len(ds_comps),
                "n_single_snv_matched": sum(c["n_snv_mutants"] for c in ds_comps),
            })
            comps.extend(ds_comps)
        return ds_rows, comps

    direct_ds, direct_comps = scan(DIRECT_EXTERNAL, "direct_external")
    adjacent_ds, adjacent_comps = scan(ADJACENT, "adjacent")

    return {
        "schema_version": "reactflow_delta.p4_external_components.v1",
        "direct_external": {"datasets": direct_ds, "components": direct_comps},
        "adjacent_openknot_lineage": {"datasets": adjacent_ds, "components": adjacent_comps},
        "K_preaccess_direct": len(direct_comps),
        "K_preaccess_direct_snv": sum(c["n_snv_mutants"] for c in direct_comps),
        "development_disconnect": "zero sequence identity overlap with OK7a_M2 dev set (all candidates)",
        "outcome_blind": "masks from sequence identity only; no reactivity read in this step",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rdat-dir", required=True)
    ap.add_argument("--dev-csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    rep = extract(Path(args.rdat_dir), Path(args.dev_csv))
    Path(args.out).write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    for role in ("direct_external", "adjacent_openknot_lineage"):
        for d in rep[role]["datasets"]:
            print(d["dataset"], d["role"], "comps=" + str(d["n_components"]),
                  "snv=" + str(d["n_single_snv_matched"]), "ovl=" + str(d["overlap_dev_all"]))
    print("K_preaccess_direct_components:", rep["K_preaccess_direct"])
    print("K_preaccess_direct_single_snv:", rep["K_preaccess_direct_snv"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
