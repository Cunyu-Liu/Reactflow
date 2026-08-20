#!/usr/bin/env python3
"""recompute_external_k_joint_v1: outcome-blind K_joint for the consumed external
components (audit P0-4 / P0-6).

Loads the P4 (24 anchors, 3 datasets) and P5b (694 anchors, 4 datasets) component
manifests, attaches provenance from external_provenance_registry_v1 (dataset ->
batch/library/study/publication), and computes K_joint via the joint_dependency
union-find graph. Reports N_rows, N_SNV, N_WT_anchor, N_dataset, N_batch,
N_study, N_publication, K_joint.

This is READ-ONLY aggregate remapping; it does NOT read any external outcome.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.reactflow_delta.joint_dependency_component_v1 import (
    ComponentCandidate, compute_k_joint, PROVENANCE_BY_DATASET,
)


# P4 manifest: components nested under direct_external, no per-component dataset
# field; the extractor iterates datasets in order and the frozen protocol records
# the per-dataset counts (M2SL5=3, M3SARS=3, 15KLIB=18).
P4_DATASET_ORDER = ["M2SL5_2A3_0000", "M3SARS_2A3_0000", "15KLIB_2A3_0000"]
P4_DATASET_COUNTS = {"M2SL5_2A3_0000": 3, "M3SARS_2A3_0000": 3, "15KLIB_2A3_0000": 18}


def _load_comps(manifest: Path, role: str) -> list[ComponentCandidate]:
    doc = json.loads(manifest.read_text())
    if role == "p4":
        comps_raw = doc.get("direct_external", {}).get("components", [])
        # assign dataset by documented contiguous split (3/3/18)
        ds_assign = []
        for ds in P4_DATASET_ORDER:
            ds_assign.extend([ds] * P4_DATASET_COUNTS[ds])
        if len(ds_assign) != len(comps_raw):
            raise ValueError(f"P4 component count {len(comps_raw)} != documented "
                             f"{len(ds_assign)}")
        comps_raw = [dict(c, dataset=ds_assign[i]) for i, c in enumerate(comps_raw)]
    else:
        comps_raw = doc.get("components", [])
    out = []
    for i, c in enumerate(comps_raw):
        ds = c.get("dataset", "")
        prov = PROVENANCE_BY_DATASET.get(ds)
        if prov is None:
            # dataset not in registry => provenance unresolved (fail-closed)
            out.append(ComponentCandidate(
                component_id=f"{role}:{ds}:{c.get('wt_name', str(i))}",
                dataset=ds, n_snv_mutants=c.get("n_snv_mutants", 0),
                n_rows=len(c.get("mutants", [])), development_disconnected=True,
                provenance_resolved=False))
            continue
        out.append(ComponentCandidate(
            component_id=f"{role}:{ds}:{c.get('wt_name', str(i))}",
            publication=prov.publication, study=prov.study, batch=prov.sequencing_batch,
            library=prov.library, dataset=ds,
            n_snv_mutants=c.get("n_snv_mutants", 0),
            n_rows=len(c.get("mutants", [])),
            development_disconnected=True, provenance_resolved=True))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p4-manifest", required=True)
    ap.add_argument("--p5b-manifest", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    comps = _load_comps(Path(args.p4_manifest), "p4")
    comps += _load_comps(Path(args.p5b_manifest), "p5b")

    res = compute_k_joint(comps, development_component_ids=set())
    res["schema_version"] = "reactflow_delta.external_k_joint.v1"
    res["source"] = {"p4": str(args.p4_manifest), "p5b": str(args.p5b_manifest)}
    res["note"] = ("All 718 WT anchors trace to the Das-lab (Stanford) M2-seq "
                   "2A3-MaP system: 2 studies / 2 publications (SL5 PNAS 2024; "
                   "Ribonanza 2024), 3 distinct NovaSeq batches. K_joint is "
                   "computed at the highest merged dependency level.")
    Path(args.out).write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps({k: res[k] for k in
                      ("K_preaccess", "K_joint", "N_rows", "N_SNV", "N_WT_anchor",
                       "N_dataset", "N_batch", "N_study", "N_publication")},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
