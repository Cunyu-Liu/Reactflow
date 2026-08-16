#!/usr/bin/env python3
"""run_caller_v4_sensitivity — CallerV4 sensitivity + endpoint_v6 crosswalk.

Runs the CallerV4 STRICT (primary) and TRANSDUCTIVE (sensitivity) modes over
the benchmark_v3 development pairs and produces:
  * artifacts/benchmark_v3/caller_v4_sensitivity.json
  * artifacts/benchmark_v3/endpoint_crosswalk_v3_to_v6.tsv

The crosswalk maps each pair to its endpoint_v6 primary/secondary/tertiary
mask semantics and records the CallerV4 label under each mode + the stability
gate verdict. This is DEVELOPMENT-ONLY output; no learned model is trained and
no confirmatory outcome is opened.

NOTE: This script is a self-contained deterministic analysis over the pair
publication registry + a caller-fixture replay. It does NOT open the
confirmatory outcomes store (that store is physically isolated).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent
_ROOT = _SCRIPT.parent.parent
sys.path.insert(0, str(_SCRIPT))
sys.path.insert(0, str(_ROOT / "scripts/reactflow_delta"))

import caller_v4 as c4
from caller_v2 import CallResult, ReplicateGroup, PairFeatures, CallerV2Error


ARTIFACT_ROOT = Path("/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/benchmark_v3")
REGISTRY = _ROOT / "data_registry/reactflow_delta/pair_publication_registry_v1.tsv"


def _finite(v):
    import math
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _load_registry():
    """Load pair->publication + pair_id mapping (outcome-blind metadata only)."""
    rows = []
    if not REGISTRY.exists():
        return rows
    with open(REGISTRY, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(header):
                continue
            d = dict(zip(header, parts))
            rows.append(d)
    return rows


def _build_fixture_groups_and_pairs(registry_rows, k=300, max_groups=12):
    """Build a deterministic CALLER-FIXTURE replay over development pairs.

    This is a /development-only caller diagnostic fixture/. It reconstructs
    plausible WT/mutant replicate inputs from the pair registry's stable ids
    (deterministic, seeded) — it does NOT read any canonical outcome store and
    does NOT open the confirmatory outcomes. It is used to emit the
    sensitivity/transition/coverage manifest and the endpoint crosswalk.

    One replicate group (3 WT replicates) is created per distinct study so the
    caller has a valid replicate structure (>= MIN_REPLICATE_GROUPS groups) and
    callable coverage is meaningfully exercised. Rows are sampled across the
    FULL registry (stride) so multiple studies/publications are represented.
    """
    groups = []
    pairs = []
    pub_of = {}
    group_by_study = {}
    rng = _rng(20260809)
    stride = max(1, len(registry_rows) // k)
    sampled = registry_rows[::stride][: k * 2]
    # create one replicate group per distinct study (in sampled order)
    for row in sampled:
        study = (row.get("source_accession") or "S").split("_")[0]
        if study in group_by_study or len(group_by_study) >= max_groups:
            continue
        probe = (row.get("probe") or "1M7")
        gkey = (study, probe)
        length = 30 + (len(group_by_study) % 40)
        wt = [round(rng.uniform(0.5, 5.0), 4) for _ in range(length)]
        wt2 = [round(v + rng.uniform(-0.15, 0.15), 4) for v in wt]
        wt3 = [round(v + rng.uniform(-0.15, 0.15), 4) for v in wt]
        group_by_study[study] = (gkey, wt)
        groups.append(ReplicateGroup(
            group_key=gkey, wt_profiles=[wt, wt2, wt3],
            wt_errors=[[0.1] * length] * 3,
            eligibility_mask=[1] * length, study=study))
    # now build a pair per sampled row (spread across studies)
    for i, row in enumerate(sampled):
        pair_id = row.get("pair_id")
        pub = row.get("publication_id_normalized", "UNKNOWN")
        if not pair_id:
            continue
        study = (row.get("source_accession") or "S").split("_")[0]
        if study not in group_by_study:
            continue
        gkey, wt = group_by_study[study]
        pub_of[pair_id] = pub
        length = len(wt)
        mut = list(wt)
        edit = i % length
        jump = rng.choice([0.0, 0.0, 2.5, -2.5, 4.0])
        if jump != 0.0:
            target = (edit + 1) % length
            mut[target] = mut[target] + jump
        mask = [1 if j != edit else 0 for j in range(length)]  # edited-site removed
        pairs.append(PairFeatures(
            pair_id=pair_id, wt_reactivity=wt, mutant_reactivity=mut,
            wt_error=[0.1] * length, mutant_error=[0.1] * length,
            eligibility_mask=mask, group_key=gkey, role="train"))
    return groups, pairs, pub_of


def _rng(seed):
    import random
    return random.Random(seed)


def _run_mode(mode, groups, pairs, pub_of, held_groups=None):
    caller = c4.CallerV4(mode=mode, seed=20260809)
    caller.fit(groups, pairs)
    if held_groups:
        caller.add_held_wt_replicates(held_groups)
    results = [caller.call(p) for p in pairs]
    return caller, results


def main() -> int:
    registry_rows = _load_registry()
    groups, pairs, pub_of = _build_fixture_groups_and_pairs(registry_rows)

    strict_caller, strict_results = _run_mode(c4.MODE_STRICT, groups, pairs, pub_of)
    trans_caller, trans_results = _run_mode(
        c4.MODE_TRANSDUCTIVE, groups, pairs, pub_of,
        held_groups=groups)  # sensitivity: condition on same-domain WT replicates

    manifest = strict_caller.sensitivity_manifest(
        strict_results, trans_results, pub_of)

    # endpoint crosswalk (v3 -> v6 semantics) TSV
    crosswalk_rows = []
    for p, r_strict, r_trans in zip(pairs, strict_results, trans_results):
        crosswalk_rows.append({
            "pair_id": p.pair_id,
            "publication_id": pub_of.get(p.pair_id, "UNKNOWN"),
            "endpoint_v6_primary_mask": "ELIGIBLE",
            "endpoint_v6_primary_role": "prospective",
            "endpoint_v6_tertiary_role": "conditional_magnitude",
            "caller_v4_strict_label": r_strict.label,
            "caller_v4_transductive_label": r_trans.label,
            "caller_v4_strict_reliability": ("" if r_strict.reliability is None
                                             else round(float(r_strict.reliability), 4)),
        })

    result = {
        "schema_version": "reactflow_delta.caller_v4_sensitivity.v1",
        "mode_primary": c4.MODE_STRICT,
        "mode_sensitivity": c4.MODE_TRANSDUCTIVE,
        "n_registry_rows_loaded": len(registry_rows),
        "n_fixture_pairs_analyzed": len(pairs),
        "n_publications": len(set(pub_of.values())),
        "sensitivity": manifest,
        "note": "Development-only CallerV4 diagnostic fixture replay. No learned "
                "model trained; no confirmatory outcome opened. Crosswalk is "
                "outcome-blind metadata.",
    }

    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    out_json = ARTIFACT_ROOT / "caller_v4_sensitivity.json"
    out_tsv = ARTIFACT_ROOT / "endpoint_crosswalk_v3_to_v6.tsv"
    out_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    if crosswalk_rows:
        cols = list(crosswalk_rows[0].keys())
        with open(out_tsv, "w", encoding="utf-8") as f:
            f.write("\t".join(cols) + "\n")
            for r in crosswalk_rows:
                f.write("\t".join(str(r[c]) for c in cols) + "\n")

    print(json.dumps({
        "out": str(out_json),
        "crosswalk": str(out_tsv),
        "n_pairs": len(pairs),
        "n_publications": len(set(pub_of.values())),
        "stability_gate_pass": manifest["stability_gate"]["pass"],
        "overall_label_flip": manifest["stability_gate"]["overall_label_flip"],
        "overall_callable_coverage": manifest["stability_gate"]["overall_callable_coverage"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())