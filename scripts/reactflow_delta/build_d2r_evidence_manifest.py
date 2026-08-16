#!/usr/bin/env python3
"""D2-R evidence audit (v3.2 §2-§6): build d2r_evidence_manifest.json.

Audits each of the 7,761 D0-R v2 annotation-only candidate relations for the
two §2.1 evidence paths:

  1. per_profile_sequence  — RDAT carries a per-profile SEQUENCE:N line or a
     ``sequence:`` annotation token resolving alt=X to a concrete base.
  2. same_parent_replicate — a T-D1.6 replicate group (same parent + same
     annotated edit + same FULL experimental condition) observed in >=2
     distinct RDAT files (independent measurements, not reuploads).

§3.1 anti-fabrication rules enforced:
  - Same-RDAT-file profiles are NOT auto-promoted to replicates (bullet 1);
    only cross-file independence counts here (the data has 0 same-file
    multi-profile groups, but the rule is still applied explicitly).
  - Same-study different-construct is NOT same-parent replicate (bullet 2);
    enforced by requiring the RDAT NAME to match across the group.
  - D2 file-internal parent/reference consistency is NOT §3.2 corroboration
    (bullet 3); enforced by requiring >=2 distinct rdat_sha256.

§3.5 one-to-one: a single piece of evidence may not cover candidates with
DIFFERENT alts; replicate corroboration only covers candidates sharing the
exact same (parent, pos, ref, alt, full-condition) key.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

_REPO = Path("/home/cunyuliu/reactflow_delta_goal_20260729")
sys.path.insert(0, str(_REPO / "src"))

from reactflow.delta.rdat import parse_rdat  # noqa: E402

RELATIONS_PATH = _REPO / "artifacts/reactflow_delta/d0r/d0r_reaudit_tierA_relations.json"
D2_LINEAGE_PATH = _REPO / "artifacts/reactflow_delta/d2/d2_lineage_verification.json"
OUT_DIR = _REPO / "artifacts/reactflow_delta/d2r"
OUT_PATH = OUT_DIR / "d2r_evidence_manifest.json"

SCHEMA_VERSION = "reactflow-delta-d2r-evidence-manifest-v1"


def _merged_global_annotations(doc: dict) -> dict[str, list[str]]:
    """Merge the list of global annotation dicts into one mapping."""
    merged: dict[str, list[str]] = {}
    for g in doc["global_annotations"]:
        for k, v in g.items():
            merged.setdefault(k, list(v))
    return merged


def _condition_key(merged_ann: dict[str, list[str]]) -> tuple:
    """Full experimental condition from RDAT global annotations (T-D1.6
    CONDITION_MATCH_FIELDS proxy: probe/modifier, ligand+buffer=chemical,
    temperature, experimentType)."""
    return (
        tuple(merged_ann.get("modifier", [])),
        tuple(sorted(merged_ann.get("chemical", []))),
        tuple(merged_ann.get("temperature", [])),
        tuple(merged_ann.get("experimentType", [])),
    )


def _reactivity_tuple(doc: dict, profile_index: int) -> tuple:
    return tuple(doc["profiles"][profile_index]["reactivity"])


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rel_doc = json.load(open(RELATIONS_PATH))
    rels = rel_doc["relations"]
    print(f"[d2r] loaded {len(rels)} candidate relations")

    # ---- Parse each distinct RDAT file referenced by the relations ----
    sha_to_path: dict[str, str] = {}
    for r in rels:
        sha_to_path[r["rdat_sha256"]] = r["rdat_path"]
    print(f"[d2r] {len(sha_to_path)} distinct RDAT files to parse")

    docs: dict[str, dict] = {}
    for sha, path in sha_to_path.items():
        doc = parse_rdat(path)
        merged_ann = _merged_global_annotations(doc)
        docs[sha] = {
            "name": Path(path).name,
            "path": path,
            "sha256": doc["sha256"],
            "rdat_name": doc["headers"].get("NAME"),
            "comments": doc["comments"],
            "merged_ann": merged_ann,
            "condition_key": _condition_key(merged_ann),
            "profiles_by_index": {p["index"]: p for p in doc["profiles"]},
        }

    # ---- D2 lineage verification lookup (parent_lineage_verified) ----
    lv_doc = json.load(open(D2_LINEAGE_PATH))
    lineage_lookup = {
        (v["rdat_sha256"], int(v["wt_profile_index"]), int(v["mutant_profile_index"])):
        bool(v["parent_lineage_verified"])
        for v in lv_doc.get("verifications", [])
    }

    # ---- Enrich each relation with RDAT-level condition + reactivity ----
    enriched: list[dict] = []
    for r in rels:
        sha = r["rdat_sha256"]
        d = docs[sha]
        mm = r.get("matched_mutation") or {}
        wt_idx = r["wt_profile_index"]
        mut_idx = r["mutant_profile_index"]
        wt_p = d["profiles_by_index"].get(wt_idx, {})
        mut_p = d["profiles_by_index"].get(mut_idx, {})
        e = dict(r)
        e["_rdat_name"] = d["rdat_name"]
        e["_condition_key"] = d["condition_key"]
        e["_merged_ann"] = d["merged_ann"]
        e["_wt_reactivity"] = tuple(wt_p.get("reactivity", []))
        e["_mut_reactivity"] = tuple(mut_p.get("reactivity", []))
        e["_mut_profile_sequence"] = mut_p.get("profile_sequence")
        e["_mut_profile_sequence_source"] = mut_p.get("profile_sequence_source")
        e["_wt_profile_sequence"] = wt_p.get("profile_sequence")
        e["_lineage_verified"] = lineage_lookup.get(
            (sha, wt_idx, mut_idx), False
        )
        enriched.append(e)

    # ---- T-D1.6 rigorous replicate grouping ----
    # Key = (parent_id, edit_positions, wt_alleles, mut_alleles, FULL_CONDITION).
    # study_id is deliberately excluded (data.py L1013-1015).
    def replicate_key(e: dict) -> tuple:
        mm = e.get("matched_mutation") or {}
        return (
            e.get("parent_prefix"),
            mm.get("encoded_position_1indexed"),
            mm.get("encoded_ref"),
            mm.get("encoded_alt"),
            e["_condition_key"],
        )

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for e in enriched:
        groups[replicate_key(e)].append(e)

    print(f"[d2r] {len(groups)} distinct T-D1.6 replicate keys (full condition)")

    # ---- Classify each group ----
    group_classification: dict[tuple, dict] = {}
    for gk, members in groups.items():
        sha_count = len({m["rdat_sha256"] for m in members})
        name_set = {m["_rdat_name"] for m in members}
        # cross-file corroboration requires >=2 distinct RDAT files
        cross_file = sha_count >= 2
        # §3.1 bullet 2: same-study different construct is NOT same-parent
        # replicate. RDAT NAME must match across the group.
        same_name = len(name_set) == 1
        # reactivity non-identity check (not reupload): for each member, is
        # there >=1 OTHER member in a DIFFERENT file whose mutant reactivity
        # differs? Build per-sha reactivity sets.
        reacts_by_sha: dict[str, set] = defaultdict(set)
        for m in members:
            reacts_by_sha[m["rdat_sha256"]].add(m["_mut_reactivity"])
        all_reacts: set = set()
        for s in reacts_by_sha.values():
            all_reacts |= s
        reactivity_differs = len(all_reacts) > 1
        # Per-member: which distinct other shas have non-identical reactivity?
        # A member is corroborated if >=1 other sha exists with a different
        # mutant reactivity (independent measurement).
        group_classification[gk] = {
            "member_count": len(members),
            "distinct_sha_count": sha_count,
            "cross_file": cross_file,
            "same_name": same_name,
            "reactivity_differs": reactivity_differs,
            "reacts_by_sha": reacts_by_sha,
            "all_reacts": all_reacts,
            "names": name_set,
        }

    # ---- Build per-candidate manifest ----
    candidates: list[dict] = []
    stats = {
        "total_candidates": len(enriched),
        "evidence_found": 0,
        "no_evidence_in_scope": 0,
        "by_evidence_type": {"per_profile_sequence": 0, "same_parent_replicate": 0, "none": 0},
        "by_miss_reason": defaultdict(int),
        "groups_total": len(groups),
        "groups_cross_file": sum(1 for g in group_classification.values() if g["cross_file"]),
        "groups_qualifying_replicate": 0,
    }

    for e in enriched:
        gk = replicate_key(e)
        gc = group_classification[gk]
        sha = e["rdat_sha256"]
        wt_idx = e["wt_profile_index"]
        mut_idx = e["mutant_profile_index"]
        mm = e.get("matched_mutation") or {}

        # --- Path 1: per-profile sequence ---
        per_profile_seq = e["_mut_profile_sequence"]
        per_profile_seq_source = e["_mut_profile_sequence_source"]
        per_profile_seq_resolves_alt = False
        if per_profile_seq is not None:
            # alt=X resolution requires the mutant sequence AND a concrete alt
            # base derivable from sequence vs WT anchor. annotation-only alt=X
            # means the encoded alt is "X"; per-profile sequence could resolve
            # it only if a WT anchor sequence is also present and the edit set
            # yields a single functional edit with a concrete mutant base.
            # The parser already attempts this; if profile_sequence is present
            # it would have been used in D0-R. Here we only record availability.
            # For alt resolution, the mutant base at the edit position must be
            # a concrete base (not X). We check the WT sequence too.
            wt_seq = e["_wt_profile_sequence"]
            pos_1idx = mm.get("encoded_position_1indexed")
            if wt_seq and per_profile_seq and pos_1idx is not None:
                if len(per_profile_seq) >= pos_1idx and len(wt_seq) >= pos_1idx:
                    mut_base = per_profile_seq[pos_1idx - 1]
                    wt_base = wt_seq[pos_1idx - 1]
                    if mut_base in "ACGU" and mut_base != wt_base:
                        per_profile_seq_resolves_alt = True

        # --- Path 2: same-parent replicate corroboration ---
        replicate_corroborated = False
        corroborating_shas: list[str] = []
        if gc["cross_file"] and gc["same_name"] and gc["reactivity_differs"]:
            # this member is corroborated if >=1 OTHER distinct sha has a
            # non-identical mutant reactivity
            my_react = e["_mut_reactivity"]
            for other_sha, other_reacts in gc["reacts_by_sha"].items():
                if other_sha == sha:
                    continue
                for or_ in other_reacts:
                    if or_ != my_react:
                        replicate_corroborated = True
                        corroborating_shas.append(other_sha)
                        break

        # --- Determine evidence type & status ---
        if per_profile_seq_resolves_alt:
            evidence_type = "per_profile_sequence"
            status = "evidence_found"
            stats["by_evidence_type"]["per_profile_sequence"] += 1
        elif replicate_corroborated:
            evidence_type = "same_parent_replicate"
            status = "evidence_found"
            stats["by_evidence_type"]["same_parent_replicate"] += 1
        else:
            evidence_type = "none"
            status = "no_evidence_in_scope"
            stats["by_evidence_type"]["none"] += 1
            # miss reason
            if not per_profile_seq:
                stats["by_miss_reason"]["no_per_profile_sequence_in_rdat"] += 1
            elif not per_profile_seq_resolves_alt:
                stats["by_miss_reason"]["per_profile_sequence_does_not_resolve_alt"] += 1
            if not gc["cross_file"]:
                stats["by_miss_reason"]["no_cross_file_replicate_single_file_only"] += 1
            elif not gc["same_name"]:
                stats["by_miss_reason"]["replicate_files_differ_in_construct_name"] += 1
            elif not gc["reactivity_differs"]:
                stats["by_miss_reason"]["replicate_reactivity_identical_reupload"] += 1
            elif not replicate_corroborated:
                stats["by_miss_reason"]["no_non_identical_corroborating_file"] += 1

        if status == "evidence_found":
            stats["evidence_found"] += 1
        else:
            stats["no_evidence_in_scope"] += 1

        # --- parent / condition / probe alignment (§3 cond 3) ---
        # For replicate path: parent_prefix, condition (full), probe (modifier)
        # must align across the candidate and its corroborating files.
        alignment_ok = True
        alignment_detail = {
            "parent_prefix": e.get("parent_prefix"),
            "rdat_name": e["_rdat_name"],
            "condition_key": list(e["_condition_key"]),
            "corroborating_files_same_parent": None,
            "corroborating_files_same_condition": None,
        }
        if replicate_corroborated:
            corrob_parents = {docs[s]["rdat_name"] for s in corroborating_shas}
            corrob_conds = {docs[s]["condition_key"] for s in corroborating_shas}
            alignment_detail["corroborating_files_same_parent"] = (
                corrob_parents == {e["_rdat_name"]}
            )
            alignment_detail["corroborating_files_same_condition"] = (
                corrob_conds == {e["_condition_key"]}
            )
            alignment_ok = (
                alignment_detail["corroborating_files_same_parent"]
                and alignment_detail["corroborating_files_same_condition"]
            )

        # --- §3.2 evidence detail ---
        s32_evidence = {
            "per_profile_sequence": {
                "available": per_profile_seq is not None,
                "source": per_profile_seq_source,
                "resolves_alt": per_profile_seq_resolves_alt,
            },
            "same_parent_replicate": {
                "replicate_group_member_count": gc["member_count"],
                "distinct_rdat_sha256_count": gc["distinct_sha_count"],
                "cross_file": gc["cross_file"],
                "same_construct_name": gc["same_name"],
                "reactivity_non_identical": gc["reactivity_differs"],
                "corroborating_rdat_sha256": sorted(set(corroborating_shas)),
                "corroborating_file_names": sorted(
                    {docs[s]["name"] for s in corroborating_shas}
                ),
            },
        }

        # --- miss reason (machine-readable, primary) ---
        miss_reason = None
        if status == "no_evidence_in_scope":
            if not per_profile_seq:
                miss_reason = "no_per_profile_sequence_in_rdat"
            elif not per_profile_seq_resolves_alt:
                miss_reason = "per_profile_sequence_does_not_resolve_alt"
            if not gc["cross_file"]:
                miss_reason = miss_reason or "no_cross_file_replicate_single_file_only"
            elif not gc["same_name"]:
                miss_reason = "replicate_files_differ_in_construct_name"
            elif not gc["reactivity_differs"]:
                miss_reason = "replicate_reactivity_identical_reupload"
            elif not replicate_corroborated:
                miss_reason = "no_non_identical_corroborating_file"

        candidates.append({
            "rdat_sha256": sha,
            "wt_profile_index": wt_idx,
            "mutant_profile_index": mut_idx,
            "parent_prefix": e.get("parent_prefix"),
            "rmdb_id": e.get("rmdb_id"),
            "citation_doi": e.get("citation_doi"),
            "owner": e.get("owner"),
            "rdat_file": docs[sha]["name"],
            "rdat_name": e["_rdat_name"],
            "modifier_relation": e.get("modifier"),
            "matched_mutation": mm,
            "status": status,
            "evidence_type": evidence_type,
            "source": {
                "rdat_file": docs[sha]["name"],
                "rdat_sha256": sha,
                "rdat_path": e["rdat_path"],
                "citation_doi": e.get("citation_doi"),
                "owner": e.get("owner"),
                "rmdb_id": e.get("rmdb_id"),
                "release": "RMDB rdat_tierA_20260730",
                "corroborating_rdat_sha256": sorted(set(corroborating_shas)),
                "corroborating_rdat_files": sorted(
                    {docs[s]["name"] for s in corroborating_shas}
                ),
            },
            "alignment": alignment_detail,
            "alignment_ok": alignment_ok,
            "s32_evidence": s32_evidence,
            "parent_lineage_verified_d2": e["_lineage_verified"],
            "miss_reason": miss_reason,
        })

    # qualifying replicate groups (for stats)
    stats["groups_qualifying_replicate"] = sum(
        1 for g in group_classification.values()
        if g["cross_file"] and g["same_name"] and g["reactivity_differs"]
    )
    stats["by_miss_reason"] = dict(stats["by_miss_reason"])

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "stage": "D2-R evidence-supplemented audit (v3.2 §2-§6)",
        "basis": "v3.1 §3.2 (annotation-only upgrade: per-profile sequence OR same-parent replicate) + v3.2 §3.1 anti-fabrication",
        "input_relations": str(RELATIONS_PATH),
        "input_d2_lineage": str(D2_LINEAGE_PATH),
        "candidate_total": len(enriched),
        "rdat_file_count": len(docs),
        "replicate_grouping": "T-D1.6 REPLICATE_PAIR_IDENTITY_FIELDS: (parent_id, edit_positions, wt_alleles, mut_alleles, FULL_CONDITION). FULL_CONDITION from RDAT global annotations (modifier, chemical, temperature, experimentType). study_id excluded per data.py L1013-1015.",
        "anti_fabrication_rules": [
            "§3.1 bullet 1: same-RDAT-file profiles NOT auto-promoted; only cross-file (>=2 distinct rdat_sha256) counts.",
            "§3.1 bullet 2: same-study different-construct NOT same-parent replicate; RDAT NAME must match across group.",
            "§3.1 bullet 3: D2 file-internal consistency NOT §3.2 corroboration; cross-file independence required.",
            "§3.5: one evidence may not cover candidates with different alts; replicate key includes alt.",
            "reupload guard: cross-file replicate requires >=1 non-identical mutant reactivity array.",
        ],
        "stats": stats,
        "candidates": candidates,
    }

    with OUT_PATH.open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[d2r] wrote {OUT_PATH}")
    print(f"[d2r] stats: {json.dumps(stats, indent=2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
