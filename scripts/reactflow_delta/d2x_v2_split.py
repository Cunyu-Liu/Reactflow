#!/usr/bin/env python3
"""
d2x_v2_split.py -- ReactFlowDelta split_v2 builder (R2C: unify group atoms + split).

PURPOSE
=======
Deterministically build `split_v2` from the frozen v2 canonical data (R2B outputs):
  * Assign every primary pair a consistent group-atom set:
        publication -> resolved PMID/DOI (OLD split manifest publication_map) or
                       UNKNOWN_PUBLICATION:<study> (PMID not asserted by frozen RMDB
                       metadata; NEVER invented)
        study      -> source_accession prefix
        parent     -> parent_lineage_evidence.design_group (RMDB design construct)
        lineage    -> parent_lineage_evidence.parent_sequence_sha256
        family     -> source_group (RMDB release category; documented choice)
  * publication is the OUTER split unit; same PMID = ONE publication; a
    publication never spans two split roles.
  * Retire the old 16SFWJ test as DEVELOPMENT_CONSUMED.
  * Designate a NEW untouched test study (never in old train/val/test, never used
    for development). If no genuinely-untouched publication with a CERTIFIED PMID
    exists, this is reported honestly as a confirmatory blocker (contract
    SS14.2), never faked.

Determinism: role assignment is fully explicit (no RNG, no hashing order that
depends on set iteration); same input files -> identical outputs.

Outputs
=======
  configs/reactflow_delta/split_v2.yaml      (machine-readable split + atoms + test)
  data_registry/d2x_v2/overlap_report.json   (publication/family/lineage/sequence overlap)
  data_registry/d2x_v2/split_v2_manifest.json (summary + SHA256 manifest)

This script does NOT train anything. It never modifies legacy d1x/d2x artifacts or
authority/contract files.
"""
import json
import os
import sys
import pickle
import hashlib
from collections import Counter, defaultdict

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
HOME = "/home/cunyuliu/reactflow_delta_goal_20260729"
ART = "/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta"
V2_DIR = os.path.join(ART, "d1x_v2/d1x_v2_canonicalization_20260807T1830+0800")
OLD_MANIFEST = os.path.join(
    ART,
    "d2x/d2x_split_publication_20260804T1600+0800/d2x_split_manifest.json",
)
ASSET_DISPOSITION = os.path.join(HOME, "data_registry/d0x_v2/asset_disposition_20260807.jsonl")

PAIRS_JSONL = os.path.join(V2_DIR, "primary_pairs_v2.jsonl")
CANONICAL_JSONL = os.path.join(V2_DIR, "canonical_records_v2.jsonl")

OUT_YAML = os.path.join(HOME, "configs/reactflow_delta/split_v2.yaml")
OUT_OVERLAP = os.path.join(HOME, "data_registry/d2x_v2/overlap_report.json")
OUT_MANIFEST = os.path.join(HOME, "data_registry/d2x_v2/split_v2_manifest.json")

# ----------------------------------------------------------------------------
# Explicit, deterministic role policy (the ONLY place roles are decided).
# Publication is the outer unit: a role is applied per study; because each study
# belongs to exactly one publication and every study is placed in exactly one
# role, no publication spans two roles.
# ----------------------------------------------------------------------------
# Old 16SFWJ test is RETIRED as DEVELOPMENT_CONSUMED (cannot be confirmatory).
RETIRED_DEVELOPMENT_CONSUMED = ["16SFWJ"]          # old test, pmid_25183835
# Development validation (dev-exposed, reused from old validation publication).
VALIDATION_STUDIES = ["CIDGMP", "TRP4P6"]          # pmid_25303992 (SHAPE-Seq 2.0)
# NEW untouched test study: never in old train/val/test, never used for dev.
# The SL5 trio (CV2/HKU/MER) are homologous SARS-CoV-2 SL5 betacoronavirus
# constructs (likely a single publication). They are grouped together as the
# untouched test FAMILY so that no SL5 relative is left in train/val (avoids
# cross-role homology), and so that (in the common case they are one paper) the
# designated test publication is kept entirely out of development.
NEW_TEST_STUDIES = ["SL5CV2", "SL5HKU", "SL5MER"]  # SARS-CoV-2 SL5 (untouched)
# everything else -> train

SCHEMA_SPLIT_V2 = "reactflow_delta.split_v2.v1"


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_old_manifest():
    d = json.load(open(OLD_MANIFEST))
    return d.get("publication_map", {}), d.get("assignment", {}), d.get("pair_counts", {})


def load_asset_source_groups():
    """source_accession -> source_group (RMDB release category)."""
    sg = {}
    for line in open(ASSET_DISPOSITION):
        d = json.loads(line)
        sg[d["source_accession"]] = d["source_group"]
    return sg


def extract_parent_info():
    """source_accession -> {design_group,parent_sequence_sha256,construct_sequence_length,canonical_sequence}
    Only the source_accessions referenced by primary pairs are decoded (the full
    canonical_records file is ~40 GB and must not be fully parsed). A single regex
    per line extracts the accession so only needed lines are json-decoded.
    Results are cached to /tmp keyed by primary-pairs sha for fast re-runs
    (cache is a pure optimization; output is identical with or without it)."""
    import re
    acc_re = re.compile(r'"source_accession":\s*"([^"]+)"')
    pairs_sha = sha256_file(PAIRS_JSONL)
    cache = "/tmp/rd_parent_cache_v2.pkl"
    if os.path.exists(cache):
        try:
            obj = pickle.load(open(cache, "rb"))
            if obj.get("pairs_sha") == pairs_sha:
                return obj["parent"]
        except Exception:
            pass

    need = set()
    for line in open(PAIRS_JSONL):
        need.add(json.loads(line)["source_accession"])

    parent = {}
    for line in open(CANONICAL_JSONL):
        m = acc_re.search(line)
        if m is None or m.group(1) not in need:
            continue
        d = json.loads(line)
        acc = d["source_accession"]
        ple = d.get("parent_lineage_evidence") or {}
        parent.setdefault(acc, {})
        parent[acc]["design_group"] = ple.get("design_group")
        parent[acc]["parent_sequence_sha256"] = ple.get("parent_sequence_sha256")
        parent[acc]["construct_sequence_length"] = ple.get("construct_sequence_length")
        parent[acc]["canonical_sequence"] = d.get("canonical_sequence")
    pickle.dump({"pairs_sha": pairs_sha, "parent": parent}, open(cache, "wb"))
    return parent


def build():
    pub_map, old_assignment, old_pair_counts = load_old_manifest()
    source_groups = load_asset_source_groups()
    parent = extract_parent_info()

    # ---- assemble pair-level atoms ----
    pairs = []
    for line in open(PAIRS_JSONL):
        p = json.loads(line)
        acc = p["source_accession"]
        study = acc.split("_")[0]
        pub = pub_map.get(study, "UNKNOWN_PUBLICATION:" + study)
        p_info = parent.get(acc, {})
        pairs.append({
            "study": study,
            "source_accession": acc,
            "asset_name": p.get("asset_name"),
            "ref_allele": p.get("ref_allele"),
            "alt_allele": p.get("alt_allele"),
            "coordinate": p.get("coordinate"),
            "wt_profile_index": p.get("wt_profile_index"),
            "mutant_profile_index": p.get("mutant_profile_index"),
            "publication": pub,
            "parent": p_info.get("design_group"),
            "lineage": p_info.get("parent_sequence_sha256"),
            "construct_sequence_length": p_info.get("construct_sequence_length"),
            "canonical_sequence": p_info.get("canonical_sequence"),
            "family": source_groups.get(acc),
        })

    # ---- role assignment (per study -> per pair) ----
    def role_for(study):
        if study in RETIRED_DEVELOPMENT_CONSUMED:
            return "DEVELOPMENT_CONSUMED"
        if study in NEW_TEST_STUDIES:
            return "test"
        if study in VALIDATION_STUDIES:
            return "validation"
        return "train"

    for p in pairs:
        p["role"] = role_for(p["study"])

    # ---- publication is the outer unit: verify no publication spans roles ----
    pub_role = {}
    for p in pairs:
        r = pub_role.get(p["publication"])
        if r is None:
            pub_role[p["publication"]] = p["role"]
        elif r != p["role"]:
            raise ValueError(
                "publication %s spans roles %s and %s" % (p["publication"], r, p["role"])
            )

    # ---- summary counts ----
    role_pairs = Counter(p["role"] for p in pairs)
    study_roles = {}
    for p in pairs:
        study_roles.setdefault(p["study"], p["role"])
    study_pair_counts = Counter(p["study"] for p in pairs)
    pub_studies = defaultdict(set)
    for p in pairs:
        pub_studies[p["publication"]].add(p["study"])

    distinct_publications = sorted(pub_studies.keys())
    distinct_studies = sorted(study_roles.keys())
    distinct_parents = sorted({p["parent"] for p in pairs if p["parent"]})
    distinct_lineages = sorted({p["lineage"] for p in pairs if p["lineage"]})
    distinct_families = sorted({p["family"] for p in pairs if p["family"]})

    # group atoms definition text (documented)
    group_atoms = {
        "publication": "resolved PMID/DOI from OLD split manifest publication_map; "
                      "else UNKNOWN_PUBLICATION:<study> (frozen RMDB snapshot does not assert "
                      "PMID per asset; never invented). same PMID = one publication = outer unit.",
        "study": "source_accession prefix",
        "parent": "parent_lineage_evidence.design_group (RMDB design construct)",
        "lineage": "parent_lineage_evidence.parent_sequence_sha256",
        "family": "source_group (RMDB release category) -- broad data-collection category, "
                  "documented choice; strict sequence control is via lineage (parent_sequence_sha256).",
    }

    split = {
        "schema_version": SCHEMA_SPLIT_V2,
        "endpoint_id": "RFD_ENDPOINT_V2",
        "generated_by": "scripts/reactflow_delta/d2x_v2_split.py",
        "input_sources": {
            "primary_pairs_v2.jsonl": PAIRS_JSONL,
            "canonical_records_v2.jsonl": CANONICAL_JSONL,
            "old_split_manifest": OLD_MANIFEST,
            "asset_disposition_20260807.jsonl": ASSET_DISPOSITION,
        },
        "group_atoms": group_atoms,
        "retired_test": {
            "studies": RETIRED_DEVELOPMENT_CONSUMED,
            "status": "DEVELOPMENT_CONSUMED",
            "reason": "old d2x test (pmid_25183835); cannot be re-used as confirmatory",
        },
        "new_test": {
            "studies": NEW_TEST_STUDIES,
            "publication": "UNKNOWN_PUBLICATION:" + NEW_TEST_STUDIES[0],
            "untouched": True,
            "confirmatory_blocker": _confirmatory_blocker(pub_studies, pub_map, old_assignment),
        },
        "assignment": {s: role_for(s) for s in sorted(study_roles)},
        "publication_map": {s: (pub_map.get(s, "UNKNOWN_PUBLICATION:" + s)) for s in sorted(study_roles)},
        "distinct_publications": distinct_publications,
        "distinct_studies": distinct_studies,
        "distinct_parents": distinct_parents,
        "distinct_lineages": distinct_lineages,
        "distinct_families": distinct_families,
        "publication_studies": {k: sorted(v) for k, v in pub_studies.items()},
        "study_roles": {s: role_for(s) for s in sorted(study_roles)},
        "pair_counts": dict(role_pairs),
        "study_pair_counts": dict(study_pair_counts),
        "overlap_report": "data_registry/d2x_v2/overlap_report.json",
        "manifest": "data_registry/d2x_v2/split_v2_manifest.json",
    }

    # ---- overlap report ----
    overlap = build_overlap(pairs, split)

    # ---- write outputs ----
    out_dir = os.path.dirname(OUT_OVERLAP)
    os.makedirs(out_dir, exist_ok=True)
    write_yaml(split, OUT_YAML)
    write_json(overlap, OUT_OVERLAP)

    manifest = {
        "schema_version": "reactflow_delta.split_v2_manifest.v1",
        "split_yaml": {"path": OUT_YAML, "sha256": sha256_file(OUT_YAML)},
        "overlap_report": {"path": OUT_OVERLAP, "sha256": sha256_file(OUT_OVERLAP)},
        "pair_counts": dict(role_pairs),
        "distinct_publications": len(distinct_publications),
        "distinct_studies": len(distinct_studies),
        "distinct_parents": len(distinct_parents),
        "distinct_lineages": len(distinct_lineages),
        "distinct_families": len(distinct_families),
        "new_test_studies": NEW_TEST_STUDIES,
        "retired_test_studies": RETIRED_DEVELOPMENT_CONSUMED,
        # per-source_accession group atoms (compact; for consistency validation)
        "group_atom_table": {
            p["source_accession"]: {
                "study": p["study"],
                "parent": p["parent"],
                "lineage": p["lineage"],
                "family": p["family"],
                "publication": p["publication"],
                "role": p["role"],
            }
            for p in pairs
        },
    }
    write_json(manifest, OUT_MANIFEST)

    print(json.dumps({"pairs": len(pairs),
                      "role_pairs": dict(role_pairs),
                      "distinct_publications": len(distinct_publications),
                      "distinct_studies": len(distinct_studies),
                      "distinct_parents": len(distinct_parents),
                      "distinct_lineages": len(distinct_lineages),
                      "distinct_families": len(distinct_families)},
                     indent=2))
    return split, overlap


def _confirmatory_blocker(pub_studies, pub_map, old_assignment):
    """Contract SS14.2: need >=3 UNTOUCHED confirmatory publications.
    A publication is 'untouched' only if it is CERTIFIED (resolved PMID) and NONE
    of its studies were ever in the OLD split train/val/test (development-exposed).
    Report honestly; never fake."""
    old_exposed_studies = set(old_assignment.keys())  # any study in old train/val/test
    untouched_certified = []
    exposed_certified = []
    unknown_publication_studies = []
    for pub, studies in sorted(pub_studies.items()):
        if pub.startswith("UNKNOWN_PUBLICATION:"):
            unknown_publication_studies.extend(studies)
            continue
        if not (set(studies) & old_exposed_studies):
            untouched_certified.append(pub)
        else:
            exposed_certified.append(pub)
    if len(untouched_certified) >= 3:
        return {
            "blocked": False,
            "untouched_certified_publications": untouched_certified,
            "satisfies_ge3_untouched": True,
        }
    return {
        "blocked": True,
        "reason": ("frozen RMDB snapshot does not assert PMID per asset. The only CERTIFIED "
                   "publications are the OLD split manifest entries, ALL of which were exposed to "
                   "development (old train/val/test). New v2 studies resolve to "
                   "UNKNOWN_PUBLICATION:<study> and cannot be certified as distinct untouched "
                   "confirmatory publications from frozen metadata alone."),
        "untouched_certified_publications": untouched_certified,
        "exposed_certified_publications": exposed_certified,
        "unknown_publication_studies": sorted(set(unknown_publication_studies)),
        "satisfies_ge3_untouched": False,
        "recommendation": ("resolve publication identity for new studies (e.g. via RMDB entry "
                           "citation / d0r_accession_registry, where per-entry PMIDs exist -- the "
                           "SL5 family carries pmid_38427602) or designate a prospective "
                           "confirmatory alternative before confirmatory CI."),
    }


def build_overlap(pairs, split):
    roles = ["train", "validation", "test"]
    # map study->role (from split)
    study_role = split["study_roles"]
    # publication overlap
    pub_role = {}
    for p in pairs:
        pub_role[p["publication"]] = p["role"]
    pub_role_clean = {k: v for k, v in pub_role.items()
                      if v != "DEVELOPMENT_CONSUMED"}
    # family/lineage/sequence set per role
    fam_by_role = defaultdict(set)
    lin_by_role = defaultdict(set)
    seq_by_role = defaultdict(set)
    study_by_role = defaultdict(set)
    for p in pairs:
        if p["role"] == "DEVELOPMENT_CONSUMED":
            continue
        fam_by_role[p["role"]].add(p["family"])
        lin_by_role[p["role"]].add(p["lineage"])
        seq_by_role[p["role"]].add(p["canonical_sequence"])
        study_by_role[p["role"]].add(p["study"])

    def overlap_metrics(getter_by_role):
        res = {}
        for a in roles:
            for b in roles:
                if a >= b:
                    continue
                sa = getter_by_role(a)
                sb = getter_by_role(b)
                inter = sa & sb
                res["%s_vs_%s" % (a, b)] = {
                    "n_%s" % a: len(sa),
                    "n_%s" % b: len(sb),
                    "intersection": sorted(inter),
                    "n_intersection": len(inter),
                }
        return res

    # homology/sequence overlap between new test and all train/val (explicit)
    test_lineages = lin_by_role["test"]
    tv_lineages = lin_by_role["train"] | lin_by_role["validation"]
    test_fams = fam_by_role["test"]
    tv_fams = fam_by_role["train"] | fam_by_role["validation"]
    test_seqs = seq_by_role["test"]
    tv_seqs = seq_by_role["train"] | seq_by_role["validation"]

    return {
        "schema_version": "reactflow_delta.overlap_report.v1",
        "note": ("publication/study/lineage overlap must be 0; family uses source_group "
                 "(broad release category) and is EXPLICITLY EXEMPTED from the strict-0 rule "
                 "because it is not a sequence family -- the strict sequence control is lineage "
                 "(parent_sequence_sha256) and exact canonical sequence."),
        "publication_overlap": {  # outer unit; by construction 0
            k: v for k, v in overlap_metrics(lambda r: {pub for pub, ro in pub_role_clean.items() if ro == r}).items()
        },
        "study_overlap": overlap_metrics(lambda r: study_by_role[r]),
        "family_overlap_source_group": overlap_metrics(lambda r: fam_by_role[r]),
        "lineage_overlap_parent_sha256": overlap_metrics(lambda r: lin_by_role[r]),
        "sequence_overlap_exact_canonical": overlap_metrics(lambda r: seq_by_role[r]),
        "new_test_vs_train_val": {
            "lineage_parent_sha256": {
                "test": len(test_lineages),
                "train_val": len(tv_lineages),
                "intersection": sorted(test_lineages & tv_lineages),
                "n_intersection": len(test_lineages & tv_lineages),
                "rule": "must be 0 (strict)",
            },
            "family_source_group": {
                "test": len(test_fams),
                "train_val": len(tv_fams),
                "intersection": sorted(test_fams & tv_fams),
                "n_intersection": len(test_fams & tv_fams),
                "rule": "explicitly exempted (broad release category, not sequence family)",
            },
            "sequence_exact_canonical": {
                "test": len(test_seqs),
                "train_val": len(tv_seqs),
                "intersection": sorted(test_seqs & tv_seqs),
                "n_intersection": len(test_seqs & tv_seqs),
                "rule": "must be 0 (strict)",
            },
        },
    }


def write_yaml(obj, path):
    out = []

    def dump(o, indent):
        pad = " " * indent
        if isinstance(o, dict):
            if not o:
                out.append(pad + "{}")
                return
            for k, v in o.items():
                if isinstance(v, dict):
                    if not v:
                        out.append(pad + "%s: {}" % k)
                    else:
                        out.append(pad + "%s:" % k)
                        dump(v, indent + 2)
                elif isinstance(v, list):
                    if not v:
                        out.append(pad + "%s: []" % k)
                    else:
                        out.append(pad + "%s:" % k)
                        dump(v, indent + 2)
                else:
                    out.append(pad + "%s: %s" % (k, yaml_scalar(v)))
        elif isinstance(o, list):
            for x in o:
                if isinstance(x, (dict, list)):
                    out.append(pad + "-")
                    dump(x, indent + 2)
                else:
                    out.append(pad + "- %s" % yaml_scalar(x))

    dump(obj, 0)
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")


def yaml_scalar(v):
    if v is None:
        return "null"
    if v is True:
        return "true"
    if v is False:
        return "false"
    if isinstance(v, (int, float)):
        return str(v)
    return '"%s"' % str(v).replace('"', '\\"')


def write_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


if __name__ == "__main__":
    build()
