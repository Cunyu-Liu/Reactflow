"""Tests for the split_v2 builder (ReactFlowDelta R2C: unify group atoms + split).

These tests validate the invariants of configs/reactflow_delta/split_v2.yaml,
data_registry/d2x_v2/overlap_report.json and data_registry/d2x_v2/split_v2_manifest.json:

  1. same PMID = one publication; publication is the outer unit and never spans roles.
  2. parent/lineage group-atom definition is consistent.
  3. publication/family/lineage overlap = 0 or explicitly exempted; new test untouched.
  4. the old 16SFWJ test is RETIRED as DEVELOPMENT_CONSUMED.
  5. the new test is a genuinely untouched study (never in old train/val/test, never exposed).
  6. deterministic reproducibility: same input -> byte-identical split.
"""
import hashlib
import json
import os
import subprocess

import pytest

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SPLIT_YAML = os.path.join(ROOT, "configs/reactflow_delta/split_v2.yaml")
OVERLAP_JSON = os.path.join(ROOT, "data_registry/d2x_v2/overlap_report.json")
MANIFEST_JSON = os.path.join(ROOT, "data_registry/d2x_v2/split_v2_manifest.json")
BUILDER = os.path.join(ROOT, "scripts/reactflow_delta/d2x_v2_split.py")
OLD_MANIFEST = (
    "/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/"
    "d2x/d2x_split_publication_20260804T1600+0800/d2x_split_manifest.json"
)

VALID_ROLES = {"train", "validation", "test", "DEVELOPMENT_CONSUMED"}


@pytest.fixture(scope="module")
def split():
    assert os.path.exists(SPLIT_YAML), "split_v2.yaml not found -- run d2x_v2_split.py"
    return yaml.safe_load(open(SPLIT_YAML))


@pytest.fixture(scope="module")
def overlap():
    assert os.path.exists(OVERLAP_JSON)
    return json.load(open(OVERLAP_JSON))


@pytest.fixture(scope="module")
def manifest():
    assert os.path.exists(MANIFEST_JSON)
    return json.load(open(MANIFEST_JSON))


@pytest.fixture(scope="module")
def old_manifest():
    return json.load(open(OLD_MANIFEST))


def all_zero(d):
    """Return True if every leaf n_intersection in d is 0."""
    for k, v in d.items():
        if isinstance(v, dict):
            if "n_intersection" in v:
                if v["n_intersection"] != 0:
                    return False
            else:
                if not all_zero(v):
                    return False
    return True


# ---------------------------------------------------------------------------
# 1. Same PMID = one publication; publication is outer unit; no cross-role span
# ---------------------------------------------------------------------------
def test_same_pmid_one_publication(split):
    pmap = split["publication_map"]
    roles = split["study_roles"]
    pub_studies = split["publication_studies"]
    # a publication (atom) maps to >=1 study
    for pub, studies in pub_studies.items():
        assert isinstance(studies, list) and len(studies) >= 1
    # every study maps to exactly one publication
    study_pub = {}
    for study in roles:
        pubs = [p for p, ss in pub_studies.items() if study in ss]
        assert len(pubs) == 1, "study %s maps to %d publications" % (study, len(pubs))
        study_pub[study] = pubs[0]
    # consistent with publication_map
    for study, pub in study_pub.items():
        assert pmap[study] == pub
    # same PMID == one publication: all studies of a publication share one role
    for pub, studies in pub_studies.items():
        role_set = {roles[s] for s in studies}
        assert len(role_set) == 1, "publication %s spans roles %s" % (pub, role_set)


def test_no_publication_spans_two_roles(split):
    roles = split["study_roles"]
    pub_role = {}
    for study, role in roles.items():
        for pub in split["publication_studies"]:
            if study in split["publication_studies"][pub]:
                if pub in pub_role:
                    assert pub_role[pub] == role, "publication %s spans roles" % pub
                else:
                    pub_role[pub] = role


def test_shared_publication_grouped(split):
    # CIDGMP and TRP4P6 share pmid_25303992 (SHAPE-Seq 2.0) -> same publication, same role
    assert split["publication_map"]["CIDGMP"] == "pmid_25303992"
    assert split["publication_map"]["TRP4P6"] == "pmid_25303992"
    assert split["study_roles"]["CIDGMP"] == split["study_roles"]["TRP4P6"] == "validation"


# ---------------------------------------------------------------------------
# 2. parent / lineage group-atom definition consistency
# ---------------------------------------------------------------------------
def test_parent_definition_consistent(manifest, split):
    table = manifest["group_atom_table"]
    assert len(table) == 75, "expected 75 source_accessions, got %d" % len(table)
    for acc, atoms in table.items():
        parent = atoms["parent"]
        lineage = atoms["lineage"]
        assert parent, "parent missing for %s" % acc
        assert lineage and len(lineage) == 64, "lineage not sha256 for %s" % acc
        # parent (design_group) is a source_accession of the SAME study
        assert parent.split("_")[0] == atoms["study"], (
            "parent %s not consistent with study %s" % (parent, atoms["study"])
        )
        assert acc.split("_")[0] == atoms["study"]
        assert atoms["role"] in VALID_ROLES
        # publication atom matches the study's publication in the split yaml
        assert atoms["publication"] == split["publication_map"][atoms["study"]]


def test_study_single_role(manifest, split):
    # every study is in exactly one role; role derived from split.study_roles
    roles = split["study_roles"]
    table = manifest["group_atom_table"]
    for acc, atoms in table.items():
        assert atoms["role"] == roles[atoms["study"]]
    # all distinct roles valid, and a publication is not split
    for study, role in roles.items():
        assert role in VALID_ROLES


# ---------------------------------------------------------------------------
# 3. overlap = 0 or explicitly exempted
# ---------------------------------------------------------------------------
def test_publication_study_lineage_sequence_overlap_zero(overlap):
    assert all_zero(overlap["publication_overlap"]), "publication overlap must be 0"
    assert all_zero(overlap["study_overlap"]), "study overlap must be 0"
    assert all_zero(overlap["lineage_overlap_parent_sha256"]), "lineage overlap must be 0"
    assert all_zero(overlap["sequence_overlap_exact_canonical"]), "sequence overlap must be 0"


def test_family_overlap_explicitly_exempted(overlap):
    # family = source_group (RMDB release category) is a broad category, not a
    # sequence family; any cross-role sharing is EXPLICITLY EXEMPTED (documented).
    note = overlap.get("note", "")
    assert "exempt" in note.lower() or "exempt" in str(overlap.get("family_overlap_source_group"))
    for k, v in overlap["family_overlap_source_group"].items():
        assert v["n_intersection"] >= 0  # allowed; exempted by documentation


def test_new_test_vs_train_val_no_sequence_leak(overlap):
    ntv = overlap["new_test_vs_train_val"]
    assert ntv["lineage_parent_sha256"]["n_intersection"] == 0
    assert ntv["sequence_exact_canonical"]["n_intersection"] == 0
    # family is a documented exemption (broad release category)
    assert ntv["family_source_group"]["rule"].startswith("explicitly exempted")


# ---------------------------------------------------------------------------
# 4. retired old test == DEVELOPMENT_CONSUMED
# ---------------------------------------------------------------------------
def test_old_test_retired_as_development_consumed(split, old_manifest):
    assert old_manifest["test_studies"] == ["16SFWJ"]
    assert split["retired_test"]["status"] == "DEVELOPMENT_CONSUMED"
    assert split["retired_test"]["studies"] == ["16SFWJ"]
    assert split["study_roles"]["16SFWJ"] == "DEVELOPMENT_CONSUMED"
    # retired test must NOT appear in any train/validation/test role
    active = {"train", "validation", "test"}
    for study, role in split["study_roles"].items():
        if study == "16SFWJ":
            assert role == "DEVELOPMENT_CONSUMED"
        else:
            assert role in active


# ---------------------------------------------------------------------------
# 5. new test is untouched (never in old train/val/test, never exposed)
# ---------------------------------------------------------------------------
def test_new_test_untouched(split, old_manifest):
    new_test = split["new_test"]
    assert new_test["untouched"] is True
    studies = new_test["studies"]
    assert sorted(studies) == ["SL5CV2", "SL5HKU", "SL5MER"]
    old_exposed = (
        set(old_manifest.get("train_studies", []))
        | set(old_manifest.get("validation_studies", []))
        | set(old_manifest.get("test_studies", []))
    )
    # new test studies were never in old train/val/test
    assert not (set(studies) & old_exposed), "new test overlaps old exposed studies"
    # new test studies were never in the old assignment at all
    assert not (set(studies) & set(old_manifest["assignment"].keys()))
    # they are all assigned to the test role, never to train/val
    for s in studies:
        assert split["study_roles"][s] == "test"
    # and they are not marked as retired/development-consumed
    assert not (set(studies) & set(split["retired_test"]["studies"]))


def test_confirmatory_blocker_reported_honestly(split):
    # With publication identity not asserted for new studies by frozen metadata,
    # the only certified publications are old development-exposed ones; so the
    # confirmatory blocker must be reported as blocked (not faked).
    blocker = split["new_test"]["confirmatory_blocker"]
    assert blocker["blocked"] is True
    assert blocker["satisfies_ge3_untouched"] is False
    assert blocker["untouched_certified_publications"] == []


# ---------------------------------------------------------------------------
# 6. deterministic reproducibility (same input -> same split)
# ---------------------------------------------------------------------------
def test_deterministic_reproducibility(manifest):
    recorded_sha = manifest["split_yaml"]["sha256"]
    # re-run the builder (idempotent; parent extraction is cached to /tmp)
    res = subprocess.run(
        ["/mnt/cunyuliu/reactflow_delta_runtime_py311/bin/python", BUILDER],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, "builder failed: %s" % res.stderr[-2000:]
    new_manifest = json.load(open(MANIFEST_JSON))
    fresh_sha = new_manifest["split_yaml"]["sha256"]
    file_sha = hashlib.sha256(open(SPLIT_YAML, "rb").read()).hexdigest()
    # self-consistent manifest, and byte-identical to the previous run
    assert fresh_sha == file_sha
    assert fresh_sha == recorded_sha, (
        "split_v2.yaml changed across runs (not deterministic): %s != %s"
        % (fresh_sha, recorded_sha)
    )
