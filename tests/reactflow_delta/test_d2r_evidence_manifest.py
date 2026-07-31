"""D2-R evidence manifest tests (v3.2 §2-§4, v3.1 §3.1/§3.2).

Two layers:

  1. Unit tests for the pure helper functions in
     ``scripts/reactflow_delta/build_d2r_evidence_manifest.py``
     (``_merged_global_annotations``, ``_condition_key``) — fast, no I/O.
  2. Integration tests against the committed manifest artifact
     (``artifacts/reactflow_delta/d2r/d2r_evidence_manifest.json``) verifying
     the §3.1 anti-fabrication invariants, §3.2 evidence-matching rules, and
     §4 stop conditions hold on the real 7,761-candidate pool.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_manifest_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "reactflow_delta"
        / "build_d2r_evidence_manifest.py"
    )
    spec = importlib.util.spec_from_file_location("build_d2r_evidence_manifest", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


mod = _load_manifest_module()

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST_PATH = _REPO_ROOT / "artifacts" / "reactflow_delta" / "d2r" / "d2r_evidence_manifest.json"


# ---------------------------------------------------------------------------
# Unit tests: _merged_global_annotations
# ---------------------------------------------------------------------------

def test_merged_global_annotations_merges_multiple_dicts():
    doc = {
        "global_annotations": [
            {"modifier": ["DMS"]},
            {"chemical": ["Mg2+"], "temperature": ["37"]},
            {"chemical": ["EDTA"], "experimentType": ["MutateAndMap"]},
        ]
    }
    merged = mod._merged_global_annotations(doc)
    assert merged["modifier"] == ["DMS"]
    # chemical from two dicts: first occurrence wins (setdefault).
    assert merged["chemical"] == ["Mg2+"]
    assert merged["temperature"] == ["37"]
    assert merged["experimentType"] == ["MutateAndMap"]


def test_merged_global_annotations_empty():
    assert mod._merged_global_annotations({"global_annotations": []}) == {}


# ---------------------------------------------------------------------------
# Unit tests: _condition_key
# ---------------------------------------------------------------------------

def test_condition_key_structure_is_four_tuple():
    merged = {
        "modifier": ["DMS"],
        "chemical": ["EDTA", "Mg2+"],
        "temperature": ["37"],
        "experimentType": ["MutateAndMap"],
    }
    ck = mod._condition_key(merged)
    assert isinstance(ck, tuple)
    assert len(ck) == 4
    # modifier, sorted chemical, temperature, experimentType
    assert ck[0] == ("DMS",)
    assert ck[1] == ("EDTA", "Mg2+")  # sorted
    assert ck[2] == ("37",)
    assert ck[3] == ("MutateAndMap",)


def test_condition_key_missing_fields_default_empty():
    ck = mod._condition_key({})
    assert ck == ((), (), (), ())


def test_condition_key_chemical_sorted_for_stability():
    # Order in source must not affect the key (sorted for stable grouping).
    ck_a = mod._condition_key({"chemical": ["EDTA", "Mg2+"]})
    ck_b = mod._condition_key({"chemical": ["Mg2+", "EDTA"]})
    assert ck_a == ck_b


# ---------------------------------------------------------------------------
# Integration tests against the committed manifest artifact
# (skipped if the artifact is absent, e.g. fresh clone before D2-R run)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def manifest():
    if not _MANIFEST_PATH.exists():
        pytest.skip(f"manifest not built yet: {_MANIFEST_PATH}")
    with _MANIFEST_PATH.open() as f:
        return json.load(f)


def test_manifest_schema_and_stage(manifest):
    assert manifest["schema_version"] == "reactflow-delta-d2r-evidence-manifest-v1"
    assert "stage" in manifest
    assert manifest["candidate_total"] == 7761


def test_manifest_anti_fabrication_rules_present(manifest):
    rules = manifest.get("anti_fabrication_rules", [])
    assert len(rules) >= 4
    joined = " ".join(rules)
    # §3.1 bullet 1: same-file not auto-promoted
    assert "cross-file" in joined.lower() or "cross_file" in joined.lower()
    # §3.1 bullet 2: same-study different-construct not same-parent
    assert "NAME" in joined or "construct" in joined.lower()
    # reupload guard
    assert "reupload" in joined.lower() or "non-identical" in joined.lower()


def test_manifest_stats_consistency(manifest):
    stats = manifest["stats"]
    total = stats["total_candidates"]
    found = stats["evidence_found"]
    miss = stats["no_evidence_in_scope"]
    assert total == found + miss == 7761
    # per_profile_sequence path is exhausted (§4 stop condition).
    assert stats["by_evidence_type"]["per_profile_sequence"] == 0
    assert stats["by_evidence_type"]["same_parent_replicate"] == found
    assert stats["by_evidence_type"]["none"] == miss


def test_manifest_stop_condition_no_per_profile_sequence(manifest):
    # §4 stop condition: no candidate carries per-profile sequence evidence.
    for c in manifest["candidates"]:
        assert c["evidence_type"] != "per_profile_sequence"
    assert manifest["stats"]["by_evidence_type"]["per_profile_sequence"] == 0


def test_manifest_evidence_found_all_cross_file(manifest):
    # §3.1 bullet 1 / bullet 3: every evidence_found candidate must have >=1
    # corroborating RDAT with a DISTINCT sha (cross-file independence).
    for c in manifest["candidates"]:
        if c["status"] == "evidence_found":
            corrob = c["source"]["corroborating_rdat_sha256"]
            assert len(corrob) >= 1, (
                f"evidence_found without corroborating sha: {c['rdat_sha256']}"
            )
            # corrob sha must differ from the candidate's own sha
            for s in corrob:
                assert s != c["rdat_sha256"]


def test_manifest_evidence_found_all_same_parent_replicate(manifest):
    for c in manifest["candidates"]:
        if c["status"] == "evidence_found":
            assert c["evidence_type"] == "same_parent_replicate"


def test_manifest_no_evidence_has_empty_corroborating(manifest):
    for c in manifest["candidates"]:
        if c["status"] == "no_evidence_in_scope":
            assert c["source"]["corroborating_rdat_sha256"] == []


def test_manifest_reupload_guard_records_reuploads(manifest):
    # 104 candidates miss due to reactivity-identical reupload (§3.1 reupload
    # guard). They must be in no_evidence_in_scope.
    by_miss = manifest["stats"]["by_miss_reason"]
    reupload_count = by_miss.get("replicate_reactivity_identical_reupload", 0)
    assert reupload_count == 104
    # cross-file but reactivity-identical → not independent corroboration.
    reupload_candidates = [
        c for c in manifest["candidates"]
        if c["status"] == "no_evidence_in_scope"
        and "replicate_reactivity_identical_reupload"
        in manifest["stats"]["by_miss_reason"]
    ]
    # at least the reupload candidates exist; each has empty corrob.
    for c in reupload_candidates[:10]:
        assert c["source"]["corroborating_rdat_sha256"] == []


def test_manifest_single_file_miss_reason_present(manifest):
    # §3.1 bullet 1: single-file candidates (no cross-file replicate) are
    # NOT auto-promoted; they land in no_evidence with single_file miss.
    by_miss = manifest["stats"]["by_miss_reason"]
    assert by_miss.get("no_cross_file_replicate_single_file_only", 0) == 6121


def test_manifest_qualifying_groups_match_evidence(manifest):
    stats = manifest["stats"]
    # 768 qualifying groups produced 1536 evidence_found (2 candidates per
    # group on average — each member of a 2-file group is corroborated).
    assert stats["groups_qualifying_replicate"] == 768
    assert stats["evidence_found"] == 1536
    assert stats["groups_cross_file"] == 820
    # 820 cross-file groups − 768 qualifying = 52 excluded (reupload or
    # name-mismatch); 768 qualify, 55 reupload groups excluded (104 members).
