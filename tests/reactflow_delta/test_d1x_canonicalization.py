#!/usr/bin/env python3
"""D1-X contract acceptance tests for exact canonicalization and cleaning.

These tests exercise the fixed cleaning contract (V4 section 8) over the
outcome-blind canonicalization logic: exact ref/alt verification, WT-mutant
condition pairing, closed-set role assignment, null+mask reactivity layers,
and the D1-X authority preflight.  No normalization, no split, no training,
no test access.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
sys.path.insert(0, str(_SRC))

_SCRIPT = _REPO_ROOT / "scripts/reactflow_delta/d1x_canonicalize.py"
_spec = importlib.util.spec_from_file_location("d1x_canonicalize", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def _condition(modifier=("p__",), temperature=("20C",), chemical=("Buffer",), experimentType=("InVitro",)):
    return {
        "modifier": modifier,
        "temperature": temperature,
        "chemical": chemical,
        "experimentType": experimentType,
        "all_required_known": all(
            [modifier, temperature, chemical, experimentType]
        ),
    }


class CanonicalizeProfileTest(unittest.TestCase):
    def test_verified_exact_single_substitution_primary(self):
        rec = {
            "source_profile_index": 3,
            "raw_mutation_token": ["C13G"],
            "parsed_mutations": [
                {
                    "kind": "EXACT_SINGLE_SUBSTITUTION",
                    "edits": [
                        {
                            "ref_allele": "C",
                            "alt_allele": "G",
                            "source_coordinate_1_based": 13,
                            "sequence_index_0_based": 12,
                        }
                    ],
                }
            ],
            "exact_mutation_evidence_status": "EXACT_REF_ALT_TOKEN_REF_VERIFIED_PROFILE_SEQUENCE_UNAVAILABLE",
            "resolved_annotations": {
                "modifier": {"resolved_values": ["p__"]},
                "temperature": {"resolved_values": ["20C"]},
                "chemical": {"resolved_values": ["Buffer"]},
                "experimentType": {"resolved_values": ["InVitro"]},
            },
        }
        seq = "A" * 12 + "C" + "A" * 20  # position 12 0-based = 'C'
        cand = _mod._canonicalize_profile(
            rec, {"reactivity": [0.1, 0.2], "reactivity_error": [0.01, 0.02]},
            seq, 0, "ACC1", "abc123", "ACC1.rdat",
        )
        self.assertEqual(cand["data_role"], "PRIMARY_EXACT_DELTA")
        self.assertEqual(cand["ref_allele"], "C")
        self.assertEqual(cand["alt_allele"], "G")
        self.assertEqual(cand["exact_mutation_evidence_status"], "VERIFIED_EXACT_SINGLE_SUBSTITUTION")
        self.assertIsNone(cand["exclusion_reason"])
        self.assertEqual(cand["verification"]["ref_matches_seq"], True)
        self.assertEqual(cand["parent_lineage_evidence"]["construct_sequence_length"], len(seq))

    def test_ref_mismatch_is_conflicting_evidence(self):
        rec = {
            "source_profile_index": 1,
            "raw_mutation_token": ["C13G"],
            "parsed_mutations": [
                {
                    "kind": "EXACT_SINGLE_SUBSTITUTION",
                    "edits": [
                        {
                            "ref_allele": "C",
                            "alt_allele": "G",
                            "source_coordinate_1_based": 13,
                            "sequence_index_0_based": 12,
                        }
                    ],
                }
            ],
            "exact_mutation_evidence_status": "EXACT_REF_ALT_TOKEN_REF_VERIFIED_PROFILE_SEQUENCE_UNAVAILABLE",
            "resolved_annotations": {},
        }
        seq = "A" * 13 + "A" * 20  # position 12 = 'A', ref 'C' does not match
        cand = _mod._canonicalize_profile(
            rec, {"reactivity": [], "reactivity_error": []},
            seq, 0, "ACC1", "abc123", "ACC1.rdat",
        )
        self.assertEqual(cand["exact_mutation_evidence_status"], "CONFLICTING_EVIDENCE")
        self.assertEqual(cand["exclusion_reason"], "REF_MISMATCH")
        self.assertIsNone(cand["data_role"])

    def test_latent_alt_is_auxiliary(self):
        rec = {
            "source_profile_index": 2,
            "raw_mutation_token": ["C13X"],
            "parsed_mutations": [
                {
                    "kind": "EXACT_SINGLE_SUBSTITUTION",
                    "edits": [
                        {
                            "ref_allele": "C",
                            "alt_allele": "X",
                            "source_coordinate_1_based": 13,
                            "sequence_index_0_based": 12,
                        }
                    ],
                }
            ],
            "exact_mutation_evidence_status": "LATENT_ALT_X_REF_CHECKED",
            "resolved_annotations": {},
        }
        seq = "A" * 12 + "C" + "A" * 20
        cand = _mod._canonicalize_profile(
            rec, {"reactivity": [], "reactivity_error": []},
            seq, 0, "ACC1", "abc123", "ACC1.rdat",
        )
        self.assertEqual(cand["data_role"], "AUXILIARY_LATENT_ALT")
        self.assertEqual(cand["alt_allele"], "X")

    def test_multi_edit_is_rescue(self):
        rec = {
            "source_profile_index": 4,
            "raw_mutation_token": ["C13G;G14A"],
            "parsed_mutations": [
                {
                    "kind": "EXACT_SINGLE_SUBSTITUTION",
                    "edits": [
                        {"ref_allele": "C", "alt_allele": "G", "sequence_index_0_based": 12},
                        {"ref_allele": "G", "alt_allele": "A", "sequence_index_0_based": 13},
                    ],
                }
            ],
            "exact_mutation_evidence_status": "MULTIPLE_MUTATION_ANNOTATION_VALUES",
            "resolved_annotations": {},
        }
        seq = "A" * 40
        cand = _mod._canonicalize_profile(
            rec, {"reactivity": [], "reactivity_error": []},
            seq, 0, "ACC1", "abc123", "ACC1.rdat",
        )
        self.assertEqual(cand["data_role"], "RESCUE_MULTI_EDIT")

    def test_wt_control_has_no_role(self):
        rec = {
            "source_profile_index": 0,
            "raw_mutation_token": ["WT"],
            "parsed_mutations": [{"kind": "WT", "edits": []}],
            "exact_mutation_evidence_status": "WT_CONTROL_CANDIDATE",
            "resolved_annotations": {},
        }
        seq = "A" * 30
        cand = _mod._canonicalize_profile(
            rec, {"reactivity": [], "reactivity_error": []},
            seq, 0, "ACC1", "abc123", "ACC1.rdat",
        )
        self.assertEqual(cand["exact_mutation_evidence_status"], "WT_CONTROL_CANDIDATE")
        self.assertIsNone(cand["data_role"])

    def test_reactivity_layers_mask_not_zero_fill(self):
        rec = {
            "source_profile_index": 1,
            "raw_mutation_token": ["C13G"],
            "parsed_mutations": [
                {
                    "kind": "EXACT_SINGLE_SUBSTITUTION",
                    "edits": [
                        {"ref_allele": "C", "alt_allele": "G", "sequence_index_0_based": 12},
                    ],
                }
            ],
            "exact_mutation_evidence_status": "EXACT_REF_ALT_TOKEN_REF_VERIFIED_PROFILE_SEQUENCE_UNAVAILABLE",
            "resolved_annotations": {
                "modifier": {"resolved_values": ["p__"]},
                "temperature": {"resolved_values": ["20C"]},
                "chemical": {"resolved_values": ["Buffer"]},
                "experimentType": {"resolved_values": ["InVitro"]},
            },
        }
        seq = "A" * 12 + "C" + "A" * 20
        profile = {"reactivity": [0.1, None, 0.3], "reactivity_error": [0.01, 0.02, 0.03]}
        cand = _mod._canonicalize_profile(
            rec, profile, seq, 0, "ACC1", "abc123", "ACC1.rdat",
        )
        layers = cand["reactivity_layers"]
        self.assertEqual(layers["position_mask"], [1, 0, 1])
        self.assertEqual(layers["missing_positions"], [1])
        self.assertEqual(layers["missing_reason"], "MASKED_UNMEASURED")

    def test_condition_matching_strict(self):
        a = _condition()
        b = _condition()
        self.assertEqual(_mod._same_condition(a, b), "MATCHED_ALL_REQUIRED")
        b_t = _condition(temperature=("37C",))
        self.assertEqual(_mod._same_condition(a, b_t), "MISMATCH_TEMPERATURE")
        b_missing = _condition(modifier=())
        self.assertEqual(_mod._same_condition(a, b_missing), "MISSING_REQUIRED_FIELD")


class AuthorityPreflightTest(unittest.TestCase):
    def test_authority_validator_imports(self):
        # The validator must be present and importable.
        vp = _REPO_ROOT / "scripts/reactflow_delta/d1x_validate_authority.py"
        self.assertTrue(vp.is_file())
        spec = importlib.util.spec_from_file_location("d1x_validate_authority", vp)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertEqual(mod.validate.__name__, "validate")


if __name__ == "__main__":
    unittest.main(verbosity=2)