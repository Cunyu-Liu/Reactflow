from __future__ import annotations

import unittest
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from reactflow.delta.d0x import (
    D0XContractError,
    audit_rdat_candidate_profiles,
    parse_mutation_value,
    strict_seqpos_to_indices,
)


FIXTURES = REPO_ROOT / "tests" / "fixtures" / "reactflow_delta" / "d0x"


class D0XContractParserTests(unittest.TestCase):
    def test_exact_c13g_and_profile_override_are_losslessly_retained(self) -> None:
        result = audit_rdat_candidate_profiles(
            FIXTURES / "exact_profile_override.rdat",
            source_accession="FIXTURE_EXACT",
        )
        self.assertEqual(len(result["profile_records"]), 2)
        mutant = result["profile_records"][1]
        self.assertEqual(mutant["raw_mutation_token"], ["C13G"])
        self.assertEqual(mutant["ref_allele"], "C")
        self.assertEqual(mutant["alt_allele"], "G")
        self.assertEqual(
            mutant["exact_mutation_evidence_status"],
            "EXACT_REF_ALT_PROFILE_SEQUENCE_VERIFIED",
        )
        self.assertEqual(mutant["provisional_data_role"], "PRIMARY_EXACT_DELTA")
        self.assertIsNone(mutant["data_role"])
        condition = mutant["resolved_annotations"]["condition"]
        self.assertEqual(condition["construct_values"], ["construct_condition"])
        self.assertEqual(condition["profile_values"], ["profile_override"])
        self.assertEqual(condition["resolved_values"], ["profile_override"])
        self.assertEqual(condition["resolution"], "PROFILE_OVERRIDE")
        self.assertEqual(condition["profile_raw_tokens"][0]["line_number"], 11)

    def test_latent_multi_invalid_missing_and_orphans_are_all_accounted(self) -> None:
        result = audit_rdat_candidate_profiles(
            FIXTURES / "mixed_dispositions_and_orphans.rdat",
            source_accession="FIXTURE_MIXED",
        )
        records = {row["source_profile_index"]: row for row in result["profile_records"]}
        self.assertEqual(records[2]["provisional_data_role"], "AUXILIARY_LATENT_ALT")
        self.assertIsNone(records[2]["alt_allele"])
        self.assertEqual(records[3]["provisional_data_role"], "RESCUE_MULTI_EDIT")
        self.assertEqual(records[4]["exclusion_reason"], "INVALID_MUTATION_TOKEN")
        self.assertEqual(records[5]["exclusion_reason"], "MISSING_MUTATION_ANNOTATION")
        self.assertEqual(records[9]["exclusion_reason"], "MISSING_MUTATION_ANNOTATION")
        accounting = result["profile_accounting"]
        self.assertEqual(accounting["all_indexed_profile_indices"], list(range(1, 10)))
        self.assertEqual(accounting["orphan_indices_by_kind"]["ANNOTATION_DATA"], [6])
        self.assertEqual(accounting["orphan_indices_by_kind"]["SEQUENCE"], [7])
        self.assertEqual(accounting["orphan_indices_by_kind"]["REACTIVITY_ERROR"], [8])
        self.assertEqual(accounting["missing_profile_annotation_indices"], [9])
        self.assertEqual(accounting["accounting_equation"]["silent_drop_count"], 0)

    def test_malformed_seqpos_fails_instead_of_silent_skip(self) -> None:
        with self.assertRaisesRegex(D0XContractError, "malformed SEQPOS"):
            audit_rdat_candidate_profiles(
                FIXTURES / "malformed_seqpos.rdat",
                source_accession="FIXTURE_BAD_SEQPOS",
            )
        with self.assertRaises(D0XContractError):
            strict_seqpos_to_indices(["A1", "not-a-coordinate", "G3"])

    def test_duplicate_profile_index_fails_closed(self) -> None:
        with self.assertRaisesRegex(D0XContractError, "duplicate REACTIVITY index 1"):
            audit_rdat_candidate_profiles(
                FIXTURES / "duplicate_profile_index.rdat",
                source_accession="FIXTURE_DUPLICATE",
            )

    def test_reference_and_alternate_mismatch_are_not_exact_evidence(self) -> None:
        ref_mismatch = parse_mutation_value(
            "A13G",
            header_sequence="AAAAAAAAAAAACAAA",
            offset=0,
            profile_sequence="AAAAAAAAAAAAGAAA",
        )
        self.assertIn("reference_allele_mismatch", ref_mismatch["issues"])
        alt_mismatch = parse_mutation_value(
            "C13U",
            header_sequence="AAAAAAAAAAAACAAA",
            offset=0,
            profile_sequence="AAAAAAAAAAAAGAAA",
        )
        self.assertIn("alternate_allele_profile_sequence_mismatch", alt_mismatch["issues"])


if __name__ == "__main__":
    unittest.main()
