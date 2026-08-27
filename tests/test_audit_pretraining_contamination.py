"""Unit tests for ``scripts/audit_pretraining_contamination.py`` (C1-1 Task 5)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# Import from the script module (scripts/ is not a package, so use importlib).
_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "audit_pretraining_contamination.py"
_spec = importlib.util.spec_from_file_location("audit_pretraining_contamination", _SCRIPT_PATH)
audit_module = importlib.util.module_from_spec(_spec)
sys.modules["audit_pretraining_contamination"] = audit_module
_spec.loader.exec_module(audit_module)


class TestKnownModels:
    def test_four_models(self):
        assert len(audit_module.KNOWN_MODELS) == 4

    def test_rinalmo_present(self):
        names = [m.name for m in audit_module.KNOWN_MODELS]
        assert "RiNALMo" in names

    def test_rna_fm_present(self):
        names = [m.name for m in audit_module.KNOWN_MODELS]
        assert "RNA-FM" in names

    def test_ernie_rna_present(self):
        names = [m.name for m in audit_module.KNOWN_MODELS]
        assert "ERNIE-RNA" in names

    def test_ribonanza_net2_present(self):
        names = [m.name for m in audit_module.KNOWN_MODELS]
        assert "RibonanzaNet2" in names

    def test_models_have_known_rna_databases(self):
        for m in audit_module.KNOWN_MODELS:
            assert len(m.known_rna_databases) > 0, f"{m.name} missing known_rna_databases"

    def test_rinalmo_audit_uses_the_published_giga_asset(self):
        model = next(m for m in audit_module.KNOWN_MODELS if m.name == "RiNALMo")
        assert model.version == "giga-v1"
        assert model.paper == (
            "Penić et al., Nature Communications 2025 "
            "(https://doi.org/10.1038/s41467-025-60872-5)"
        )
        assert model.weights_url == (
            "https://zenodo.org/records/15043668/files/"
            "rinalmo_giga_pretrained.pt"
        )

    def test_rna_fm_audit_uses_the_pinned_v4_checkpoint(self):
        model = next(m for m in audit_module.KNOWN_MODELS if m.name == "RNA-FM")
        assert model.version == "rna_fm_t12"
        assert "12 layers, 640 dim" in model.training_data_description
        assert model.weights_url == (
            "https://huggingface.co/cuhkaih/rnafm/resolve/"
            "91d4a46d28d8054a7b429955e8fc0c253ba0afd6/"
            "RNA-FM_pretrained.pth"
        )


class TestSplitSequences:
    def test_default_empty(self):
        s = audit_module.SplitSequences()
        assert s.test_count == 0
        assert s.novel_count == 0
        assert len(s.test_sequences) == 0
        assert len(s.novel_sequences) == 0
        assert len(s.test_families) == 0
        assert len(s.novel_families) == 0


class TestComputeWeightHash:
    def test_none_path(self):
        sha, status = audit_module.compute_weight_hash(None)
        assert sha == "not_downloaded"
        assert status == "not_downloaded"

    def test_nonexistent_path(self):
        sha, status = audit_module.compute_weight_hash("/nonexistent/weights.pt")
        assert sha == "not_downloaded"
        assert "not found" in status

    def test_existing_file(self, tmp_path: Path):
        path = tmp_path / "weights.pt"
        path.write_bytes(b"model weights content")
        sha, status = audit_module.compute_weight_hash(str(path))
        assert status == "computed"
        assert len(sha) == 64  # SHA-256 hex digest

    def test_known_content(self, tmp_path: Path):
        path = tmp_path / "weights.pt"
        path.write_bytes(b"hello world")
        sha, status = audit_module.compute_weight_hash(str(path))
        assert status == "computed"
        # Known SHA-256 of "hello world"
        assert sha == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"


class TestAuditModel:
    def test_rinalmo_is_contaminated(self):
        """RiNALMo trained on RNAcentral which includes Rfam; should be contaminated."""
        spec = audit_module.KNOWN_MODELS[0]  # RiNALMo
        assert spec.name == "RiNALMo"
        split_seqs = audit_module.SplitSequences(
            test_families={"RF00001", "RF00002"},
            novel_families={"RF00003"},
        )
        result = audit_module.audit_model(spec, split_seqs)
        assert result.contamination_status == "contaminated"
        # Should report exact overlap (database-level)
        assert isinstance(result.exact_overlap_test, list)
        assert "efold_train" in result.exact_overlap_test
        # Family overlap should be reported
        assert "all" in str(result.family_overlap_test)

    def test_rna_fm_is_contaminated(self):
        """RNA-FM trained on RNAcentral; should be contaminated."""
        spec = next(s for s in audit_module.KNOWN_MODELS if s.name == "RNA-FM")
        split_seqs = audit_module.SplitSequences(
            test_families={"RF00001"},
        )
        result = audit_module.audit_model(spec, split_seqs)
        assert result.contamination_status == "contaminated"

    def test_ernie_rna_is_contaminated(self):
        """ERNIE-RNA trained on RNAcentral; should be contaminated."""
        spec = next(s for s in audit_module.KNOWN_MODELS if s.name == "ERNIE-RNA")
        split_seqs = audit_module.SplitSequences(
            test_families={"RF00001"},
        )
        result = audit_module.audit_model(spec, split_seqs)
        assert result.contamination_status == "contaminated"

    def test_ribonanza_net2_is_unknown_or_contaminated(self):
        """RibonanzaNet2 trained on Ribonanza + bpRNA + RNAStrAlign.

        bpRNA and RNAStrAlign overlap with efold_train, so it should be
        contaminated.
        """
        spec = next(s for s in audit_module.KNOWN_MODELS if s.name == "RibonanzaNet2")
        split_seqs = audit_module.SplitSequences(
            test_families={"RF00001"},
        )
        result = audit_module.audit_model(spec, split_seqs)
        # bpRNA and RNAStrAlign overlap with efold_train
        assert result.contamination_status == "contaminated"

    def test_weight_hash_not_downloaded_by_default(self):
        """By default, weight_hash should be 'not_downloaded' for all models."""
        spec = audit_module.KNOWN_MODELS[0]
        assert spec.weights_path is None
        split_seqs = audit_module.SplitSequences()
        result = audit_module.audit_model(spec, split_seqs)
        assert result.weight_hash == "not_downloaded"
        assert result.weight_hash_computation_status != "computed"
        assert "download" in result.notes.lower()

    def test_weight_hash_computed_when_path_provided(self, tmp_path: Path):
        """When weights_path is set and file exists, weight_hash should be computed."""
        weights_path = tmp_path / "weights.pt"
        weights_path.write_bytes(b"fake weights")
        spec = audit_module.PretrainedModelSpec(
            name="TestModel",
            version="1.0",
            paper="test",
            training_data_description="test",
            training_data_available=True,
            training_data_url="http://example.com",
            weights_url="http://example.com/weights",
            known_rna_databases=["RNAcentral"],
            weights_path=str(weights_path),
        )
        split_seqs = audit_module.SplitSequences()
        result = audit_module.audit_model(spec, split_seqs)
        assert result.weight_hash_computation_status == "computed"
        assert len(result.weight_hash) == 64

    def test_unavailable_training_data_marks_unknown(self):
        spec = audit_module.PretrainedModelSpec(
            name="TestModel",
            version="1.0",
            paper="test",
            training_data_description="test",
            training_data_available=False,
            training_data_url=None,
            weights_url=None,
            known_rna_databases=[],
        )
        split_seqs = audit_module.SplitSequences()
        result = audit_module.audit_model(spec, split_seqs)
        assert result.contamination_status == "unknown_contamination"
        assert result.exact_overlap_test == "not_computed"


class TestDatabaseToReactflowOverlap:
    def test_rnacentral_includes_efold_train(self):
        assert "efold_train" in audit_module.DATABASE_TO_REACTFLOW_OVERLAP["RNAcentral"]

    def test_rfam_includes_efold_train(self):
        assert "efold_train" in audit_module.DATABASE_TO_REACTFLOW_OVERLAP["Rfam"]

    def test_bprna_includes_efold_train(self):
        assert "efold_train" in audit_module.DATABASE_TO_REACTFLOW_OVERLAP["bpRNA"]

    def test_rnastralign_includes_efold_train(self):
        assert "efold_train" in audit_module.DATABASE_TO_REACTFLOW_OVERLAP["RNAStrAlign"]

    def test_ribonanza_empty(self):
        # Ribonanza has no documented ReactFlow overlap (would need verification).
        assert audit_module.DATABASE_TO_REACTFLOW_OVERLAP["Ribonanza"] == []
