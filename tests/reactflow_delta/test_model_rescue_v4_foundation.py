from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.reactflow_delta import precompute_model_rescue_v4_rnafm as F


def test_sequence_loader_requests_only_frozen_outcome_blind_columns(monkeypatch) -> None:
    requested = {}

    def fake_read_csv(path, *, usecols):
        requested["usecols"] = usecols
        return pd.DataFrame(
            {
                "id": ["P01_m1_wt", "P01_m1_mm_1_C_A"],
                "puzzle": ["P01", "P01"],
                "method": ["m1", "m1"],
                "sequence": ["ACTG", "AATG"],
            }
        )

    monkeypatch.setattr(F.pd, "read_csv", fake_read_csv)
    entries = F.load_outcome_blind_sequences("unused.csv")
    assert requested["usecols"] == list(F.OUTCOME_BLIND_COLUMNS)
    assert [entry.sequence for entry in entries] == ["ACUG", "AAUG"]
    assert entries[1].row_id == "P01_m1_mm_1_C_A"


class _FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, tokens, repr_layers):
        batch, token_length = tokens.shape
        representation = torch.arange(
            batch * token_length * F.REPRESENTATION_WIDTH, dtype=torch.float32
        ).reshape(batch, token_length, F.REPRESENTATION_WIDTH)
        return {"representations": {F.REPRESENTATION_LAYER: representation}}


def _fake_converter(data):
    length = max(len(sequence) for _, sequence in data) + 2
    return None, None, torch.zeros(len(data), length, dtype=torch.long)


def test_official_loader_requires_and_passes_explicit_checkpoint(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "RNA-FM_pretrained.pth"
    checkpoint.write_bytes(b"fixture")
    model = _FakeModel()
    calls = []
    alphabet = SimpleNamespace(get_batch_converter=lambda: _fake_converter)
    fake_fm = SimpleNamespace(
        pretrained=SimpleNamespace(
            rna_fm_t12=lambda location: calls.append(location) or (model, alphabet)
        )
    )
    monkeypatch.setitem(sys.modules, "fm", fake_fm)

    loaded, converter = F.load_official_rnafm(checkpoint)

    assert loaded is model
    assert converter is _fake_converter
    assert calls == [str(checkpoint.resolve())]
    assert loaded.training is False
    assert all(parameter.requires_grad is False for parameter in loaded.parameters())


def test_official_loader_fails_closed_when_checkpoint_is_absent(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="official RNA-FM checkpoint is absent"):
        F.load_official_rnafm(tmp_path / "missing.pth")


def test_embedding_extractor_trims_special_tokens_and_foundation_is_frozen() -> None:
    entries = [
        F.SequenceEntry("a", "P01", "m1", "ACGU"),
        F.SequenceEntry("b", "P01", "m1", "ACG"),
    ]
    model = _FakeModel()
    F.freeze_foundation(model)
    assert model.training is False
    assert all(parameter.requires_grad is False for parameter in model.parameters())
    rows = F.extract_batch_embeddings(model, _fake_converter, entries, "cpu")
    assert rows[0].shape == (4, F.REPRESENTATION_WIDTH)
    assert rows[1].shape == (3, F.REPRESENTATION_WIDTH)


def test_cache_manifest_discloses_columns_and_unknown_sequence_overlap(tmp_path) -> None:
    pytest.importorskip("h5py")
    entries = [
        F.SequenceEntry("wt", "P01", "m1", "ACGU"),
        F.SequenceEntry("mut", "P01", "m1", "AUGU"),
    ]
    rows = [[
        np.ones((4, F.REPRESENTATION_WIDTH), dtype=np.float32),
        np.zeros((4, F.REPRESENTATION_WIDTH), dtype=np.float32),
    ]]
    cache = tmp_path / "cache.h5"
    manifest_path = tmp_path / "manifest.json"
    manifest = F.write_cache(
        entries=entries,
        embeddings=rows,
        cache_path=cache,
        manifest_path=manifest_path,
        model_location=tmp_path / "RNA-FM_pretrained.pth",
    )
    assert cache.exists()
    assert manifest["csv_columns_read"] == list(F.OUTCOME_BLIND_COLUMNS)
    assert manifest["mutant_outcome_columns_loaded"] is False
    assert manifest["exact_openknot_pretraining_overlap"] == "UNKNOWN_NOT_ASSERTED"
    assert manifest["official_checkpoint_source"] == F.OFFICIAL_CHECKPOINT_SOURCE
    assert manifest["checkpoint_path_used"].endswith("RNA-FM_pretrained.pth")
    assert json.loads(manifest_path.read_text())["n_sequences"] == 2
