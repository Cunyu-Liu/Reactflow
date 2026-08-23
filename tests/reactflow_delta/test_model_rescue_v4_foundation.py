from __future__ import annotations

import json

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
    )
    assert cache.exists()
    assert manifest["csv_columns_read"] == list(F.OUTCOME_BLIND_COLUMNS)
    assert manifest["mutant_outcome_columns_loaded"] is False
    assert manifest["exact_openknot_pretraining_overlap"] == "UNKNOWN_NOT_ASSERTED"
    assert json.loads(manifest_path.read_text())["n_sequences"] == 2
