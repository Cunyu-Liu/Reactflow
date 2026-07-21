import importlib.util
import json
import sys
from pathlib import Path

import pytest

from reactflow.npio import NdArray

# The exporter lives under scripts/ and is deliberately NOT importable as part
# of the reactflow package (it must never pull torch into the library import
# graph).  Load it by file path so its standard-library surface can be tested.
_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_frozen_features.py"
_spec = importlib.util.spec_from_file_location("export_frozen_features", _SCRIPT)
export_frozen_features = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(export_frozen_features)  # type: ignore[union-attr]

DryRunBackend = export_frozen_features.DryRunBackend
tokenize = export_frozen_features.tokenize
read_sequences = export_frozen_features.read_sequences
build_provenance = export_frozen_features.build_provenance
export_sharded = export_frozen_features.export_sharded


def test_tokenize_maps_bases_and_folds_t_to_u():
    assert tokenize("ACGU") == [0, 1, 2, 3]
    assert tokenize("acgt") == [0, 1, 2, 3]  # lowercase + T->U
    with pytest.raises(ValueError, match="non-ACGU"):
        tokenize("ACGN")


def test_read_sequences_parses_ids_and_families(tmp_path):
    path = tmp_path / "seqs.jsonl"
    path.write_text(
        '{"id": "r1", "sequence": "ACGU", "family": "CL1"}\n'
        "\n"  # blank line skipped
        '{"sequence": "GGCC", "clan": "CL2"}\n'
        '{"record_id": "r3", "sequence": "AUAU"}\n',
        encoding="utf-8",
    )
    records = read_sequences(path)
    assert len(records) == 3
    assert records[0] == ("r1", "ACGU", "CL1")
    assert records[1][0] == "seq000002"  # synthesized id from line number
    assert records[1][2] == "CL2"  # 'clan' alias
    assert records[2] == ("r3", "AUAU", None)


def test_read_sequences_respects_limit(tmp_path):
    path = tmp_path / "seqs.jsonl"
    path.write_text("".join(f'{{"sequence": "ACGU"}}\n' for _ in range(5)), encoding="utf-8")
    assert len(read_sequences(path, limit=2)) == 2


def test_dry_run_backend_shapes_and_optional_arrays():
    backend = DryRunBackend(d_single=8, d_pair=4, n_probe=2, seed=0)
    arrays = backend.encode("rec", "ACGUAC")
    assert arrays["single"].shape == (6, 8)
    assert arrays["pair"].shape == (6, 6, 4)
    assert arrays["react_logits"].shape == (6, 2)

    minimal = DryRunBackend(d_single=8, d_pair=None, n_probe=None)
    only = minimal.encode("rec", "ACGU")
    assert set(only) == {"single"}


def test_dry_run_backend_is_deterministic_and_order_independent():
    a = DryRunBackend(d_single=4, d_pair=None, n_probe=None, seed=42)
    b = DryRunBackend(d_single=4, d_pair=None, n_probe=None, seed=42)
    # same id + seed -> identical features regardless of call order
    fa = a.encode("x", "ACGU")["single"]
    _ = a.encode("y", "GGGG")  # consume more of the stream
    fb = b.encode("x", "ACGU")["single"]
    assert list(fa.data) == list(fb.data)
    # different seed -> different features
    c = DryRunBackend(d_single=4, d_pair=None, n_probe=None, seed=7)
    assert list(c.encode("x", "ACGU")["single"].data) != list(fa.data)


def test_dry_run_backend_is_not_real():
    assert DryRunBackend.is_real is False
    assert DryRunBackend.weights_sha256 == ""


def test_cli_dry_run_writes_labelled_shard(tmp_path, capsys):
    seqs = tmp_path / "seqs.jsonl"
    seqs.write_text('{"id": "r1", "sequence": "ACGUACGU", "family": "CL1"}\n', encoding="utf-8")
    out = tmp_path / "shard"
    rc = export_frozen_features.main(
        [
            "--sequences", str(seqs),
            "--out", str(out),
            "--backend", "dry-run",
            "--d-single", "8",
            "--d-pair", "4",
            "--n-probe", "2",
        ]
    )
    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["backend"] == "dry-run"
    assert summary["records"] == 1
    assert summary["weights_sha256"] == ""

    from reactflow.frozen import read_frozen_shard

    shard = read_frozen_shard(out)
    # honesty red line: the shard is unmistakably stamped as a dry run
    assert "DRY RUN" in shard.provenance.notes
    assert shard.provenance.weights_sha256 == ""
    assert shard.records[0].record_id == "r1"


def test_cli_dry_run_can_omit_optional_arrays(tmp_path, capsys):
    seqs = tmp_path / "seqs.jsonl"
    seqs.write_text('{"sequence": "ACGU"}\n', encoding="utf-8")
    out = tmp_path / "shard"
    rc = export_frozen_features.main(
        ["--sequences", str(seqs), "--out", str(out), "--d-single", "6", "--d-pair", "0", "--n-probe", "0"]
    )
    assert rc == 0
    capsys.readouterr()
    from reactflow.frozen import read_frozen_shard

    shard = read_frozen_shard(out)
    assert shard.records[0].pair() is None
    assert shard.records[0].react_logits() is None


def test_cli_dry_run_can_write_sharded_export(tmp_path, capsys):
    seqs = tmp_path / "seqs.jsonl"
    seqs.write_text(
        '{"id": "r1", "sequence": "ACGU"}\n'
        '{"id": "r2", "sequence": "CCCC"}\n'
        '{"id": "r3", "sequence": "GGGG"}\n',
        encoding="utf-8",
    )
    out = tmp_path / "sharded"

    rc = export_frozen_features.main(
        [
            "--sequences", str(seqs),
            "--out", str(out),
            "--backend", "dry-run",
            "--d-single", "6",
            "--d-pair", "0",
            "--n-probe", "0",
            "--shard-size", "2",
        ]
    )
    summary = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert summary["records"] == 3
    assert summary["shard_count"] == 2
    assert (out / "sharded_manifest.json").exists()

    from reactflow.features import load_frozen_features

    lookup = load_frozen_features(out)
    assert len(lookup) == 3
    assert lookup.has("ACGU")
    assert lookup.single_rows("GGGG") is not None


def test_sharded_resume_skips_complete_shards_and_rewrites_missing(tmp_path):
    sequences = [
        ("r1", "ACGU", "CL1"),
        ("r2", "CCCC", "CL1"),
        ("r3", "GGGG", "CL2"),
    ]
    provenance = build_provenance(
        model_name="RibonanzaNet2",
        model_version="alpha-v1",
        weights_sha256="abc123",
        d_single=6,
        d_pair=None,
        n_probe=None,
        notes="test",
    )
    out = tmp_path / "sharded"
    export_sharded(
        sequences,
        DryRunBackend(d_single=6, d_pair=None, n_probe=None, seed=0),
        provenance,
        out,
        shard_size=2,
    )
    first_hash = json.loads((out / "shard_00000" / "provenance.json").read_text())["content_sha256"]
    (out / "shard_00001" / "features.npz").unlink()

    class CountingBackend(DryRunBackend):
        def __init__(self):
            super().__init__(d_single=6, d_pair=None, n_probe=None, seed=0)
            self.calls = 0

        def encode(self, record_id, sequence):
            self.calls += 1
            return super().encode(record_id, sequence)

    backend = CountingBackend()
    manifest = export_sharded(
        sequences,
        backend,
        provenance,
        out,
        shard_size=2,
        resume=True,
    )

    assert backend.calls == 1
    assert manifest["record_count"] == 3
    assert manifest["shard_count"] == 2
    assert manifest["shards"][0]["content_sha256"] == first_hash
    assert (out / "shard_00001" / "features.npz").exists()


def test_sharded_export_uses_backend_batch_path(tmp_path):
    sequences = [
        ("r1", "ACGU", None),
        ("r2", "CCCC", None),
    ]
    provenance = build_provenance(
        model_name="RibonanzaNet2",
        model_version="alpha-v1",
        weights_sha256="abc123",
        d_single=2,
        d_pair=None,
        n_probe=None,
        notes="test",
    )

    class BatchOnlyBackend:
        def __init__(self):
            self.calls = []

        def encode(self, record_id, sequence):
            raise AssertionError("encode should not be used when batch_size > 1")

        def encode_many(self, records, *, batch_size):
            self.calls.append(([record_id for record_id, _, _ in records], batch_size))
            return [
                {"single": NdArray.from_nested([[float(row), 0.0] for row in range(len(sequence))], kind="float32")}
                for _, sequence, _ in records
            ]

    backend = BatchOnlyBackend()
    manifest = export_sharded(
        sequences,
        backend,
        provenance,
        tmp_path / "sharded",
        shard_size=2,
        batch_size=4,
    )

    assert backend.calls == [(["r1", "r2"], 4)]
    assert manifest["record_count"] == 2


def test_cli_torch_backend_requires_paths(tmp_path):
    seqs = tmp_path / "seqs.jsonl"
    seqs.write_text('{"sequence": "ACGU"}\n', encoding="utf-8")
    with pytest.raises(SystemExit, match="requires"):
        export_frozen_features.main(
            ["--sequences", str(seqs), "--out", str(tmp_path / "s"), "--backend", "torch"]
        )


def test_missing_matplotlib_gets_import_only_stub(monkeypatch, tmp_path):
    monkeypatch.delitem(sys.modules, "matplotlib", raising=False)
    monkeypatch.delitem(sys.modules, "matplotlib.pyplot", raising=False)
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "matplotlib":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    export_frozen_features._install_import_only_matplotlib_stub()

    network_py = tmp_path / "Network.py"
    network_py.write_text(
        "import matplotlib.pyplot as plt\n"
        "VALUE = 7\n",
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location("stubbed_network_for_test", network_py)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]

    assert module.VALUE == 7
    assert hasattr(module.plt, "__file__") is False
    with pytest.raises(RuntimeError, match="import-only matplotlib.pyplot stub"):
        module.plt.figure()
