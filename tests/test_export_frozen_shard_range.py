import importlib.util
import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))


def _load_script(name):
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


export_range = _load_script("export_frozen_shard_range")
rebuild_manifest = _load_script("rebuild_frozen_sharded_manifest")


def test_range_export_and_manifest_rebuild_round_trip(tmp_path):
    seqs = tmp_path / "seqs.jsonl"
    seqs.write_text(
        '{"id": "r1", "sequence": "ACGU"}\n'
        '{"id": "r2", "sequence": "CCCC"}\n'
        '{"id": "r3", "sequence": "GGGG"}\n',
        encoding="utf-8",
    )
    out = tmp_path / "sharded"

    common = [
        "--sequences", str(seqs),
        "--out", str(out),
        "--backend", "dry-run",
        "--d-single", "6",
        "--d-pair", "0",
        "--n-probe", "0",
        "--shard-size", "2",
    ]
    assert export_range.main(common + ["--shard-start", "1", "--shard-end", "2"]) == 0
    assert not (out / "sharded_manifest.json").exists()
    assert export_range.main(common + ["--shard-start", "0", "--shard-end", "1"]) == 0

    manifest = rebuild_manifest.rebuild_manifest(out, shard_size=2)

    assert manifest["record_count"] == 3
    assert manifest["shard_count"] == 2
    assert [row["path"] for row in manifest["shards"]] == ["shard_00000", "shard_00001"]

    from reactflow.features import load_frozen_features

    lookup = load_frozen_features(out)
    assert len(lookup) == 3
    assert lookup.has("ACGU")
    assert lookup.has("GGGG")
