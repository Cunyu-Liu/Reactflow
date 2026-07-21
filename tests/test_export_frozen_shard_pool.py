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


pool = _load_script("export_frozen_shard_pool")
rebuild_manifest = _load_script("rebuild_frozen_sharded_manifest")


def test_pool_exports_missing_shards_and_skips_completed(tmp_path):
    seqs = tmp_path / "seqs.jsonl"
    seqs.write_text(
        '{"id": "r1", "sequence": "ACGU"}\n'
        '{"id": "r2", "sequence": "CCCC"}\n'
        '{"id": "r3", "sequence": "GGGG"}\n'
        '{"id": "r4", "sequence": "UUUU"}\n',
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
        "--batch-size", "4",
        "--stale-claim-seconds", "1",
    ]

    assert pool.main(common + ["--worker-id", "w1"]) == 0
    assert pool.main(common + ["--worker-id", "w2"]) == 0
    manifest = rebuild_manifest.rebuild_manifest(out, shard_size=2)

    assert manifest["record_count"] == 4
    assert manifest["shard_count"] == 2

    from reactflow.features import load_frozen_features

    lookup = load_frozen_features(out)
    assert len(lookup) == 4
    assert lookup.has("ACGU")
    assert lookup.has("UUUU")
