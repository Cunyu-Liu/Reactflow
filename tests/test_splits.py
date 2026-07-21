"""Tests for clan-disjoint splitting and leakage validation in reactflow.splits."""

import json

import pytest

from reactflow.constraints import pairs_to_matrix
from reactflow.splits import (
    SPLIT_NAMES,
    SplitAssignment,
    SplitManifest,
    SplitRecord,
    build_split_manifest,
    length_bucket_label,
    manifest_from_json,
    manifest_to_json,
    read_split_metadata_tsv,
    split_efold_cache_by_clan,
    validate_split_leakage,
)


def _clan_records(num_clans=12, per_clan=3, base_len=10):
    records = []
    for clan in range(num_clans):
        for k in range(per_clan):
            records.append(
                SplitRecord(
                    record_id=f"r{clan}_{k}",
                    length=base_len * (clan + 1),
                    clan=f"CL{clan:05d}",
                    cluster=f"clu{clan}_{k}",
                )
            )
    return records


def test_pairs_to_matrix_is_symmetric_and_rejects_bad_pairs():
    matrix = pairs_to_matrix([(0, 3), (1, 2)], 4)
    assert matrix[0][3] == 1 and matrix[3][0] == 1
    assert matrix[1][2] == 1 and matrix[2][1] == 1
    with pytest.raises(ValueError, match="diagonal"):
        pairs_to_matrix([(1, 1)], 4)
    with pytest.raises(ValueError, match="out of range"):
        pairs_to_matrix([(0, 9)], 4)
    with pytest.raises(ValueError, match="non-negative"):
        pairs_to_matrix([], -1)


def test_build_split_manifest_is_clan_disjoint_and_covers_all_records():
    records = _clan_records()
    manifest = build_split_manifest(records, seed=7)

    # every record is assigned exactly once
    assert sum(manifest.counts_by_split().values()) == len(records)
    # clan sets pairwise disjoint (validation would raise otherwise)
    validate_split_leakage(manifest)
    clans = manifest.clans_by_split()
    for i in range(len(SPLIT_NAMES)):
        for j in range(i + 1, len(SPLIT_NAMES)):
            assert not (clans[SPLIT_NAMES[i]] & clans[SPLIT_NAMES[j]])
    # a non-trivial novel holdout exists
    assert manifest.counts_by_split()["novel"] > 0


def test_build_split_manifest_is_deterministic_across_calls():
    records = _clan_records()
    a = build_split_manifest(records, seed=13)
    b = build_split_manifest(records, seed=13)
    assert a.assignments == b.assignments


def test_build_split_manifest_is_order_invariant():
    records = _clan_records()
    reversed_records = list(reversed(records))
    a = build_split_manifest(records, seed=3)
    b = build_split_manifest(reversed_records, seed=3)
    # same record -> same split irrespective of input ordering
    split_a = {x.record_id: x.split for x in a.assignments}
    split_b = {x.record_id: x.split for x in b.assignments}
    assert split_a == split_b


def test_clanless_records_become_singletons_and_never_leak():
    records = [
        SplitRecord(record_id="solo1", length=20, clan=None),
        SplitRecord(record_id="solo2", length=20, clan=None),
    ] + _clan_records(num_clans=6)
    manifest = build_split_manifest(records, seed=1)
    validate_split_leakage(manifest)
    solo_clans = {a.clan for a in manifest.assignments if a.record_id.startswith("solo")}
    assert solo_clans == {"singleton:solo1", "singleton:solo2"}


def test_empty_records_produce_empty_manifest():
    manifest = build_split_manifest([], seed=0)
    assert manifest.assignments == tuple()
    assert manifest.counts_by_split() == {name: 0 for name in SPLIT_NAMES}


def test_zero_novel_fraction_leaves_novel_empty():
    manifest = build_split_manifest(_clan_records(), novel_clan_fraction=0.0, seed=2)
    assert manifest.counts_by_split()["novel"] == 0


@pytest.mark.parametrize(
    "length, expected",
    [
        (10, "len_le_50"),
        (50, "len_le_50"),
        (51, "len_51_200"),
        (200, "len_51_200"),
        (201, "len_gt_200"),
    ],
)
def test_length_bucket_label_is_monotone_step(length, expected):
    assert length_bucket_label(length, (50, 200)) == expected


def test_length_bucket_label_edge_cases():
    assert length_bucket_label(100, ()) == "all"
    with pytest.raises(ValueError, match="strictly increasing"):
        length_bucket_label(10, (50, 50))


def test_invalid_fraction_configuration_raises():
    with pytest.raises(ValueError, match="exactly train/val/test"):
        build_split_manifest(_clan_records(), fractions={"train": 1.0})
    with pytest.raises(ValueError, match="positive"):
        build_split_manifest(_clan_records(), fractions={"train": 0.0, "val": 0.0, "test": 0.0})
    with pytest.raises(ValueError, match="novel_clan_fraction"):
        build_split_manifest(_clan_records(), novel_clan_fraction=1.0)


def test_validate_split_leakage_detects_clan_leak():
    bad = SplitManifest(
        assignments=(
            SplitAssignment("a", "CL1", None, 10, "b", "train"),
            SplitAssignment("b", "CL1", None, 10, "b", "test"),
        ),
        fractions={"train": 0.8, "val": 0.1, "test": 0.1},
        novel_clan_fraction=0.0,
        length_bucket_boundaries=(50, 200),
        seed=0,
    )
    with pytest.raises(ValueError, match="clan leakage"):
        validate_split_leakage(bad)


def test_validate_split_leakage_detects_cluster_spanning_splits():
    bad = SplitManifest(
        assignments=(
            SplitAssignment("a", "CLa", "clu9", 10, "b", "train"),
            SplitAssignment("b", "CLb", "clu9", 10, "b", "test"),
        ),
        fractions={"train": 0.8, "val": 0.1, "test": 0.1},
        novel_clan_fraction=0.0,
        length_bucket_boundaries=(50, 200),
        seed=0,
    )
    with pytest.raises(ValueError, match="spans splits"):
        validate_split_leakage(bad)


def test_manifest_json_round_trip_and_revalidates(tmp_path):
    manifest = build_split_manifest(_clan_records(), seed=5)
    path = tmp_path / "split_manifest.json"
    manifest_to_json(manifest, path)

    loaded = manifest_from_json(path)
    assert set(loaded.assignments) == set(manifest.assignments)
    assert loaded.seed == manifest.seed
    assert loaded.novel_clan_fraction == manifest.novel_clan_fraction


def test_manifest_from_json_rejects_leaky_file(tmp_path):
    path = tmp_path / "leaky.json"
    path.write_text(
        '{"fractions": {"train": 0.8, "val": 0.1, "test": 0.1},'
        ' "novel_clan_fraction": 0.0, "length_bucket_boundaries": [50, 200], "seed": 0,'
        ' "assignments": ['
        '{"record_id": "a", "clan": "CL1", "cluster": null, "length": 10,'
        ' "length_bucket": "len_le_50", "split": "train"},'
        '{"record_id": "b", "clan": "CL1", "cluster": null, "length": 10,'
        ' "length_bucket": "len_le_50", "split": "test"}]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="clan leakage"):
        manifest_from_json(path)


def test_counts_by_bucket_partitions_records():
    manifest = build_split_manifest(_clan_records(), seed=9)
    by_bucket = manifest.counts_by_bucket()
    total = sum(sum(buckets.values()) for buckets in by_bucket.values())
    assert total == len(manifest.assignments)


def _write_cache(path, families=("CL0", "CL1", "CL2", "CL3")):
    rows = []
    for index, family in enumerate(families):
        rows.append(
            {
                "source_id": f"r{index}",
                "family": family,
                "sequence": "GGGAAACCC",
                "pairs": [[0, 8], [1, 7], [2, 6]],
                "probe": "2A3",
                "reactivity": [0.0 for _ in range(9)],
                "reactivity_source": "structure_forward_proxy",
            }
        )
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    return rows


def test_split_efold_cache_by_clan_writes_manifest_and_split_files(tmp_path):
    cache = tmp_path / "cache.jsonl"
    rows = _write_cache(cache, families=("CL0", "CL0", "CL1", "CL2", "CL3", "CL4"))
    out = tmp_path / "split"

    summary = split_efold_cache_by_clan(
        [cache],
        out,
        novel_clan_fraction=0.2,
        length_bucket_boundaries=(9,),
        seed=3,
    )
    manifest = manifest_from_json(out / "split_manifest.json")

    assert summary.input_records == len(rows)
    assert summary.manifest_path == str(out / "split_manifest.json")
    assert set(summary.split_paths) == set(SPLIT_NAMES)
    assert sum(summary.counts_by_split.values()) == len(rows)
    validate_split_leakage(manifest)
    emitted = 0
    for split in SPLIT_NAMES:
        path = out / f"{split}.jsonl"
        assert path.exists()
        emitted += len([line for line in path.read_text(encoding="utf-8").splitlines() if line])
    assert emitted == len(rows)


def test_split_efold_cache_uses_metadata_tsv_override(tmp_path):
    cache = tmp_path / "cache.jsonl"
    _write_cache(cache, families=(None, None, None, None))
    metadata = tmp_path / "metadata.tsv"
    metadata.write_text(
        "record_id\tclan\tcluster\n"
        "r0\tCL_A\tcluster_a\n"
        "r1\tCL_A\tcluster_b\n"
        "r2\tCL_B\tcluster_c\n"
        "r3\tCL_C\tcluster_d\n",
        encoding="utf-8",
    )

    parsed = read_split_metadata_tsv(metadata)
    summary = split_efold_cache_by_clan(
        [cache],
        tmp_path / "split_meta",
        metadata_tsv=metadata,
        novel_clan_fraction=0.0,
        seed=1,
    )
    manifest = manifest_from_json(tmp_path / "split_meta" / "split_manifest.json")

    assert parsed["r0"] == ("CL_A", "cluster_a")
    assert summary.metadata_records == 4
    assert {assignment.clan for assignment in manifest.assignments} == {"CL_A", "CL_B", "CL_C"}
    assert {assignment.cluster for assignment in manifest.assignments} == {
        "cluster_a",
        "cluster_b",
        "cluster_c",
        "cluster_d",
    }
