from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

import scripts.reactflow_delta.assemble_independent_rnet_distill_formal as assembler
from scripts.reactflow_delta.merge_independent_rnet_distill import (
    EXPECTED_MERGED_FIELDS,
    MERGE_INTEGRITY,
    SCHEMA as MERGED_SCHEMA,
    STATUS as MERGED_STATUS,
)
from scripts.reactflow_delta.run_independent_rnet_distill_downstream import (
    EXPECTED_PREDICTION_FIELDS,
    PREDICTION_SCHEMA,
)


def _source_payload(fold: int, seed: int) -> dict[str, np.ndarray]:
    keys = np.asarray([f"P{fold}:a", f"P{fold}:b"], dtype=object)
    n_rows = len(keys)
    arrays: dict[str, np.ndarray] = {}
    for name in EXPECTED_PREDICTION_FIELDS:
        if name == "schema_version":
            arrays[name] = np.asarray(PREDICTION_SCHEMA)
        elif name in {"keys", "biological_scoring_key"}:
            arrays[name] = keys.copy()
        elif name == "outer_fold":
            arrays[name] = np.full(n_rows, fold, dtype=np.int64)
        elif name == "seed":
            arrays[name] = np.full(n_rows, seed, dtype=np.int64)
        elif name == "registered_status":
            arrays[name] = np.full(n_rows, "covered", dtype=object)
        elif name == "feature41_point":
            arrays[name] = np.asarray([0.1, 0.2], dtype=np.float64)
        elif name == "v8_point":
            arrays[name] = np.asarray([0.3, 0.4], dtype=np.float64)
        elif name == "candidate_point":
            arrays[name] = np.asarray([seed, seed + 0.5], dtype=np.float64)
        elif name == "null_point":
            arrays[name] = np.asarray([seed + 10.0, seed + 10.5], dtype=np.float64)
        elif name.endswith("_weights"):
            arrays[name] = np.tile(
                np.asarray([[0.25, 0.75]], dtype=np.float64), (n_rows, 1)
            )
        elif name.endswith("_locations"):
            if name.startswith("feature41_"):
                offset = 1.0
            elif name.startswith("historical_v10_"):
                offset = 2.0
            elif name.startswith("candidate_"):
                offset = float(seed)
            else:
                offset = float(seed + 5)
            arrays[name] = np.tile(
                np.asarray([[offset, offset + 1.0]], dtype=np.float64),
                (n_rows, 1),
            )
        elif name.endswith("_scales"):
            arrays[name] = np.tile(
                np.asarray([[0.5, 1.0]], dtype=np.float64), (n_rows, 1)
            )
        elif name.endswith("_expected_absolute_delta"):
            if name.startswith("feature41_"):
                value = 1.0
            elif name.startswith("historical_v10_"):
                value = 2.0
            elif name.startswith("candidate_"):
                value = float(seed + 1)
            else:
                value = float(seed + 6)
            arrays[name] = np.full(n_rows, value, dtype=np.float64)
        else:
            raise AssertionError(f"unhandled source field: {name}")
    return arrays


def _write_source_universe(source_dir: Path) -> dict:
    source_dir.mkdir(parents=True)
    rows = []
    for seed in range(5):
        for fold in range(20):
            path = source_dir / f"rnet_distill_predictions_fold{fold}_seed{seed}.npz"
            np.savez_compressed(path, **_source_payload(fold, seed))
            rows.append(
                {
                    "outer_fold": fold,
                    "seed": seed,
                    "held_puzzle": f"P{fold + 1:02d}",
                    "n_registered_prediction_rows": 2,
                    "prediction_artifact": str(path.resolve()),
                }
            )
    merged = {
        "schema_version": MERGED_SCHEMA,
        "phase": "RND6P",
        "status": MERGED_STATUS["RND6P"],
        "folds": rows,
        "merge_integrity": dict(MERGE_INTEGRITY),
    }
    assert frozenset(merged) == EXPECTED_MERGED_FIELDS
    return merged


def _rewrite_with_target(path: Path) -> None:
    with np.load(path, allow_pickle=True) as handle:
        arrays = {name: np.asarray(handle[name]) for name in handle.files}
    arrays["target"] = np.zeros(len(arrays["keys"]), dtype=np.float64)
    np.savez_compressed(path, **arrays)


def test_rnd6p_atomic_assembly_uses_exact100_and_fixed_equal_seed_mixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    formal_dir = tmp_path / "rnd6_formal_seeds0_4"
    merged_path = formal_dir / "rnet_distill_complete_unscored_merge.json"
    assembly_dir = formal_dir / "assembled"
    assembly_path = (
        assembly_dir / "rnet_distill_five_seed_prediction_only_assembly.json"
    )
    merged = _write_source_universe(tmp_path / "sources")
    formal_dir.mkdir()
    merged_path.write_text(
        json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(assembler, "FORMAL_DIR", formal_dir)
    monkeypatch.setattr(assembler, "MERGED_PATH", merged_path)
    monkeypatch.setattr(assembler, "ASSEMBLY_DIR", assembly_dir)
    monkeypatch.setattr(assembler, "ASSEMBLY_PATH", assembly_path)
    authority_calls: list[tuple[Path, str]] = []
    monkeypatch.setattr(
        assembler,
        "assert_run_authority",
        lambda root, phase: authority_calls.append((root, phase)),
    )

    assert assembler.main(
        [
            "--repo-root",
            str(tmp_path),
            "--merged-json",
            str(merged_path),
            "--out-dir",
            str(assembly_dir),
            "--out-json",
            str(assembly_path),
        ]
    ) == 0
    assert authority_calls == [(tmp_path.resolve(), "RND6P")]

    assert assembly_dir.is_dir()
    assert len(list(assembly_dir.glob("*.npz"))) == 20
    assert sorted(path.name for path in assembly_dir.iterdir()) == sorted(
        [
            *[
                f"rnet_distill_formal_predictions_fold{fold}_seeds0_4.npz"
                for fold in range(20)
            ],
            "rnet_distill_five_seed_prediction_only_assembly.json",
        ]
    )
    assert not list(formal_dir.glob(".assembled.*"))
    manifest = json.loads(assembly_path.read_text(encoding="utf-8"))
    assert frozenset(manifest) == assembler.EXPECTED_ASSEMBLY_FIELDS
    assert manifest == {
        **manifest,
        "schema_version": assembler.SCHEMA,
        "phase": "RND6P",
        "status": assembler.ASSEMBLY_STATUS,
        "equal_seed_mixture": True,
        "equal_seed_weight": 0.2,
        "best_seed_selection_performed": False,
        "score_computed": False,
        "target_accessed": False,
        "external_outcome_accessed": False,
    }
    assert len(manifest["folds"]) == 20
    assert all(
        frozenset(row) == assembler.EXPECTED_FOLD_MANIFEST_FIELDS
        for row in manifest["folds"]
    )
    assert manifest["folds"][0] == {
        "outer_fold": 0,
        "seeds": [0, 1, 2, 3, 4],
        "prediction_artifact": str(
            (assembly_dir / "rnet_distill_formal_predictions_fold0_seeds0_4.npz").resolve()
        ),
        "n_registered_prediction_rows": 2,
        "candidate_components_per_distribution": 10,
        "null_components_per_distribution": 10,
        "feature41_components_per_distribution": 2,
        "historical_v10_components_per_distribution": 2,
    }

    with np.load(
        assembly_dir / "rnet_distill_formal_predictions_fold0_seeds0_4.npz",
        allow_pickle=True,
    ) as handle:
        assert frozenset(handle.files) == assembler.EXPECTED_FORMAL_PREDICTION_FIELDS
        assert str(handle["schema_version"].item()) == assembler.FORMAL_PREDICTION_SCHEMA
        assert np.array_equal(handle["seed"], np.full(2, -1, dtype=np.int64))
        assert np.array_equal(
            handle["assembled_seed_count"], np.full(2, 5, dtype=np.int64)
        )
        assert np.allclose(handle["candidate_point"], [2.0, 2.5])
        assert np.allclose(handle["null_point"], [12.0, 12.5])
        assert handle["candidate_weights"].shape == (2, 10)
        assert handle["null_weights"].shape == (2, 10)
        assert np.allclose(handle["candidate_weights"].sum(axis=1), 1.0)
        assert np.allclose(handle["null_weights"].sum(axis=1), 1.0)
        assert handle["feature41_weights"].shape == (2, 2)
        assert handle["historical_v10_weights"].shape == (2, 2)
        expected_candidate_absolute = assembler._expected_absolute(
            np.asarray(handle["candidate_weights"]),
            np.asarray(handle["candidate_locations"]),
            np.asarray(handle["candidate_scales"]),
        )
        assert np.allclose(
            handle["candidate_expected_absolute_delta"],
            expected_candidate_absolute,
        )

    with pytest.raises(FileExistsError, match="refuses to overwrite"):
        assembler.main(
            [
                "--repo-root",
                str(tmp_path),
                "--merged-json",
                str(merged_path),
                "--out-dir",
                str(assembly_dir),
                "--out-json",
                str(assembly_path),
            ]
        )


def test_rnd6p_assembler_validates_all100_sources_before_any_output(
    tmp_path: Path,
) -> None:
    merged = _write_source_universe(tmp_path / "sources")
    out_dir = tmp_path / "assembled"
    incomplete = copy.deepcopy(merged)
    incomplete["folds"] = incomplete["folds"][:-1]
    with pytest.raises(ValueError, match="exactly 100"):
        assembler.assemble(incomplete, out_dir)
    assert not out_dir.exists()

    last = Path(merged["folds"][-1]["prediction_artifact"])
    _rewrite_with_target(last)
    with pytest.raises(ValueError, match="field universe differs"):
        assembler.assemble(merged, out_dir)
    assert not out_dir.exists()


@pytest.mark.parametrize(
    "field",
    (
        "feature41_point",
        "v8_point",
        "feature41_weights",
        "historical_v10_locations",
    ),
)
def test_rnd6p_assembler_requires_fixed_comparators_exact_across_seeds(
    field: str,
) -> None:
    sources = {(0, seed): _source_payload(0, seed) for seed in range(5)}
    changed = np.asarray(sources[(0, 4)][field]).copy()
    changed.flat[0] += 0.125
    sources[(0, 4)][field] = changed

    with pytest.raises(ValueError, match=f"fixed comparator {field} differs"):
        assembler._assemble_fold_payload(sources, fold=0)


def test_rnd6p_assembler_rejects_noncanonical_cli_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[Path, str]] = []
    monkeypatch.setattr(
        assembler,
        "assert_run_authority",
        lambda root, phase: calls.append((root, phase)),
    )
    with pytest.raises(RuntimeError, match="merged_json differs"):
        assembler.validate_cli_binding(
            tmp_path,
            tmp_path / "wrong-merge.json",
            assembler.ASSEMBLY_DIR,
            assembler.ASSEMBLY_PATH,
        )
    with pytest.raises(RuntimeError, match="out_dir differs"):
        assembler.validate_cli_binding(
            tmp_path,
            assembler.MERGED_PATH,
            tmp_path / "wrong-assembled",
            assembler.ASSEMBLY_PATH,
        )
    with pytest.raises(RuntimeError, match="out_json differs"):
        assembler.validate_cli_binding(
            tmp_path,
            assembler.MERGED_PATH,
            assembler.ASSEMBLY_DIR,
            tmp_path / "wrong-assembly.json",
        )
    assert calls == [(tmp_path.resolve(), "RND6P")] * 3
