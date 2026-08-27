from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from scripts.reactflow_delta.post_v14_branch5_route_probe import (
    PROBE_FEATURE_WIDTH,
    RAW_SUMMARY_WIDTH,
    ProbeRidgeStats,
    fit_probe_ridge,
    source_receiver_features,
)
from scripts.reactflow_delta.run_post_v14_branch5_route_probe import (
    EXPECTED_PARENT_STATE,
    EXPECTED_PROJECT_TASK,
    FROZEN_RUNTIME_PATHS,
    PREDICTION_PHASE,
    PREDICTION_TOKEN,
    SOURCE_MANIFEST_SCHEMA,
    SOURCE_MANIFEST_STATUS,
    _atomic_write_prediction,
    _encode_puzzle,
    _fit_probe_models,
    _held_prediction,
    _load_source_registry,
    add_weighted_grid_to_cuda_stats,
    add_weighted_grid_to_stats,
    assert_run_authority,
)


def _active() -> dict:
    return {
        "project_task_id": EXPECTED_PROJECT_TASK,
        "authority": {
            "current_phase": PREDICTION_PHASE,
            "source_manifest_status": SOURCE_MANIFEST_STATUS,
            **{name: str(path) for name, path in FROZEN_RUNTIME_PATHS.items()},
        },
        "runnable_phases": [PREDICTION_PHASE],
        "training_allowed": PREDICTION_TOKEN,
        "candidate_model_training_allowed": PREDICTION_TOKEN,
        "held_score_read_allowed": False,
        "partial_fold_score_read_allowed": False,
        "new_external_outcome_access_allowed": False,
        "parent_state": dict(EXPECTED_PARENT_STATE),
    }


def test_branch5_runner_is_default_closed_and_requires_exact_parent(
    tmp_path: Path,
) -> None:
    config = tmp_path / "configs/reactflow_delta"
    config.mkdir(parents=True)
    current = _active()
    (config / "active_contract.yaml").write_text(yaml.safe_dump(current))
    assert_run_authority(tmp_path)
    cli = {
        "source_manifest": FROZEN_RUNTIME_PATHS["source_manifest_path"],
        "m2_csv": FROZEN_RUNTIME_PATHS["m2_csv_path"],
        "tic2a_merged_registry": FROZEN_RUNTIME_PATHS["tic2a_merged_registry_path"],
        "unconstrained_feature_cache": FROZEN_RUNTIME_PATHS[
            "unconstrained_feature_cache_path"
        ],
        "constrained_feature_cache": FROZEN_RUNTIME_PATHS[
            "constrained_feature_cache_path"
        ],
        "prediction_dir": FROZEN_RUNTIME_PATHS["prediction_dir"],
    }
    assert_run_authority(tmp_path, **cli)
    with pytest.raises(RuntimeError, match="differs"):
        assert_run_authority(
            tmp_path,
            **{
                **cli,
                "source_manifest": (tmp_path / "different.json").resolve(),
            },
        )

    for mutation in (
        lambda value: value.__setitem__(
            "project_task_id", "reactflow_delta_model_rescue_v14"
        ),
        lambda value: value["parent_state"].__setitem__(
            "post_v14_first_matching_branch_id", "4"
        ),
        lambda value: value.__setitem__("held_score_read_allowed", True),
        lambda value: value.__setitem__("training_allowed", False),
        lambda value: value["authority"].__setitem__(
            "source_manifest_status", "NOT_BOUND"
        ),
        lambda value: value["authority"].__setitem__(
            "m2_csv_path", "/mnt/cunyuliu/wrong.csv"
        ),
    ):
        changed = _active()
        mutation(changed)
        (config / "active_contract.yaml").write_text(yaml.safe_dump(changed))
        with pytest.raises(RuntimeError):
            assert_run_authority(tmp_path)


def _safe_manifest(tmp_path: Path) -> dict:
    rows = []
    for fold in range(20):
        v13 = tmp_path / f"v13_candidate_point_fold{fold}_seed0.pt"
        v14 = tmp_path / f"v14_candidate_point_fold{fold}_seed0.pt"
        v13.touch()
        v14.touch()
        rows.append(
            {
                "outer_fold": fold,
                "held_puzzle": f"P{fold + 1:02d}",
                "seed": 0,
                "v13_source_phase": "V13M3",
                "v13_candidate_checkpoint": str(v13),
                "v14_source_phase": "V14M3",
                "v14_arm": "CANDIDATE",
                "v14_candidate_checkpoint": str(v14),
                "held_score_closed_at_projection": True,
                "external_outcome_accessed": False,
            }
        )
    return {
        "schema_version": SOURCE_MANIFEST_SCHEMA,
        "status": SOURCE_MANIFEST_STATUS,
        "parent_state": dict(EXPECTED_PARENT_STATE),
        "folds": rows,
    }


def test_safe_source_manifest_is_exact_and_forbids_history_or_score(
    tmp_path: Path,
) -> None:
    manifest = _safe_manifest(tmp_path)
    path = tmp_path / "safe.json"
    path.write_text(json.dumps(manifest))
    assert sorted(_load_source_registry(path)) == list(range(20))
    assert sorted(
        _load_source_registry(
            path,
            expected_checkpoint_dirs={
                "v13_checkpoint_dir": tmp_path,
                "v14_checkpoint_dir": tmp_path,
            },
        )
    ) == list(range(20))
    with pytest.raises(ValueError, match="checkpoint directory differs"):
        _load_source_registry(
            path,
            expected_checkpoint_dirs={
                "v13_checkpoint_dir": tmp_path,
                "v14_checkpoint_dir": tmp_path / "wrong",
            },
        )

    manifest["training_history"] = [1.0]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="field boundary"):
        _load_source_registry(path)

    manifest = _safe_manifest(tmp_path)
    manifest["unregistered_metadata"] = "not accepted"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="field boundary"):
        _load_source_registry(path)


def test_grid_sufficient_statistics_equal_explicit_520d_rows() -> None:
    rng = np.random.default_rng(14)
    summary = rng.normal(size=(5, RAW_SUMMARY_WIDTH))
    edit = np.asarray([0, 3], dtype=np.int64)
    residual = rng.normal(size=(2, 5))
    weight = np.asarray(
        [[1.0, 0.0, 2.0, 0.5, 0.0], [0.0, 1.5, 0.0, 0.25, 3.0]],
        dtype=np.float64,
    )
    optimized = ProbeRidgeStats.zeros()
    add_weighted_grid_to_stats(
        optimized,
        summary=summary,
        edit_index=edit,
        residual=residual,
        weight=weight,
    )

    explicit = ProbeRidgeStats.zeros()
    features = source_receiver_features(
        np_to_torch(summary), np_to_torch(edit, integer=True)
    ).numpy()
    qualified = weight > 0
    explicit.add_rows(
        features[qualified].reshape(-1, PROBE_FEATURE_WIDTH),
        residual[qualified],
        weight[qualified],
    )
    assert optimized.sum_weight == pytest.approx(explicit.sum_weight)
    for name in ("sum_x", "sum_x2", "xtx", "sum_y", "xty"):
        np.testing.assert_allclose(
            np.asarray(getattr(optimized, name)),
            np.asarray(getattr(explicit, name)),
            atol=1e-10,
            rtol=1e-10,
        )


def test_near_zero_weighted_std_uses_the_frozen_inactive_rule() -> None:
    stats = ProbeRidgeStats.zeros(width=2)
    stats.add_rows(
        np.asarray([[0.0, 0.0], [1.0e-9, 1.0]], dtype=np.float64),
        np.asarray([0.0, 1.0], dtype=np.float64),
        np.ones(2, dtype=np.float64),
    )
    model = fit_probe_ridge(stats)
    assert model["scale_x"][0] == 1.0
    assert model["scale_x"][1] == pytest.approx(0.5)


def np_to_torch(value: np.ndarray, *, integer: bool = False):
    import torch

    return torch.tensor(value, dtype=torch.int64 if integer else torch.float64)


def test_content_contrast_removes_position_and_region_only_arm_signal() -> None:
    import torch

    class PositionBiasedEncoder(torch.nn.Module):
        def encode(self, context, _corruption_mask=None):
            sequence, reactivity, precision, observed, position, region = context
            base = (
                position[:, None] * 2.0
                + region[:, :1] * 3.0
                + sequence.sum(dim=1, keepdim=True) * 5.0
                + reactivity[:, None]
                + precision[:, None]
                + observed[:, None]
            )
            return base.expand(-1, 256)

    length = 9
    position = torch.arange(length, dtype=torch.float64)
    region = torch.column_stack([position.remainder(2), 1.0 - position.remainder(2)])
    zero_context = (
        torch.zeros(length, 4, dtype=torch.float64),
        torch.zeros(length, dtype=torch.float64),
        torch.zeros(length, dtype=torch.float64),
        torch.zeros(length, dtype=torch.float64),
        position,
        region,
    )
    construct_ids = [f"C{index}" for index in range(8)]
    hidden, reactivity, observed = _encode_puzzle(
        model=PositionBiasedEncoder().eval(),
        construct_ids=construct_ids,
        context_cache={name: zero_context for name in construct_ids},
    )
    assert torch.equal(hidden, torch.zeros_like(hidden))
    from scripts.reactflow_delta.post_v14_branch5_route_probe import (
        nonfocal_linear_summary,
    )

    aligned = nonfocal_linear_summary(
        hidden, reactivity, observed, focal_index=0, shift=0
    )
    shifted = nonfocal_linear_summary(
        hidden, reactivity, observed, focal_index=0, shift=17
    )
    assert torch.equal(aligned, torch.zeros_like(aligned))
    assert torch.equal(aligned, shifted)
    edit = torch.tensor([1, 7], dtype=torch.int64)
    aligned_features = source_receiver_features(aligned, edit)
    shifted_features = source_receiver_features(shifted, edit)
    assert torch.equal(aligned_features, torch.zeros_like(aligned_features))
    assert torch.equal(aligned_features, shifted_features)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a real CUDA GPU")
def test_production_fit_keeps_ridge_inputs_stats_and_solve_on_cuda(
    monkeypatch,
) -> None:
    import scripts.reactflow_delta.run_post_v14_branch5_route_probe as runtime
    import torch

    device = torch.device("cuda:0")
    length = 5
    records = []
    contexts = {}
    for construct_index in range(8):
        construct_id = f"P01_M{construct_index}"
        for mutant_index in range(2 if construct_index == 0 else 1):
            records.append(
                SimpleNamespace(
                    puzzle="P01",
                    method=f"M{construct_index}",
                    construct_id=construct_id,
                    design_pos=mutant_index,
                    full_pos=mutant_index + 1,
                    ref="A",
                    alt="C",
                )
            )
        position = torch.arange(length, dtype=torch.float32, device=device)
        sequence = torch.zeros((length, 4), dtype=torch.float32, device=device)
        sequence[:, construct_index % 4] = 1.0
        contexts[construct_id] = (
            sequence,
            torch.full(
                (length,),
                float(construct_index + 1),
                dtype=torch.float32,
                device=device,
            ),
            torch.ones(length, dtype=torch.float32, device=device),
            torch.ones(length, dtype=torch.float32, device=device),
            position,
            torch.column_stack([position.remainder(2), 1.0 - position.remainder(2)]),
        )

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros((), device=device))

        def encode(self, context, _corruption_mask=None):
            sequence, reactivity, precision, observed, position, region = context
            value = (
                sequence.sum(dim=1) + reactivity + precision + observed + self.anchor
            )
            return value[:, None].expand(-1, 256)

    qualified = torch.tensor(
        [[1, 1, 0, 1, 0], [0, 1, 1, 0, 1]],
        dtype=torch.bool,
        device=device,
    )
    cells = [
        {
            "construct_id": "P01_M0",
            "qualified_mask": qualified,
            "target": torch.tensor(
                [[1.0, 2.0, 0.0, 3.0, 0.0], [0.0, 1.5, 2.5, 0.0, 4.0]],
                dtype=torch.float32,
                device=device,
            ),
            "wt": torch.zeros(length, dtype=torch.float32, device=device),
            "parent_point": torch.full(
                (2, length), 0.25, dtype=torch.float32, device=device
            ),
            "edit": torch.tensor([1, 2], dtype=torch.int64, device=device),
        }
    ]

    grid_calls = []
    real_add_grid = add_weighted_grid_to_cuda_stats

    def checked_add_grid(stats, *, summary, edit_index, residual, weight):
        grid_calls.append(
            (
                summary.device,
                edit_index.device,
                residual.device,
                weight.device,
                summary.dtype,
                residual.dtype,
                weight.dtype,
            )
        )
        return real_add_grid(
            stats,
            summary=summary,
            edit_index=edit_index,
            residual=residual,
            weight=weight,
        )

    solve_calls = []
    real_solve = torch.linalg.solve

    def checked_solve(matrix, rhs):
        solve_calls.append((matrix.device, rhs.device, matrix.dtype, rhs.dtype))
        return real_solve(matrix, rhs)

    monkeypatch.setattr(runtime, "add_weighted_grid_to_cuda_stats", checked_add_grid)
    monkeypatch.setattr(torch.linalg, "solve", checked_solve)
    models, counts = _fit_probe_models(
        train_records=records,
        cells=cells,
        context_cache=contexts,
        v14_encoder=Encoder().eval(),
    )

    assert len(grid_calls) == 2
    assert all(
        call
        == (
            device,
            device,
            device,
            device,
            torch.float64,
            torch.float64,
            torch.float64,
        )
        for call in grid_calls
    )
    assert solve_calls == [
        (device, device, torch.float64, torch.float64),
        (device, device, torch.float64, torch.float64),
    ]
    assert counts["n_outer_train_qualified_rows"] == int(qualified.sum().item())
    for model in models.values():
        for name in ("mean_x", "scale_x", "mean_y", "coefficient"):
            assert model[name].is_cuda
            assert model[name].dtype == torch.float64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a real CUDA GPU")
def test_held_prediction_never_reads_mutant_outcome(
    monkeypatch, tmp_path: Path
) -> None:
    import torch
    import scripts.reactflow_delta.run_post_v14_branch5_route_probe as runtime

    device = torch.device("cuda:0")
    length = 4
    records = []
    contexts = {}
    constructs = {}
    for index in range(8):
        construct_id = f"P01_M{index}"
        records.append(
            SimpleNamespace(
                puzzle="P01",
                method=f"M{index}",
                construct_id=construct_id,
                design_pos=0,
                full_pos=1,
                ref="A",
                alt="C",
            )
        )
        sequence = torch.zeros(length, 4, device=device)
        sequence[:, 0] = 1.0
        contexts[construct_id] = (
            sequence,
            torch.full((length,), float(index + 1), device=device),
            torch.ones(length, device=device),
            torch.ones(length, device=device),
            torch.arange(length, dtype=torch.float32, device=device),
            torch.column_stack(
                [torch.ones(length, device=device), torch.zeros(length, device=device)]
            ),
        )
        constructs[construct_id] = SimpleNamespace(
            sequence="AAAA",
            wt_observed=np.ones(length, dtype=bool),
        )

    class NoOutcomeUniverse:
        def get_construct(self, construct_id):
            return constructs[construct_id]

        def mutant_full_profile(self, *_args):
            raise AssertionError("held mutant outcome entered prediction path")

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros((), device=device))

        def encode(self, context, _corruption_mask=None):
            sequence, reactivity, precision, observed, position, region = context
            value = (
                sequence.sum(dim=1)
                + reactivity
                + precision
                + observed
                + position
                + region[:, 0]
                + self.anchor
            )
            return value[:, None].expand(-1, 256)

    class Parent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)

        def forward_point(
            self,
            _context,
            edit,
            _distance,
            _refs,
            _alts,
            prediction_mask,
            _feature41,
        ):
            return torch.zeros(
                len(edit), prediction_mask.shape[1], device=self.anchor.device
            )

    monkeypatch.setattr(
        runtime,
        "_feature41_matrix",
        lambda construct, rows, *_args: (
            np.zeros((len(rows), len(construct.sequence), 41), dtype=np.float32),
            np.zeros((len(rows), len(construct.sequence)), dtype=np.float32),
        ),
    )
    ridge = {
        "mean_x": torch.zeros(PROBE_FEATURE_WIDTH, dtype=torch.float64, device=device),
        "scale_x": torch.ones(PROBE_FEATURE_WIDTH, dtype=torch.float64, device=device),
        "mean_y": torch.zeros((), dtype=torch.float64, device=device),
        "coefficient": torch.zeros(
            PROBE_FEATURE_WIDTH, dtype=torch.float64, device=device
        ),
        "alpha": 1.0,
    }
    prediction_devices = []
    real_predict = runtime.predict_probe_ridge_cuda

    def checked_predict(model, features):
        prediction_devices.append((features.device, features.dtype))
        result = real_predict(model, features)
        assert result.is_cuda
        assert result.dtype == torch.float64
        return result

    monkeypatch.setattr(runtime, "predict_probe_ridge_cuda", checked_predict)
    prediction = _held_prediction(
        univ=NoOutcomeUniverse(),
        held_records=records,
        context_cache=contexts,
        feature41_model={},
        unconstrained=object(),
        constrained=object(),
        v13_parent=Parent().to(device),
        v14_encoder=Encoder().eval(),
        ridge_models={"aligned": ridge, "shift17": ridge},
        fold_id=0,
    )
    assert len(prediction["keys"]) == 8 * length
    assert not {
        "target",
        "target_error",
        "qualified_target_mask",
        "score",
        "loss",
    } & set(prediction)
    assert prediction_devices == [(device, torch.float64) for _ in range(8 * 2)]
    for name in ("parent_point", "aligned_point", "shift17_point"):
        assert prediction[name].is_cuda
        assert prediction[name].dtype == torch.float64
        torch.testing.assert_close(prediction[name], torch.zeros_like(prediction[name]))
    output_path = tmp_path / "prediction.npz"
    _atomic_write_prediction(output_path, prediction)
    with np.load(output_path, allow_pickle=True) as serialized:
        for name in ("parent_point", "aligned_point", "shift17_point"):
            assert serialized[name].dtype == np.float64
            np.testing.assert_array_equal(serialized[name], 0.0)
