#!/usr/bin/env python3
"""Run the pre-registered post-V12 diagnostics after a complete screen FAIL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import ndtr
import torch
import yaml

from scripts.reactflow_delta.diagnose_model_rescue_v11 import (
    directional_summary,
    method_balanced_weights,
    weighted_correlation,
)
from scripts.reactflow_delta.m2_universe_v1 import M2Universe
from scripts.reactflow_delta.merge_model_rescue_v12 import SCHEMA as MERGED_SCHEMA
from scripts.reactflow_delta.model_rescue_v1 import weighted_gaussian_mixture_crps
from scripts.reactflow_delta.model_rescue_v12 import (
    MonotoneRegimeGate,
    PREDICTION_SCHEMA,
)
from scripts.reactflow_delta.qualify_model_rescue_v12 import (
    SCHEMA as QUALIFICATION_SCHEMA,
)
from scripts.reactflow_delta.run_p2_v3 import _bio_key
from scripts.reactflow_delta.score_model_rescue_v12 import SCHEMA as SCORE_SCHEMA
from scripts.reactflow_delta.split_v4_lopo_puzzle import build_split_v4


SCHEMA = "reactflow_delta.model_rescue_v12_post_screen_diagnostics.v1"
GATE_GRID = np.linspace(0.0, 1.0, 101, dtype=np.float64)
LEVELS = (0.50, 0.68, 0.80, 0.90, 0.95)
DISTANCE_GRID = (0.0, 1.0, 5.0, 10.0, 20.0, 40.0)
MAGNITUDE_GRID = (0.0, 0.01, 0.05, 0.10, 0.20, 0.40)


def assert_diagnostic_authority(repo_root: Path) -> None:
    active = yaml.safe_load(
        (repo_root / "configs/reactflow_delta/active_contract.yaml").read_text()
    )
    if active["authority"]["current_phase"] != "V12M3" or active.get(
        "runnable_phases"
    ) != ["V12M3"]:
        raise RuntimeError("post-V12 diagnostics are closed outside V12M3")
    if active.get("training_allowed") is not False or active.get(
        "candidate_model_training_allowed"
    ) is not False:
        raise RuntimeError("post-V12 diagnostics require training closed")
    if active.get("held_score_read_allowed") != (
        "V12_COMPLETE_SCORE_POSTSCREEN_DIAGNOSTICS_ONLY"
    ):
        raise RuntimeError("post-V12 diagnostic authority is closed")
    if active.get("partial_fold_score_read_allowed") is not False:
        raise RuntimeError("post-V12 partial diagnostics remain prohibited")
    if active.get("new_external_outcome_access_allowed") is not False:
        raise RuntimeError("post-V12 diagnostics require external outcomes locked")


def _load_prediction(path: Path, fold: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as handle:
        prediction = {name: np.asarray(handle[name]) for name in handle.files}
    if str(prediction["schema_version"].item()) != PREDICTION_SCHEMA:
        raise ValueError(f"invalid V12 prediction schema in {path}")
    if set(map(int, prediction["outer_fold"])) != {fold} or set(
        map(int, prediction["seed"])
    ) != {0}:
        raise ValueError(f"V12 diagnostic fold or seed mismatch in {path}")
    keys = list(map(str, prediction["keys"]))
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate V12 diagnostic keys in {path}")
    return prediction


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, quantile: float
) -> float:
    order = np.argsort(values, kind="stable")
    ordered = np.asarray(values, dtype=np.float64)[order]
    cumulative = np.cumsum(np.asarray(weights, dtype=np.float64)[order])
    index = int(np.searchsorted(cumulative, quantile, side="left"))
    return float(ordered[min(index, len(ordered) - 1)])


def _registered_labels(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    methods = []
    mutants = []
    for key in map(str, keys):
        raw = key.split("|")
        if len(raw) != 7:
            raise ValueError(f"invalid V12 biological key {key}")
        methods.append(raw[2])
        mutants.append(f"{raw[3]}|{raw[4]}|{raw[5]}")
    return np.asarray(methods, dtype=object), np.asarray(mutants, dtype=object)


def gate_geometry(
    prediction: dict[str, np.ndarray], gate_parameters: dict[str, float]
) -> dict[str, Any]:
    methods, mutants = _registered_labels(prediction["keys"])
    weights = method_balanced_weights(methods, mutants)
    gate = np.asarray(prediction["gate_value"], dtype=np.float64)
    distance = np.asarray(prediction["gate_distance_factor"], dtype=np.float64)
    magnitude = np.asarray(prediction["gate_magnitude_factor"], dtype=np.float64)
    quantiles = (0.05, 0.25, 0.50, 0.75, 0.95)
    module = MonotoneRegimeGate().to(dtype=torch.float64)
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            parameter.copy_(torch.as_tensor(float(gate_parameters[name]), dtype=parameter.dtype))
        d = torch.tensor(
            [value for value in DISTANCE_GRID for _ in MAGNITUDE_GRID],
            dtype=torch.float64,
        )
        m = torch.tensor(
            [value for _ in DISTANCE_GRID for value in MAGNITUDE_GRID],
            dtype=torch.float64,
        )
        surface = module(d, m).numpy().reshape(len(DISTANCE_GRID), len(MAGNITUDE_GRID))
    return {
        "parameters": {name: float(value) for name, value in gate_parameters.items()},
        "weighted_quantiles": {
            "gate": {str(q): _weighted_quantile(gate, weights, q) for q in quantiles},
            "distance_factor": {
                str(q): _weighted_quantile(distance, weights, q) for q in quantiles
            },
            "magnitude_factor": {
                str(q): _weighted_quantile(magnitude, weights, q) for q in quantiles
            },
        },
        "weighted_fractions": {
            "gate_lt_0.10": float(weights[gate < 0.10].sum()),
            "gate_lt_0.25": float(weights[gate < 0.25].sum()),
            "gate_gt_0.75": float(weights[gate > 0.75].sum()),
            "gate_gt_0.90": float(weights[gate > 0.90].sum()),
        },
        "surface": {
            "distance": list(DISTANCE_GRID),
            "absolute_feature41": list(MAGNITUDE_GRID),
            "gate": surface.tolist(),
        },
    }


def _descriptive_summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) != 20 or not np.isfinite(array).all():
        raise ValueError("gate-geometry dispersion requires twenty finite folds")
    return {
        "n_puzzles": 20,
        "mean": float(array.mean()),
        "standard_deviation": float(array.std(ddof=1)),
        "minimum": float(array.min()),
        "median": float(np.median(array)),
        "maximum": float(array.max()),
    }


def summarize_gate_geometries(folds: list[dict[str, Any]]) -> dict[str, Any]:
    if len(folds) != 20:
        raise ValueError("gate-geometry summary requires folds0-19")
    parameter_names = tuple(sorted(folds[0]["gate_geometry"]["parameters"]))
    distance = tuple(folds[0]["gate_geometry"]["surface"]["distance"])
    magnitude = tuple(
        folds[0]["gate_geometry"]["surface"]["absolute_feature41"]
    )
    surfaces = []
    for row in folds:
        geometry = row["gate_geometry"]
        if tuple(sorted(geometry["parameters"])) != parameter_names:
            raise ValueError("gate parameter universe differs across folds")
        if tuple(geometry["surface"]["distance"]) != distance or tuple(
            geometry["surface"]["absolute_feature41"]
        ) != magnitude:
            raise ValueError("gate diagnostic grid differs across folds")
        surface = np.asarray(geometry["surface"]["gate"], dtype=np.float64)
        if surface.shape != (len(distance), len(magnitude)):
            raise ValueError("gate diagnostic surface has the wrong shape")
        surfaces.append(surface)
    stacked = np.stack(surfaces)
    grid = {}
    for distance_index, distance_value in enumerate(distance):
        for magnitude_index, magnitude_value in enumerate(magnitude):
            grid[f"distance={distance_value}|absolute_feature41={magnitude_value}"] = (
                _descriptive_summary(
                    stacked[:, distance_index, magnitude_index].tolist()
                )
            )
    return {
        "parameters": {
            name: _descriptive_summary(
                [float(row["gate_geometry"]["parameters"][name]) for row in folds]
            )
            for name in parameter_names
        },
        "surface_grid": grid,
    }


def _observations(
    univ: M2Universe,
    held_records: list[Any],
    prediction: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    index = {str(key): row for row, key in enumerate(prediction["keys"])}
    fields: dict[str, list[Any]] = {
        name: []
        for name in (
            "method",
            "mutant",
            "distance",
            "target",
            "feature41_point",
            "parent_point",
            "candidate_point",
            "gate_value",
        )
    }
    for name in ("feature41", "parent", "candidate", "historical_v10"):
        for suffix in (
            "weights",
            "locations",
            "scales",
            "expected_absolute_delta",
        ):
            fields[f"{name}_{suffix}"] = []
    for record in held_records:
        construct = univ.get_construct(record.construct_id)
        target, _error = univ.mutant_full_profile(
            record.wt_id, record.design_pos, record.ref, record.alt
        )
        if target is None:
            continue
        positions = np.flatnonzero(construct.wt_observed & np.isfinite(target))
        mutant = f"{record.construct_id}|{record.mutation_key}"
        for position in positions:
            key = _bio_key(univ, record, int(position))
            if key not in index:
                raise ValueError("V12 diagnostic prediction universe is incomplete")
            row = index[key]
            fields["method"].append(str(record.method))
            fields["mutant"].append(mutant)
            fields["distance"].append(abs(int(position) - int(record.full_pos)))
            fields["target"].append(
                float(target[position] - construct.wt_reactivity[position])
            )
            fields["feature41_point"].append(float(prediction["feature41_point"][row]))
            fields["parent_point"].append(float(prediction["v11_parent_point"][row]))
            fields["candidate_point"].append(float(prediction["candidate_point"][row]))
            fields["gate_value"].append(float(prediction["gate_value"][row]))
            for name in ("feature41", "parent", "candidate", "historical_v10"):
                for suffix in ("weights", "locations", "scales"):
                    fields[f"{name}_{suffix}"].append(
                        np.asarray(prediction[f"{name}_{suffix}"][row], dtype=np.float64)
                    )
                fields[f"{name}_expected_absolute_delta"].append(
                    float(prediction[f"{name}_expected_absolute_delta"][row])
                )
    result = {}
    for name, values in fields.items():
        if name in ("method", "mutant"):
            result[name] = np.asarray(values, dtype=object)
        elif name.endswith(("_weights", "_locations", "_scales")):
            result[name] = np.stack(values)
        else:
            result[name] = np.asarray(values, dtype=np.float64)
    return result


def _regime_masks(observations: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    distance = observations["distance"]
    magnitude = np.abs(observations["feature41_point"])
    return {
        "distance": {
            "0": distance == 0,
            "1_5": (distance >= 1) & (distance <= 5),
            "6_20": (distance >= 6) & (distance <= 20),
            "gt20": distance > 20,
        },
        "magnitude": {
            "0_0.05": magnitude < 0.05,
            "0.05_0.10": (magnitude >= 0.05) & (magnitude < 0.10),
            "0.10_0.20": (magnitude >= 0.10) & (magnitude < 0.20),
            "0.20_inf": magnitude >= 0.20,
        },
    }


def _best_gate(
    observations: dict[str, np.ndarray], selected: np.ndarray
) -> float:
    if not selected.any():
        return 0.0
    weights = method_balanced_weights(
        observations["method"], observations["mutant"], selected
    )
    target = observations["target"]
    feature = observations["feature41_point"]
    residual = observations["parent_point"] - feature
    losses = [
        float(np.sum(weights * np.abs(target - (feature + gate * residual))))
        for gate in GATE_GRID
    ]
    return float(GATE_GRID[int(np.argmin(losses))])


def _oracle(
    observations: dict[str, np.ndarray], groups: dict[str, np.ndarray]
) -> dict[str, Any]:
    prediction = np.asarray(observations["feature41_point"], dtype=np.float64).copy()
    residual = observations["parent_point"] - observations["feature41_point"]
    gate_by_row = np.zeros(len(prediction), dtype=np.float64)
    gates = {}
    covered = np.zeros(len(prediction), dtype=bool)
    for label, selected in groups.items():
        selected = np.asarray(selected, dtype=bool)
        if selected.any():
            gate = _best_gate(observations, selected)
            prediction[selected] += gate * residual[selected]
            gate_by_row[selected] = gate
            covered |= selected
            gates[label] = gate
        else:
            gates[label] = None
    if not covered.all():
        raise RuntimeError("V12 diagnostic oracle groups do not partition observations")
    weights = method_balanced_weights(observations["method"], observations["mutant"])
    target = observations["target"]
    return {
        "gates": gates,
        "signed_delta_mae": float(np.sum(weights * np.abs(target - prediction))),
        "point_absolute_delta_mae": float(
            np.sum(weights * np.abs(np.abs(target) - np.abs(prediction)))
        ),
        "prediction": prediction,
        "gate_by_row": gate_by_row,
    }


def oracle_diagnostics(
    observations: dict[str, np.ndarray], score_row: dict[str, Any]
) -> dict[str, Any]:
    all_rows = np.ones(len(observations["target"]), dtype=bool)
    regimes = _regime_masks(observations)
    two_dimensional = {
        f"distance={distance}|magnitude={magnitude}": dmask & mmask
        for distance, dmask in regimes["distance"].items()
        for magnitude, mmask in regimes["magnitude"].items()
    }
    oracles = {
        "global": _oracle(observations, {"all": all_rows}),
        "distance": _oracle(observations, regimes["distance"]),
        "magnitude": _oracle(observations, regimes["magnitude"]),
        "distance_by_magnitude": _oracle(observations, two_dimensional),
    }
    weights = method_balanced_weights(observations["method"], observations["mutant"])
    target = observations["target"]
    point_losses = {
        "feature41_signed_delta_mae": float(
            np.sum(weights * np.abs(target - observations["feature41_point"]))
        ),
        "parent_signed_delta_mae": float(
            np.sum(weights * np.abs(target - observations["parent_point"]))
        ),
        "candidate_signed_delta_mae": float(
            np.sum(weights * np.abs(target - observations["candidate_point"]))
        ),
        "candidate_point_absolute_delta_mae": float(
            np.sum(
                weights
                * np.abs(np.abs(target) - np.abs(observations["candidate_point"]))
            )
        ),
        "feature41_absolute_delta_mae": float(score_row["feature41_absolute_delta_mae"]),
        "parent_point_absolute_delta_mae": float(
            np.sum(
                weights
                * np.abs(np.abs(target) - np.abs(observations["parent_point"]))
            )
        ),
    }
    comparison_losses = {
        "feature41": {
            "signed_delta_mae": point_losses["feature41_signed_delta_mae"],
            "point_absolute_delta_mae": point_losses[
                "feature41_absolute_delta_mae"
            ],
        },
        "parent_v11": {
            "signed_delta_mae": point_losses["parent_signed_delta_mae"],
            "point_absolute_delta_mae": point_losses[
                "parent_point_absolute_delta_mae"
            ],
        },
        "candidate_v12": {
            "signed_delta_mae": point_losses["candidate_signed_delta_mae"],
            "point_absolute_delta_mae": point_losses[
                "candidate_point_absolute_delta_mae"
            ],
        },
    }
    for name, current in oracles.items():
        current["signed_relative_gain_vs_feature41"] = (
            point_losses["feature41_signed_delta_mae"] - current["signed_delta_mae"]
        ) / point_losses["feature41_signed_delta_mae"]
        current["point_absolute_relative_gain_vs_feature41"] = (
            point_losses["feature41_absolute_delta_mae"]
            - current["point_absolute_delta_mae"]
        ) / point_losses["feature41_absolute_delta_mae"]
        current["gains"] = {}
        for comparator, comparator_losses in comparison_losses.items():
            current["gains"][comparator] = {}
            for metric in ("signed_delta_mae", "point_absolute_delta_mae"):
                absolute_gain = comparator_losses[metric] - current[metric]
                current["gains"][comparator][f"{metric}_absolute_gain"] = float(
                    absolute_gain
                )
                current["gains"][comparator][f"{metric}_relative_gain"] = float(
                    absolute_gain / comparator_losses[metric]
                )
    oracle_gate_correlation = weighted_correlation(
        observations["gate_value"],
        oracles["distance_by_magnitude"]["gate_by_row"],
        weights,
    )
    comparisons = {}
    ordered_pairs = (
        ("candidate_v12", None, "global", "candidate_minus_global"),
        (None, "global", "distance", "global_minus_distance"),
        (None, "global", "magnitude", "global_minus_magnitude"),
        (None, "global", "distance_by_magnitude", "global_minus_2d"),
    )
    for left_point, left_oracle, right_oracle, label in ordered_pairs:
        comparisons[label] = {}
        for metric, baseline_key in (
            ("signed_delta_mae", "feature41_signed_delta_mae"),
            ("point_absolute_delta_mae", "feature41_absolute_delta_mae"),
        ):
            if left_point is not None:
                left = comparison_losses[left_point][metric]
            else:
                left = oracles[str(left_oracle)][metric]
            right = oracles[right_oracle][metric]
            comparisons[label][f"{metric}_absolute_headroom"] = float(left - right)
            comparisons[label][f"{metric}_relative_headroom"] = float(
                (left - right) / point_losses[baseline_key]
            )
    residual_associations = {}
    target_residual = target - observations["feature41_point"]
    parent_residual = observations["parent_point"] - observations["feature41_point"]
    for dimension, masks in _regime_masks(observations).items():
        residual_associations[dimension] = {}
        for label, selected in masks.items():
            selected = np.asarray(selected, dtype=bool)
            if selected.any():
                selected_weights = method_balanced_weights(
                    observations["method"], observations["mutant"], selected
                )
                residual_associations[dimension][label] = {
                    "n_rows": int(selected.sum()),
                    "weighted_residual_correlation": weighted_correlation(
                        target_residual, parent_residual, selected_weights
                    ),
                }
            else:
                residual_associations[dimension][label] = {"available": False}
    for current in oracles.values():
        current.pop("prediction")
        current.pop("gate_by_row")
    return {
        "observed_point_losses": point_losses,
        "oracles": oracles,
        "comparisons": comparisons,
        "v12_gate_to_2d_oracle_gate_weighted_correlation": oracle_gate_correlation,
        "regime_residual_associations": residual_associations,
    }


def _distribution(
    observations: dict[str, np.ndarray], name: str, selected: np.ndarray
) -> dict[str, Any]:
    weights = method_balanced_weights(
        observations["method"], observations["mutant"], selected
    )
    target = observations["target"]
    mixture_weights = observations[f"{name}_weights"]
    locations = observations[f"{name}_locations"]
    scales = observations[f"{name}_scales"]
    cdf = np.sum(mixture_weights * ndtr((target[:, None] - locations) / scales), axis=1)
    crps = weighted_gaussian_mixture_crps(
        locations, scales, mixture_weights, target
    )
    mean_location = np.sum(mixture_weights * locations, axis=1)
    scale_mean = np.sum(mixture_weights * scales, axis=1)
    current: dict[str, Any] = {
        "crps": float(np.sum(weights * crps)),
        "distribution_absolute_delta_mae": float(
            np.sum(
                weights
                * np.abs(
                    np.abs(target)
                    - observations[f"{name}_expected_absolute_delta"]
                )
            )
        ),
        "pit_mean": float(np.sum(weights * cdf)),
        "pit_variance": float(np.sum(weights * (cdf - np.sum(weights * cdf)) ** 2)),
        "mean_scale_error_association": weighted_correlation(
            scale_mean, np.abs(target - mean_location), weights
        ),
    }
    if mixture_weights.shape[1] > 1:
        current["first_component_weight_error_association"] = weighted_correlation(
            mixture_weights[:, 0], np.abs(target - mean_location), weights
        )
    for level in LEVELS:
        lower = (1.0 - level) / 2.0
        upper = 1.0 - lower
        tag = str(int(round(level * 100)))
        covered = (cdf >= lower) & (cdf <= upper)
        current[f"coverage{tag}"] = float(np.sum(weights * covered))
        current[f"coverage{tag}_error"] = current[f"coverage{tag}"] - level
        current[f"lower_miss{tag}"] = float(np.sum(weights * (cdf < lower)))
        current[f"upper_miss{tag}"] = float(np.sum(weights * (cdf > upper)))
    return current


def distribution_diagnostics(observations: dict[str, np.ndarray]) -> dict[str, Any]:
    all_rows = np.ones(len(observations["target"]), dtype=bool)
    result = {
        "global": {
            name: _distribution(observations, name, all_rows)
            for name in ("feature41", "parent", "candidate", "historical_v10")
        },
        "regimes": {},
    }
    for dimension, masks in _regime_masks(observations).items():
        result["regimes"][dimension] = {}
        for label, selected in masks.items():
            if selected.any():
                result["regimes"][dimension][label] = {
                    name: _distribution(observations, name, selected)
                    for name in ("feature41", "parent", "candidate", "historical_v10")
                }
            else:
                result["regimes"][dimension][label] = {"available": False}
    return result


def _summary(folds: list[dict[str, Any]], key: str) -> dict[str, Any]:
    return directional_summary([float(row[key]) for row in folds])


def route_summary(
    folds: list[dict[str, Any]], qualification: dict[str, Any]
) -> dict[str, Any]:
    derived = []
    for row in folds:
        losses = row["oracle"]["observed_point_losses"]
        oracle = row["oracle"]["oracles"]
        signed_baseline = losses["feature41_signed_delta_mae"]
        absolute_baseline = losses["feature41_absolute_delta_mae"]
        derived.append(
            {
                "v12_to_global_signed_relative_headroom": (
                    losses["candidate_signed_delta_mae"]
                    - oracle["global"]["signed_delta_mae"]
                )
                / signed_baseline,
                "global_to_2d_signed_relative_headroom": (
                    oracle["global"]["signed_delta_mae"]
                    - oracle["distance_by_magnitude"]["signed_delta_mae"]
                )
                / signed_baseline,
                "global_to_2d_point_absolute_relative_headroom": (
                    oracle["global"]["point_absolute_delta_mae"]
                    - oracle["distance_by_magnitude"]["point_absolute_delta_mae"]
                )
                / absolute_baseline,
                "v12_to_2d_signed_relative_headroom": (
                    losses["candidate_signed_delta_mae"]
                    - oracle["distance_by_magnitude"]["signed_delta_mae"]
                )
                / signed_baseline,
                "v12_to_2d_point_absolute_relative_headroom": (
                    losses["candidate_point_absolute_delta_mae"]
                    - oracle["distance_by_magnitude"]["point_absolute_delta_mae"]
                )
                / absolute_baseline,
                "oracle_2d_signed_relative_gain_vs_feature41": oracle[
                    "distance_by_magnitude"
                ]["signed_relative_gain_vs_feature41"],
                "oracle_2d_point_absolute_relative_gain_vs_feature41": oracle[
                    "distance_by_magnitude"
                ]["point_absolute_relative_gain_vs_feature41"],
                "candidate_tail_asymmetry90": row["distribution"]["global"][
                    "candidate"
                ]["lower_miss90"]
                - row["distribution"]["global"]["candidate"]["upper_miss90"],
                "candidate_tail_asymmetry95": row["distribution"]["global"][
                    "candidate"
                ]["lower_miss95"]
                - row["distribution"]["global"]["candidate"]["upper_miss95"],
            }
        )
    summaries = {
        key: _summary(derived, key)
        for key in derived[0]
    }
    coordinate = summaries["global_to_2d_signed_relative_headroom"]
    coordinate_absolute = summaries[
        "global_to_2d_point_absolute_relative_headroom"
    ]
    transfer = summaries["v12_to_global_signed_relative_headroom"]
    v12_to_2d_signed = summaries["v12_to_2d_signed_relative_headroom"]
    v12_to_2d_absolute = summaries[
        "v12_to_2d_point_absolute_relative_headroom"
    ]
    oracle_signed = summaries["oracle_2d_signed_relative_gain_vs_feature41"]
    oracle_absolute = summaries[
        "oracle_2d_point_absolute_relative_gain_vs_feature41"
    ]
    coordinate_limited = bool(
        coordinate.get("stable_nonzero_direction")
        and coordinate.get("mean", 0.0) >= 0.01
        and coordinate.get("positive_puzzles", 0) >= 14
        and coordinate_absolute.get("stable_nonzero_direction")
        and coordinate_absolute.get("mean", 0.0) >= 0.01
        and coordinate_absolute.get("positive_puzzles", 0) >= 14
    )
    transfer_limited = bool(
        transfer.get("stable_nonzero_direction")
        and transfer.get("mean", 0.0) >= 0.02
        and transfer.get("positive_puzzles", 0) >= 14
        and coordinate.get("mean", float("inf")) < 0.01
    )
    adequate = bool(
        v12_to_2d_signed.get("mean", float("inf")) <= 0.01
        and v12_to_2d_absolute.get("mean", float("inf")) <= 0.01
    )
    residual_signal_absent = bool(
        oracle_signed.get("mean", 0.0) < 0.10
        and oracle_absolute.get("mean", 0.0) < 0.05
    )
    point_gate_names = (
        "signed_gain_vs_feature41_ge_10pct",
        "signed_gain_vs_parent_v11_ge_1pct",
        "signed_ci_lower_each_gt_zero",
        "signed_positive_puzzles_vs_feature41_ge_16",
        "signed_positive_puzzles_vs_parent_v11_ge_14",
        "point_absolute_gain_vs_feature41_ge_5pct",
        "point_absolute_gain_vs_parent_v11_ge_1pct",
        "point_absolute_ci_lower_each_gt_zero",
        "point_absolute_positive_puzzles_vs_feature41_ge_16",
        "point_absolute_positive_puzzles_vs_parent_v11_ge_14",
    )
    point_gates_pass = all(qualification["gates"].get(name) is True for name in point_gate_names)
    tail_stable = any(
        summaries[name].get("stable_nonzero_direction") is True
        for name in ("candidate_tail_asymmetry90", "candidate_tail_asymmetry95")
    )
    distribution_only_supported = bool(point_gates_pass and tail_stable)
    if distribution_only_supported:
        route = "RESIDUAL_DISTRIBUTION_ONLY_AMENDMENT_ELIGIBLE_NOT_AUTHORIZED"
    elif residual_signal_absent:
        route = "TERMINATE_V11_RESIDUAL_ARCHITECTURE_FAMILY"
    elif adequate:
        route = "TERMINATE_SHRINKAGE_GATE_CAPACITY_ROUTE"
    elif transfer_limited:
        route = "TRANSFER_LIMITED_DO_NOT_INCREASE_GATE_CAPACITY"
    elif coordinate_limited:
        route = "LOW_CAPACITY_NON_PRODUCT_GATE_AMENDMENT_ELIGIBLE_NOT_AUTHORIZED"
    else:
        route = "NO_NEW_MODEL_ROUTE_IDENTIFIED"
    return {
        "per_puzzle_derived": derived,
        "summaries": summaries,
        "decisions": {
            "coordinate_limited": coordinate_limited,
            "transfer_limited": transfer_limited,
            "monotone_product_adequate": adequate,
            "residual_signal_absent": residual_signal_absent,
            "point_gates_pass": point_gates_pass,
            "stable_tail_asymmetry": tail_stable,
            "distribution_only_supported": distribution_only_supported,
            "route": route,
            "new_model_authorized": False,
        },
    }


def diagnose(
    merged: dict[str, Any],
    scores: dict[str, Any],
    qualification: dict[str, Any],
    m2_csv: Path,
) -> dict[str, Any]:
    if merged.get("schema_version") != MERGED_SCHEMA or merged.get("status") != (
        "V12M3_COMPLETE_UNSCORED_MERGE_PASS"
    ):
        raise ValueError("post-V12 diagnostics require the complete V12M3 merge")
    if scores.get("schema_version") != SCORE_SCHEMA or scores.get("status") != (
        "V12M3_COMPLETE_SCORE_PASS"
    ):
        raise ValueError("post-V12 diagnostics require the one complete score")
    if qualification.get("schema_version") != QUALIFICATION_SCHEMA or qualification.get(
        "status"
    ) != "V12M3_TOP_JOURNAL_SCREEN_FAIL" or qualification.get("gate_passed") is not False:
        raise ValueError("post-V12 diagnostics run only after exact V12M3 FAIL")
    merged_rows = {int(row["outer_fold"]): row for row in merged["folds"]}
    score_rows = {int(row["outer_fold"]): row for row in scores["scores"]}
    if sorted(merged_rows) != list(range(20)) or sorted(score_rows) != list(range(20)):
        raise ValueError("post-V12 diagnostics require folds0-19")
    univ = M2Universe(m2_csv)
    identity = univ.build()
    if identity.get("canonical_mutant_full_profile_identity") != (
        "EXACT_PUZZLE_METHOD_MUTATION"
    ):
        raise RuntimeError("post-V12 diagnostics require exact target identity")
    records = univ.get_records()
    split = build_split_v4(sorted({record.puzzle for record in records}), seed=20260813)
    folds = {int(fold.outer_fold): fold for fold in split["folds"]}
    fold_results = []
    for fold_id in range(20):
        fold = folds[fold_id]
        held_records = [record for record in records if record.puzzle == fold.held_puzzle]
        prediction = _load_prediction(
            Path(merged_rows[fold_id]["prediction_artifact"]), fold_id
        )
        observations = _observations(univ, held_records, prediction)
        fold_results.append(
            {
                "outer_fold": fold_id,
                "held_puzzle": str(fold.held_puzzle),
                "gate_geometry": gate_geometry(
                    prediction, merged_rows[fold_id]["gate_parameters"]
                ),
                "oracle": oracle_diagnostics(observations, score_rows[fold_id]),
                "distribution": distribution_diagnostics(observations),
            }
        )
    route = route_summary(fold_results, qualification)
    return {
        "schema_version": SCHEMA,
        "status": "POST_V12_DIAGNOSTICS_COMPLETE",
        "evidence_status": "POST_HOC_DEVELOPMENT_DIAGNOSTIC_ONLY",
        "folds": fold_results,
        "gate_geometry_across_folds": summarize_gate_geometries(fold_results),
        "route_summary": route,
        "independent_units": 20,
        "oracle_is_in_sample_upper_bound_not_model_performance": True,
        "new_model_authorized": False,
        "partial_score_inspected": False,
        "external_outcome_accessed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--merged-json", type=Path, required=True)
    parser.add_argument("--score-json", type=Path, required=True)
    parser.add_argument("--qualification-json", type=Path, required=True)
    parser.add_argument("--m2-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args(argv)
    assert_diagnostic_authority(args.repo_root.resolve())
    if args.out_json.exists():
        raise FileExistsError(f"post-V12 diagnostics refuse to overwrite {args.out_json}")
    result = diagnose(
        json.loads(args.merged_json.read_text()),
        json.loads(args.score_json.read_text()),
        json.loads(args.qualification_json.read_text()),
        args.m2_csv,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "result": str(args.out_json)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
