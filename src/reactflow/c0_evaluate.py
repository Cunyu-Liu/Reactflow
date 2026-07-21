"""C0 structure metrics and validation-locked decoder manifests."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Optional, Sequence

from reactflow.constraints import matrix_to_pairs
from reactflow.metrics import pair_confusion


DISTANCE_BINS = (
    ("short", 1, 11),
    ("medium", 12, 23),
    ("long", 24, None),
)


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return 0.0 if denominator == 0 else 2 * tp / denominator


def mcc_from_counts(tp: int, fp: int, fn: int, tn: int) -> float:
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return 0.0 if denominator == 0 else (tp * tn - fp * fn) / denominator


def shifted_pair_counts(
    predicted: Sequence[Sequence[float]],
    target: Sequence[Sequence[float]],
    *,
    tolerance: int = 1,
) -> dict:
    """Match predicted and target pairs one-to-one within endpoint tolerance."""

    predicted_pairs = list(matrix_to_pairs(predicted))
    target_pairs = list(matrix_to_pairs(target))
    unmatched = set(range(len(target_pairs)))
    tp = 0
    for left, right in predicted_pairs:
        candidates = [
            index
            for index in unmatched
            if abs(left - target_pairs[index][0]) <= tolerance
            and abs(right - target_pairs[index][1]) <= tolerance
        ]
        if not candidates:
            continue
        chosen = min(
            candidates,
            key=lambda index: (
                abs(left - target_pairs[index][0]) + abs(right - target_pairs[index][1]),
                target_pairs[index],
            ),
        )
        unmatched.remove(chosen)
        tp += 1
    return {"tp": tp, "fp": len(predicted_pairs) - tp, "fn": len(target_pairs) - tp}


def pair_confusion_by_distance(
    predicted: Sequence[Sequence[float]],
    target: Sequence[Sequence[float]],
    *,
    min_distance: int,
    max_distance: Optional[int],
) -> dict:
    """Return pair confusion over one disjoint sequence-distance slice."""

    if len(predicted) != len(target) or any(len(a) != len(b) for a, b in zip(predicted, target)):
        raise ValueError("predicted and target matrices must have the same shape")
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for left in range(len(predicted)):
        for right in range(left + 1, len(predicted)):
            distance = right - left
            if distance < min_distance or (max_distance is not None and distance > max_distance):
                continue
            pred = float(predicted[left][right]) > 0.5
            truth = float(target[left][right]) > 0.5
            if pred and truth:
                counts["tp"] += 1
            elif pred:
                counts["fp"] += 1
            elif truth:
                counts["fn"] += 1
            else:
                counts["tn"] += 1
    return counts


def metrics_from_confusion(confusion: Mapping[str, int]) -> dict:
    tp, fp, fn, tn = (int(confusion[key]) for key in ("tp", "fp", "fn", "tn"))
    return {
        "precision": 0.0 if tp + fp == 0 else tp / (tp + fp),
        "recall": 0.0 if tp + fn == 0 else tp / (tp + fn),
        "f1": f1_from_counts(tp, fp, fn),
        "mcc": mcc_from_counts(tp, fp, fn, tn),
        "confusion": dict(confusion),
    }


def structure_record_metrics(
    predicted: Sequence[Sequence[float]],
    target: Sequence[Sequence[float]],
) -> dict:
    confusion = pair_confusion(predicted, target)
    tp, fp, fn, tn = (confusion[key] for key in ("tp", "fp", "fn", "tn"))
    shifted = shifted_pair_counts(predicted, target)
    predicted_count = len(matrix_to_pairs(predicted))
    target_count = len(matrix_to_pairs(target))
    distance_bins = {
        label: metrics_from_confusion(
            pair_confusion_by_distance(
                predicted,
                target,
                min_distance=min_distance,
                max_distance=max_distance,
            )
        )
        for label, min_distance, max_distance in DISTANCE_BINS
    }
    return {
        "precision": 0.0 if tp + fp == 0 else tp / (tp + fp),
        "recall": 0.0 if tp + fn == 0 else tp / (tp + fn),
        "exact_f1": f1_from_counts(tp, fp, fn),
        "shifted_f1": f1_from_counts(shifted["tp"], shifted["fp"], shifted["fn"]),
        "mcc": mcc_from_counts(tp, fp, fn, tn),
        "predicted_pair_count": predicted_count,
        "target_pair_count": target_count,
        "pair_count_ratio": None if target_count == 0 else predicted_count / target_count,
        "confusion": confusion,
        "distance_bins": distance_bins,
    }


def aggregate_structure_records(records: Sequence[Mapping[str, object]]) -> dict:
    if not records:
        return {"count": 0}
    scalar_names = ("precision", "recall", "exact_f1", "shifted_f1", "mcc")
    pooled = {key: 0 for key in ("tp", "fp", "fn", "tn")}
    predicted_pairs = target_pairs = 0
    legal = 0
    runtimes = []
    distance_pooled = {
        label: {key: 0 for key in ("tp", "fp", "fn", "tn")}
        for label, _min_distance, _max_distance in DISTANCE_BINS
    }
    distance_macro = {label: [] for label, _min_distance, _max_distance in DISTANCE_BINS}
    for record in records:
        metrics = record["metrics"]
        for key in pooled:
            pooled[key] += int(metrics["confusion"][key])
        predicted_pairs += int(metrics["predicted_pair_count"])
        target_pairs += int(metrics["target_pair_count"])
        legal += int(bool(record.get("legal")))
        runtimes.append(float(record.get("runtime_seconds", 0.0)))
        for label in distance_pooled:
            bin_metrics = metrics["distance_bins"][label]
            distance_macro[label].append(float(bin_metrics["f1"]))
            for key in distance_pooled[label]:
                distance_pooled[label][key] += int(bin_metrics["confusion"][key])
    runtimes.sort()
    count = len(records)
    return {
        "count": count,
        **{f"mean_{name}": sum(float(record["metrics"][name]) for record in records) / count for name in scalar_names},
        "micro_f1": f1_from_counts(pooled["tp"], pooled["fp"], pooled["fn"]),
        "micro_mcc": mcc_from_counts(pooled["tp"], pooled["fp"], pooled["fn"], pooled["tn"]),
        "predicted_pair_count": predicted_pairs,
        "target_pair_count": target_pairs,
        "pair_count_ratio": None if target_pairs == 0 else predicted_pairs / target_pairs,
        "legality_rate": legal / count,
        "runtime_seconds_total": sum(runtimes),
        "runtime_seconds_mean": sum(runtimes) / count,
        "runtime_seconds_p50": runtimes[(count - 1) // 2],
        "runtime_seconds_p95": runtimes[min(count - 1, math.ceil(0.95 * count) - 1)],
        "confusion": pooled,
        "distance_bins": {
            label: {
                **metrics_from_confusion(confusion),
                "mean_f1": sum(distance_macro[label]) / count,
            }
            for label, confusion in distance_pooled.items()
        },
    }


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_sha256() -> str:
    """Hash the installed ReactFlow Python package independent of its root path."""

    digest = hashlib.sha256()
    package_root = Path(__file__).resolve().parent
    for path in sorted(package_root.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def frozen_feature_provenance(path: Optional[Path]) -> dict:
    """Record the immutable top-level index used by a frozen-feature lookup."""

    if path is None:
        return {"present": False, "path": None, "manifest_path": None, "manifest_sha256": None}
    root = path.resolve()
    if not root.is_dir():
        raise ValueError(f"frozen feature directory does not exist: {root}")
    candidates = (root / "sharded_manifest.json", root / "manifest.json", root / "provenance.json")
    manifest = next((candidate for candidate in candidates if candidate.is_file()), None)
    if manifest is None:
        raise ValueError("frozen feature directory has no auditable top-level manifest")
    return {
        "present": True,
        "path": str(root),
        "manifest_path": str(manifest),
        "manifest_sha256": sha256_path(manifest),
    }


def verify_frozen_feature_provenance(expected: Mapping[str, object], path: Optional[Path]) -> dict:
    actual = frozen_feature_provenance(path)
    for key in ("present", "manifest_sha256"):
        if actual.get(key) != expected.get(key):
            raise ValueError(f"frozen feature provenance mismatch: {key}")
    return actual


def read_decoder_manifest(path: Path, *, checkpoint_path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("unsupported decoder manifest schema")
    expected = payload.get("checkpoint_sha256")
    actual = sha256_path(checkpoint_path)
    if expected != actual:
        raise ValueError("decoder manifest checkpoint hash mismatch")
    if payload.get("fitted_split") != "validation":
        raise ValueError("decoder manifest was not fitted on validation")
    expected_code = payload.get("code_sha256")
    if not expected_code:
        raise ValueError("decoder manifest is missing code hash")
    if expected_code != code_sha256():
        raise ValueError("decoder manifest code hash mismatch")
    return payload
