#!/usr/bin/env python3
"""Evaluate external RNA 2D baseline predictions on ReactFlow splits.

This script is intentionally a thin protocol bridge: it does not run eFold,
RNADiffFold, or any other external model.  It scores their exported predictions
against ReactFlow's cache JSONL split files and emits a ``baseline_*_results``
artifact that ``build_sota_alignment_table.py`` can consume without mixing
cited-only numbers into same-split local claims.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple


MMSEQS_TIERS = {"in_clan", "cross_clan", "novel_clan"}
PUBLIC_TIERS = {"archiveII", "PDB", "viral", "lncRNA", "human_mRNA"}
DISTANCE_BINS: Tuple[Tuple[str, int, Optional[int]], ...] = (
    ("short", 1, 11),
    ("medium", 12, 23),
    ("long", 24, None),
)
ALLOWED_PROTOCOLS = ("same_split_local", "local_closest_protocol")


class RunningMetrics:
    """Streaming macro and micro pair metrics for one tier or distance bin."""

    def __init__(self) -> None:
        self.count = 0
        self.sum_f1 = 0.0
        self.sum_mcc = 0.0
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    def add(self, confusion: Mapping[str, int]) -> None:
        tp = int(confusion["tp"])
        fp = int(confusion["fp"])
        fn = int(confusion["fn"])
        tn = int(confusion["tn"])
        self.count += 1
        self.sum_f1 += _f1_from_counts(tp, fp, fn)
        self.sum_mcc += _mcc_from_counts(tp, fp, fn, tn)
        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.tn += tn

    def as_dict(self) -> dict:
        return {
            "count": self.count,
            "mean_f1": 0.0 if self.count == 0 else self.sum_f1 / self.count,
            "mean_mcc": 0.0 if self.count == 0 else self.sum_mcc / self.count,
            "micro_f1": _f1_from_counts(self.tp, self.fp, self.fn),
            "micro_mcc": _mcc_from_counts(self.tp, self.fp, self.fn, self.tn),
            "confusion": {"tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn},
        }


def _f1_from_counts(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return 0.0 if denominator == 0 else 2 * tp / denominator


def _mcc_from_counts(tp: int, fp: int, fn: int, tn: int) -> float:
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return 0.0 if denominator == 0 else (tp * tn - fp * fn) / denominator


def _candidate_count(length: int, *, min_distance: int = 1, max_distance: Optional[int] = None) -> int:
    if length < 2:
        return 0
    upper = length - 1 if max_distance is None else min(max_distance, length - 1)
    if upper < min_distance:
        return 0
    return sum(length - distance for distance in range(min_distance, upper + 1))


def _normalize_pair(raw_pair: Sequence[object], length: int, *, one_based: bool) -> Tuple[int, int]:
    if len(raw_pair) != 2:
        raise ValueError(f"pair must have exactly two indices, got {raw_pair!r}")
    i = int(raw_pair[0])
    j = int(raw_pair[1])
    if one_based:
        i -= 1
        j -= 1
    if i == j:
        raise ValueError(f"self-pair ({i}, {j}) is invalid")
    if not (0 <= i < length and 0 <= j < length):
        raise ValueError(f"pair ({i}, {j}) is out of range for length {length}")
    return (i, j) if i < j else (j, i)


def _pairs_from_obj(obj: Mapping[str, object], *, length: int, one_based: bool) -> frozenset[Tuple[int, int]]:
    raw_pairs = None
    for key in ("predicted_pairs", "prediction", "pairs", "structure"):
        if key in obj:
            raw_pairs = obj[key]
            break
    if raw_pairs is None:
        raise ValueError("record is missing predicted_pairs/prediction/pairs/structure")
    if not isinstance(raw_pairs, (list, tuple)):
        raise ValueError("pairs field must be a list of [i, j] entries")
    return frozenset(_normalize_pair(pair, length, one_based=one_based) for pair in raw_pairs)  # type: ignore[arg-type]


def _record_key(obj: Mapping[str, object], *, ordinal: int) -> str:
    for key in ("source_id", "id", "record_id", "reference"):
        value = obj.get(key)
        if value not in (None, ""):
            return str(value)
    sequence = obj.get("sequence")
    if sequence not in (None, ""):
        return f"sequence:{sequence}"
    return f"ordinal:{ordinal}"


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            yield obj


def _load_predictions(path: Path, *, one_based: bool) -> Dict[str, dict]:
    predictions: Dict[str, dict] = {}
    for ordinal, obj in enumerate(_iter_jsonl(path)):
        sequence = str(obj.get("sequence") or "").upper()
        if not sequence:
            raise ValueError(f"{path} prediction #{ordinal} is missing sequence")
        key = _record_key(obj, ordinal=ordinal)
        if key in predictions:
            raise ValueError(f"duplicate prediction key {key!r} in {path}")
        predictions[key] = {
            "sequence": sequence,
            "pairs": _pairs_from_obj(obj, length=len(sequence), one_based=one_based),
        }
    return predictions


def _pair_confusion(
    predicted: frozenset[Tuple[int, int]],
    target: frozenset[Tuple[int, int]],
    *,
    length: int,
    min_distance: int = 1,
    max_distance: Optional[int] = None,
) -> dict:
    def in_bin(pair: Tuple[int, int]) -> bool:
        distance = pair[1] - pair[0]
        return distance >= min_distance and (max_distance is None or distance <= max_distance)

    predicted_bin = {pair for pair in predicted if in_bin(pair)}
    target_bin = {pair for pair in target if in_bin(pair)}
    tp = len(predicted_bin & target_bin)
    fp = len(predicted_bin - target_bin)
    fn = len(target_bin - predicted_bin)
    tn = _candidate_count(length, min_distance=min_distance, max_distance=max_distance) - tp - fp - fn
    return {"tp": tp, "fp": fp, "fn": fn, "tn": max(tn, 0)}


def _parse_tier_path(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"expected tier=path, got {value!r}")
    tier, path = value.split("=", 1)
    tier = tier.strip()
    if not tier:
        raise ValueError(f"empty tier in {value!r}")
    return tier, Path(path)


def _split_label(tier: str) -> str:
    if tier in MMSEQS_TIERS:
        return f"MMseqs:{tier}"
    if tier in PUBLIC_TIERS:
        return f"eFold-RNAndria:{tier}"
    return tier


def evaluate_tier(
    *,
    tier: str,
    gold_path: Path,
    prediction_path: Optional[Path],
    model: str,
    protocol: str,
    seed_count: str,
    output_path: Path,
    one_based_predictions: bool,
    emit_partial_rows: bool,
) -> Tuple[dict, Optional[dict]]:
    """Score one tier and return its detailed summary plus optional SOTA row."""

    if prediction_path is None:
        gold_count = sum(1 for _ in _iter_jsonl(gold_path))
        return (
            {
                "status": "missing_predictions",
                "gold": str(gold_path),
                "predictions": None,
                "gold_count": gold_count,
                "matched_count": 0,
                "missing_count": gold_count,
                "extra_prediction_count": None,
                "message": "no prediction JSONL was supplied for this tier",
            },
            None,
        )

    predictions = _load_predictions(prediction_path, one_based=one_based_predictions)
    aggregate = RunningMetrics()
    distance = {label: RunningMetrics() for label, _min_d, _max_d in DISTANCE_BINS}
    gold_keys = set()
    missing = []
    sequence_mismatches = []
    duplicate_gold = []

    for ordinal, obj in enumerate(_iter_jsonl(gold_path)):
        sequence = str(obj.get("sequence") or "").upper()
        if not sequence:
            raise ValueError(f"{gold_path} gold record #{ordinal} is missing sequence")
        key = _record_key(obj, ordinal=ordinal)
        if key in gold_keys:
            duplicate_gold.append(key)
            continue
        gold_keys.add(key)
        prediction = predictions.get(key)
        if prediction is None:
            missing.append(key)
            continue
        if prediction["sequence"] != sequence:
            sequence_mismatches.append(key)
            continue
        target = _pairs_from_obj(obj, length=len(sequence), one_based=False)
        predicted = prediction["pairs"]
        aggregate.add(_pair_confusion(predicted, target, length=len(sequence)))
        for label, min_distance, max_distance in DISTANCE_BINS:
            distance[label].add(
                _pair_confusion(
                    predicted,
                    target,
                    length=len(sequence),
                    min_distance=min_distance,
                    max_distance=max_distance,
                )
            )

    extra = sorted(set(predictions) - gold_keys)
    gold_count = len(gold_keys)
    matched = aggregate.count
    if duplicate_gold:
        status = "duplicate_gold"
    elif gold_count == 0:
        status = "empty_gold"
    elif matched == gold_count and not extra and not sequence_mismatches:
        status = "ok"
    elif matched > 0:
        status = "partial"
    else:
        status = "no_matches"

    summary = {
        "status": status,
        "gold": str(gold_path),
        "predictions": str(prediction_path),
        "gold_count": gold_count,
        "matched_count": matched,
        "missing_count": len(missing),
        "extra_prediction_count": len(extra),
        "sequence_mismatch_count": len(sequence_mismatches),
        "duplicate_gold_count": len(duplicate_gold),
        "missing_examples": missing[:10],
        "extra_prediction_examples": extra[:10],
        "sequence_mismatch_examples": sequence_mismatches[:10],
        **aggregate.as_dict(),
        "distance_bins": {label: metrics.as_dict() for label, metrics in distance.items()},
    }

    row = None
    if matched > 0 and (status == "ok" or emit_partial_rows):
        metrics = aggregate.as_dict()
        row = {
            "model": model,
            "protocol": protocol,
            "split": _split_label(tier),
            "seed_count": seed_count,
            "tier": tier,
            "status": "ok" if status == "ok" else "partial",
            "mean_f1": metrics["mean_f1"],
            "mean_mcc": metrics["mean_mcc"],
            "long_f1": distance["long"].as_dict()["mean_f1"],
            "long_recall": _recall_from_confusion(distance["long"].as_dict()["confusion"]),
            "artifact": str(output_path),
        }
    return summary, row


def _recall_from_confusion(confusion: Mapping[str, int]) -> float:
    tp = int(confusion["tp"])
    fn = int(confusion["fn"])
    return 0.0 if tp + fn == 0 else tp / (tp + fn)


def evaluate_baselines(
    *,
    gold_paths: Mapping[str, Path],
    prediction_paths: Mapping[str, Path],
    model: str,
    protocol: str,
    seed_count: str,
    output_path: Path,
    one_based_predictions: bool,
    emit_partial_rows: bool = False,
) -> dict:
    tiers: MutableMapping[str, dict] = {}
    rows = []
    for tier, gold_path in sorted(gold_paths.items()):
        summary, row = evaluate_tier(
            tier=tier,
            gold_path=gold_path,
            prediction_path=prediction_paths.get(tier),
            model=model,
            protocol=protocol,
            seed_count=seed_count,
            output_path=output_path,
            one_based_predictions=one_based_predictions,
            emit_partial_rows=emit_partial_rows,
        )
        tiers[tier] = summary
        if row is not None:
            rows.append(row)
    return {
        "schema_version": 1,
        "model": model,
        "protocol": protocol,
        "seed_count": seed_count,
        "rows": rows,
        "tiers": tiers,
        "notes": (
            "Rows are emitted only for complete tiers by default. Partial tiers remain in "
            "the detailed summary but are withheld from SOTA alignment unless "
            "--emit-partial-rows is set."
        ),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="external baseline model name")
    parser.add_argument(
        "--gold-json",
        action="append",
        default=[],
        help="gold ReactFlow cache split as tier=path; may repeat",
    )
    parser.add_argument(
        "--prediction-json",
        action="append",
        default=[],
        help="external prediction JSONL as tier=path; may repeat",
    )
    parser.add_argument("--output", required=True, help="output baseline results JSON")
    parser.add_argument("--protocol", choices=ALLOWED_PROTOCOLS, default="same_split_local")
    parser.add_argument("--seed-count", default="single_seed")
    parser.add_argument("--one-based-predictions", action="store_true")
    parser.add_argument(
        "--emit-partial-rows",
        action="store_true",
        help="also emit SOTA rows for partially covered tiers; off by default",
    )
    args = parser.parse_args(argv)

    if not args.gold_json:
        parser.error("at least one --gold-json tier=path is required")
    gold_paths = dict(_parse_tier_path(value) for value in args.gold_json)
    prediction_paths = dict(_parse_tier_path(value) for value in args.prediction_json)
    output_path = Path(args.output)
    payload = evaluate_baselines(
        gold_paths=gold_paths,
        prediction_paths=prediction_paths,
        model=args.model,
        protocol=args.protocol,
        seed_count=args.seed_count,
        output_path=output_path,
        one_based_predictions=args.one_based_predictions,
        emit_partial_rows=args.emit_partial_rows,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "rows": len(payload["rows"])}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
