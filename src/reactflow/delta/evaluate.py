"""Unified evaluator for ReactFlow-Delta B0 baselines (v3.3 §12).

Primary metric (§12.1)::

    Skill = 1 - WMAE(pred, true) / WMAE(0, true)

where ``WMAE`` is the pair-quality-weighted mean absolute error computed on the
§12.1 endpoint mask (unedited + aligned + probe-eligible + profile-valid
positions). ``WMAE(0, true)`` is the zero-change reference: the error of
predicting ``Delta r = 0`` everywhere.

Aggregation order (§12.1):

1. Within pair (position-level WMAE/Skill).
2. Parent macro-average (mean over pairs sharing a ``parent_prefix``).
3. Study macro-average (mean over parents sharing a ``citation_doi``).

Secondary metrics (§12.2): WMAE, RMSE, Pearson r, Spearman rho, sign accuracy,
affected-position AUPRC, local/mid/remote WMAE by sequence distance from the
edit position.

The evaluator is deterministic, stdlib + numpy only, and fail-closed: a pair
with an empty endpoint mask or zero ``WMAE(0, true)`` is excluded from the
Skill macro-average but retained for WMAE/RMSE.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

EVALUATOR_SCHEMA_VERSION = "reactflow-delta-b0-evaluator-v1"

# Distance bands for local/mid/remote reporting (mirrors PH0 build_thermo_features).
LOCAL_WINDOW = 10  # |seq_dist| <= 10
REMOTE_THRESHOLD = 20  # |seq_dist| > 20


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairRecord:
    """A single evaluation pair with ground-truth Delta r and endpoint mask."""

    pair_id: str
    parent: str  # parent_prefix
    study: str  # citation_doi
    rdat_path: str
    wt_profile_index: int
    mutant_profile_index: int
    edit_arr_idx: int  # 0-indexed position in delta array
    edit_pos_1indexed: int  # 1-indexed SEQUENCE position
    encoded_ref: str
    aligned_length: int
    delta_true: np.ndarray  # float array, NaN for missing
    endpoint_mask: np.ndarray  # bool array, True = include in eval
    pair_quality_weight: float
    # Sequence-coordinate mapping (array index -> 1-indexed SEQUENCE position).
    seq_positions: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    # Optional WT sequence (lowercase RNA), only loaded when needed by baselines.
    wt_sequence: str | None = None
    # Optional WT reactivity array (aligned to delta array), for the static
    # reactivity baseline (F(seq) -> r_wt). Loaded alongside wt_sequence.
    wt_reactivity: np.ndarray | None = None
    # Optional precomputed WT thermo features from PH0 manifest.
    wt_features: dict[str, Any] | None = None


@dataclass
class PairMetrics:
    """Per-pair metric bundle produced by ``compute_pair_metrics``."""

    pair_id: str
    parent: str
    study: str
    n_positions: int  # number of endpoint-mask positions
    weight: float  # pair_quality_weight
    # Primary
    wmae_pred: float
    wmae_zero: float
    skill: float  # NaN if wmae_zero == 0 (excluded from Skill macro)
    # Secondary
    rmse: float
    pearson_r: float
    spearman_rho: float
    sign_accuracy: float
    affected_auprc: float
    local_wmae: float
    mid_wmae: float
    remote_wmae: float
    local_n: int
    mid_n: int
    remote_n: int


# ---------------------------------------------------------------------------
# Array helpers
# ---------------------------------------------------------------------------


def to_float_array(values: Sequence[Any]) -> np.ndarray:
    """Convert a list (possibly containing ``None``) to a float array.

    ``None`` and non-finite strings become ``NaN`` so that ``np.isnan`` can be
    used uniformly for the missing-position mask.
    """

    arr = np.empty(len(values), dtype=float)
    for i, v in enumerate(values):
        if v is None:
            arr[i] = float("nan")
        else:
            arr[i] = float(v)
    return arr


def build_endpoint_mask(
    delta: np.ndarray,
    wt_reactivity: np.ndarray,
    mut_reactivity: np.ndarray,
    edit_arr_idx: int | None,
) -> np.ndarray:
    """Build the §12.1 endpoint mask.

    A position ``i`` is included iff all of:

    * ``i`` is not the edit position (``unedited``);
    * WT and mutant reactivity are both non-missing (``probe-eligible``,
      ``profile valid``, ``aligned``);
    * ``delta[i]`` is non-missing.
    """

    n = len(delta)
    mask = np.ones(n, dtype=bool)
    if edit_arr_idx is not None and 0 <= edit_arr_idx < n:
        mask[edit_arr_idx] = False
    if len(wt_reactivity) == n:
        mask &= ~np.isnan(wt_reactivity)
    if len(mut_reactivity) == n:
        mask &= ~np.isnan(mut_reactivity)
    mask &= ~np.isnan(delta)
    return mask


# ---------------------------------------------------------------------------
# Skill and secondary metrics
# ---------------------------------------------------------------------------


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation, NaN if undefined (constant input or n < 2)."""

    if x.size < 2:
        return float("nan")
    sx = x - x.mean()
    sy = y - y.mean()
    denom = math.sqrt(float((sx * sx).sum()) * float((sy * sy).sum()))
    if denom == 0.0:
        return float("nan")
    return float((sx * sy).sum() / denom)


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho via rank substitution, NaN if undefined."""

    if x.size < 2:
        return float("nan")
    rx = _rank(x)
    ry = _rank(y)
    return _safe_pearson(rx, ry)


def _rank(a: np.ndarray) -> np.ndarray:
    """Average-rank of values in ``a`` (ties share the mean rank)."""

    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    n = len(a)
    i = 0
    sorted_vals = a[order]
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # ranks are 1-indexed
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _sign_accuracy(pred: np.ndarray, true: np.ndarray) -> float:
    """Fraction of positions where sign(pred) == sign(true). Zero == positive."""

    if true.size == 0:
        return float("nan")
    s_pred = np.sign(pred)
    s_true = np.sign(true)
    # sign(0) == 0; treat 0 as matching 0 only.
    matches = (s_pred == s_true).astype(float)
    return float(matches.mean())


def _affected_auprc(
    pred: np.ndarray, true: np.ndarray, *, threshold: float = 0.05
) -> float:
    """Affected-position AUPRC.

    A position is "affected" iff ``|true| >= threshold``. We compute the area
    under the precision-recall curve induced by thresholding ``|pred|``.
    Returns ``NaN`` if there are no affected positions.
    """

    if true.size == 0:
        return float("nan")
    affected = np.abs(true) >= threshold
    n_pos = int(affected.sum())
    if n_pos == 0 or n_pos == true.size:
        return float("nan")
    order = np.argsort(-np.abs(pred))  # descending |pred|
    tp = 0
    ap_sum = 0.0
    for rank, idx in enumerate(order, start=1):
        if affected[idx]:
            tp += 1
            precision = tp / rank
            ap_sum += precision
    return float(ap_sum / n_pos)


def _distance_band_wmae(
    pred: np.ndarray,
    true: np.ndarray,
    seq_dist: np.ndarray,
    lo: int,
    hi: int | None,
) -> tuple[float, int]:
    """WMAE on positions with ``lo <= |seq_dist| <= hi`` (``hi=None`` means +inf)."""

    d = np.abs(seq_dist)
    if hi is None:
        sel = (d >= lo) & (d <= 1_000_000)
    else:
        sel = (d >= lo) & (d <= hi)
    if not np.any(sel):
        return float("nan"), 0
    err = np.abs(pred - true)[sel]
    return float(err.mean()), int(sel.sum())


def compute_pair_metrics(
    record: PairRecord,
    delta_pred: np.ndarray,
) -> PairMetrics:
    """Compute the full metric bundle for one pair.

    ``delta_pred`` must be aligned to ``record.delta_true`` (same length as the
    delta array). Positions outside ``endpoint_mask`` are ignored.
    """

    mask = record.endpoint_mask
    if mask.size == 0 or not np.any(mask):
        # Empty mask: emit NaN metrics, will be filtered upstream.
        return _empty_pair_metrics(record)

    true = record.delta_true[mask]
    pred = np.asarray(delta_pred, dtype=float)[mask]
    w = float(record.pair_quality_weight)

    # Replace any NaN in pred with 0 (defensive; baselines should already fill).
    pred = np.where(np.isnan(pred), 0.0, pred)

    wmae_pred = float(np.mean(np.abs(pred - true)))
    wmae_zero = float(np.mean(np.abs(true)))
    skill = 1.0 - wmae_pred / wmae_zero if wmae_zero > 0 else float("nan")

    rmse = float(math.sqrt(np.mean((pred - true) ** 2)))
    pearson = _safe_pearson(pred, true)
    spearman = _safe_spearman(pred, true)
    sign_acc = _sign_accuracy(pred, true)
    auprc = _affected_auprc(pred, true)

    # Distance bands (sequence distance from edit position).
    seq_dist = record.seq_positions - float(record.edit_pos_1indexed)
    # On masked positions only.
    full_dist = np.where(mask, seq_dist, np.nan)
    # Build band masks on the masked subset directly.
    d_masked = np.abs(seq_dist[mask])
    local_sel = d_masked <= LOCAL_WINDOW
    mid_sel = (d_masked > LOCAL_WINDOW) & (d_masked <= REMOTE_THRESHOLD)
    remote_sel = d_masked > REMOTE_THRESHOLD

    def _band(sel: np.ndarray) -> tuple[float, int]:
        if not np.any(sel):
            return float("nan"), 0
        return float(np.mean(np.abs(pred - true)[sel])), int(sel.sum())

    local_wmae, local_n = _band(local_sel)
    mid_wmae, mid_n = _band(mid_sel)
    remote_wmae, remote_n = _band(remote_sel)

    return PairMetrics(
        pair_id=record.pair_id,
        parent=record.parent,
        study=record.study,
        n_positions=int(mask.sum()),
        weight=w,
        wmae_pred=wmae_pred,
        wmae_zero=wmae_zero,
        skill=skill,
        rmse=rmse,
        pearson_r=pearson,
        spearman_rho=spearman,
        sign_accuracy=sign_acc,
        affected_auprc=auprc,
        local_wmae=local_wmae,
        mid_wmae=mid_wmae,
        remote_wmae=remote_wmae,
        local_n=local_n,
        mid_n=mid_n,
        remote_n=remote_n,
    )


def _empty_pair_metrics(record: PairRecord) -> PairMetrics:
    return PairMetrics(
        pair_id=record.pair_id,
        parent=record.parent,
        study=record.study,
        n_positions=0,
        weight=record.pair_quality_weight,
        wmae_pred=float("nan"),
        wmae_zero=float("nan"),
        skill=float("nan"),
        rmse=float("nan"),
        pearson_r=float("nan"),
        spearman_rho=float("nan"),
        sign_accuracy=float("nan"),
        affected_auprc=float("nan"),
        local_wmae=float("nan"),
        mid_wmae=float("nan"),
        remote_wmae=float("nan"),
        local_n=0,
        mid_n=0,
        remote_n=0,
    )


# ---------------------------------------------------------------------------
# Aggregation: pair -> parent macro -> study macro
# ---------------------------------------------------------------------------


def _nanmean(values: list[float]) -> float:
    """Mean over non-NaN values; NaN if all are NaN or list is empty."""

    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    mask = ~np.isnan(arr)
    if not np.any(mask):
        return float("nan")
    return float(arr[mask].mean())


_AGG_FIELDS = (
    "skill",
    "wmae_pred",
    "wmae_zero",
    "rmse",
    "pearson_r",
    "spearman_rho",
    "sign_accuracy",
    "affected_auprc",
    "local_wmae",
    "mid_wmae",
    "remote_wmae",
)


def aggregate_metrics(pair_metrics: list[PairMetrics]) -> dict[str, Any]:
    """Aggregate pair metrics: pair -> parent macro -> study macro -> final.

    Returns a nested dict with ``per_pair``, ``per_parent``, ``per_study``,
    and ``final`` sections. Skill is aggregated only over pairs with a
    finite, non-NaN skill value; WMAE/RMSE are aggregated over all pairs with
    a finite value.
    """

    # ---- per_pair ----
    per_pair = [_pair_metrics_to_dict(m) for m in pair_metrics]

    # ---- per_parent (macro over pairs within a parent) ----
    by_parent: dict[str, list[PairMetrics]] = defaultdict(list)
    for m in pair_metrics:
        by_parent[m.parent].append(m)

    per_parent: dict[str, dict[str, Any]] = {}
    for parent, members in by_parent.items():
        per_parent[parent] = {
            "n_pairs": len(members),
            "study": members[0].study,  # parent maps to exactly one study
            **{f: _nanmean([getattr(m, f) for m in members]) for f in _AGG_FIELDS},
        }

    # ---- per_study (macro over parents within a study) ----
    parents_by_study: dict[str, list[str]] = defaultdict(list)
    for parent, agg in per_parent.items():
        parents_by_study[agg["study"]].append(parent)

    per_study: dict[str, dict[str, Any]] = {}
    for study, parents in parents_by_study.items():
        per_study[study] = {
            "n_parents": len(parents),
            "n_pairs": sum(per_parent[p]["n_pairs"] for p in parents),
            **{
                f: _nanmean([per_parent[p][f] for p in parents])
                for f in _AGG_FIELDS
            },
        }

    # ---- final (macro over studies) ----
    final = {
        "n_studies": len(per_study),
        "n_parents": len(per_parent),
        "n_pairs": len(pair_metrics),
        "n_pairs_skill": sum(
            1 for m in pair_metrics if not np.isnan(m.skill)
        ),
        **{
            f: _nanmean([per_study[s][f] for s in per_study])
            for f in _AGG_FIELDS
        },
    }

    return {
        "schema_version": EVALUATOR_SCHEMA_VERSION,
        "per_pair": per_pair,
        "per_parent": per_parent,
        "per_study": per_study,
        "final": final,
    }


def _pair_metrics_to_dict(m: PairMetrics) -> dict[str, Any]:
    return {
        "pair_id": m.pair_id,
        "parent": m.parent,
        "study": m.study,
        "n_positions": m.n_positions,
        "weight": m.weight,
        "skill": m.skill,
        "wmae_pred": m.wmae_pred,
        "wmae_zero": m.wmae_zero,
        "rmse": m.rmse,
        "pearson_r": m.pearson_r,
        "spearman_rho": m.spearman_rho,
        "sign_accuracy": m.sign_accuracy,
        "affected_auprc": m.affected_auprc,
        "local_wmae": m.local_wmae,
        "mid_wmae": m.mid_wmae,
        "remote_wmae": m.remote_wmae,
        "local_n": m.local_n,
        "mid_n": m.mid_n,
        "remote_n": m.remote_n,
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _pair_id_from_entry(entry: Mapping[str, Any]) -> str:
    """Reconstruct the canonical pair_id from a registry entry.

    Format: ``"{rdat_basename}:{wt_idx}:{mut_idx}:{edit_pos_1idx}"``.
    """

    import os

    rdat_name = os.path.basename(entry["rdat_path"])
    mut = entry["matched_mutation"]
    return "{}:{}:{}:{}".format(
        rdat_name,
        entry["wt_profile_index"],
        entry["mutant_profile_index"],
        mut["encoded_position_1indexed"],
    )


def load_split_pairs(
    split_name: str,
    *,
    registry_path: str | Path,
    split_members_path: str | Path,
    thermo_manifest_path: str | Path | None = None,
    rdat_loader=None,
    include_annotation_only: bool = False,
) -> list[PairRecord]:
    """Load ``PairRecord`` objects for a named split.

    Parameters
    ----------
    split_name : {"train", "validation", "test"}
        Split to load.
    registry_path : path
        Path to ``d1_true_pair_registry.json``.
    split_members_path : path
        Path to ``split_members.json`` (PH0).
    thermo_manifest_path : path, optional
        Path to ``thermo_features_manifest.json`` (PH0). When provided, the
        ``wt_features`` dict and ``edit_arr_idx`` are attached to each pair
        (the registry's ``matched_mutation`` does not carry ``edit_arr_idx``).
    rdat_loader : callable, optional
        ``rdat_loader(rdat_path) -> (wt_sequence, seqpos, offset)``. When
        provided, the WT sequence and SEQPOS mapping are attached so that
        thermo/learned baselines can construct mutant sequences. The loader
        is responsible for caching.
    """

    if split_name not in {"train", "validation", "test"}:
        raise ValueError(f"split_name must be train/validation/test, got {split_name!r}")

    with open(registry_path) as f:
        registry_doc = json.load(f)
    with open(split_members_path) as f:
        splits_doc = json.load(f)

    split_key = {"train": "train", "validation": "validation", "test": "test"}[split_name]
    split_pids = set(splits_doc[split_key]["pair_ids"])

    # Index thermo manifest by pair_id for optional attachment.
    thermo_by_pid: dict[str, dict] = {}
    if thermo_manifest_path is not None:
        with open(thermo_manifest_path) as f:
            thermo_doc = json.load(f)
        for entry in thermo_doc["per_pair"]:
            thermo_by_pid[entry["pair_id"]] = entry

    records: list[PairRecord] = []
    for entry in registry_doc["registry"]:
        is_true = entry.get("true_pair")
        is_safe_anno = (
            not is_true
            and include_annotation_only
            and entry.get("exclusion_reasons") == ["annotation_only_alt_not_verifiable"]
            and entry.get("parent_lineage_verified") is True
            and entry.get("has_wt_anchor") is True
            and entry.get("comparable_fraction", 0) >= 0.6
            and entry.get("normalization_domain_compatible") is True
            and entry.get("condition_match_status") == "match"
            and entry.get("in_vivo_in_vitro_mixed") is False
            and entry.get("edit_count") == 1
            and entry.get("edit_type") == "substitution"
        )
        if not is_true and not is_safe_anno:
            continue
        pid = _pair_id_from_entry(entry)
        if pid not in split_pids:
            continue

        mut = entry["matched_mutation"]
        edit_pos_1idx = int(mut["encoded_position_1indexed"])
        encoded_ref = str(mut["encoded_ref"])
        delta = to_float_array(entry["delta_reactivity_normalized"])
        wt_react = to_float_array(entry["wt_reactivity_project"])
        mut_react = to_float_array(entry["mut_reactivity_project"])
        aligned_length = int(entry["aligned_length"])

        # edit_arr_idx from thermo manifest if available, else None.
        thermo_entry = thermo_by_pid.get(pid)
        edit_arr_idx: int | None
        if thermo_entry is not None:
            edit_arr_idx = int(thermo_entry["edit_arr_idx"])
        else:
            edit_arr_idx = None

        mask = build_endpoint_mask(delta, wt_react, mut_react, edit_arr_idx)

        # Sequence positions (1-indexed). Will be filled if rdat_loader given.
        seq_positions = np.full(aligned_length, np.nan, dtype=float)

        wt_sequence: str | None = None
        wt_features: dict | None = None
        if thermo_entry is not None:
            wt_features = dict(thermo_entry.get("wt_features", {}))

        wt_reactivity_arr: np.ndarray | None = None
        if rdat_loader is not None:
            wt_sequence, seqpos_tokens, offset = rdat_loader(entry["rdat_path"])
            # Map array index -> 1-indexed SEQUENCE position.
            from reactflow.delta.thermo_state import seqpos_to_sequence_positions

            sp = seqpos_to_sequence_positions(seqpos_tokens, offset)
            for i, s in enumerate(sp):
                if i < aligned_length and s is not None:
                    seq_positions[i] = float(s)
            # WT reactivity aligned to the delta array (already projected by
            # the registry builder).
            wt_reactivity_arr = wt_react

        records.append(
            PairRecord(
                pair_id=pid,
                parent=entry["parent_prefix"],
                study=entry["citation_doi"],
                rdat_path=entry["rdat_path"],
                wt_profile_index=int(entry["wt_profile_index"]),
                mutant_profile_index=int(entry["mutant_profile_index"]),
                edit_arr_idx=edit_arr_idx if edit_arr_idx is not None else -1,
                edit_pos_1indexed=edit_pos_1idx,
                encoded_ref=encoded_ref,
                aligned_length=aligned_length,
                delta_true=delta,
                endpoint_mask=mask,
                pair_quality_weight=float(entry.get("pair_quality_weight", 1.0)),
                seq_positions=seq_positions,
                wt_sequence=wt_sequence,
                wt_reactivity=wt_reactivity_arr,
                wt_features=wt_features,
            )
        )

    # Preserve split order from split_members.json.
    pid_order = {pid: i for i, pid in enumerate(splits_doc[split_key]["pair_ids"])}
    records.sort(key=lambda r: pid_order.get(r.pair_id, 1_000_000))
    return records


def make_rdat_loader():
    """Build a cached RDAT loader returning ``(wt_sequence, seqpos, offset)``.

    Imports ``parse_rdat`` lazily so the evaluator module stays importable in
    environments without the RDAT parser (e.g. GPU envs without ViennaRNA).
    """

    from reactflow.delta.rdat import parse_rdat  # type: ignore

    cache: dict[str, tuple[str, list[str], int]] = {}

    def _load(rdat_path: str) -> tuple[str, list[str], int]:
        if rdat_path not in cache:
            doc = parse_rdat(rdat_path)
            wt_seq = doc["headers"]["SEQUENCE"]
            seqpos = list(doc["seqpos"])
            offset = int(doc["headers"]["OFFSET"])
            cache[rdat_path] = (wt_seq, seqpos, offset)
        return cache[rdat_path]

    return _load


# ---------------------------------------------------------------------------
# Top-level evaluation entry point
# ---------------------------------------------------------------------------


def evaluate_predictions(
    records: list[PairRecord],
    predictions: Mapping[str, Sequence[float]],
    *,
    baseline_name: str = "unknown",
    runtime_seconds: float | None = None,
    peak_gpu_mb: float | None = None,
    param_count: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate a baseline's predictions against ground truth.

    ``predictions`` maps ``pair_id -> delta_pred array`` (aligned to the delta
    array of the corresponding pair). Pairs missing from ``predictions`` are
    skipped and recorded in ``missing_pair_ids``.
    """

    pair_metrics: list[PairMetrics] = []
    missing: list[str] = []
    shape_errors: list[dict[str, Any]] = []

    for rec in records:
        if rec.pair_id not in predictions:
            missing.append(rec.pair_id)
            continue
        pred = np.asarray(predictions[rec.pair_id], dtype=float)
        if pred.shape[0] != rec.delta_true.shape[0]:
            shape_errors.append(
                {
                    "pair_id": rec.pair_id,
                    "expected": int(rec.delta_true.shape[0]),
                    "got": int(pred.shape[0]),
                }
            )
            continue
        pair_metrics.append(compute_pair_metrics(rec, pred))

    agg = aggregate_metrics(pair_metrics)

    result = {
        "schema_version": EVALUATOR_SCHEMA_VERSION,
        "baseline_name": baseline_name,
        "n_pairs_evaluated": len(pair_metrics),
        "n_pairs_total": len(records),
        "missing_pair_ids": missing,
        "shape_errors": shape_errors,
        "runtime_seconds": runtime_seconds,
        "peak_gpu_mb": peak_gpu_mb,
        "param_count": param_count,
        "aggregation": agg,
    }
    if extra:
        result["extra"] = extra
    return result
