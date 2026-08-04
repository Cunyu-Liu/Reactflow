#!/usr/bin/env python3
"""B0-X data loading and feature construction (contract §20.8).

Loads the frozen D1-X canonical records + D2-X publication-level split and
builds per-PRIMARY_EXACT_DELTA-pair feature/target tensors.  The primary
endpoint is the *full-position continuous delta* (mutant reactivity minus WT
reactivity) on the eligible position mask (contract §12.3).  The eligible mask
excludes edited site, alignment-changed, probe-eligibility-changed, missing,
invalid and unmeasured positions (contract §12.3).

No test split is touched: only train and validation splits are consumed.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_NUC = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}


@dataclass
class Pair:
    pair_id: str
    study: str
    split: str
    parent: str
    seq: str
    mutation_pos: int  # 0-based index into canonical_sequence
    ref_allele: str
    alt_allele: str
    wt_reactivity: list[float]
    mutant_reactivity: list[float]
    mask: list[int]
    delta: list[float]  # raw-scale mutant - wt
    n_eligible: int = 0
    source: str = ""


def _finite(v: Any) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(v)


def _study_of(sa: str) -> str:
    return sa.split("_")[0]


def _parent_of(r: dict) -> str:
    ple = r.get("parent_lineage_evidence") or {}
    return str(ple.get("parent_sequence_sha256", "") or r.get("source_accession", ""))


def _mutation_pos(r: dict) -> int | None:
    coord = r.get("mutation_coordinate_system") or {}
    idx = coord.get("sequence_index_0_based")
    if isinstance(idx, str):
        try:
            return int(idx)
        except ValueError:
            return None
    return idx if isinstance(idx, int) else None


def compute_delta_and_mask(r: dict) -> tuple[list[float | None], list[int]]:
    """Return (delta_reactivity, eligible_mask) at aligned positions."""
    rl = r.get("reactivity_layers", {})
    tf = rl.get("train_frozen", {}).get("reactivity") or []
    wt = r.get("wt_anchor_reactivity") or []
    mask = rl.get("position_mask") or []
    n = min(len(tf), len(wt))
    if not mask:
        mask = [1 if (_finite(tf[i]) and _finite(wt[i])) else 0 for i in range(n)]
    delta: list[float | None] = []
    for i in range(n):
        if _finite(tf[i]) and _finite(wt[i]):
            delta.append(float(tf[i] - wt[i]))
        else:
            delta.append(None)
    return delta, mask[:n]


def load_pairs(
    canonical_jsonl: Path,
    split_manifest: Path,
    *,
    splits: set[str] | None = None,
) -> list[Pair]:
    split = json.loads(split_manifest.read_text(encoding="utf-8"))
    assignment = split.get("assignment", {})
    pairs: list[Pair] = []
    with open(canonical_jsonl, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("data_role") != "PRIMARY_EXACT_DELTA":
                continue
            study = _study_of(r.get("source_accession") or "")
            split_name = assignment.get(study, "UNASSIGNED")
            if splits is not None and split_name not in splits:
                continue
            mpos = _mutation_pos(r)
            if mpos is None:
                continue
            delta, mask = compute_delta_and_mask(r)
            n_eligible = sum(1 for i in range(len(mask)) if mask[i] and _finite(delta[i]))
            if n_eligible == 0:
                continue
            seq = (r.get("canonical_sequence") or "").upper().replace("T", "U")
            rl = r.get("reactivity_layers", {})
            mutant_reactivity = [float(v) for v in rl.get("train_frozen", {}).get("reactivity") or []]
            pairs.append(Pair(
                pair_id=f"{r.get('source_accession')}_{r.get('source_profile_index')}",
                study=study,
                split=split_name,
                parent=_parent_of(r),
                seq=seq,
                mutation_pos=mpos,
                ref_allele=(r.get("ref_allele") or "").upper().replace("T", "U"),
                alt_allele=(r.get("alt_allele") or "").upper().replace("T", "U"),
                wt_reactivity=[float(v) for v in r.get("wt_anchor_reactivity") or []],
                mutant_reactivity=mutant_reactivity,
                mask=mask,
                delta=[float(v) if _finite(v) else 0.0 for v in delta],
                n_eligible=n_eligible,
                source=r.get("source_accession", ""),
            ))
    return pairs


def split_groups(pairs: list[Pair]) -> dict[str, list[Pair]]:
    out: dict[str, list[Pair]] = defaultdict(list)
    for p in pairs:
        out[p.split].append(p)
    return dict(out)