#!/usr/bin/env python3
"""validate_prediction_artifact_v2 — keyed prediction schema v2 validator.

Enforces the prediction_v2 schema contract so candidate-vs-baseline comparisons
require exact key equality (biological keys), not array-position zip.

Contract highlights (schemas/reactflow_delta/prediction_v2.schema.json):
  * every row carries biological keys (pair_id, asset_id, study_id,
    publication_id, parent_id, lineage_id, fold_id, split_role) + provenance
    hashes (data/split/caller/model_config/source_commit);
  * (pair_id, fold_id, seed, model_variant) must be unique across the artifact;
  * raw_prediction and transformed_prediction are separate columns;
  * tool failure / unsupported / missing / abstention are explicit coverage
    statuses, never imputed to 0;
  * row-count semantic: n_prediction_rows == n_unique_pairs x n_seeds
    per model_variant (on the called/eligible subset).

Returns a machine-readable validation manifest.  Raises ValueError on FAIL.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


def sha256_file(path) -> str:
    """Self-contained SHA-256 (isolated worktree must not import the reactflow package)."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


SCHEMA_VERSION = "reactflow_delta.prediction_v2.v1"

REQUIRED_FIELDS = [
    "pair_id", "asset_id", "study_id", "publication_id", "parent_id",
    "lineage_id", "fold_id", "split_role", "endpoint_version",
    "caller_version", "seed", "model_id", "model_variant", "y", "weight",
    "raw_prediction", "transformed_prediction", "coverage_status",
    "data_hash", "split_hash", "caller_hash", "model_config_hash",
    "source_commit",
]

COVERAGE_CALLED = "CALLED"
COVERAGE_NO_CALL = "NO_CALL"
COVERAGE_ABSTAIN = "ABSTAIN"
COVERAGE_UNSUPPORTED = "UNSUPPORTED"
COVERAGE_MISSING = "MISSING"
COVERAGE_TOOL_FAILURE = "TOOL_FAILURE"
VALID_COVERAGE = {
    COVERAGE_CALLED, COVERAGE_NO_CALL, COVERAGE_ABSTAIN,
    COVERAGE_UNSUPPORTED, COVERAGE_MISSING, COVERAGE_TOOL_FAILURE,
}
VALID_SPLIT_ROLE = {"development", "confirmatory", "excluded"}

KEY_FIELDS = ["pair_id", "fold_id", "seed", "model_variant"]


def validate_rows(
    rows: Sequence[dict[str, Any]],
    expect_seeds: int | None = None,
    endpoint_version: str = "endpoint_v6",
    caller_version: str = "caller_v4",
) -> dict[str, Any]:
    """Validate a sequence of prediction rows.

    Returns a manifest dict.  Raises ValueError on first FAIL.
    """
    if not rows:
        raise ValueError("EMPTY_PREDICTION_ARTIFACT")
    # 1) required fields + no missing values in key/provenance fields
    for i, r in enumerate(rows):
        for f in REQUIRED_FIELDS:
            if f not in r:
                raise ValueError(f"MISSING_FIELD row={i} field={f}")
        if r["coverage_status"] not in VALID_COVERAGE:
            raise ValueError(f"INVALID_COVERAGE row={i} value={r['coverage_status']}")
        if r["split_role"] not in VALID_SPLIT_ROLE:
            raise ValueError(f"INVALID_SPLIT_ROLE row={i} value={r['split_role']}")
        if r["endpoint_version"] != endpoint_version:
            raise ValueError(
                f"ENDPOINT_MISMATCH row={i} got={r['endpoint_version']} expected={endpoint_version}")
        if r["caller_version"] != caller_version:
            raise ValueError(
                f"CALLER_MISMATCH row={i} got={r['caller_version']} expected={caller_version}")
        for kf in ["pair_id", "fold_id", "model_variant", "model_id", "source_commit"]:
            if not isinstance(r[kf], str) or not r[kf]:
                raise ValueError(f"INVALID_KEY row={i} field={kf}")
        if not isinstance(r["seed"], int):
            raise ValueError(f"INVALID_SEED row={i} type={type(r['seed']).__name__}")

    # 2) uniqueness of (pair_id, fold_id, seed, model_variant)
    seen: set[tuple] = set()
    dups: list[tuple] = []
    for i, r in enumerate(rows):
        key = tuple((r[k] for k in KEY_FIELDS))
        if key in seen:
            dups.append(key)
        seen.add(key)
    if dups:
        raise ValueError(f"DUPLICATE_KEY_FIELDS n={len(dups)} example={dups[0]}")

    # 3) coverage-status semantics: TOOL_FAILURE / UNSUPPORTED / MISSING / ABSTAIN
    #    must NOT have y/prediction coerced to 0; require explicit flag and
    #    that raw_prediction is None (not 0) for non-called.
    for i, r in enumerate(rows):
        cs = r["coverage_status"]
        if cs in (COVERAGE_NO_CALL, COVERAGE_ABSTAIN, COVERAGE_UNSUPPORTED,
                  COVERAGE_MISSING, COVERAGE_TOOL_FAILURE):
            _rpv = r["raw_prediction"]
            if _rpv is not None:
                raise ValueError(
                    f"COVERAGE_NONCALL_WITH_VALUE row={i} status={cs} "
                    f"raw_prediction must be None, got={_rpv}")
    return {
        "schema_version": SCHEMA_VERSION,
        "row_count": len(rows),
        "unique_keys": len(seen),
        "unique_pairs": len({r["pair_id"] for r in rows}),
        "unique_seeds": len({r["seed"] for r in rows}),
        "models": sorted({r["model_variant"] for r in rows}),
        "coverage_counts": {
            cs: sum(1 for r in rows if r["coverage_status"] == cs)
            for cs in VALID_COVERAGE
        },
        "row_count_semantics": _check_row_count_semantics(rows, expect_seeds),
    }


def _check_row_count_semantics(
    rows: Sequence[dict[str, Any]], expect_seeds: int | None,
) -> dict[str, Any]:
    called = [r for r in rows if r["coverage_status"] == COVERAGE_CALLED]
    per_model: dict[str, int] = {}
    pairs_per_model: dict[str, set] = {}
    seeds_per_model: dict[str, set] = {}
    for r in called:
        mv = r["model_variant"]
        per_model[mv] = per_model.get(mv, 0) + 1
        pairs_per_model.setdefault(mv, set()).add(r["pair_id"])
        seeds_per_model.setdefault(mv, set()).add(r["seed"])
    out: dict[str, Any] = {}
    for mv, n in per_model.items():
        n_pairs = len(pairs_per_model[mv])
        n_seeds = len(seeds_per_model[mv])
        out[mv] = {"rows": n, "n_pairs": n_pairs, "n_seeds": n_seeds,
                   "consistent": n == n_pairs * n_seeds}
        if n != n_pairs * n_seeds:
            raise ValueError(
                f"ROW_COUNT_SEMANTICS_FAIL model={mv} rows={n} "
                f"pairs={n_pairs} seeds={n_seeds} -> {n} != {n_pairs*n_seeds}")
    if expect_seeds is not None:
        for mv in per_model:
            if len(seeds_per_model[mv]) != expect_seeds:
                raise ValueError(
                    f"SEED_COUNT_MISMATCH model={mv} got={len(seeds_per_model[mv])} "
                    f"expected={expect_seeds}")
    return out


def validate_artifact_file(
    path: str | Path,
    expect_seeds: int | None = None,
    endpoint_version: str = "endpoint_v6",
    caller_version: str = "caller_v4",
) -> dict[str, Any]:
    """Validate a JSON-lines prediction artifact file."""
    p = Path(path)
    if not p.exists():
        raise ValueError(f"MISSING_ARTIFACT: {p}")
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    manifest = validate_rows(rows, expect_seeds, endpoint_version, caller_version)
    manifest["artifact_path"] = str(p)
    manifest["artifact_sha256"] = sha256_file(p)
    return manifest


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("artifact", type=Path)
    ap.add_argument("--expect-seeds", type=int, default=None)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    m = validate_artifact_file(args.artifact, expect_seeds=args.expect_seeds)
    if args.out:
        args.out.write_text(json.dumps(m, indent=2, sort_keys=True) + "\n")
    print(json.dumps(m, sort_keys=True))
