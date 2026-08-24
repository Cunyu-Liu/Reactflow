#!/usr/bin/env python3
"""Fresh target-identity-corrected B1/MeanAligned outer experts for V8M1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from scripts.reactflow_delta.m2_universe_v1 import M2Universe
from scripts.reactflow_delta.run_model_rescue_v3_expert_rebuild import run_expert_fold
from scripts.reactflow_delta.split_v4_lopo_puzzle import build_split_v4


SCHEMA = "reactflow_delta.model_rescue_v8_corrected_expert_rebuild.v1"
PREDICTION_SCHEMA = "reactflow_delta.model_rescue_v8_corrected_expert_prediction.v1"
ARTIFACT_PREFIX = "v8_corrected"


def assert_v8m1_authority(repo_root: Path) -> None:
    active = yaml.safe_load(
        (repo_root / "configs/reactflow_delta/active_contract.yaml").read_text(
            encoding="utf-8"
        )
    )
    if active["authority"]["current_phase"] != "V8M1":
        raise RuntimeError("V8 corrected expert rebuild is closed outside V8M1")
    if active.get("runnable_phases") != ["V8M1"]:
        raise RuntimeError("V8M1 must be the only runnable phase")
    required = "TARGET_IDENTITY_CORRECTED_B1_AND_MEANALIGNED_FRESH_REBUILD_ONLY"
    if active.get("training_allowed") != required:
        raise RuntimeError("V8M1 fresh corrected expert authority is absent")
    if active.get("held_score_read_allowed") is not False:
        raise RuntimeError("V8M1 requires held scores closed")
    if active.get("partial_fold_score_read_allowed") is not False:
        raise RuntimeError("V8M1 requires partial scores closed")
    if active.get("new_external_outcome_access_allowed") is not False:
        raise RuntimeError("V8M1 requires external outcomes locked")
    if active.get("legacy_v3_expert_reuse_allowed") is not False:
        raise RuntimeError("V8M1 forbids legacy v3 expert reuse")


def _parse_folds(raw: str) -> list[int]:
    folds = [int(value) for value in raw.split(",") if value.strip()]
    if not folds or len(folds) != len(set(folds)) or not set(folds) <= set(range(20)):
        raise ValueError("V8M1 folds must be unique members of 0 through 19")
    return sorted(folds)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--m2-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--folds", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    assert_v8m1_authority(repo_root)
    if args.seed != 0 or args.epochs != 40:
        raise ValueError("V8M1 is frozen to seed 0 and 40 epochs")
    if args.learning_rate != 1e-3 or args.weight_decay != 0.0:
        raise ValueError("V8M1 optimizer configuration is frozen")
    selected = _parse_folds(args.folds)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for fold_id in selected:
        result_path = args.out_dir / (
            f"v8_corrected_expert_fold_result_fold{fold_id}_seed0.json"
        )
        if result_path.exists():
            raise FileExistsError(f"refusing to overwrite V8M1 fold {fold_id}")

    device = args.device if torch.cuda.is_available() else "cpu"
    universe = M2Universe(args.m2_csv)
    identity = universe.build()
    if identity.get("n_canonical_mutant_full_profiles") != 13976 or identity.get(
        "canonical_mutant_full_profile_identity"
    ) != "EXACT_PUZZLE_METHOD_MUTATION":
        raise RuntimeError("V8M1 requires exact canonical target identity")
    records = universe.get_records()
    split = build_split_v4(
        sorted({record.puzzle for record in records}), seed=20260813
    )
    folds = [fold for fold in split["folds"] if int(fold.outer_fold) in selected]
    if len(folds) != len(selected):
        raise ValueError("one or more requested V8M1 folds are absent")

    for fold in folds:
        fold_id = int(fold.outer_fold)
        print(f"[V8M1] fold={fold_id} held={fold.held_puzzle} start", flush=True)
        result = run_expert_fold(
            univ=universe,
            fold=fold,
            records=records,
            device=device,
            out_dir=args.out_dir,
            epochs=40,
            learning_rate=1e-3,
            weight_decay=0.0,
            seed=0,
            artifact_prefix=ARTIFACT_PREFIX,
            result_schema=SCHEMA,
            prediction_schema=PREDICTION_SCHEMA,
        )
        result.update(
            {
                "target_profile_identity": "EXACT_PUZZLE_METHOD_MUTATION",
                "canonical_mutant_full_profiles": 13976,
                "legacy_v3_checkpoint_reused": False,
                "legacy_v3_prediction_reused": False,
            }
        )
        path = args.out_dir / (
            f"v8_corrected_expert_fold_result_fold{fold_id}_seed0.json"
        )
        path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"[V8M1] fold={fold_id} complete", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
