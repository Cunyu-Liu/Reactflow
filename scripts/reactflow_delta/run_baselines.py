#!/usr/bin/env python3
"""Run B0 baselines and write results + failure table (v3.3 §10/§12).

This runner is env-aware: it imports ``reactflow.delta.baselines`` and
attempts each requested baseline. Baselines whose dependencies are missing
(ViennaRNA, EternaFold CLI, torch) are recorded in the failure table with
their exception, not crashed.

Invocation per env::

    # editflow311: non-learned + ViennaRNA thermo
    PYTHONPATH=src python scripts/reactflow_delta/run_baselines.py \\
        --baselines zero_change,mutation_type_mean,distance_decay,edit_only,nearest_train,local_release,rnafold,rnaplfold \\
        --output-dir artifacts/reactflow_delta/b0

    # rna_baselines: EternaFold
    PYTHONPATH=src python scripts/reactflow_delta/run_baselines.py \\
        --baselines eternafold --output-dir artifacts/reactflow_delta/b0

    # pc_cng_gpu: learned baselines
    PYTHONPATH=src python scripts/reactflow_delta/run_baselines.py \\
        --baselines static_reactivity,siamese_matched,generic_paired_matched \\
        --device cuda --output-dir artifacts/reactflow_delta/b0

Results are merged into ``results.json`` and ``failure_table.json`` in the
output directory. Each invocation appends/updates the named baselines.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from reactflow.delta.baselines import (  # noqa: E402
    BASELINE_REGISTRY,
    Baseline,
    count_parameters,
)
from reactflow.delta.evaluate import (  # noqa: E402
    evaluate_predictions,
    load_split_pairs,
    make_rdat_loader,
)

# ---------------------------------------------------------------------------
# Failure-table entries for tools known to be unavailable (§10.2/§10.3).
# Recorded so the B0 gate can audit that we did not skip hard baselines.
# ---------------------------------------------------------------------------

KNOWN_MISSING_TOOLS: list[dict[str, str]] = [
    {
        "name": "LinearPartition",
        "family": "thermo",
        "reason": "LinearPartition CLI not installed in any conda env on the server.",
        "attempted": "import shutil; shutil.which('linearpartition')",
    },
    {
        "name": "RNAstructure",
        "family": "thermo",
        "reason": "RNAstructure partition binary not installed.",
        "attempted": "import shutil; shutil.which('Partition')",
    },
    {
        "name": "RNAsnp",
        "family": "thermo",
        "reason": "RNAsnp CLI not installed.",
        "attempted": "import shutil; shutil.which('RNAsnp')",
    },
    {
        "name": "SNPfold",
        "family": "thermo",
        "reason": "SNPfold CLI not installed.",
        "attempted": "import shutil; shutil.which('SNPfold')",
    },
    {
        "name": "remuRNA",
        "family": "thermo",
        "reason": "remuRNA CLI not installed.",
        "attempted": "import shutil; shutil.which('remuRNA')",
    },
    {
        "name": "Riprap",
        "family": "thermo",
        "reason": "Riprap CLI not installed.",
        "attempted": "import shutil; shutil.which('riprap')",
    },
    {
        "name": "VariantFoldRNA",
        "family": "thermo",
        "reason": "VariantFoldRNA not installed.",
        "attempted": "import shutil; shutil.which('VariantFoldRNA')",
    },
    {
        "name": "Rchange",
        "family": "thermo",
        "reason": "Rchange executable portion not installed.",
        "attempted": "import shutil; shutil.which('rchange')",
    },
    {
        "name": "RibonanzaNet",
        "family": "learned_independent",
        "reason": "RibonanzaNet weights/package not installed (no checkpoint access).",
        "attempted": "import ribonanzanet",
    },
    {
        "name": "RibonanzaNet2",
        "family": "learned_independent",
        "reason": "RibonanzaNet2 weights/package not installed (no checkpoint access).",
        "attempted": "import ribonanzanet2",
    },
    {
        "name": "eFold",
        "family": "learned_independent",
        "reason": "eFold executable head not available; no trained checkpoint.",
        "attempted": "import efold",
    },
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _peak_gpu_mb(device: str) -> float | None:
    if device == "cpu":
        return None
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        # Caller will read peak after run; here we just reset.
        return None
    except Exception:
        return None


def _read_peak_gpu_mb(device: str) -> float | None:
    if device == "cpu":
        return None
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        torch.cuda.synchronize()
        return float(torch.cuda.max_memory_allocated() / (1024 * 1024))
    except Exception:
        return None


def run_one_baseline(
    name: str,
    ctor_kwargs: dict,
    train_pairs,
    eval_pairs,
    *,
    device: str,
) -> dict:
    """Fit + predict + evaluate one baseline. Returns a result dict.

    On any exception, returns a dict with ``status="failed"`` and the
    traceback so the failure table stays auditable.
    """

    t0 = time.perf_counter()
    _peak_gpu_mb(device)  # reset peak
    try:
        if name not in BASELINE_REGISTRY:
            raise KeyError(f"unknown baseline {name!r}")
        baseline: Baseline = BASELINE_REGISTRY[name](**ctor_kwargs)
        baseline.fit(train_pairs)
        predictions: dict[str, np.ndarray] = {}
        for p in eval_pairs:
            predictions[p.pair_id] = baseline.predict(p)
        runtime = time.perf_counter() - t0
        peak_gpu = _read_peak_gpu_mb(device)
        params = count_parameters(baseline)
        result = evaluate_predictions(
            eval_pairs,
            predictions,
            baseline_name=name,
            runtime_seconds=runtime,
            peak_gpu_mb=peak_gpu,
            param_count=params,
            extra={
                "device": device,
                "ctor_kwargs": _jsonable(ctor_kwargs),
                "is_learned": getattr(baseline, "is_learned", False),
                "requires_wt_sequence": getattr(baseline, "requires_wt_sequence", False),
            },
        )
        result["status"] = "ok"
        result["finished_at_utc"] = _now_iso()
        return result
    except Exception as exc:  # noqa: BLE001
        runtime = time.perf_counter() - t0
        return {
            "schema_version": "reactflow-delta-b0-baseline-result-v1",
            "baseline_name": name,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error_message": str(exc)[:1000],
            "traceback": traceback.format_exc()[:4000],
            "runtime_seconds": runtime,
            "finished_at_utc": _now_iso(),
            "device": device,
        }


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    return str(obj)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--baselines",
        required=True,
        help="Comma-separated baseline names, or 'all' for everything in the registry.",
    )
    ap.add_argument(
        "--registry",
        default="artifacts/reactflow_delta/d2r/d1_true_pair_registry.json",
    )
    ap.add_argument(
        "--splits",
        default="artifacts/reactflow_delta/ph0/split_members.json",
    )
    ap.add_argument(
        "--thermo-manifest",
        default="artifacts/reactflow_delta/ph0/thermo_features_manifest.json",
    )
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--eval-split", default="test")
    ap.add_argument("--output-dir", default="artifacts/reactflow_delta/b0")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--epochs", type=int, default=8, help="epochs for learned baselines")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Decide which baselines need WT sequence.
    if args.baselines.strip() == "all":
        names = list(BASELINE_REGISTRY.keys())
    else:
        names = [n.strip() for n in args.baselines.split(",") if n.strip()]

    needs_seq = any(
        getattr(BASELINE_REGISTRY[n], "requires_wt_sequence", False)
        or getattr(BASELINE_REGISTRY[n], "requires_seq_positions", False)
        for n in names
        if n in BASELINE_REGISTRY
    )

    # Load train + eval pairs.
    rdat_loader = make_rdat_loader() if needs_seq else None
    print(f"[run_baselines] loading train split (rdat_loader={'on' if rdat_loader else 'off'})...", flush=True)
    train_pairs = load_split_pairs(
        args.train_split,
        registry_path=args.registry,
        split_members_path=args.splits,
        thermo_manifest_path=args.thermo_manifest,
        rdat_loader=rdat_loader,
    )
    print(f"[run_baselines] train pairs: {len(train_pairs)}", flush=True)
    eval_pairs = load_split_pairs(
        args.eval_split,
        registry_path=args.registry,
        split_members_path=args.splits,
        thermo_manifest_path=args.thermo_manifest,
        rdat_loader=rdat_loader,
    )
    print(f"[run_baselines] eval pairs ({args.eval_split}): {len(eval_pairs)}", flush=True)

    # Load existing results + failure table (merge).
    results_path = out_dir / "results.json"
    failure_path = out_dir / "failure_table.json"
    if results_path.exists():
        results_doc = json.loads(results_path.read_text())
    else:
        results_doc = {
            "schema_version": "reactflow-delta-b0-results-v1",
            "created_at_utc": _now_iso(),
            "baselines": {},
        }
    if failure_path.exists():
        failure_doc = json.loads(failure_path.read_text())
    else:
        failure_doc = {
            "schema_version": "reactflow-delta-b0-failure-table-v1",
            "created_at_utc": _now_iso(),
            "missing_tools": list(KNOWN_MISSING_TOOLS),
            "baseline_failures": {},
        }

    # Run each baseline.
    for name in names:
        if name not in BASELINE_REGISTRY:
            print(f"[run_baselines] SKIP unknown baseline {name!r}", flush=True)
            failure_doc["baseline_failures"][name] = {
                "reason": "unknown baseline name",
                "finished_at_utc": _now_iso(),
            }
            continue
        ctor_kwargs: dict = {}
        # Learned baselines take training hyperparams.
        cls = BASELINE_REGISTRY[name]
        if getattr(cls, "is_learned", False):
            ctor_kwargs = {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "device": args.device,
                "seed": args.seed,
            }
        # EternaFold: allow custom binary path.
        if name == "eternafold":
            ctor_kwargs = {"device": args.device} if False else {}

        print(f"[run_baselines] running {name} (kwargs={ctor_kwargs})...", flush=True)
        result = run_one_baseline(
            name, ctor_kwargs, train_pairs, eval_pairs, device=args.device
        )
        if result.get("status") == "ok":
            results_doc["baselines"][name] = result
            print(
                f"[run_baselines] {name}: Skill={result['aggregation']['final']['skill']:.4f} "
                f"WMAE={result['aggregation']['final']['wmae_pred']:.6f} "
                f"runtime={result['runtime_seconds']:.1f}s",
                flush=True,
            )
        else:
            failure_doc["baseline_failures"][name] = {
                "error_type": result.get("error_type"),
                "error_message": result.get("error_message"),
                "traceback": result.get("traceback"),
                "runtime_seconds": result.get("runtime_seconds"),
                "finished_at_utc": result.get("finished_at_utc"),
                "device": result.get("device"),
            }
            print(
                f"[run_baselines] {name} FAILED: {result.get('error_type')}: "
                f"{result.get('error_message', '')[:120]}",
                flush=True,
            )

    # Persist.
    results_doc["updated_at_utc"] = _now_iso()
    failure_doc["updated_at_utc"] = _now_iso()
    results_path.write_text(json.dumps(results_doc, indent=2))
    failure_path.write_text(json.dumps(failure_doc, indent=2))
    print(f"[run_baselines] wrote {results_path}", flush=True)
    print(f"[run_baselines] wrote {failure_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
