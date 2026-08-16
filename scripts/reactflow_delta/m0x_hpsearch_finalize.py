#!/usr/bin/env python3
"""EPRO_DEV_10 finalizer: write run_manifest.json for the completed HP search.

The training loop in m0x_epro_hpsearch.py completed all 11 configs and saved
best_model.pt, but crashed during a bootstrap step (dummy comparator lacked a
'score' key). The crash is purely in manifest bookkeeping; the model and scores
are intact. This finalizer reconstructs the manifest from the confirmed results
actually produced by that run (read from hpsearch.log) so the iteration is
recorded without re-running the expensive feature build + retraining.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

SEED = 20260804
SCHEMA = "reactflow_delta.m0x_run_manifest.v1"
RUN_ID = "epro_dev_10_hpsearch_20260805"
ITERATION_ID = "EPRO_DEV_10"
HYPOTHESIS_ID = "m0x_h10_hpsearch_changer"
REF_DEV06_AUPRC = 0.7353243279593717


def main() -> int:
    log_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else log_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    text = log_path.read_text(encoding="utf-8")

    # Parse per-config train lines.
    cfg_re = re.compile(
        r"\[cfg\] (\{.*?\}) -> val_auprc=([0-9.]+)@(\d+) \((\d+)s\) params=([0-9,]+)")
    results = []
    for m in cfg_re.finditer(text):
        cfg = json.loads(m.group(1).replace("'", '"'))
        results.append({
            "config": cfg,
            "study_macro_auprc": float(m.group(2)),
            "best_epoch": int(m.group(3)),
            "train_s": int(m.group(4)),
            "param_count": int(m.group(5).replace(",", "")),
        })
    results.sort(key=lambda r: -r["study_macro_auprc"])
    if not results:
        print(f"FATAL: no [cfg] lines parsed from {log_path}", file=sys.stderr)
        return 2

    best = results[0]
    improved = best["study_macro_auprc"] > REF_DEV06_AUPRC

    manifest = {
        "schema_version": SCHEMA,
        "run_id": RUN_ID,
        "iteration_id": ITERATION_ID,
        "hypothesis_id": HYPOTHESIS_ID,
        "authority_amendment": "reactflow_delta_v4_m0x_epro_scope_20260805 (epoch 13)",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "warning": "NOT_FINAL",
        "data": {"train_pairs": 3516, "validation_pairs": 548,
                 "test_sealed": True, "test_accessed": False,
                 "pooled_train_positions": 471344, "feat_dim": 42},
        "split": "publication (train 3516 / val 548)",
        "dev06_reference": {"study_macro_auprc": REF_DEV06_AUPRC},
        "grid": [r["config"] for r in results],
        "results": results,
        "best": {"config": best["config"], "study_macro_auprc": best["study_macro_auprc"],
                 "best_epoch": best["best_epoch"], "param_count": best["param_count"]},
        "improved_vs_dev06": improved,
        "point_gain_vs_dev06": best["study_macro_auprc"] - REF_DEV06_AUPRC,
        "comparison_table": {
            "structure_aware_hpsearch": {"study_macro_auprc": best["study_macro_auprc"],
                                         "config": best["config"]},
            "structure_aware_dev06": {"study_macro_auprc": REF_DEV06_AUPRC},
        },
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(f"manifest written: {out_dir/'run_manifest.json'}")
    print(f"best: {best['config']} -> {best['study_macro_auprc']:.4f} "
          f"(improved_vs_dev06={improved}, gain={best['study_macro_auprc']-REF_DEV06_AUPRC:+.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())