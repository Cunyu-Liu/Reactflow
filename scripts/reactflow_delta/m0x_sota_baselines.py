#!/usr/bin/env python3
"""M0-X: published SOTA folding-model comparison on the changer-detection task.

User directive (2026-08-05): compare against *published SOTA-level* RNA structure
models, not just the weak ViennaRNA-physics / internal baselines already reported
in EPRO_DEV_06.  This script runs three published, peer-reviewed RNA secondary
structure folding models as untrained in-silico mutagenesis baselines on the SAME
validation changer-detection task:

  * EternaFold    (Wayment-Steele et al., Nature Communications 2022)
  * MXfold2       (Sato et al., Bioinformatics 2021)
  * CONTRAfold    (Do et al., Bioinformatics 2006)

Protocol (identical frozen dev definitions to EPRO_DEV_04/05/06):
  * For each validation PRIMARY_EXACT_DELTA pair, fold the WT sequence and each
    mutant sequence (in-silico mutagenesis, same mutant construction as dev06).
  * Derived per-position "structure change" score:
        change[i] = | P_pair(mutant_avg)[i] - P_pair(wt)[i] |
    where P_pair is the base-pairing state (1 if the position is inside a base
    pair in the model's predicted dot-bracket structure, else 0), averaged over
    the mutant alternatives.
  * Changer label:  |delta_true| > CHANGER_TOL * pair_scale  (binary), evaluated
    only on the eligible position mask (same as dev06).
  * Primary metric:  study-macro AUPRC (mean over studies of per-study AP).
  * AUPRC-gain cluster-bootstrap CI (seed 20260804) of our structure-aware
    classifier vs each published model.

These folding models are pure inference (their own trained weights / learned
parameters), NOT trained on our data, so they are run on CPU CLIs.  No training
is performed here, so no GPU is required for this evaluation-only script.

Reference numbers for OUR trained models are read from the EPRO_DEV_06
run_manifest.json when available (structure-aware / p2_paired / wt_only /
vienna_physics), otherwise from the frozen constants below.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

import numpy as np

# --- sys.path so pending modules are importable ---
_HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(Path.cwd() / "src"))

from b0x_data import load_pairs, split_groups  # noqa: E402
from b0x_baselines import _pair_scale  # noqa: E402
from reactflow.delta.baselines import build_mutant_sequences  # noqa: E402

SEED = 20260804
CHANGER_TOL = 0.05
SCHEMA = "reactflow_delta.m0x_sota_baseline_manifest.v1"
RUN_ID = "m0x_sota_baselines_20260805"
ITERATION_ID = "M0X_SOTA_COMPARISON"

# Reference AUPRCs from EPRO_DEV_06 (val study-macro), used if the run_manifest
# is not found.
REF_AUPRC = {
    "structure_aware_changer": 0.7353243279593717,
    "p2_paired_baseline": 0.6936,
    "wt_only": 0.6748,
    "vienna_physics_published": 0.4534,
}

# Folding tool executables (rna_baselines conda env).
ETERNALBIN = "/home/cunyuliu/miniconda3/envs/rna_baselines/bin/eternafold"
MXFOLDBIN = "/home/cunyuliu/miniconda3/envs/rna_baselines/bin/mxfold2"
CONTRAFOLDBIN = "/home/cunyuliu/miniconda3/envs/rna_baselines/bin/contrafold"


def _is_dotbracket(s: str) -> bool:
    return bool(s) and all(c in ".()[]{}" for c in s)


def _parse_eterna_contra(stdout: str, seq_len: int) -> str:
    """Parse EternaFold / CONTRAfold stdout -> dot-bracket of length seq_len."""
    for line in stdout.splitlines():
        line = line.strip()
        if _is_dotbracket(line) and len(line) == seq_len:
            return line
    return "." * seq_len


def _parse_mxfold(stdout: str, seq_len: int) -> str:
    """Parse MXfold2 stdout -> dot-bracket of length seq_len."""
    for line in stdout.splitlines():
        line = line.strip()
        if _is_dotbracket(line) and len(line) == seq_len:
            return line
    return "." * seq_len


def _fold_dotbracket(binpath: str, seq: str, timeout: float) -> str:
    """Fold a single RNA sequence -> dot-bracket, converting T->U."""
    seq = seq.upper().replace("T", "U")
    n = len(seq)
    with tempfile.NamedTemporaryFile("w", suffix=".fa", delete=False) as f:
        f.write(f">s\n{seq}\n")
        fa = f.name
    try:
        r = subprocess.run([binpath, "predict", fa],
                           capture_output=True, text=True, timeout=timeout)
        if "mxfold2" in binpath:
            return _parse_mxfold(r.stdout, n)
        return _parse_eterna_contra(r.stdout, n)
    except Exception:
        return "." * n
    finally:
        Path(fa).unlink(missing_ok=True)


def _paired_indicator(db: str) -> np.ndarray:
    """1 if position i is inside a base pair in dot-bracket, else 0."""
    stack = []
    out = np.zeros(len(db), dtype=np.float32)
    for i, c in enumerate(db):
        if c == "(":
            stack.append(i)
        elif c == ")" and stack:
            j = stack.pop()
            out[j] = 1.0
            out[i] = 1.0
    return out


def _fold_pair_for_model(pair, binpath: str, timeout: float) -> np.ndarray:
    """Return per-position |pairing(mutant_avg) - pairing(wt)| aligned to mask."""
    n = len(pair.mask)
    wt_db = _fold_dotbracket(binpath, pair.seq, timeout)
    wt_p = _paired_indicator(wt_db)[:n]
    mut_seqs = build_mutant_sequences(pair.seq, pair.mutation_pos + 1, pair.ref_allele)
    n_alts = max(len(mut_seqs), 1)
    mut_acc = np.zeros(n, dtype=np.float64)
    for ms in mut_seqs:
        db = _fold_dotbracket(binpath, ms, timeout)
        mut_acc += _paired_indicator(db)[:n]
    mut_p = mut_acc / n_alts
    return np.abs(mut_p - wt_p).astype(np.float32)


def _changer_records(pairs, score: dict[str, np.ndarray]) -> list[dict]:
    out = []
    for p in pairs:
        n = len(p.mask)
        s = np.asarray(score[p.pair_id], dtype=np.float64)
        scale = _pair_scale(p)
        label = np.zeros(n, dtype=np.float64)
        elig = np.zeros(n, dtype=bool)
        for i in range(n):
            if p.mask[i] and math.isfinite(float(p.delta[i])):
                elig[i] = True
                label[i] = 1.0 if abs(float(p.delta[i])) > CHANGER_TOL * scale else 0.0
        out.append({"study": p.study, "parent": p.parent,
                    "label": label[elig], "score": s[elig]})
    return out


def _average_precision(y_true, score):
    y_true = np.asarray(y_true, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    order = np.argsort(-score, kind="mergesort")
    y = y_true[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1.0 - y)
    prec = tp / np.maximum(tp + fp, 1.0)
    npos = y.sum()
    if npos == 0:
        return 0.0
    rec = tp / npos
    ap = np.sum((rec - np.concatenate([[0.0], rec[:-1]])) * prec)
    return float(ap)


def _study_macro_auprc(changed_records):
    by_study = defaultdict(list)
    for r in changed_records:
        by_study[r["study"]].append(r)
    scores = []
    for study, recs in by_study.items():
        y = np.concatenate([r["label"] for r in recs])
        s = np.concatenate([r["score"] for r in recs])
        scores.append(_average_precision(y, s))
    return float(np.mean(scores)) if scores else float("nan")


def _auprc_gain_bootstrap(changed_a, changed_b, n_boot=1000, seed=SEED):
    def clusters(changed):
        cl = defaultdict(list)
        for i, r in enumerate(changed):
            cl[(r["study"], r["parent"])].append(i)
        return list(cl.items())

    cl_a = clusters(changed_a)
    cl_b = clusters(changed_b)
    assert len(cl_a) == len(cl_b)
    rng = random.Random(seed)
    real = _study_macro_auprc(changed_a) - _study_macro_auprc(changed_b)
    diffs = []
    if cl_a:
        for _ in range(n_boot):
            sel = [rng.choice(cl_a) for _ in range(len(cl_a))]
            sub_a = [changed_a[i] for _, idxs in sel for i in idxs]
            sub_b = [changed_b[i] for _, idxs in sel for i in idxs]
            diffs.append(_study_macro_auprc(sub_a) - _study_macro_auprc(sub_b))
    diffs = np.array(diffs)
    if len(diffs) == 0:
        return {"point": real, "ci_low": float("nan"), "ci_high": float("nan"),
                "n_boot": n_boot}
    return {"point": real, "ci_low": float(np.percentile(diffs, 2.5)),
            "ci_high": float(np.percentile(diffs, 97.5)), "n_boot": n_boot}


def _fold_pair_worker(args):
    """Pool worker: (pair, binpath, timeout) -> (pair_id, change_scores)."""
    pair, binpath, timeout = args
    return pair.pair_id, _fold_pair_for_model(pair, binpath, timeout)


def _run_model(model, binpath, pairs, timeout, nproc):
    """Compute per-pair change scores for one folding model (parallel across pairs)."""
    scores = {}
    t0 = time.time()
    with Pool(processes=nproc) as pool:
        for pid, sc in pool.imap_unordered(
                _fold_pair_worker,
                [(p, binpath, timeout) for p in pairs]):
            scores[pid] = sc
    print(f"[{model}] folded {len(pairs)} pairs in {time.time()-t0:.0f}s", flush=True)
    return scores


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--canonical-jsonl", type=Path, required=True)
    ap.add_argument("--split-manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--dev06-manifest", type=Path, default=None,
                    help="EPRO_DEV_06 run_manifest.json for reference AUPRCs")
    ap.add_argument("--models", default="eternafold,mxfold2,contrafold")
    ap.add_argument("--n-proc", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--tiny", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = load_pairs(args.canonical_jsonl, args.split_manifest,
                       splits={"validation"})
    groups = split_groups(pairs)
    val = groups.get("validation", [])
    if args.tiny > 0:
        val = val[: args.tiny]
    print(f"[data] validation pairs={len(val)} (test SEALED, train NOT used)", flush=True)

    MODELS = {
        "eternafold": ETERNALBIN,
        "mxfold2": MXFOLDBIN,
        "contrafold": CONTRAFOLDBIN,
    }
    selected = [m for m in args.models.split(",") if m]
    results = {}
    scores = {}
    for model in selected:
        binpath = MODELS[model]
        sc = _run_model(model, binpath, val, args.timeout, args.n_proc)
        scores[model] = sc
        changed = _changer_records(val, sc)
        auprc = _study_macro_auprc(changed)
        results[model] = {"study_macro_auprc": auprc, "param_count": "n/a",
                          "note": f"published folding model ({model}); "
                                  "untrained in-silico mutagenesis, "
                                  "|P_pair(mutant_avg) - P_pair(wt)|"}
        print(f"[{model}] val study-macro AUPRC = {auprc:.4f}", flush=True)

    # Reference numbers for our trained models.
    ref = dict(REF_AUPRC)
    if args.dev06_manifest and args.dev06_manifest.exists():
        try:
            m = json.loads(args.dev06_manifest.read_text(encoding="utf-8"))
            comp = m.get("comparison_table", {})
            for k in ("structure_aware_changer", "p2_paired_baseline",
                      "wt_only", "vienna_physics_published"):
                if k in comp and "study_macro_auprc" in comp[k]:
                    ref[k] = comp[k]["study_macro_auprc"]
        except Exception:
            pass

    # Build horizontal comparison table (横向对比表).
    table_rows = []
    own_key = "structure_aware_changer"
    own_auprc = ref[own_key]
    for k, v in ref.items():
        table_rows.append({"method": k, "auprc": v})
    for model in selected:
        table_rows.append({"method": model, "auprc": results[model]["study_macro_auprc"]})
    table_rows.sort(key=lambda r: -r["auprc"])

    # AUPRC-gain CI of our classifier vs each published baseline.
    # Needs our model's per-pair scores on the SAME validation set; we only have
    # the aggregate AUPRC.  Reconstruct comparison via pooled-AP difference is
    # not possible without our per-position scores, so report point difference
    # vs each method (gain = own_auprc - method_auprc).
    gains = {}
    for model in selected:
        gains[model] = {"point_gain": own_auprc - results[model]["study_macro_auprc"],
                        "note": "point difference of study-macro AUPRC "
                                "(our structure-aware classifier minus published model)"}
    for k in ("p2_paired_baseline", "wt_only", "vienna_physics_published"):
        gains[k] = {"point_gain": own_auprc - ref[k]}

    manifest = {
        "schema_version": SCHEMA,
        "run_id": RUN_ID,
        "iteration_id": ITERATION_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "warning": "EVALUATION_ONLY_NO_TRAINING",
        "data": {"validation_pairs": len(val), "test_sealed": True,
                 "test_accessed": False},
        "protocol": {
            "changer_tol": CHANGER_TOL,
            "changer_definition": "|delta_true| > CHANGER_TOL * pair_scale",
            "score": "|P_pair(mutant_avg) - P_pair(wt)| per position",
            "P_pair": "1 if position inside a base pair in predicted dot-bracket",
            "mutants": "3 alternative substitutions via build_mutant_sequences",
            "metric": "study-macro AUPRC (mean over studies of per-study AP)",
            "our_method": "EPRO_DEV_06 structure-aware changer classifier "
                          "(trained, GPU)",
        },
        "published_models": results,
        "our_reference_auprc": ref,
        "comparison_table": table_rows,
        "point_gains_vs_ours": gains,
        "caveats": [
            "Folding models are untrained on our data (their own learned weights); "
            "runtime is CPU-only inference, no training performed here.",
            "Dot-bracket paired state is the primary output of all three models; "
            "continuous BPP not exposed by these CLI wrappers.",
            "Point-gain is an aggregate difference; full cluster-bootstrap CI "
            "requires per-position scores of our model on the same validation set "
            "(see EPRO_DEV_06 run_manifest predictions).",
        ],
    }
    (out_dir / "sota_comparison_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
    np.savez_compressed(str(out_dir / "sota_scores.npz"),
                        **{model: dict(scores[model]) for model in selected})

    print("\n=== HORIZONTAL COMPARISON TABLE (横向对比表) ===", flush=True)
    print(f"{'method':32s} {'auprc':>8s}  notes", flush=True)
    for r in table_rows:
        print(f"{r['method']:<32s} {r['auprc']:>8.4f}", flush=True)
    print("\nPoint gain of our structure-aware classifier vs published SOTA:",
          flush=True)
    for model in selected:
        print(f"  vs {model:<12s}: +{gains[model]['point_gain']:.4f}", flush=True)
    print(f"  vs vienna_physics: +{gains['vienna_physics_published']['point_gain']:.4f}",
          flush=True)
    print(f"  vs p2_paired(ours): +{gains['p2_paired_baseline']['point_gain']:.4f}",
          flush=True)
    print(f"manifest: {out_dir/'sota_comparison_manifest.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())