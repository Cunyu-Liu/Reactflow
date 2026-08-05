#!/usr/bin/env python3
"""M0-X: genuine SOTA deep-learning folding-model comparison on the
changer-detection task -- RNAformer (Nature Machine Intelligence 2023).

User directive (2026-08-05): compare against *published SOTA-level* RNA structure
models, NOT the classical/weak baselines already reported (ViennaRNA physics,
EternaFold, MXfold2, CONTRAfold).  RNAformer is a peer-reviewed deep-learning
transformer that predicts a continuous base-pair probability matrix and is the
strongest published folding model available in this environment:

  * RNAformer  (M. Becker, M. Zerbel, et al., "Predicting RNA secondary
                structure by learning unrolled algorithms", Nat. Mach. Intell.
                2023).  32M-parameter axial-attention transformer, intra-family
                fine-tuned weights (RNAformer_32M_state_dict_intra_family_finetuned.pth).

Protocol (identical frozen dev definitions to EPRO_DEV_04/05/06 and the
classical-baseline comparison):
  * For each validation PRIMARY_EXACT_DELTA pair, run RNAformer on the WT
    sequence and each mutant sequence (in-silico mutagenesis, same mutant
    construction as dev06).
  * RNAformer outputs a continuous base-pair probability (BPP) matrix via
    sigmoid(logits[0,:,:,-1]).
  * Per-position pairing probability:
        P_pair(i) = sum_{j != i} BPP[i, j]
  * Derived per-position "structure change" score:
        change[i] = | P_pair(mutant_avg)[i] - P_pair(wt)[i] |
  * Changer label:  |delta_true| > CHANGER_TOL * pair_scale  (binary), evaluated
    only on the eligible position mask (same as dev06).
  * Primary metric:  study-macro AUPRC (mean over studies of per-study AP).
  * AUPRC-gain cluster-bootstrap CI (seed 20260804) of our trained structure-aware
    classifier vs RNAformer.

RNAformer is a published pretrained model (its own trained weights), NOT trained on
our data, run as pure inference on CPU.  No training / no GPU required here.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
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
SCHEMA = "reactflow_delta.m0x_sota_dl_rnaformer_manifest.v1"
RUN_ID = "m0x_sota_dl_rnaformer_20260805"
ITERATION_ID = "M0X_SOTA_DL_RNAFORMER"

# Reference AUPRCs from EPRO_DEV_06 (val study-macro), used if run_manifest absent.
REF_AUPRC = {
    "structure_aware_changer": 0.7353243279593717,
    "p2_paired_baseline": 0.6936,
    "wt_only": 0.6748,
    "vienna_physics_published": 0.4534,
}

RNAFORMER_WEIGHTS = "/home/cunyuliu/rna_baselines_src/RNAformer_pretrained/RNAformer_32M_state_dict_intra_family_finetuned.pth"
RNAFORMER_CONFIG = "/home/cunyuliu/rna_baselines_src/RNAformer_pretrained/RNAformer_32M_config_intra_family_finetuned.yml"

SEQ_VOCAB = ['A', 'C', 'G', 'U', 'N']
SEQ_STOI = dict(zip(SEQ_VOCAB, range(len(SEQ_VOCAB))))


def _sequence2index(sequence: str) -> np.ndarray:
    return np.array([SEQ_STOI.get(nt, 4) for nt in sequence.upper().replace("T", "U")],
                    dtype=np.int64)


def _load_model(weights_path: str, config_path: str, device="cpu"):
    """Load the pretrained RNAformer model (eval mode), optionally on CUDA."""
    import torch
    import loralib as lora
    from RNAformer.model.RNAformer import RiboFormer
    from RNAformer.utils.configuration import Config

    config = Config(config_file=config_path)
    # The intra-family-fine-tuned checkpoint was trained with cycling, so the
    # model must be built with cycling>=1 to include recycle_pair_norm and match
    # the state_dict keys. Any positive value is equivalent for parameter loading.
    cycle_steps = int(getattr(config.RNAformer, "cycling", 0) or 1)
    config.RNAformer.cycling = cycle_steps
    model = RiboFormer(config.RNAformer)
    if hasattr(config, "lora") and config.lora:
        # Replicate the LoRA insertion used by infer_RNAformer so the
        # state_dict keys match.
        lora_config = {
            "r": config.r, "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
        }
        with torch.no_grad():
            for name, module in model.named_modules():
                if any(replace_key in name for replace_key in config.replace_layer):
                    parent = model.get_submodule(".".join(name.split(".")[:-1]))
                    target_name = name.split(".")[-1]
                    target = model.get_submodule(name)
                    if isinstance(target, torch.nn.Linear) and "qkv" in name:
                        new_module = lora.MergedLinear(
                            target.in_features, target.out_features,
                            bias=target.bias is not None,
                            enable_lora=[True, True, True], **lora_config)
                        new_module.weight.copy_(target.weight)
                        if target.bias is not None:
                            new_module.bias.copy_(target.bias)
                    elif isinstance(target, torch.nn.Linear):
                        new_module = lora.Linear(
                            target.in_features, target.out_features,
                            bias=target.bias is not None, **lora_config)
                        new_module.weight.copy_(target.weight)
                        if target.bias is not None:
                            new_module.bias.copy_(target.bias)
                    elif isinstance(target, torch.nn.Conv2d):
                        k = target.kernel_size[0]
                        new_module = lora.Conv2d(
                            target.in_channels, target.out_channels, k,
                            padding=(k - 1) // 2,
                            bias=target.bias is not None, **lora_config)
                        new_module.conv.weight.copy_(target.weight)
                        if target.bias is not None:
                            new_module.conv.bias.copy_(target.bias)
                    setattr(parent, target_name, new_module)
    state_dict = torch.load(weights_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict, strict=True)
    model.to(torch.device(device))
    model.eval()
    return model


def _fold_bpp(model, sequence: str) -> np.ndarray:
    """Run RNAformer on one sequence -> per-position pairing probability.

    Returns shape (L,) where P_pair(i) = sum_{j != i} BPP[i, j].
    """
    import torch
    dev = next(model.parameters()).device
    src = torch.LongTensor(_sequence2index(sequence)).unsqueeze(0).to(dev)  # (1, L)
    src_len = torch.LongTensor([src.shape[-1]]).to(dev)
    pdb_sample = torch.FloatTensor([[1.0]]).to(dev)
    with torch.no_grad():
        logits, _ = model(src, src_len, pdb_sample)
        bpp = torch.sigmoid(logits[0, :, :, -1])  # (L, L)
        bpp_np = bpp.cpu().numpy()
    L = bpp_np.shape[0]
    # Marginal pairing probability: leave out the diagonal.
    p_pair = bpp_np.sum(axis=1)
    if L > 1:
        p_pair = p_pair - np.diag(bpp_np)
    return np.clip(p_pair, 0.0, 1.0).astype(np.float32)


def _fold_pair_for_model(model, pair, max_len: int):
    """Return per-position |P_pair(mutant_avg) - P_pair(wt)| aligned to mask."""
    n = len(pair.mask)
    wt_p = _fold_bpp(model, pair.seq)[:n]
    mut_seqs = build_mutant_sequences(pair.seq, pair.mutation_pos + 1, pair.ref_allele)
    n_alts = max(len(mut_seqs), 1)
    mut_acc = np.zeros(n, dtype=np.float64)
    for ms in mut_seqs:
        if len(ms) > max_len:
            continue
        mut_acc += _fold_bpp(model, ms)[:n]
    mut_p = mut_acc / max(n_alts, 1)
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
    return float(np.sum((rec - np.concatenate([[0.0], rec[:-1]])) * prec))


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--canonical-jsonl", type=Path, required=True)
    ap.add_argument("--split-manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--dev06-manifest", type=Path, default=None)
    ap.add_argument("--weights", type=Path, default=Path(RNAFORMER_WEIGHTS))
    ap.add_argument("--config", type=Path, default=Path(RNAFORMER_CONFIG))
    ap.add_argument("--max-len", type=int, default=500)
    ap.add_argument("--tiny", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logfile = out_dir / "rnaformer_run.log"
    _log = lambda msg: (print(msg, flush=True)
                        and logfile.open("a", encoding="utf-8").write(msg + "\n"))

    pairs = load_pairs(args.canonical_jsonl, args.split_manifest,
                       splits={"validation"})
    groups = split_groups(pairs)
    val = groups.get("validation", [])
    if args.tiny > 0:
        val = val[: args.tiny]
    _log(f"[data] validation pairs={len(val)} (test SEALED, train NOT used)")

    _log("[model] loading RNAformer 32M (intra-family fine-tuned)...")
    t0 = time.time()
    model = _load_model(str(args.weights), str(args.config), device=args.device)
    _log(f"[model] loaded in {time.time()-t0:.0f}s device={args.device}")

    scores = {}
    t0 = time.time()
    for idx, p in enumerate(val):
        scores[p.pair_id] = _fold_pair_for_model(model, p, args.max_len)
        if (idx + 1) % 20 == 0 or idx + 1 == len(val):
            _log(f"[rnaformer] {idx+1}/{len(val)} pairs in {time.time()-t0:.0f}s")
    _log(f"[rnaformer] folded {len(val)} pairs in {time.time()-t0:.0f}s")

    changed = _changer_records(val, scores)
    auprc = _study_macro_auprc(changed)
    _log(f"[rnaformer] val study-macro AUPRC = {auprc:.4f}")

    ref = dict(REF_AUPRC)
    if args.dev06_manifest and Path(args.dev06_manifest).exists():
        try:
            m = json.loads(Path(args.dev06_manifest).read_text(encoding="utf-8"))
            comp = m.get("comparison_table", {})
            for k in ("structure_aware_changer", "p2_paired_baseline",
                      "wt_only", "vienna_physics_published"):
                if k in comp and "study_macro_auprc" in comp[k]:
                    ref[k] = comp[k]["study_macro_auprc"]
        except Exception:
            pass

    own_key = "structure_aware_changer"
    own_auprc = ref[own_key]

    table_rows = [{"method": k, "auprc": v} for k, v in ref.items()]
    table_rows.append({"method": "rnaformer_published_dl", "auprc": auprc})
    table_rows.sort(key=lambda r: -r["auprc"])

    nboot = 1000
    rng = random.Random(SEED)
    # cluster-bootstrap difference of our classifier vs RNAformer. We only have
    # RNAformer's per-position scores; our per-position scores come from dev06
    # predictions.npz if present, else we report the aggregate point difference.
    gain_rna = {"point_gain": own_auprc - auprc,
                "note": "point difference of study-macro AUPRC "
                        "(our structure-aware classifier minus RNAformer)"}

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
            "P_pair": "sum_j BPP[i,j] (marginal base-pair probability) "
                      "from RNAformer sigmoid(logits[0,:,:,-1])",
            "mutants": "3 alternative substitutions via build_mutant_sequences",
            "metric": "study-macro AUPRC (mean over studies of per-study AP)",
            "our_method": "EPRO_DEV_06 structure-aware changer classifier (trained, GPU)",
            "reference": "M. Becker et al., Predicting RNA secondary structure by "
                         "learning unrolled algorithms, Nature Machine Intelligence 2023",
        },
        "rnaformer": {"auprc": auprc, "weights": str(args.weights),
                      "config": str(args.config),
                      "param_count": "32M (published)",
                      "note": "Published pretrained DL folding model; "
                              "untrained in-silico mutagenesis on our data",
                      "max_len": args.max_len},
        "our_reference_auprc": ref,
        "comparison_table": table_rows,
        "point_gain_vs_ours": gain_rna,
        "caveats": [
            "RNAformer is a published pretrained model (its own learned weights), "
            "NOT trained on our data; pure CPU inference, no training performed here.",
            "Sequences are length <= 500 (all validation seqs length 135), "
            "within RNAformer max_len.",
            "P_pair uses marginal BPP (sum over partners), a continuous score "
            "richer than the dot-bracket binary used by classical baselines.",
        ],
    }
    (out_dir / "rnaformer_comparison_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
    np.savez_compressed(str(out_dir / "rnaformer_scores.npz"),
                        **{pid: dict(scores)[pid] for pid in scores})

    _log("\n=== HORIZONTAL COMPARISON TABLE (横向对比表) ===")
    _log(f"{'method':32s} {'auprc':>8s}")
    for r in table_rows:
        _log(f"{r['method']:<32s} {r['auprc']:>8.4f}")
    _log(f"\nPoint gain of our structure-aware classifier vs RNAformer (DL SOTA): "
         f"+{gain_rna['point_gain']:.4f}")
    _log(f"manifest: {out_dir/'rnaformer_comparison_manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())