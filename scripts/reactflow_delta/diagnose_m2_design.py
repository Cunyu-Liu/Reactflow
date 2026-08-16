#!/usr/bin/env python3
"""diagnose_m2_design — position-wise diagnosis of one held-out M2 design.

For a chosen design (puzzle x method), report how the residual model vs the
per-position median prior behaves at each WINDOW position around the edit site.
The window is centered on the edit site: window position k maps to sequence index
idx = edit_pos - HALF + k (k = HALF is the edit site itself).

For each window position k, over the design's held-out changers we report:
  * mean |y|   (observed response magnitude -> where the signal lives),
  * prior      (the per-position median prior the model starts from),
  * baseline & residual weighted MAE,
  * improvement = baseline_MAE_k - resid_MAE_k  (>0 = residual helps there),
  * mean |delta| the model learned at that position.

Output: JSON + printed position table.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np

WINDOW = 21
HALF = WINDOW // 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--design", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # single pass over the predictions
    y_by_pair = {}
    weight_by_pair = {}
    base_pred_by_pair = {}
    res_pred_by_pair = {}
    pids = set()
    with open(args.pred, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["fold_id"] != args.design or r["coverage_status"] != "CALLED":
                continue
            pid = r["pair_id"]
            pids.add(pid)
            y_by_pair.setdefault(pid, np.asarray(r["y"], dtype=float))
            weight_by_pair.setdefault(pid, np.asarray(r["weight"], dtype=float))
            var = r["model_variant"]
            if var == "wmed_spectrum":
                base_pred_by_pair[pid] = np.asarray(r["raw_prediction"], dtype=float)
            elif var == "wmae_resid_spectrum":
                res_pred_by_pair.setdefault(pid, []).append(
                    np.asarray(r["raw_prediction"], dtype=float))

    pids = sorted(pids)
    res_avg = {pid: np.mean(res_pred_by_pair[pid], axis=0)
               for pid in pids if pid in res_pred_by_pair}

    rows = []
    for k in range(WINDOW):
        wb = wr = 0.0
        mae_b = mae_r = 0.0
        absy = 0.0
        prior_sum = 0.0
        delta_sum = 0.0
        n = 0
        for pid in pids:
            w = weight_by_pair[pid]
            if w[k] <= 0:
                continue
            y = y_by_pair[pid][k]
            bpred = base_pred_by_pair.get(pid)
            rpred = res_avg.get(pid)
            n += 1
            absy += abs(y)
            if bpred is not None:
                mae_b += w[k] * abs(y - bpred[k])
                wb += w[k]
                prior_sum += bpred[k]
                if rpred is not None:
                    mae_r += w[k] * abs(y - rpred[k])
                    wr += w[k]
                    delta_sum += abs(rpred[k] - bpred[k])
        if n == 0:
            continue
        rows.append({
            "position": k, "seq_offset": k - HALF, "n_eligible": n,
            "mean_abs_y": round(absy / n, 4),
            "prior": round(prior_sum / n, 4),
            "baseline_mae": round(mae_b / wb if wb > 0 else float("nan"), 4),
            "resid_mae": round(mae_r / wr if wr > 0 else float("nan"), 4),
            "improvement": round(mae_b / wb - mae_r / wr if wb > 0 and wr > 0
                                 else float("nan"), 4),
            "mean_abs_delta": round(delta_sum / n, 4),
        })

    out = {"design": args.design, "window": WINDOW, "half": HALF,
           "n_changers": len(pids), "positions": rows}
    Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")

    print(f"=== POSITION-WISE DIAGNOSIS: {args.design}  (n_changers={len(pids)}) ===")
    print(f"{'pos':>3}{'off':>5}{'n':>5}{'|y|':>7}{'prior':>7}{'baseMAE':>9}"
          f"{'resMAE':>9}{'impr':>8}{'|del|':>7}")
    for r in rows:
        print(f"{r['position']:>3}{r['seq_offset']:>5}{r['n_eligible']:>5}"
              f"{r['mean_abs_y']:>7}{r['prior']:>7}{r['baseline_mae']:>9}"
              f"{r['resid_mae']:>9}{r['improvement']:>8}{r['mean_abs_delta']:>7}")
    good = [r for r in rows if r["improvement"] is not None and r["improvement"] > 0]
    bad = [r for r in rows if r["improvement"] is not None and r["improvement"] < 0]
    print(f"\npositions residual helps (+): {[r['position'] for r in good]}")
    print(f"positions residual hurts (-): {[r['position'] for r in bad]}")
    if good:
        best = max(good, key=lambda r: r["improvement"])
        print(f"best-learned position: k={best['position']} "
              f"(seq_offset={best['seq_offset']}, 0 = edit site) "
              f"improvement={best['improvement']}")
    print(f"\nWROTE -> {args.out}")


if __name__ == "__main__":
    main()
