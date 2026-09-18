#!/usr/bin/env python
"""DeltaFlow seed1 completion handoff (Task 9 pre-step; engineering-only).

Runs after all 20 seed1 folds are published. Three stages:
  1. VERIFY  - 20/20 fold completeness (7 files + joint_samples per fold).
  2. MERGE   - canonical unscored merge for seed1 (merge-once enforced).
  3. PREVIEW - two-seed (seed0 vs seed1) engineering consistency readout.

ZERO held-target access: only model outputs are summarized. This is not a
scientific verdict; formal judgement stays with score-once/qualifier-once
after seeds 2-4 are restored.

Exit codes: 0 = handoff complete; 3 = folds incomplete (wait); 4 = merge failed.
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REQUIRED_FILES = (
    "deltaflow_candidate_flow_fold{f}_seed1.pt",
    "deltaflow_feature41_asymmetric_fold{f}_seed1.pt",
    "deltaflow_flow_audit_fold{f}_seed1.json",
    "deltaflow_fold_result_fold{f}_seed1.json",
    "deltaflow_null_flow_fold{f}_seed1.pt",
    "deltaflow_predictions_fold{f}_seed1.npz",
    "deltaflow_stage1_point_fold{f}_seed1.pt",
)


def stage1_verify(dflow3_dir: Path, n_folds: int = 20) -> None:
    missing = []
    joint_empty = []
    for f in range(n_folds):
        for pat in REQUIRED_FILES:
            p = dflow3_dir / pat.format(f=f)
            if not p.exists():
                missing.append(str(p.name))
        jdir = dflow3_dir / f"deltaflow_predictions_fold{f}_seed1_joint_samples"
        if (not jdir.is_dir()) or (not any(jdir.glob("*.npz"))):
            joint_empty.append(f)
    if missing or joint_empty:
        print(f"[handoff] VERIFY incomplete: {len(missing)} missing files, "
              f"{len(joint_empty)} empty joint dirs")
        for name in missing[:10]:
            print(f"  MISSING {name}")
        print(f"  EMPTY_JOINT folds {joint_empty}")
        sys.exit(3)
    print(f"[handoff] VERIFY ok: all {n_folds} folds complete "
          f"(7 files + joint_samples each)")


def stage2_merge(dflow3_dir: Path) -> None:
    marker = dflow3_dir / "deltaflow_merged_predictions_seed1.npz"
    if marker.exists():
        print("[handoff] MERGE skipped (canonical merge already published)")
        return
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "merge_deltaflow_predictions.py"),
        "--out-dir", str(dflow3_dir),
        "--seed", "1",
    ]
    print(f"[handoff] MERGE run: {' '.join(cmd)}")
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"[handoff] MERGE failed rc={rc}")
        sys.exit(4)
    print("[handoff] MERGE ok")


def _mix(x, w):
    return (x * w).sum(-1)


def _pearson(a, b):
    """Pearson r; returns 'const' (string) if either series is constant."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size < 4 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return "const"
    return float(np.corrcoef(a, b)[0, 1])


def arm_stats(d, point=True):
    out = {}
    prefix = "" if point else "flow_"
    for arm in ("candidate", "null"):
        key = f"{prefix}{arm}"
        e = d[f"{key}_expected_absolute_delta"]
        w = d[f"{key}_weights"].astype(np.float64)
        s = d[f"{key}_scales"].astype(np.float64)
        l = d[f"{key}_locations"].astype(np.float64)
        wsum = w.sum(-1)
        ok = (np.isfinite(e) & np.isfinite(wsum) & (wsum > 0)
              & np.isfinite(s).all(-1) & np.isfinite(l).all(-1))
        out[arm] = dict(
            n=int(e.size),
            n_finite=int(ok.sum()),
            mean_ead=float(np.mean(e[ok])),
            mean_scale=float(np.mean(_mix(s, w)[ok])),
            mean_abs_loc=float(np.mean(np.abs(_mix(l, w))[ok])),
        )
    return out


def stage3_preview(seed0_npz: Path, seed1_npz: Path, out_dir: Path) -> dict:
    print(f"[handoff] PREVIEW seed0={seed0_npz} seed1={seed1_npz}")
    d0 = np.load(seed0_npz, allow_pickle=True)
    d1 = np.load(seed1_npz, allow_pickle=True)

    report = {"stamp": datetime.now(timezone.utc).isoformat(),
              "seed0": str(seed0_npz), "seed1": str(seed1_npz)}
    report["seed0_point"] = arm_stats(d0, point=True)
    report["seed1_point"] = arm_stats(d1, point=True)
    report["seed0_flow"] = arm_stats(d0, point=False)
    report["seed1_flow"] = arm_stats(d1, point=False)

    # elementwise seed-consistency on a stride subsample (same key universes
    # across seeds, enforced by each merge's fold audits)
    n0, n1 = int(d0["keys"].size), int(d1["keys"].size)
    report["row_counts"] = {"seed0": n0, "seed1": n1}
    report["cross_seed"] = {}
    if n0 == n1:
        idx = np.arange(0, n0, 97)
        for arm in ("candidate", "null"):
            e0 = d0[f"{arm}_expected_absolute_delta"][idx]
            e1 = d1[f"{arm}_expected_absolute_delta"][idx]
            s0 = _mix(d0[f"{arm}_scales"].astype(np.float64),
                      d0[f"{arm}_weights"].astype(np.float64))[idx]
            s1 = _mix(d1[f"{arm}_scales"].astype(np.float64),
                      d1[f"{arm}_weights"].astype(np.float64))[idx]
            report["cross_seed"][arm] = dict(
                n_subsample=int(idx.size),
                r_expected_abs_delta=_pearson(e0, e1),
                r_scale=_pearson(s0, s1))
        n0f, n1f = int(d0["flow_keys"].size), int(d1["flow_keys"].size)
        if n0f == n1f:
            idxf = np.arange(0, n0f, 97)
            for arm in ("candidate", "null"):
                e0 = np.ravel(d0[f"flow_{arm}_expected_absolute_delta"][idxf])
                e1 = np.ravel(d1[f"flow_{arm}_expected_absolute_delta"][idxf])
                s0 = np.ravel(d0[f"flow_{arm}_scales"][idxf])
                s1 = np.ravel(d1[f"flow_{arm}_scales"][idxf])
                report["cross_seed"][f"flow_{arm}"] = dict(
                    n_subsample=int(idxf.size),
                    r_expected_abs_delta=_pearson(e0, e1),
                    r_scale=_pearson(s0, s1))

    # arm ordering (candidate should be tighter than null in both seeds)
    ord_report = {}
    for name, d in (("seed0", d0), ("seed1", d1)):
        e_c = d["candidate_expected_absolute_delta"].astype(np.float64)
        e_n = d["null_expected_absolute_delta"].astype(np.float64)
        m = np.isfinite(e_c) & np.isfinite(e_n)
        folds = np.asarray(d["outer_fold"]).astype(int)
        s_c = _mix(d["candidate_scales"].astype(np.float64),
                   d["candidate_weights"].astype(np.float64))
        s_n = _mix(d["null_scales"].astype(np.float64),
                   d["null_weights"].astype(np.float64))
        fold_c = np.array([np.nanmean(s_c[folds == f]) for f in range(20)])
        fold_n = np.array([np.nanmean(s_n[folds == f]) for f in range(20)])
        ord_report[name] = dict(
            row_frac_candidate_below=float(np.mean(e_c[m] < e_n[m])),
            row_frac_equal=float(np.mean(e_c[m] == e_n[m])),
            folds_candidate_scale_below=int(np.sum(fold_c < fold_n)),
        )
    flow_ord = {}
    for name, d in (("seed0", d0), ("seed1", d1)):
        sc = np.ravel(d["flow_candidate_scales"].astype(np.float64))
        sn = np.ravel(d["flow_null_scales"].astype(np.float64))
        ec = np.ravel(d["flow_candidate_expected_absolute_delta"])
        en = np.ravel(d["flow_null_expected_absolute_delta"])
        m_s = np.isfinite(sc) & np.isfinite(sn)
        m_e = np.isfinite(ec) & np.isfinite(en)
        flow_ord[name] = dict(
            flow_row_frac_scale_c_and_below=float(np.mean(sc[m_s] <= sn[m_s])),
            flow_row_frac_ead_c_and_below=float(np.mean(ec[m_e] <= en[m_e])),
        )
    report["arm_ordering_flow"] = flow_ord
    report["arm_ordering"] = ord_report

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        md = out_dir / "deltaflow_seed1_handoff_preview.md"
        js = out_dir / "deltaflow_seed1_handoff_preview.json"
        md.write_text(_render_md(report), encoding="utf-8")
        js.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                      encoding="utf-8")
        print(f"[handoff] PREVIEW written -> {md} / {js}")
        flag = out_dir / "deltaflow_seed1_handoff_done.flag"
        flag.write_text(report["stamp"] + "\n", encoding="utf-8")
    return report


def _render_md(r: dict) -> str:
    lines = ["# DeltaFlow seed1 handoff preview (engineering-only)",
             "",
             f"- run: {r['stamp']}",
             "- **zero held-target access** — model outputs only; not a "
             "scientific verdict (formal judgement stays with score-once)",
             f"- row counts — seed0: {r['row_counts']['seed0']:,} | "
             f"seed1: {r['row_counts']['seed1']:,}",
             "",
             "## point-level arm stats", "",
             "| seed | arm | n | mean E\\|Δ\\| | mean scale | mean \\|loc\\| |",
             "|---|---|---|---|---|---|"]
    for seed in ("seed0", "seed1"):
        for arm in ("candidate", "null"):
            s = r[f"{seed}_point"][arm]
            lines.append(f"| {seed} | {arm} | {s['n']:,} | {s['mean_ead']:.6f} | "
                         f"{s['mean_scale']:.6f} | {s['mean_abs_loc']:.6f} |")
    lines += ["", "## flow-level arm stats", "",
              "| seed | arm | n | mean E\\|Δ\\| | mean scale |",
              "|---|---|---|---|---|"]
    for seed in ("seed0", "seed1"):
        for arm in ("candidate", "null"):
            s = r[f"{seed}_flow"][arm]
            lines.append(f"| {seed} | {arm} | {s['n']:,} | {s['mean_ead']:.6f} | "
                         f"{s['mean_scale']:.6f} |")
    lines += ["", "## cross-seed consistency (stride-97 subsample)", "",
              "| field | r(E\\|Δ\\|) | r(scale) |",
              "|---|---|---|"]
    for key in ("candidate", "null", "flow_candidate", "flow_null"):
        c = r["cross_seed"].get(key, {})
        r_ead = c.get("r_expected_abs_delta", "n/a")
        r_sc = c.get("r_scale", "n/a")
        r_ead = f"{r_ead:.4f}" if isinstance(r_ead, float) else str(r_ead)
        r_sc = f"{r_sc:.4f}" if isinstance(r_sc, float) else str(r_sc)
        lines.append(f"| {key} | {r_ead} | {r_sc} |")
    lines += ["", "## arm ordering", "",
              "| seed | row-frac candidate<E\\|Δ\\| null | fold-mean-scale "
              "candidate<null (of 20) |", "|---|---|---|"]
    for seed in ("seed0", "seed1"):
        o = r["arm_ordering"][seed]
        lines.append(f"| {seed} | {o['row_frac_candidate_below']:.3f} | "
                     f"{o['folds_candidate_scale_below']} |")
    lines += ["", "## flow-level arm ordering (the hypothesis-relevant layer)", "",
              "| seed | row-frac flow candidate scale ≤ null | row-frac flow candidate E\|Δ\| ≤ null |",
              "|---|---|---|"]
    for seed in ("seed0", "seed1"):
        o = r.get("arm_ordering_flow", {}).get(seed, {})
        lines.append(f"| {seed} | {o.get('flow_row_frac_scale_c_and_below', float('nan')):.3f} | "
                     f"{o.get('flow_row_frac_ead_c_and_below', float('nan')):.3f} |")
    lines += ["", ": warning: `point-level arm ordering` above is expected to be 0/0 — stage1 point outputs are arm-identical by construction; the arms diverge only at the flow layer.", "",
              "> Interpretation: this preview only checks that the two "
              "seeds agree on *predictive* statistics (locations, scales) and "
              "that the candidate/null arm ordering is stable. It reads no "
              "targets.", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dflow3-dir", required=True)
    ap.add_argument("--seed0-merged", required=True)
    ap.add_argument("--preview-only", action="store_true")
    ap.add_argument("--seed1-merged", default=None)
    ap.add_argument("--preview-out", default=None)
    args = ap.parse_args()

    dflow3 = Path(args.dflow3_dir)
    seed1_npz = (Path(args.seed1_merged) if args.seed1_merged
                 else dflow3 / "deltaflow_merged_predictions_seed1.npz")
    out = (Path(args.preview_out) if args.preview_out else dflow3)

    if not args.preview_only:
        stage1_verify(dflow3)
        stage2_merge(dflow3)
    else:
        print("[handoff] PREVIEW-ONLY mode (stages 1-2 skipped)")

    report = stage3_preview(Path(args.seed0_merged), seed1_npz, out)
    for seed in ("seed0", "seed1"):
        for arm in ("candidate", "null"):
            s = report[f"{seed}_point"][arm]
            print(f"  {seed:6s} {arm:9s} point: E|Δ|={s['mean_ead']:.6f} "
                  f"scale={s['mean_scale']:.6f} |loc|={s['mean_abs_loc']:.6f}")
    cs = report["cross_seed"]

    def _fmt(v):
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    for key in ("candidate", "null", "flow_candidate", "flow_null"):
        if key not in cs:
            continue
        print(f"  cross-seed {key:15s}: r(E|Δ|)={_fmt(cs[key].get('r_expected_abs_delta', 'n/a'))} "
              f"r(scale)={_fmt(cs[key].get('r_scale', 'n/a'))}")
    print("[handoff] DONE")


if __name__ == "__main__":
    main()
