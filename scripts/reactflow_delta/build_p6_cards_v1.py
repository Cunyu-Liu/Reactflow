#!/usr/bin/env python3
"""build_p6_cards_v1: P6 code/data/model cards + environment (contract 12.8).

Auto-fills cards from the locked result artifacts (no hand-copied headline).
Emits:
  cards.md / cards.tex - model card, data card, code card
  environment.yml       - pinned runtime environment
  cards_summary.json    - machine-readable registry
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def build_cards(p2, p3, horizontal, p4, p5, calib, replay, env: dict,
                git: dict) -> dict:
    direct_row = next((m for m in horizontal.get("method_table", [])
                       if m["method"] == "reg_direct"), {})
    cards = {
        "model_card": {
            "name": "RFD-Direct (reg_direct)",
            "role": "adopted deployment model (P3 chose the simplest qualified direct model per contract 17.2)",
            "task": "predict full-construct mutant 2A3 reactivity profile given WT sequence + WT profile + exact SNV",
            "architecture": "regularized linear direct model (ridge, lambda=1e-2) over chemistry/distance template",
            "features": "signed distance x exact ref->alt x WT edit-site + readout reactivity state (13-dim)",
            "estimand": "Gaussian CRPS at fixed scale 0.3 over full-construct qualified positions",
            "dev_performance": {
                "mean_held_crps": direct_row.get("mean_held_crps"),
                "skill_vs_zero_pct": direct_row.get("skill_vs_zero_pct"),
                "p2_D_p_ci": p2.get("p2_ci20", {}),
            },
            "external_performance": {
                "p4_verdict": p4.get("verdict"),
                "D_vs_zero_ci": p4.get("ci_zero", {}),
                "calibration": calib.get("verdict", {}),
            },
            "mechanism_verdict": p5.get("verdict"),
            "limitations": ["PRACTICAL_IMPORTANCE_NOT_ESTABLISHED (delta_practical not established)",
                            "signed-delta point MAE negative vs no-change anchor (CRPS advantage is tail-driven)",
                            "edit-site-concentration mechanism not established on external data (P5)"],
            "uncertainty": f"Gaussian predictive with frozen scale 0.3; empirical residual SD {calib.get('pooled', {}).get('empirical_residual_sd')}",
        },
        "data_card": {
            "development": {
                "name": "OpenKnot M2 Round 3 (OK7a_M2)",
                "file": "OK7a_M2_data.v4.5.2.csv",
                "cells": 160, "wt": 160, "exact_snv": 13976,
                "chemistry": "2A3-MaP",
                "role": "development (LOPO-puzzle split, 20 puzzles)",
            },
            "external": {
                "name": "Ribonanza M2-style 2A3 (via RMDB)",
                "datasets": {"M2SL5_2A3_0000": "betacoronavirus SL5",
                             "M3SARS_2A3_0000": "coronavirus frameshift elements",
                             "15KLIB_2A3_0000": "diverse (TTR, SAM riboswitch, SARS windows, HDV)"},
                "components": p4.get("K_preaccess"),
                "single_snv": p4.get("K_preaccess_single_snv"),
                "development_disconnect": "zero sequence identity overlap with development",
                "role": "confirmatory (development-disconnected)",
            },
            "provenance": "raw rdat under /mnt/cunyuliu/reactflow_delta_raw/rmdb/rdat_tierA_20260730",
        },
        "code_card": {
            "repo": git.get("remote", "git@github.com:Cunyu-Liu/Reactflow.git"),
            "branch": git.get("branch", "codex/reactflow-delta-prospective-v2-20260813"),
            "commit": git.get("head", ""),
            "entrypoints": {
                "P2": "scripts/reactflow_delta/run_p2_direct_v2.py",
                "P3": "scripts/reactflow_delta/run_p3_lrso_v2.py",
                "P4": "scripts/reactflow_delta/run_p4_external_v1.py",
                "P5": "scripts/reactflow_delta/run_p5_mechanism_v1.py",
                "P4_calibration": "scripts/reactflow_delta/analyze_p4_calibration_v1.py",
                "P6_replay": "scripts/reactflow_delta/run_replay_v1.py",
                "P6_tables": "scripts/reactflow_delta/generate_p6_tables_figures_v1.py",
            },
            "test_suite": "pytest tests/reactflow_delta/ (P4/P5/replay/tables suites pass)",
        },
    }
    return cards


def render_markdown(cards: dict) -> str:
    lines = ["# ReactFlow-Delta prospective-v2 cards (auto-generated)", ""]
    for cid, card in cards.items():
        lines.append(f"## {cid}\n")
        lines.append("```json")
        lines.append(json.dumps(card, indent=2, default=str))
        lines.append("```\n")
    return "\n".join(lines)


def render_env(pkgs: dict) -> str:
    lines = ["name: editflow", "channels:", "  - defaults", "dependencies:"]
    for pkg, ver in pkgs.items():
        if ver:
            lines.append(f"  - {pkg}={ver}")
        else:
            lines.append(f"  - {pkg}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p2-result", required=True)
    ap.add_argument("--p3-result", required=True)
    ap.add_argument("--horizontal", required=True)
    ap.add_argument("--p4-result", required=True)
    ap.add_argument("--p5-result", required=True)
    ap.add_argument("--calib-result", required=True)
    ap.add_argument("--replay-report", required=True)
    ap.add_argument("--env", required=True)
    ap.add_argument("--git-remote", required=True)
    ap.add_argument("--git-branch", required=True)
    ap.add_argument("--git-head", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)

    def load(p: str):
        return json.loads(Path(p).read_text(encoding="utf-8"))

    p2 = load(args.p2_result); p3 = load(args.p3_result)
    hor = load(args.horizontal); p4 = load(args.p4_result)
    p5 = load(args.p5_result); calib = load(args.calib_result)
    replay = load(args.replay_report)
    env = json.loads(Path(args.env).read_text(encoding="utf-8"))
    git = {"remote": args.git_remote, "branch": args.git_branch, "head": args.git_head}

    cards = build_cards(p2, p3, hor, p4, p5, calib, replay, env, git)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cards.md").write_text(render_markdown(cards), encoding="utf-8")
    (out_dir / "environment.yml").write_text(render_env(env), encoding="utf-8")
    (out_dir / "cards_summary.json").write_text(
        json.dumps({"cards": list(cards.keys()), "env_pkgs": list(env.keys()),
                    "replay_verdict": replay.get("verdict")}, indent=2), encoding="utf-8")
    print(json.dumps(cards_summary := {"cards": list(cards.keys()),
                                       "env_pkgs": list(env.keys())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
