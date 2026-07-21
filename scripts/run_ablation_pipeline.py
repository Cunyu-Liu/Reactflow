#!/usr/bin/env python3
"""Run ReactFlow full-scale ablation experiments with retry bookkeeping.

The script is intentionally stdlib-only.  It orchestrates existing ReactFlow CLI
commands, retries OOM/convergence failures with smaller batch sizes or lower
learning rates, then writes a filled ablation table and compact SVG charts.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
PYTHON = os.environ.get("PYTHON", sys.executable)
TORCH_PYTHON = os.environ.get("TORCH_PYTHON", "/home/zhaoyihao/anaconda3/envs/esm/bin/python")
if not Path(TORCH_PYTHON).exists():
    TORCH_PYTHON = PYTHON

RAW_EFOLD = Path(os.environ.get("RAW_EFOLD", ROOT / "data/raw/efold/dryad_20260129"))
MODEL_DIR = Path(os.environ.get("RIBONANZANET2_DIR", ROOT / "data/models/ribonanzanet2"))
RUN_ID = os.environ.get("RUN_ID", time.strftime("full_ablation_%Y%m%d_%H%M%S"))
OUT = Path(os.environ.get("OUT_DIR", ROOT / "artifacts/full_runs" / RUN_ID))
EPOCHS = int(os.environ.get("EPOCHS", "5"))
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "256"))
WINDOW_SIZE = int(os.environ.get("WINDOW_SIZE", "256"))
WINDOW_STRIDE = int(os.environ.get("WINDOW_STRIDE", "128"))
BUCKETS = os.environ.get("BUCKETS", "64,128,256,384")
TRAIN_LIMIT = os.environ.get("TRAIN_LIMIT", "")
EVAL_LIMIT = os.environ.get("EVAL_LIMIT", "")
TRAIN_EVAL_LIMIT = int(os.environ.get("TRAIN_EVAL_LIMIT", "1000000000"))
TIER_EVAL_LIMIT = int(os.environ.get("TIER_EVAL_LIMIT", "1000000000"))
EXPORT_WARM = os.environ.get("EXPORT_WARM", "1") != "0"
ALLOW_WARM_FALLBACK = os.environ.get("ALLOW_WARM_FALLBACK", "0") == "1"
TORCH_DEVICE = os.environ.get("TORCH_DEVICE", "cuda")
FROZEN_SHARD_SIZE = int(os.environ.get("FROZEN_SHARD_SIZE", "256"))
USE_RFAM_METADATA = os.environ.get("USE_RFAM_METADATA", "1") != "0"
RFAM_CLUSTER_METHOD = os.environ.get("RFAM_CLUSTER_METHOD", "auto")
RFAM_SPLIT_NAME = os.environ.get("RFAM_SPLIT_NAME", "rfam_current_seed0")
RUN_BASE = os.environ.get("RUN_BASE", "1") != "0"
RUN_WARM = os.environ.get("RUN_WARM", "1") != "0"
RUN_ADAPTER = os.environ.get("RUN_ADAPTER", "1") != "0"
RUN_THERMO = os.environ.get("RUN_THERMO", "1") != "0"
RUN_TORCH = os.environ.get("RUN_TORCH", "1") != "0"
RUN_CONTACT = os.environ.get("RUN_CONTACT", "1") != "0"
CONTACT_LAMBDA = float(os.environ.get("CONTACT_LAMBDA", "0.2"))
CONTACT_NEGATIVE_WEIGHT = float(os.environ.get("CONTACT_NEGATIVE_WEIGHT", "0.25"))


def env_for_python() -> Dict[str, str]:
    env = os.environ.copy()
    current = env.get("PYTHONPATH", "")
    src = str(ROOT / "src")
    env["PYTHONPATH"] = src if not current else src + os.pathsep + current
    return env


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def run_command(
    name: str,
    cmd: Sequence[str],
    *,
    cwd: Path = ROOT,
    extra_env: Optional[Dict[str, str]] = None,
) -> Tuple[int, str, str, float]:
    start = time.perf_counter()
    env = env_for_python()
    if extra_env:
        env.update(extra_env)
    log_dir = OUT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        list(cmd),
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    elapsed = time.perf_counter() - start
    (log_dir / f"{name}.stdout").write_text(proc.stdout, encoding="utf-8")
    (log_dir / f"{name}.stderr").write_text(proc.stderr, encoding="utf-8")
    write_json(
        log_dir / f"{name}.command.json",
        {"command": list(cmd), "cwd": str(cwd), "elapsed_seconds": elapsed, "returncode": proc.returncode},
    )
    return proc.returncode, proc.stdout, proc.stderr, elapsed


def parse_first_json(text: str) -> dict:
    start = text.find("{")
    if start < 0:
        raise ValueError("command stdout did not contain JSON")
    return json.loads(text[start:])


def oom_like(stderr: str, stdout: str = "") -> bool:
    text = (stderr + "\n" + stdout).lower()
    markers = ["out of memory", "cuda out", "memoryerror", "killed", "oom"]
    return any(marker in text for marker in markers)


def checkpoint_history(output_dir: Path) -> List[dict]:
    ckpt = output_dir / "training_checkpoint.json"
    if not ckpt.exists():
        return []
    payload = json.loads(ckpt.read_text(encoding="utf-8"))
    return list(payload.get("history") or [])


def convergence_bad(output_dir: Path) -> bool:
    history = checkpoint_history(output_dir)
    if len(history) < 2:
        return False
    first = float(history[0]["total"])
    last = float(history[-1]["total"])
    if not (math.isfinite(first) and math.isfinite(last)):
        return True
    return last > first * 1.25


def prepare_cache(name: str, raw_name: str, *, long: bool = False) -> Path:
    cache = OUT / "cache" / f"{name}.jsonl"
    if cache.exists() and cache.stat().st_size > 0:
        return cache
    cmd = [
        PYTHON,
        "-m",
        "reactflow.cli",
        "prepare-efold-cache",
        str(RAW_EFOLD / raw_name),
        "--output",
        str(cache),
        "--max-length",
        str(MAX_LENGTH),
        "--bucket-boundaries",
        BUCKETS,
    ]
    if long:
        cmd += ["--window-size", str(WINDOW_SIZE), "--window-stride", str(WINDOW_STRIDE)]
    if name.startswith("efold_train") and TRAIN_LIMIT:
        cmd += ["--limit", TRAIN_LIMIT]
    if not name.startswith("efold_train") and EVAL_LIMIT:
        cmd += ["--limit", EVAL_LIMIT]
    code, stdout, stderr, _ = run_command(f"cache_{name}", cmd)
    if code != 0:
        raise RuntimeError(f"cache {name} failed: {stderr[-1000:]}")
    summary = parse_first_json(stdout)
    summary["sha256"] = sha256_path(cache)
    write_json(OUT / "cache" / f"{name}.summary.json", summary)
    return cache


def build_split_metadata(train_cache: Path) -> Optional[Path]:
    if not USE_RFAM_METADATA:
        return None
    metadata_dir = OUT / "metadata"
    metadata_tsv = metadata_dir / "rfam_current_metadata.tsv"
    manifest = metadata_dir / "rfam_current_metadata.manifest.json"
    if metadata_tsv.exists() and manifest.exists():
        return metadata_tsv
    cmd = [
        PYTHON,
        "scripts/build_rfam_metadata.py",
        str(train_cache),
        "--output",
        str(metadata_tsv),
        "--manifest",
        str(manifest),
        "--rfam-download-dir",
        str(metadata_dir / "rfam_database_files"),
        "--cluster-method",
        RFAM_CLUSTER_METHOD,
        "--threads",
        os.environ.get("RFAM_METADATA_THREADS", "8"),
    ]
    code, stdout, stderr, _ = run_command("build_rfam_metadata", cmd)
    if code != 0:
        raise RuntimeError(f"Rfam metadata failed: {stderr[-1000:]}")
    summary = parse_first_json(stdout)
    summary["metadata_sha256"] = sha256_path(metadata_tsv)
    write_json(metadata_dir / "rfam_current_metadata.summary.json", summary)
    return metadata_tsv


def split_cache(train_cache: Path) -> Path:
    metadata_tsv = build_split_metadata(train_cache)
    split_dir = OUT / "splits" / (RFAM_SPLIT_NAME if metadata_tsv is not None else "rfam_seed0")
    manifest = split_dir / "split_manifest.json"
    if manifest.exists():
        return split_dir
    cmd = [
        PYTHON,
        "-m",
        "reactflow.cli",
        "split-efold-cache",
        str(train_cache),
        "--output-dir",
        str(split_dir),
        "--bucket-boundaries",
        BUCKETS,
        "--novel-clan-fraction",
        "0.15",
        "--seed",
        "0",
    ]
    if metadata_tsv is not None:
        cmd += ["--metadata-tsv", str(metadata_tsv)]
    code, stdout, stderr, _ = run_command("split_cache", cmd)
    if code != 0:
        raise RuntimeError(f"split failed: {stderr[-1000:]}")
    summary = parse_first_json(stdout)
    summary["manifest_sha256"] = sha256_path(manifest)
    if metadata_tsv is not None:
        summary["metadata_tsv"] = str(metadata_tsv)
        summary["metadata_sha256"] = sha256_path(metadata_tsv)
    write_json(split_dir / "split_summary.json", summary)
    return split_dir


def sequences_from_cache(cache: Path, out: Path) -> Path:
    seen = set()
    out.parent.mkdir(parents=True, exist_ok=True)
    with cache.open(encoding="utf-8") as handle, out.open("w", encoding="utf-8") as dst:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            seq = row["sequence"]
            if seq in seen:
                continue
            seen.add(seq)
            payload = {"id": row.get("source_id") or f"seq{len(seen)}", "sequence": seq, "family": row.get("family")}
            dst.write(json.dumps(payload, sort_keys=True) + "\n")
    return out


def export_warm_features(train_cache: Path) -> Optional[Path]:
    existing = ROOT / "artifacts/efold_scale/frozen/ribonanzanet2_efold_train64_single"
    full = OUT / "frozen" / "ribonanzanet2_train_single"
    if full.exists() and (full / "provenance.json").exists():
        return full
    if not EXPORT_WARM:
        return existing if ALLOW_WARM_FALLBACK and existing.exists() else None
    seqs = sequences_from_cache(train_cache, OUT / "clean" / "train_sequences_for_frozen.jsonl")
    cmd = [
        TORCH_PYTHON,
        "scripts/export_frozen_features.py",
        "--sequences",
        str(seqs),
        "--out",
        str(full),
        "--backend",
        "torch",
        "--model",
        "RibonanzaNet2",
        "--model-version",
        "alpha-v1",
        "--network-dir",
        str(MODEL_DIR),
        "--config",
        str(MODEL_DIR / "pairwise.yaml"),
        "--weights",
        str(MODEL_DIR / "pytorch_model_fsdp.bin"),
        "--device",
        TORCH_DEVICE,
        "--d-pair",
        "0",
        "--n-probe",
        "0",
        "--shard-size",
        str(FROZEN_SHARD_SIZE),
    ]
    code, stdout, stderr, _ = run_command("export_warm_features", cmd)
    if code != 0:
        write_json(OUT / "frozen" / "export_failure.json", {"stderr": stderr[-4000:], "stdout": stdout[-4000:]})
        if ALLOW_WARM_FALLBACK and existing.exists():
            return existing
        return None
    summary = parse_first_json(stdout)
    write_json(OUT / "frozen" / "export_summary.json", summary)
    return full


def run_eval_with_retries(
    run_id: str,
    *,
    train_json: Path,
    eval_specs: Sequence[Tuple[str, Path]],
    epochs: int,
    lambda_react: float = 0.0,
    lambda_thermo: float = 0.0,
    lambda_contact: float = 0.0,
    contact_negative_weight: float = 0.25,
    thermo_mode: str = "mse",
    adapter_dim: int = 0,
    frozen_dir: Optional[Path] = None,
    backend: str = "stdlib",
    hidden_size: int = 8,
) -> dict:
    batch_candidates: List[Optional[int]] = [None, 32, 16, 8, 4, 2, 1]
    lr_candidates = [0.2, 0.1, 0.05]
    attempts = []
    for lr in lr_candidates:
        for batch in batch_candidates:
            out_dir = OUT / "runs" / f"{run_id}_lr{lr}_bs{batch or 'full'}"
            out_dir.mkdir(parents=True, exist_ok=True)
            profile = out_dir / "profile.jsonl"
            cmd = [
                TORCH_PYTHON if backend == "torch" else PYTHON,
                "-m",
                "reactflow.cli",
                "evaluate-efold",
                "--train-json",
                str(train_json),
                "--train-limit",
                str(TRAIN_EVAL_LIMIT),
                "--eval-limit",
                str(TIER_EVAL_LIMIT),
                "--epochs",
                str(epochs),
                "--learning-rate",
                str(lr),
                "--hidden-size",
                str(hidden_size),
                "--lambda-react",
                str(lambda_react),
                "--lambda-thermo",
                str(lambda_thermo),
                "--lambda-contact",
                str(lambda_contact),
                "--contact-negative-weight",
                str(contact_negative_weight),
                "--thermo-mode",
                thermo_mode,
                "--bucket-boundaries",
                BUCKETS,
                "--profile-path",
                str(profile),
                "--output-dir",
                str(out_dir),
                "--backend",
                backend,
                "--torch-device",
                TORCH_DEVICE,
            ]
            if batch is not None:
                cmd += ["--batch-size", str(batch)]
            if adapter_dim:
                cmd += ["--adapter-dim", str(adapter_dim), "--adapter-lr", "0.05"]
                if frozen_dir is not None:
                    cmd += ["--frozen-dir", str(frozen_dir)]
            for tier, path in eval_specs:
                cmd += ["--eval-json", f"{tier}={path}"]
            code, stdout, stderr, elapsed = run_command(f"run_{run_id}_lr{lr}_bs{batch or 'full'}", cmd)
            attempt = {"lr": lr, "batch_size": batch, "returncode": code, "elapsed_seconds": elapsed}
            if code == 0:
                payload = parse_first_json(stdout)
                write_json(out_dir / "eval_summary.json", payload)
                attempt["status"] = "ok"
                attempts.append(attempt)
                if convergence_bad(out_dir):
                    attempt["status"] = "convergence_retry"
                    continue
                payload["run_id"] = run_id
                payload["selected_attempt"] = attempt
                payload["output_dir"] = str(out_dir)
                payload["attempts"] = attempts
                return payload
            attempt["status"] = "oom_retry" if oom_like(stderr, stdout) else "failed_retry"
            attempt["stderr_tail"] = stderr[-1000:]
            attempts.append(attempt)
            if not oom_like(stderr, stdout):
                break
    return {"run_id": run_id, "status": "failed", "attempts": attempts}


def metric_rows(results: Sequence[dict]) -> List[dict]:
    rows = []
    for result in results:
        run_id = result.get("run_id", "unknown")
        if result.get("status") == "failed":
            rows.append({"run_id": run_id, "status": "failed"})
            continue
        for tier, metrics in sorted(result.get("tiers", {}).items()):
            rows.append(
                {
                    "run_id": run_id,
                    "status": "ok",
                    "tier": tier,
                    "mean_f1": metrics.get("mean_f1"),
                    "micro_f1": metrics.get("micro_f1"),
                    "mean_mcc": metrics.get("mean_mcc"),
                    "micro_mcc": metrics.get("micro_mcc"),
                    "runtime": result.get("selected_attempt", {}).get("elapsed_seconds"),
                    "output_dir": result.get("output_dir"),
                }
            )
    return rows


def write_markdown_results(results: Sequence[dict]) -> None:
    rows = metric_rows(results)
    lines = [
        "# ReactFlow 消融实验结果",
        "",
        f"- Run ID: `{RUN_ID}`",
        f"- Host: `{socket.gethostname()}`",
        f"- Python: `{platform.python_version()}`",
        f"- Raw data: `{RAW_EFOLD}`",
        "",
        "| Run ID | Status | Tier | Mean F1 | Micro F1 | Mean MCC | Micro MCC | Runtime(s) | Artifact |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {run_id} | {status} | {tier} | {mean_f1} | {micro_f1} | {mean_mcc} | {micro_mcc} | {runtime} | {output_dir} |".format(
                run_id=row.get("run_id", ""),
                status=row.get("status", ""),
                tier=row.get("tier", ""),
                mean_f1="" if row.get("mean_f1") is None else row.get("mean_f1"),
                micro_f1="" if row.get("micro_f1") is None else row.get("micro_f1"),
                mean_mcc="" if row.get("mean_mcc") is None else row.get("mean_mcc"),
                micro_mcc="" if row.get("micro_mcc") is None else row.get("micro_mcc"),
                runtime="" if row.get("runtime") is None else round(float(row.get("runtime")), 2),
                output_dir=row.get("output_dir", ""),
            )
        )
    (OUT / "ablation_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_svg_chart(results: Sequence[dict]) -> None:
    rows = [row for row in metric_rows(results) if row.get("status") == "ok" and row.get("mean_f1") is not None]
    if not rows:
        return
    width = 1100
    height = max(260, 40 * len(rows) + 80)
    max_value = max(float(row["mean_f1"]) for row in rows) or 1.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="24" y="30" font-family="Arial" font-size="20" font-weight="700">ReactFlow ablation mean F1 by tier</text>',
    ]
    y = 64
    for row in rows:
        label = f"{row['run_id']} / {row['tier']}"
        value = float(row["mean_f1"])
        bar = 700 * value / max_value
        parts.append(f'<text x="24" y="{y + 16}" font-family="Arial" font-size="13">{label}</text>')
        parts.append(f'<rect x="330" y="{y}" width="{bar:.1f}" height="22" fill="#4f7cff"/>')
        parts.append(f'<text x="{340 + bar:.1f}" y="{y + 16}" font-family="Arial" font-size="13">{value:.4f}</text>')
        y += 38
    parts.append("</svg>")
    (OUT / "ablation_mean_f1.svg").write_text("\n".join(parts), encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(
        OUT / "run_metadata.json",
        {
            "run_id": RUN_ID,
            "root": str(ROOT),
            "raw_efold": str(RAW_EFOLD),
            "epochs": EPOCHS,
            "max_length": MAX_LENGTH,
            "window_size": WINDOW_SIZE,
            "window_stride": WINDOW_STRIDE,
            "buckets": BUCKETS,
            "train_limit": TRAIN_LIMIT or None,
            "eval_limit": EVAL_LIMIT or None,
            "train_eval_limit": TRAIN_EVAL_LIMIT,
            "tier_eval_limit": TIER_EVAL_LIMIT,
            "export_warm": EXPORT_WARM,
            "allow_warm_fallback": ALLOW_WARM_FALLBACK,
            "torch_python": TORCH_PYTHON,
            "torch_device": TORCH_DEVICE,
            "frozen_shard_size": FROZEN_SHARD_SIZE,
            "use_rfam_metadata": USE_RFAM_METADATA,
            "rfam_cluster_method": RFAM_CLUSTER_METHOD,
            "rfam_split_name": RFAM_SPLIT_NAME,
            "run_base": RUN_BASE,
            "run_warm": RUN_WARM,
            "run_adapter": RUN_ADAPTER,
            "run_thermo": RUN_THERMO,
            "run_torch": RUN_TORCH,
            "run_contact": RUN_CONTACT,
            "contact_lambda": CONTACT_LAMBDA,
            "contact_negative_weight": CONTACT_NEGATIVE_WEIGHT,
        },
    )

    train_cache = prepare_cache("efold_train", "efold_train.json", long=True)
    archive_cache = prepare_cache("archiveII", "archiveII.json")
    pdb_cache = prepare_cache("PDB", "PDB.json")
    viral_cache = prepare_cache("viral", "viral_fragments.json", long=True)
    lnc_cache = prepare_cache("lncRNA", "lncRNA_nonFiltered.json", long=True)
    human_cache = prepare_cache("human_mRNA", "human_mRNA.json", long=True)
    split_dir = split_cache(train_cache)
    frozen_dir = export_warm_features(split_dir / "train.jsonl")

    eval_specs = [
        ("in_clan", split_dir / "test.jsonl"),
        ("novel_clan", split_dir / "novel.jsonl"),
        ("archiveII", archive_cache),
        ("PDB", pdb_cache),
        ("viral", viral_cache),
        ("lncRNA", lnc_cache),
        ("human_mRNA", human_cache),
    ]

    experiments = []
    if RUN_BASE:
        experiments.append(("RF-A0-base", {"adapter_dim": 0, "backend": "stdlib"}))
    if RUN_WARM:
        experiments.append(("RF-A1-warm", {"adapter_dim": 8, "frozen_dir": frozen_dir, "backend": "stdlib"}))
    if RUN_ADAPTER:
        experiments.extend(
            [
                ("RF-A2-adapter4", {"adapter_dim": 4, "frozen_dir": frozen_dir, "backend": "stdlib"}),
                ("RF-A2-adapter16", {"adapter_dim": 16, "frozen_dir": frozen_dir, "backend": "stdlib"}),
            ]
        )
    if RUN_THERMO:
        experiments.append(("RF-A4-thermo", {"lambda_thermo": 0.1, "backend": "stdlib"}))
    if RUN_CONTACT:
        experiments.append(
            (
                "RF-A3-contact",
                {
                    "lambda_contact": CONTACT_LAMBDA,
                    "contact_negative_weight": CONTACT_NEGATIVE_WEIGHT,
                    "backend": "torch",
                },
            )
        )
    if RUN_TORCH:
        experiments.append(("RF-A8-torch", {"backend": "torch"}))
    results = []
    for run_id, kwargs in experiments:
        if kwargs.get("adapter_dim") and frozen_dir is None:
            results.append({"run_id": run_id, "status": "failed", "reason": "no frozen_dir"})
            continue
        result = run_eval_with_retries(
            run_id,
            train_json=split_dir / "train.jsonl",
            eval_specs=eval_specs,
            epochs=EPOCHS,
            lambda_react=0.0,
            **kwargs,
        )
        results.append(result)
        write_json(OUT / "partial_results.json", results)
        write_markdown_results(results)
        write_svg_chart(results)

    write_json(OUT / "results.json", results)
    write_markdown_results(results)
    write_svg_chart(results)
    return 0 if all(result.get("status") != "failed" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
