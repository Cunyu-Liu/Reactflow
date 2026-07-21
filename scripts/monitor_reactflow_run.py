#!/usr/bin/env python3
"""Summarize an active ReactFlow training run from profile.jsonl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reactflow.run_monitor import summarize_profile, write_monitor_markdown


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", help="run directory containing profile.jsonl/stderr.log")
    parser.add_argument("--profile", help="explicit profile.jsonl path")
    parser.add_argument("--stderr", help="explicit stderr.log path")
    parser.add_argument("--total-samples", type=int, help="known training sample count for progress/ETA")
    parser.add_argument("--output-json", help="optional path to write the JSON summary")
    parser.add_argument("--output-md", help="optional path to write a Markdown summary")
    parser.add_argument("--stderr-tail-bytes", type=int, default=2000)
    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
        profile = Path(args.profile) if args.profile else run_dir / "profile.jsonl"
        stderr = Path(args.stderr) if args.stderr else run_dir / "stderr.log"
    elif args.profile:
        profile = Path(args.profile)
        stderr = Path(args.stderr) if args.stderr else None
    else:
        parser.error("one of --run-dir or --profile is required")

    summary = summarize_profile(
        profile,
        total_samples=args.total_samples,
        stderr_path=stderr,
        stderr_tail_bytes=args.stderr_tail_bytes,
    )
    text = json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    print(text, end="")
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(text, encoding="utf-8")
    if args.output_md:
        write_monitor_markdown(summary, args.output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
