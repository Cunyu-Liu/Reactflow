#!/usr/bin/env python3
"""Export eFold predictions and score them on ReactFlow same-split gold tiers.

This is an execution wrapper around the existing protocol-safe scorer in
``evaluate_external_baseline_predictions.py``.  It runs a local eFold Python
package or CLI, converts dot-bracket predictions to zero-based pair JSONL, and
then emits ``baseline_efold_results.json`` rows only when the scorer sees full
tier coverage.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Callable, Iterable, Mapping, Optional, Sequence


MODEL_NAME = "eFold/RNAndria local rerun"
ALLOWED_PROTOCOLS = ("same_split_local", "local_closest_protocol")
OPEN_TO_CLOSE = {"(": ")", "[": "]", "{": "}", "<": ">"}
CLOSE_TO_OPEN = {close: open_ for open_, close in OPEN_TO_CLOSE.items()}
DOTBRACKET_CHARS = set(".") | set(OPEN_TO_CLOSE) | set(CLOSE_TO_OPEN)
_EFOLD_DEVICE_READY: set[str] = set()


def _load_scorer_module():
    """Load the protocol-safe baseline scorer next to this wrapper."""

    path = Path(__file__).resolve().with_name("evaluate_external_baseline_predictions.py")
    spec = importlib.util.spec_from_file_location("evaluate_external_baseline_predictions", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"cannot load scorer module from {path}")
    spec.loader.exec_module(module)
    return module


def parse_dot_bracket(dot_bracket: str) -> list[list[int]]:
    """Convert dot-bracket notation into sorted zero-based pair indices."""

    stacks = {open_: [] for open_ in OPEN_TO_CLOSE}
    pairs: list[list[int]] = []
    for index, token in enumerate(dot_bracket):
        if token == ".":
            continue
        if token in OPEN_TO_CLOSE:
            stacks[token].append(index)
            continue
        if token in CLOSE_TO_OPEN:
            open_token = CLOSE_TO_OPEN[token]
            if not stacks[open_token]:
                raise ValueError(f"unbalanced dot-bracket close token {token!r} at index {index}")
            pairs.append([stacks[open_token].pop(), index])
            continue
        raise ValueError(f"unsupported dot-bracket token {token!r} at index {index}")
    dangling = {token: stack for token, stack in stacks.items() if stack}
    if dangling:
        raise ValueError(f"unbalanced dot-bracket open tokens: {dangling}")
    return sorted(pairs)


def _extract_dot_bracket(raw: object, *, length: int) -> str:
    """Extract one dot-bracket string from a CLI/module return value."""

    if isinstance(raw, Mapping):
        for key in ("dotbracket", "dot_bracket", "structure", "prediction"):
            if key in raw:
                return _extract_dot_bracket(raw[key], length=length)
        for value in raw.values():
            try:
                return _extract_dot_bracket(value, length=length)
            except ValueError:
                continue
    if not isinstance(raw, str):
        raise ValueError(f"expected eFold prediction as string or mapping, got {type(raw)!r}")
    candidates = []
    for line in raw.splitlines() or [raw]:
        stripped = line.strip()
        if not stripped:
            continue
        candidates.append(stripped)
        candidates.extend(part.strip() for part in stripped.split())
    for candidate in candidates:
        if len(candidate) == length and set(candidate) <= DOTBRACKET_CHARS:
            return candidate
    raise ValueError(f"could not find length-{length} dot-bracket prediction in {raw!r}")


def _extract_base_pairs(raw: object, *, length: int, one_based: bool) -> list[list[int]]:
    """Extract sorted zero-based base pairs from a module return value."""

    if isinstance(raw, Mapping):
        for key in ("predicted_pairs", "pairs", "base_pairs", "basepairs", "structure"):
            if key in raw:
                return _extract_base_pairs(raw[key], length=length, one_based=one_based)
        for value in raw.values():
            try:
                return _extract_base_pairs(value, length=length, one_based=one_based)
            except ValueError:
                continue
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"expected base-pair list, got {type(raw)!r}")
    pairs: list[list[int]] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"expected pair entries of length 2, got {item!r}")
        left = int(item[0])
        right = int(item[1])
        if one_based:
            left -= 1
            right -= 1
        # eFold occasionally emits diagonal artifacts such as (51, 51) in its
        # base-pair list. They are not valid RNA pairs, so scoring should ignore
        # them rather than aborting an otherwise usable baseline rerun.
        if left == right:
            continue
        if not (0 <= left < length and 0 <= right < length):
            raise ValueError(f"invalid base pair {(item[0], item[1])!r} for sequence length {length}")
        if left > right:
            left, right = right, left
        pairs.append([left, right])
    return sorted(pairs)


def _prediction_record(raw: object, *, length: int) -> dict:
    """Normalize an eFold return value into prediction JSONL fields."""

    if isinstance(raw, Mapping) and "predicted_pairs" in raw:
        return {"predicted_pairs": _extract_base_pairs(raw["predicted_pairs"], length=length, one_based=False)}
    dot_bracket = _extract_dot_bracket(raw, length=length)
    return {"dotbracket": dot_bracket, "predicted_pairs": parse_dot_bracket(dot_bracket)}


def _predict_with_module_on_device(sequence: str, *, device: str) -> dict:
    """Run eFold directly on the requested torch device.

    Complexity: O(L^2) tensor storage for sequence length L plus eFold model
    inference cost; this avoids CPU-only default inference for long same-split
    reruns.
    """

    import importlib

    import torch

    run_module = importlib.import_module("efold.api.run")
    torch_device = torch.device(device)
    if str(torch_device) not in _EFOLD_DEVICE_READY:
        run_module.model.to(torch_device)
        run_module.model.eval()
        _EFOLD_DEVICE_READY.add(str(torch_device))

    seq = run_module.sequence_to_int(sequence).unsqueeze(0).to(torch_device)
    batch_obj = run_module.batch.Batch(
        sequence=seq,
        reference=[""],
        length=[len(seq)],
        L=len(seq),
        use_error=False,
        batch_size=1,
        data_types=["sequence"],
        dt_count={"sequence": 1},
        device=str(torch_device),
    )
    with torch.no_grad():
        pred = run_module.model(batch_obj)
        structure = (
            run_module.postprocess(
                pred["structure"],
                run_module.model.seq2oneHot(batch_obj.get("sequence")),
                0.01,
                0.1,
                100,
                1.6,
                True,
                1.5,
            )
            .detach()
            .cpu()
            .numpy()
            .round()
            .astype(int)[0]
        )
    pairs = [[int(left), int(right)] for left, right in (run_module.np.stack(run_module.np.where(run_module.np.triu(structure) == 1)) + 1).T]
    return {"predicted_pairs": _extract_base_pairs(pairs, length=len(sequence), one_based=True)}


def _predict_with_module(sequence: str, *, device: str = "cpu") -> object:
    """Run ``efold.inference`` or a direct device-aware eFold call.

    Complexity: O(L^2) tensor storage for sequence length L plus eFold inference
    cost.
    """

    try:
        from efold import inference  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised through CLI tests.
        raise RuntimeError("Python package 'efold' is not importable") from exc
    if device != "cpu":
        return _predict_with_module_on_device(sequence, device=device)
    errors = []
    for kwargs in ({"fmt": "basepair"}, {"fmt": "bp"}):
        try:
            return {"predicted_pairs": _extract_base_pairs(inference(sequence, **kwargs), length=len(sequence), one_based=True)}
        except (TypeError, ValueError, AssertionError) as exc:
            errors.append(str(exc))
            continue
    for kwargs in ({"fmt": "dotbracket"}, {"fmt": "db"}, {}):
        try:
            return _extract_dot_bracket(inference(sequence, **kwargs), length=len(sequence))
        except (TypeError, ValueError, AssertionError) as exc:
            errors.append(str(exc))
            continue
    raise RuntimeError(f"could not call efold.inference with known signatures: {errors}")


def _predict_with_cli(sequence: str, *, efold_bin: str) -> str:
    """Run the eFold CLI and return dot-bracket output."""

    attempts = ([efold_bin, sequence, "-db"], [efold_bin, sequence])
    last_error = ""
    for command in attempts:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode == 0:
            return _extract_dot_bracket(completed.stdout, length=len(sequence))
        last_error = (completed.stderr or completed.stdout).strip()
    raise RuntimeError(f"eFold CLI failed for sequence length {len(sequence)}: {last_error[:500]}")


def _prediction_function(*, backend: str, efold_bin: Optional[str], device: str) -> Callable[[str], object]:
    """Return the configured eFold predictor.

    Complexity: O(1) setup; returned callables carry the selected backend and
    device.
    """

    if backend in {"auto", "cli"}:
        resolved_bin = efold_bin or shutil.which("efold")
        if resolved_bin:
            return lambda sequence: _predict_with_cli(sequence, efold_bin=resolved_bin)
        if backend == "cli":
            raise RuntimeError("efold CLI not found; install `efold` or pass --efold-bin")
    if backend in {"auto", "module"}:
        return lambda sequence: _predict_with_module(sequence, device=device)
    raise ValueError(f"unsupported backend {backend!r}")


def _iter_jsonl(path: Path) -> Iterable[dict]:
    """Yield JSON objects from a JSONL file."""

    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            yield obj


def _parse_tier_path(value: str) -> tuple[str, Path]:
    """Parse ``tier=path`` CLI arguments."""

    if "=" not in value:
        raise ValueError(f"expected tier=path, got {value!r}")
    tier, path = value.split("=", 1)
    if not tier.strip():
        raise ValueError(f"empty tier in {value!r}")
    return tier.strip(), Path(path)


def _identity_fields(row: Mapping[str, object]) -> dict:
    """Preserve the key field used by the scorer for matching predictions."""

    for key in ("source_id", "id", "record_id", "reference"):
        value = row.get(key)
        if value not in (None, ""):
            return {key: value}
    return {}


def _jsonl_line_count(path: Path) -> int:
    """Return the number of non-empty JSONL records already written."""

    if not path.exists():
        return 0
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def export_predictions(
    *,
    tier: str,
    gold_path: Path,
    output_path: Path,
    predict_structure: Callable[[str], object],
    limit: Optional[int],
    progress_every: int,
    resume_existing: bool,
) -> dict:
    """Run eFold for one tier and write prediction JSONL."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_count = _jsonl_line_count(output_path) if resume_existing else 0
    count = existing_count
    mode = "a" if resume_existing and existing_count > 0 else "w"
    with output_path.open(mode, encoding="utf-8") as handle:
        for gold_index, row in enumerate(_iter_jsonl(gold_path)):
            if limit is not None and count >= limit:
                break
            if gold_index < existing_count:
                continue
            sequence = str(row.get("sequence") or "").upper()
            if not sequence:
                raise ValueError(f"{gold_path} contains a record without sequence")
            prediction = _prediction_record(predict_structure(sequence), length=len(sequence))
            record = {
                **_identity_fields(row),
                "sequence": sequence,
                "prediction_backend": "efold",
                **prediction,
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            count += 1
            if progress_every > 0 and count % progress_every == 0:
                print(f"[efold-baseline] tier={tier} predictions={count}", file=sys.stderr, flush=True)
    return {
        "tier": tier,
        "gold": str(gold_path),
        "predictions": str(output_path),
        "prediction_count": count,
        "resumed_from": existing_count,
    }


def run_baseline(
    *,
    gold_paths: Mapping[str, Path],
    output_dir: Path,
    results_json: Path,
    model: str,
    protocol: str,
    seed_count: str,
    backend: str,
    efold_bin: Optional[str],
    device: str,
    limit: Optional[int],
    emit_partial_rows: bool,
    progress_every: int,
    resume_existing: bool,
) -> dict:
    """Export eFold prediction JSONL files and score them."""

    predict = _prediction_function(backend=backend, efold_bin=efold_bin, device=device)
    prediction_paths = {}
    exports = {}
    for tier, gold_path in sorted(gold_paths.items()):
        output_path = output_dir / f"{tier}.efold.predictions.jsonl"
        exports[tier] = export_predictions(
            tier=tier,
            gold_path=gold_path,
            output_path=output_path,
            predict_structure=predict,
            limit=limit,
            progress_every=progress_every,
            resume_existing=resume_existing,
        )
        prediction_paths[tier] = output_path
    scorer = _load_scorer_module()
    payload = scorer.evaluate_baselines(
        gold_paths=gold_paths,
        prediction_paths=prediction_paths,
        model=model,
        protocol=protocol,
        seed_count=seed_count,
        output_path=results_json,
        one_based_predictions=False,
        emit_partial_rows=emit_partial_rows,
    )
    results_json.parent.mkdir(parents=True, exist_ok=True)
    results_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"exports": exports, "results": str(results_json), "rows": len(payload["rows"])}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-json", action="append", required=True, help="tier=path gold JSONL; may repeat")
    parser.add_argument("--output-dir", required=True, help="directory for exported prediction JSONL files")
    parser.add_argument("--results-json", required=True, help="output baseline results JSON")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--protocol", choices=ALLOWED_PROTOCOLS, default="same_split_local")
    parser.add_argument("--seed-count", default="single_seed")
    parser.add_argument("--backend", choices=("auto", "cli", "module"), default="auto")
    parser.add_argument("--efold-bin", help="path to the efold CLI binary")
    parser.add_argument("--device", default="cpu", help="torch device for module backend, e.g. cpu or cuda")
    parser.add_argument("--limit", type=int, help="optional smoke-test record limit per tier")
    parser.add_argument("--progress-every", type=int, default=1000, help="log progress every N predictions; use 0 to disable")
    parser.add_argument("--resume-existing", action="store_true", help="append after existing prediction JSONL rows instead of overwriting")
    parser.add_argument("--emit-partial-rows", action="store_true")
    args = parser.parse_args(argv)

    gold_paths = dict(_parse_tier_path(value) for value in args.gold_json)
    summary = run_baseline(
        gold_paths=gold_paths,
        output_dir=Path(args.output_dir),
        results_json=Path(args.results_json),
        model=args.model,
        protocol=args.protocol,
        seed_count=args.seed_count,
        backend=args.backend,
        efold_bin=args.efold_bin,
        device=args.device,
        limit=args.limit,
        emit_partial_rows=args.emit_partial_rows,
        progress_every=args.progress_every,
        resume_existing=args.resume_existing,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
