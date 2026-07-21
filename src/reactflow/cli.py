"""Command-line entry points for ReactFlow reproducibility."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

from reactflow.constraints import dotbracket_to_matrix
from reactflow.data import RibonanzaProfile, feature_engineering_report, normalize_profile, read_ribonanza_csv, validate_profile
from reactflow.symbolic import run_all_symbolic_checks
from reactflow.visualization import (
    write_guidance_scan_svg,
    write_pair_heatmap_svg,
    write_profile_overlay_svg,
    write_training_curves_svg,
)


def _cmd_validate_csv(args: argparse.Namespace) -> int:
    """Validate a Ribonanza-style CSV file and print JSONL reports.

    Complexity: O(NL) for N profiles and max length L.
    """

    for profile in read_ribonanza_csv(Path(args.path), limit=args.limit):
        normalized = normalize_profile(profile.reactivity, method=args.normalization)
        normalized_profile = RibonanzaProfile(
            sequence=profile.sequence,
            probe=profile.probe,
            reactivity=normalized,
            error=profile.error,
            reads=profile.reads,
            snr=profile.snr,
            sequence_id=profile.sequence_id,
        )
        report = validate_profile(normalized_profile)
        payload = {
            "validation": report.__dict__,
            "features": feature_engineering_report(normalized_profile),
        }
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


def _cmd_verify_symbolic(_: argparse.Namespace) -> int:
    """Run SymPy symbolic checks and return non-zero on residuals.

    Complexity: O(1) for the current fixed checks.
    """

    results = run_all_symbolic_checks()
    print(json.dumps(results, indent=2, ensure_ascii=False, sort_keys=True))
    residuals = []
    for check in results.values():
        residuals.extend(value for key, value in check.items() if key.startswith("residual"))
    return 0 if all(value == "0" for value in residuals) else 1


def _cmd_plot_dotbracket(args: argparse.Namespace) -> int:
    """Render dot-bracket structure as an SVG pair heatmap.

    Complexity: O(L^2) for the output matrix.
    """

    matrix = dotbracket_to_matrix(args.dotbracket)
    write_pair_heatmap_svg(matrix, Path(args.output), title=args.title)
    return 0


def _cmd_plot_profiles(args: argparse.Namespace) -> int:
    """Render comma-separated target/predicted profiles as SVG.

    Complexity: O(L).
    """

    predicted = tuple(float(v) for v in args.predicted.split(","))
    target = tuple(float(v) for v in args.target.split(","))
    write_profile_overlay_svg(predicted, target, Path(args.output), title=args.title)
    return 0


def _diagnostic_features(
    sequence: str,
    adapter: "Optional[object]" = None,
    frozen: "Optional[object]" = None,
) -> "tuple":
    """Build the per-position feature matrix for a diagnostic/eval forward pass.

    The denoiser is evaluated at flow time ``t = 1`` with every position set to
    the unpaired noised class, i.e. the clean-input encoding.  When ``adapter`` is
    ``None`` this is exactly the ``FEATURE_SIZE`` C3 encoding; when a warm-start
    :class:`~reactflow.features.FeatureAdapter` is supplied its projected frozen
    representation (looked up per sequence in ``frozen``, or a zero fallback) is
    concatenated on, so the feature width matches a warm-start denoiser.

    Complexity: O(L) base plus O(L * d_adapter * d_single) when an adapter is used.
    """

    from reactflow.features import build_augmented_features
    from reactflow.train import build_features

    base = build_features(sequence, 1.0, [0 for _ in sequence])
    if adapter is None:
        return base
    single_rows = frozen.single_rows(sequence) if frozen is not None else None
    augmented, _ = build_augmented_features(base, adapter, single_rows)
    return augmented


def _parse_bucket_boundaries(raw: str) -> tuple:
    """Parse a comma-separated sequence of increasing length boundaries."""

    if raw.strip() == "":
        return tuple()
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    for index in range(1, len(values)):
        if values[index] <= values[index - 1]:
            raise argparse.ArgumentTypeError("--bucket-boundaries must be strictly increasing")
    return values


def _train_with_backend(
    args: argparse.Namespace,
    *,
    train_pilot,
    train_pilot_torch,
    samples,
    config,
    frozen=None,
):
    """Run the selected training backend, returning ``None`` on CLI-safe errors."""

    try:
        if getattr(args, "backend", "stdlib") == "torch":
            return train_pilot_torch(samples=samples, config=config, frozen=frozen, device=args.torch_device)
        return train_pilot(samples=samples, config=config, frozen=frozen)
    except (RuntimeError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return None


def _length_bucket_counts(samples, boundaries: Sequence[int]) -> dict:
    """Return compact sample counts by configured length bucket."""

    if not boundaries:
        return {}
    from reactflow.train import bucket_samples_by_length

    return {label: len(items) for label, items in bucket_samples_by_length(samples, boundaries).items()}


def _write_training_checkpoint_artifact(output_dir: Path, *, config, result, metadata: dict) -> Path:
    """Write the standard training checkpoint artifact into ``output_dir``."""

    from reactflow.checkpoint import write_training_checkpoint

    checkpoint_path = Path(output_dir) / "training_checkpoint.json"
    write_training_checkpoint(checkpoint_path, config=config, result=result, metadata=metadata)
    return checkpoint_path


class _ProfileAppendLogger:
    """Append evaluation/finalization timing events to an existing profile JSONL.

    Formula: each event records ``(phase, seconds, sample_index, length, tier)``
    as one JSON row.  Appending preserves the training profiler's earlier
    ``epoch_total`` evidence while extending the same heartbeat stream through
    tier evaluation and artifact materialization.  Complexity: O(1) memory and
    O(event bytes) I/O per logged event.
    """

    def __init__(self, path: Optional[str]) -> None:
        """Open ``path`` in append mode, or become a no-op logger when absent.

        Complexity: O(1).
        """

        self.path = Path(path) if path else None
        self._handle = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("a", encoding="utf-8")

    def log(
        self,
        phase: str,
        seconds: float,
        *,
        sample_index: Optional[int] = None,
        length: Optional[int] = None,
        tier: Optional[str] = None,
        flush: bool = False,
    ) -> None:
        """Append one timing row and optionally flush it as a heartbeat.

        Formula: write ``max(seconds, 0)`` for phase ``p``.  The non-negative
        clamp is only defensive against clock jitter and does not affect model
        computation.  Complexity: O(1) plus one JSONL write.
        """

        if self._handle is None:
            return
        record = {
            "epoch": None,
            "length": length,
            "phase": phase,
            "sample_index": sample_index,
            "seconds": max(0.0, float(seconds)),
            "tier": tier,
        }
        self._handle.write(json.dumps(record, sort_keys=True) + "\n")
        if flush:
            self._handle.flush()

    def close(self) -> None:
        """Close the append stream when it is active.

        Complexity: O(1).
        """

        if self._handle is not None:
            self._handle.close()
            self._handle = None


def _cmd_train(args: argparse.Namespace) -> int:
    """Run the deterministic training pilot and write diagnostics.

    Trains either the base C3 pilot (``--adapter-dim 0``) or the C5 warm-start
    path (``--adapter-dim > 0`` with ``--frozen-dir`` pointing at an exported
    frozen shard).  Writes three SVG artifacts into ``--output-dir``: the
    multi-metric training curves, a reactivity overlay (calibrated prediction vs.
    synthetic target for the first sample) and the predicted pairing-marginal
    heatmap.  A JSON summary of the loss trajectory is printed to stdout.

    Complexity: O(epochs * N * L^2 H^2) for training plus O(L^2) for plots.
    """

    from reactflow.features import FeatureAdapter, load_frozen_features
    from reactflow.model import PairwiseDenoiser, marginal_pair_matrix
    from reactflow.reactivity import ReactivityForwardOperator, fit_weighted_affine_calibration
    from reactflow.synthetic import make_dataset
    from reactflow.train import (
        FEATURE_SIZE,
        TrainConfig,
        _predicted_reactivity,
        _reactivity_coefficients,
        train_pilot,
        train_pilot_torch,
    )

    if args.adapter_dim > 0 and not args.frozen_dir:
        print(
            json.dumps({"error": "--adapter-dim > 0 requires --frozen-dir"}, ensure_ascii=False),
            file=sys.stderr,
        )
        return 2
    frozen = load_frozen_features(
        Path(args.frozen_dir),
        max_loaded_shards=getattr(args, "frozen_cache_shards", 4),
    ) if args.frozen_dir else None

    config = TrainConfig(
        epochs=args.epochs,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        lambda_react=args.lambda_react,
        lambda_thermo=args.lambda_thermo,
        thermo_mode=args.thermo_mode,
        seed=args.seed,
        adapter_dim=args.adapter_dim,
        adapter_lr=args.adapter_lr,
        profile_path=args.profile_path or None,
        batch_size=args.batch_size,
        length_bucket_boundaries=args.bucket_boundaries,
        family_balanced_batches=getattr(args, "family_balanced_batches", False),
        lambda_calib=getattr(args, "lambda_calib", 0.0),
        calib_beta=getattr(args, "calib_beta", 1.0),
        calib_tau_squared=getattr(args, "calib_tau_squared", 0.05),
        lambda_contact=getattr(args, "lambda_contact", 0.0),
        contact_negative_weight=getattr(args, "contact_negative_weight", 0.25),
        contact_long_range_min_distance=getattr(args, "contact_long_range_min_distance", 24),
        contact_long_range_weight=getattr(args, "contact_long_range_weight", 1.0),
    )
    samples = make_dataset(count=args.samples, stem=args.stem, loop=args.loop, probe=args.probe, seed=1)
    result = _train_with_backend(
        args,
        train_pilot=train_pilot,
        train_pilot_torch=train_pilot_torch,
        samples=samples,
        config=config,
        frozen=frozen,
    )
    if result is None:
        return 2

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history = result.history
    curves = {
        "total": [record.total for record in history],
        "dfm": [record.dfm for record in history],
        "react_mag": [record.react_magnitude for record in history],
        "react_shape": [record.react_shape for record in history],
        "mean_f1": [record.mean_f1 for record in history],
    }
    write_training_curves_svg(curves, output_dir / "training_curves.svg", title="ReactFlow training")

    model = PairwiseDenoiser(result.parameters, min_loop=config.min_loop)
    adapter = (
        FeatureAdapter(result.adapter_parameters)
        if result.adapter_parameters is not None
        else None
    )
    operator = ReactivityForwardOperator()
    sample = samples[0]
    features = _diagnostic_features(sample.sequence, adapter, frozen)
    forward = model.forward(sample.sequence, features)
    a_values, c_values = _reactivity_coefficients(operator, sample.sequence, sample.probe)
    predicted = _predicted_reactivity(forward.marginals, a_values, c_values)
    alpha, gamma = fit_weighted_affine_calibration(predicted, sample.reactivity, sample.weights)
    calibrated = tuple(alpha * value + gamma for value in predicted)
    write_profile_overlay_svg(
        calibrated,
        sample.reactivity,
        output_dir / "reactivity_overlay.svg",
        title="Calibrated predicted vs. synthetic reactivity (sample 0)",
    )
    write_pair_heatmap_svg(
        marginal_pair_matrix(forward.marginals),
        output_dir / "pairing_marginals.svg",
        title="Predicted pairing marginals (sample 0)",
    )

    warm_start = None
    if result.adapter_parameters is not None:
        matched = sum(1 for s in samples if frozen is not None and frozen.has(s.sequence))
        warm_start = {
            "adapter_dim": result.adapter_parameters.d_adapter,
            "d_single": result.adapter_parameters.d_single,
            "frozen_dir": args.frozen_dir,
            "frozen_records": len(frozen) if frozen is not None else 0,
            "matched_pilot_sequences": matched,
        }

    checkpoint_path = _write_training_checkpoint_artifact(
        output_dir,
        config=config,
        result=result,
        metadata={
            "backend": args.backend,
            "dataset": "synthetic_pilot",
            "family_balanced_batches": config.family_balanced_batches,
            "mode": "warm_start" if warm_start is not None else "base",
            "samples": len(samples),
        },
    )
    summary = {
        "backend": args.backend,
        "mode": "warm_start" if warm_start is not None else "base",
        "epochs": config.epochs,
        "samples": len(samples),
        "feature_size": FEATURE_SIZE + config.adapter_dim,
        "family_balanced_batches": config.family_balanced_batches,
        "contact_long_range_min_distance": config.contact_long_range_min_distance,
        "contact_long_range_weight": config.contact_long_range_weight,
        "length_buckets": _length_bucket_counts(samples, args.bucket_boundaries),
        "first": history[0].__dict__,
        "last": history[-1].__dict__,
        "warm_start": warm_start,
        "profile": result.profile_summary,
        "artifacts": [
            str(output_dir / "training_curves.svg"),
            str(output_dir / "reactivity_overlay.svg"),
            str(output_dir / "pairing_marginals.svg"),
            str(checkpoint_path),
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    """Run the C5.4 evaluation protocol on a freshly trained pilot model.

    Trains a pilot (base or warm-start), then scores three deterministic
    synthetic-pilot generalization tiers -- ``in_clan`` (the training
    distribution), ``cross_clan`` (unseen sequences of the same structural
    family) and ``novel_clan`` (a structurally novel, longer family) -- to read
    off the OOD generalization gap ``F1(in_clan) - F1(novel_clan)``.  It also
    reports Pearson/Spearman/calibrated-MAE reactivity metrics per tier and emits
    an *honest* cited-vs-local comparison table: the eFold DOI numbers stay in the
    cited column, the synthetic-pilot recompute stays in the local column, and the
    two are never merged.  The eFold public-benchmark rows remain ``pending``
    because ReactFlow has not been trained on those real corpora yet.

    Returns non-zero only if the tiers cannot be scored (e.g. a tier is empty).

    Complexity: O(epochs * N * L^2 H^2) training plus O(sum_k L_k^2) scoring.
    """

    from reactflow.constraints import project_greedy_matching
    from reactflow.evaluate import (
        StructurePrediction,
        build_comparison_table,
        generalization_gap,
        reactivity_metrics,
        render_comparison_markdown,
        structure_distance_bin_metrics_by_tier,
        structure_metrics_by_tier,
    )
    from reactflow.features import FeatureAdapter, load_frozen_features
    from reactflow.model import PairwiseDenoiser, marginal_pair_matrix
    from reactflow.reactivity import ReactivityForwardOperator
    from reactflow.synthetic import make_dataset
    from reactflow.train import (
        TrainConfig,
        _predicted_reactivity,
        _reactivity_coefficients,
        train_pilot,
        train_pilot_torch,
    )

    if args.adapter_dim > 0 and not args.frozen_dir:
        print(
            json.dumps({"error": "--adapter-dim > 0 requires --frozen-dir"}, ensure_ascii=False),
            file=sys.stderr,
        )
        return 2
    frozen = load_frozen_features(
        Path(args.frozen_dir),
        max_loaded_shards=getattr(args, "frozen_cache_shards", 4),
    ) if args.frozen_dir else None

    config = TrainConfig(
        epochs=args.epochs,
        seed=args.seed,
        adapter_dim=args.adapter_dim,
        adapter_lr=args.adapter_lr,
        lambda_thermo=args.lambda_thermo,
        thermo_mode=args.thermo_mode,
        profile_path=args.profile_path or None,
        batch_size=args.batch_size,
        length_bucket_boundaries=args.bucket_boundaries,
        family_balanced_batches=getattr(args, "family_balanced_batches", False),
        lambda_calib=getattr(args, "lambda_calib", 0.0),
        calib_beta=getattr(args, "calib_beta", 1.0),
        calib_tau_squared=getattr(args, "calib_tau_squared", 0.05),
        lambda_contact=getattr(args, "lambda_contact", 0.0),
        contact_negative_weight=getattr(args, "contact_negative_weight", 0.25),
        contact_long_range_min_distance=getattr(args, "contact_long_range_min_distance", 24),
        contact_long_range_weight=getattr(args, "contact_long_range_weight", 1.0),
    )
    train_samples = make_dataset(count=args.samples, stem=args.stem, loop=args.loop, probe=args.probe, seed=1)
    result = _train_with_backend(
        args,
        train_pilot=train_pilot,
        train_pilot_torch=train_pilot_torch,
        samples=train_samples,
        config=config,
        frozen=frozen,
    )
    if result is None:
        return 2
    model = PairwiseDenoiser(result.parameters, min_loop=config.min_loop)
    adapter = (
        FeatureAdapter(result.adapter_parameters)
        if result.adapter_parameters is not None
        else None
    )
    operator = ReactivityForwardOperator()

    # Deterministic, clearly-labelled synthetic-pilot proxies for the three
    # generalization tiers.  ``in_clan`` reuses the training seed (in-distribution);
    # ``cross_clan`` draws unseen sequences of the same structural family; and
    # ``novel_clan`` uses a longer stem/loop so the structural family itself is new.
    tier_specs = {
        "in_clan": {"seed": 1, "stem": args.stem, "loop": args.loop},
        "cross_clan": {"seed": 2, "stem": args.stem, "loop": args.loop},
        "novel_clan": {"seed": 3, "stem": args.stem + 1, "loop": args.loop + 1},
    }

    predictions = []
    reactivity_by_tier = {}
    for tier, spec in tier_specs.items():
        tier_samples = make_dataset(count=args.samples, probe=args.probe, **spec)
        for sample in tier_samples:
            features = _diagnostic_features(sample.sequence, adapter, frozen)
            marginals = model.forward(sample.sequence, features).marginals
            soft = marginal_pair_matrix(marginals)
            projected = project_greedy_matching(
                sample.sequence,
                soft,
                min_loop=config.min_loop,
                allow_wobble=model.allow_wobble,
                allow_pseudoknot=True,
                min_score=1e-6,
            )
            predictions.append(
                StructurePrediction(predicted=projected, target=sample.pair_matrix, tier=tier)
            )
        head = tier_samples[0]
        head_features = _diagnostic_features(head.sequence, adapter, frozen)
        head_marginals = model.forward(head.sequence, head_features).marginals
        a_values, c_values = _reactivity_coefficients(operator, head.sequence, head.probe)
        predicted_react = _predicted_reactivity(head_marginals, a_values, c_values)
        reactivity_by_tier[tier] = reactivity_metrics(predicted_react, head.reactivity, head.weights)

    tier_metrics = structure_metrics_by_tier(predictions)
    distance_metrics = structure_distance_bin_metrics_by_tier(predictions)
    gap = generalization_gap(tier_metrics)

    cited = {
        "viral_mRNA": (0.73, "10.1126/sciadv.adz4967"),
        "lncRNA": (0.44, "10.1126/sciadv.adz4967"),
    }
    local = {"synthetic_pilot_novel_clan": tier_metrics["novel_clan"].mean_f1}
    table = build_comparison_table(cited, local)
    markdown = render_comparison_markdown(table)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "comparison_table.md"
    table_path.write_text(markdown, encoding="utf-8")
    checkpoint_path = _write_training_checkpoint_artifact(
        output_dir,
        config=config,
        result=result,
        metadata={
            "backend": args.backend,
            "dataset": "synthetic_pilot",
            "family_balanced_batches": config.family_balanced_batches,
            "mode": "warm_start" if adapter is not None else "base",
            "samples": len(train_samples),
        },
    )

    summary = {
        "backend": args.backend,
        "mode": "warm_start" if adapter is not None else "base",
        "epochs": config.epochs,
        "samples_per_tier": args.samples,
        "family_balanced_batches": config.family_balanced_batches,
        "contact_long_range_min_distance": config.contact_long_range_min_distance,
        "contact_long_range_weight": config.contact_long_range_weight,
        "length_buckets": _length_bucket_counts(train_samples, args.bucket_boundaries),
        "tiers": {
            tier: {
                "count": metrics.count,
                "mean_f1": round(metrics.mean_f1, 4),
                "mean_mcc": round(metrics.mean_mcc, 4),
                "micro_f1": round(metrics.micro_f1, 4),
                "micro_mcc": round(metrics.micro_mcc, 4),
            }
            for tier, metrics in tier_metrics.items()
        },
        "distance_bins": {
            tier: {
                label: {
                    "count": metrics.count,
                    "mean_f1": round(metrics.mean_f1, 4),
                    "mean_mcc": round(metrics.mean_mcc, 4),
                    "micro_f1": round(metrics.micro_f1, 4),
                    "micro_mcc": round(metrics.micro_mcc, 4),
                }
                for label, metrics in bins.items()
            }
            for tier, bins in distance_metrics.items()
        },
        "generalization_gap": {
            "in_clan_f1": round(gap.in_clan_f1, 4),
            "novel_clan_f1": round(gap.novel_clan_f1, 4),
            "gap": round(gap.gap, 4),
        },
        "reactivity": {
            tier: {
                "count": metrics.count,
                "pearson": round(metrics.pearson, 4),
                "spearman": round(metrics.spearman, 4),
                "calibrated_mae": round(metrics.calibrated_mae, 4),
            }
            for tier, metrics in reactivity_by_tier.items()
        },
        "profile": result.profile_summary,
        "comparison_markdown": markdown,
        "artifacts": [str(table_path), str(checkpoint_path)],
        "note": (
            "Cited F1 are eFold public-benchmark numbers from DOI "
            "10.1126/sciadv.adz4967 and are never merged with local numbers. "
            "Local numbers are recomputed on a labelled deterministic synthetic "
            "pilot (not a public benchmark); eFold public sets stay 'pending' "
            "until ReactFlow is trained on the real corpora in a full C5 run."
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


def _cmd_prepare_efold_cache(args: argparse.Namespace) -> int:
    """Materialize filtered eFold/RNAndria records into JSONL sample cache."""

    from reactflow.train import build_efold_sample_cache

    summary = build_efold_sample_cache(
        [Path(path) for path in args.json],
        Path(args.output),
        limit=args.limit,
        scan_limit=args.scan_limit,
        min_length=args.min_length,
        max_length=args.max_length,
        default_probe=args.probe,
        min_loop=args.min_loop,
        window_size=args.window_size,
        window_stride=args.window_stride,
        length_bucket_boundaries=args.bucket_boundaries,
    )
    print(json.dumps(summary.__dict__, ensure_ascii=False, sort_keys=True))
    return 0


def _cmd_split_efold_cache(args: argparse.Namespace) -> int:
    """Split eFold JSONL caches into leakage-safe train/val/test/novel files."""

    from reactflow.splits import split_efold_cache_by_clan

    fractions = {
        "train": args.train_fraction,
        "val": args.val_fraction,
        "test": args.test_fraction,
    }
    summary = split_efold_cache_by_clan(
        [Path(path) for path in args.cache],
        Path(args.output_dir),
        metadata_tsv=Path(args.metadata_tsv) if args.metadata_tsv else None,
        manifest_path=Path(args.manifest) if args.manifest else None,
        fractions=fractions,
        novel_clan_fraction=args.novel_clan_fraction,
        length_bucket_boundaries=args.bucket_boundaries,
        seed=args.seed,
    )
    print(json.dumps(summary.__dict__, ensure_ascii=False, sort_keys=True))
    return 0


def _cmd_train_efold(args: argparse.Namespace) -> int:
    """Train the ReactFlow head on real eFold/RNAndria structure JSON files.

    The eFold Dryad files contain sequence + hard 2D structure targets and, for
    some subsets, SHAPE/DMS profiles.  This command converts those records into
    the same sample contract used by ``train_pilot``.  Its default
    ``lambda_react=0`` is intentional: structure-only files must not silently turn
    the deterministic forward proxy into fabricated experimental supervision.

    Complexity: O(epochs * N * L^2 H^2) after JSON loading.
    """

    from reactflow.features import FeatureAdapter, load_frozen_features
    from reactflow.model import PairwiseDenoiser, marginal_pair_matrix
    from reactflow.reactivity import ReactivityForwardOperator, fit_weighted_affine_calibration
    from reactflow.train import (
        FEATURE_SIZE,
        TrainConfig,
        _predicted_reactivity,
        _reactivity_coefficients,
        load_efold_samples,
        train_pilot,
        train_pilot_torch,
    )

    if args.adapter_dim > 0 and not args.frozen_dir:
        print(json.dumps({"error": "--adapter-dim > 0 requires --frozen-dir"}, ensure_ascii=False), file=sys.stderr)
        return 2
    frozen = load_frozen_features(
        Path(args.frozen_dir),
        max_loaded_shards=getattr(args, "frozen_cache_shards", 4),
    ) if args.frozen_dir else None
    samples = load_efold_samples(
        [Path(path) for path in args.json],
        limit=args.limit,
        min_length=args.min_length,
        max_length=args.max_length,
        default_probe=args.probe,
        min_loop=args.min_loop,
        window_size=args.window_size,
        window_stride=args.window_stride,
        length_bucket_boundaries=args.bucket_boundaries,
    )
    if not samples:
        print(json.dumps({"error": "no eFold samples passed filters"}, ensure_ascii=False), file=sys.stderr)
        return 2

    config = TrainConfig(
        epochs=args.epochs,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        lambda_react=args.lambda_react,
        lambda_thermo=args.lambda_thermo,
        thermo_mode=args.thermo_mode,
        seed=args.seed,
        min_loop=args.min_loop,
        adapter_dim=args.adapter_dim,
        adapter_lr=args.adapter_lr,
        profile_path=args.profile_path or None,
        batch_size=args.batch_size,
        length_bucket_boundaries=args.bucket_boundaries,
        family_balanced_batches=getattr(args, "family_balanced_batches", False),
        lambda_calib=getattr(args, "lambda_calib", 0.0),
        calib_beta=getattr(args, "calib_beta", 1.0),
        calib_tau_squared=getattr(args, "calib_tau_squared", 0.05),
        lambda_contact=getattr(args, "lambda_contact", 0.0),
        contact_negative_weight=getattr(args, "contact_negative_weight", 0.25),
        contact_long_range_min_distance=getattr(args, "contact_long_range_min_distance", 24),
        contact_long_range_weight=getattr(args, "contact_long_range_weight", 1.0),
    )
    result = _train_with_backend(
        args,
        train_pilot=train_pilot,
        train_pilot_torch=train_pilot_torch,
        samples=samples,
        config=config,
        frozen=frozen,
    )
    if result is None:
        return 2

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history = result.history
    write_training_curves_svg(
        {
            "total": [record.total for record in history],
            "dfm": [record.dfm for record in history],
            "react_mag": [record.react_magnitude for record in history],
            "react_shape": [record.react_shape for record in history],
            "mean_f1": [record.mean_f1 for record in history],
        },
        output_dir / "training_curves.svg",
        title="ReactFlow eFold training",
    )

    model = PairwiseDenoiser(result.parameters, min_loop=config.min_loop)
    adapter = FeatureAdapter(result.adapter_parameters) if result.adapter_parameters is not None else None
    operator = ReactivityForwardOperator()
    sample = samples[0]
    features = _diagnostic_features(sample.sequence, adapter, frozen)
    forward = model.forward(sample.sequence, features)
    a_values, c_values = _reactivity_coefficients(operator, sample.sequence, sample.probe)
    predicted = _predicted_reactivity(forward.marginals, a_values, c_values)
    alpha, gamma = fit_weighted_affine_calibration(predicted, sample.reactivity, sample.weights)
    calibrated = tuple(alpha * value + gamma for value in predicted)
    write_profile_overlay_svg(
        calibrated,
        sample.reactivity,
        output_dir / "reactivity_overlay.svg",
        title="Calibrated predicted vs. eFold/forward-proxy reactivity",
    )
    write_pair_heatmap_svg(
        marginal_pair_matrix(forward.marginals),
        output_dir / "pairing_marginals.svg",
        title="Predicted pairing marginals (eFold sample 0)",
    )

    warm_start = None
    if result.adapter_parameters is not None:
        matched = sum(1 for s in samples if frozen is not None and frozen.has(s.sequence))
        warm_start = {
            "adapter_dim": result.adapter_parameters.d_adapter,
            "d_single": result.adapter_parameters.d_single,
            "frozen_dir": args.frozen_dir,
            "frozen_records": len(frozen) if frozen is not None else 0,
            "matched_sequences": matched,
        }
    checkpoint_path = _write_training_checkpoint_artifact(
        output_dir,
        config=config,
        result=result,
        metadata={
            "backend": args.backend,
            "dataset": "efold",
            "family_balanced_batches": config.family_balanced_batches,
            "input_json": list(args.json),
            "mode": "warm_start" if warm_start is not None else "base",
            "samples": len(samples),
            "window_size": args.window_size,
            "window_stride": args.window_stride,
        },
    )
    summary = {
        "backend": args.backend,
        "dataset": "efold",
        "input_json": args.json,
        "mode": "warm_start" if warm_start is not None else "base",
        "epochs": config.epochs,
        "samples": len(samples),
        "family_balanced_batches": config.family_balanced_batches,
        "contact_long_range_min_distance": config.contact_long_range_min_distance,
        "contact_long_range_weight": config.contact_long_range_weight,
        "length_min": min(len(sample.sequence) for sample in samples),
        "length_max": max(len(sample.sequence) for sample in samples),
        "length_buckets": _length_bucket_counts(samples, args.bucket_boundaries),
        "window_size": args.window_size,
        "window_stride": args.window_stride,
        "feature_size": FEATURE_SIZE + config.adapter_dim,
        "lambda_react": config.lambda_react,
        "reactivity_note": (
            "Records with shape/dms use real probing profiles; structure-only records use "
            "f(S) only for monitoring. Keep lambda_react=0 for structure-only training."
        ),
        "first": history[0].__dict__,
        "last": history[-1].__dict__,
        "warm_start": warm_start,
        "profile": result.profile_summary,
        "artifacts": [
            str(output_dir / "training_curves.svg"),
            str(output_dir / "reactivity_overlay.svg"),
            str(output_dir / "pairing_marginals.svg"),
            str(checkpoint_path),
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


def _cmd_evaluate_efold(args: argparse.Namespace) -> int:
    """Train on eFold JSON and score named eFold/RNAndria evaluation tiers."""

    from reactflow.constraints import project_greedy_matching
    from reactflow.evaluate import (
        StructurePrediction,
        build_comparison_table,
        reactivity_metrics,
        render_comparison_markdown,
        structure_distance_bin_metrics_by_tier,
        structure_metrics_by_tier,
    )
    from reactflow.features import FeatureAdapter, load_frozen_features
    from reactflow.model import PairwiseDenoiser, marginal_pair_matrix
    from reactflow.reactivity import ReactivityForwardOperator
    from reactflow.train import (
        TrainConfig,
        _predicted_reactivity,
        _reactivity_coefficients,
        load_efold_samples,
        train_pilot,
        train_pilot_torch,
    )

    if args.adapter_dim > 0 and not args.frozen_dir:
        print(json.dumps({"error": "--adapter-dim > 0 requires --frozen-dir"}, ensure_ascii=False), file=sys.stderr)
        return 2
    if args.inference_mode != "legacy_direct" and not args.validation_json:
        print(
            json.dumps(
                {
                    "error": (
                        "corrected evaluate-efold requires --validation-json; "
                        "use --inference-mode legacy_direct only for explicit regression"
                    )
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2
    if args.profile_path:
        profile_path = Path(args.profile_path)
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_path.write_text("", encoding="utf-8")
    load_profiler = _ProfileAppendLogger(args.profile_path or None)
    try:
        phase_start = time.perf_counter()
        load_profiler.log("load_frozen_start", 0.0, flush=True)
        frozen = load_frozen_features(
            Path(args.frozen_dir),
            max_loaded_shards=getattr(args, "frozen_cache_shards", 4),
        ) if args.frozen_dir else None
        load_profiler.log("load_frozen_total", time.perf_counter() - phase_start, flush=True)

        phase_start = time.perf_counter()
        load_profiler.log("load_train_start", 0.0, flush=True)
        train_samples = load_efold_samples(
            [Path(path) for path in args.train_json],
            limit=args.train_limit,
            min_length=args.min_length,
            max_length=args.max_length,
            default_probe=args.probe,
            min_loop=args.min_loop,
            window_size=args.window_size,
            window_stride=args.window_stride,
            length_bucket_boundaries=args.bucket_boundaries,
        )
        load_profiler.log(
            "load_train_total",
            time.perf_counter() - phase_start,
            length=len(train_samples),
            flush=True,
        )
        if not train_samples:
            print(json.dumps({"error": "no train samples passed filters"}, ensure_ascii=False), file=sys.stderr)
            return 2

        eval_specs = args.eval_json or [f"in_clan={args.train_json[0]}"]
        eval_by_tier = {}
        for spec in eval_specs:
            if "=" not in spec:
                print(json.dumps({"error": "--eval-json must be tier=path"}, ensure_ascii=False), file=sys.stderr)
                return 2
            tier, raw_path = spec.split("=", 1)
            phase_start = time.perf_counter()
            load_profiler.log("load_eval_start", 0.0, tier=tier, flush=True)
            samples = load_efold_samples(
                [Path(raw_path)],
                limit=args.eval_limit,
                min_length=args.min_length,
                max_length=args.max_length,
                default_probe=args.probe,
                min_loop=args.min_loop,
                window_size=args.window_size,
                window_stride=args.window_stride,
                length_bucket_boundaries=args.bucket_boundaries,
            )
            load_profiler.log(
                "load_eval_total",
                time.perf_counter() - phase_start,
                length=len(samples),
                tier=tier,
                flush=True,
            )
            if samples:
                eval_by_tier[tier] = samples
        if not eval_by_tier:
            print(json.dumps({"error": "no eval samples passed filters"}, ensure_ascii=False), file=sys.stderr)
            return 2
    finally:
        load_profiler.close()

    config = TrainConfig(
        epochs=args.epochs,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        min_loop=args.min_loop,
        lambda_react=args.lambda_react,
        lambda_thermo=args.lambda_thermo,
        thermo_mode=args.thermo_mode,
        adapter_dim=args.adapter_dim,
        adapter_lr=args.adapter_lr,
        profile_path=args.profile_path or None,
        batch_size=args.batch_size,
        length_bucket_boundaries=args.bucket_boundaries,
        family_balanced_batches=getattr(args, "family_balanced_batches", False),
        lambda_calib=getattr(args, "lambda_calib", 0.0),
        calib_beta=getattr(args, "calib_beta", 1.0),
        calib_tau_squared=getattr(args, "calib_tau_squared", 0.05),
        lambda_contact=getattr(args, "lambda_contact", 0.0),
        contact_negative_weight=getattr(args, "contact_negative_weight", 0.25),
        contact_long_range_min_distance=getattr(args, "contact_long_range_min_distance", 24),
        contact_long_range_weight=getattr(args, "contact_long_range_weight", 1.0),
    )
    result = _train_with_backend(
        args,
        train_pilot=train_pilot,
        train_pilot_torch=train_pilot_torch,
        samples=train_samples,
        config=config,
        frozen=frozen,
    )
    if result is None:
        return 2
    if args.inference_mode != "legacy_direct":
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = _write_training_checkpoint_artifact(
            output_dir,
            config=config,
            result=result,
            metadata={
                "backend": args.backend,
                "dataset": "efold",
                "eval_tiers": sorted(eval_by_tier),
                "inference_protocol": "c0_corrected",
                "train_json": list(args.train_json),
            },
        )
        decoder_manifest_path = output_dir / "decoder_manifest.json"
        calibration_args = argparse.Namespace(
            checkpoint=str(checkpoint_path),
            validation_json=args.validation_json,
            frozen_dir=args.frozen_dir,
            frozen_cache_shards=args.frozen_cache_shards,
            output=str(decoder_manifest_path),
            seed=args.inference_seed,
            coarse_count=args.inference_coarse_count,
            validation_count=args.inference_validation_count,
            steps_grid=args.inference_steps_grid,
            samples_grid=args.inference_samples_grid,
            temperature_grid=args.inference_temperature_grid,
            threshold_grid=args.inference_threshold_grid,
            matching_policy=None,
        )
        calibration_rc = _cmd_calibrate_inference(calibration_args)
        if calibration_rc != 0:
            return calibration_rc
        eval_args = argparse.Namespace(
            checkpoint=str(checkpoint_path),
            decoder_manifest=str(decoder_manifest_path),
            eval_json=list(args.eval_json),
            frozen_dir=args.frozen_dir,
            frozen_cache_shards=args.frozen_cache_shards,
            output_dir=str(output_dir / "corrected_evaluation"),
            limit_per_tier=args.eval_limit,
            full_tier=[],
            mode=[args.inference_mode],
        )
        return _cmd_evaluate_checkpoint(eval_args)
    model = PairwiseDenoiser(result.parameters, min_loop=config.min_loop)
    adapter = FeatureAdapter(result.adapter_parameters) if result.adapter_parameters is not None else None
    operator = ReactivityForwardOperator()

    eval_profiler = _ProfileAppendLogger(args.profile_path or None)
    predictions = []
    reactivity_by_tier = {}
    sample_event_index = 0
    try:
        for tier, samples in eval_by_tier.items():
            tier_start = time.perf_counter()
            for sample in samples:
                sample_start = time.perf_counter()
                phase_start = time.perf_counter()
                features = _diagnostic_features(sample.sequence, adapter, frozen)
                eval_profiler.log(
                    "eval_features",
                    time.perf_counter() - phase_start,
                    sample_index=sample_event_index,
                    length=len(sample.sequence),
                    tier=tier,
                )
                phase_start = time.perf_counter()
                marginals = model.forward(sample.sequence, features).marginals
                soft = marginal_pair_matrix(marginals)
                eval_profiler.log(
                    "eval_model_forward",
                    time.perf_counter() - phase_start,
                    sample_index=sample_event_index,
                    length=len(sample.sequence),
                    tier=tier,
                )
                phase_start = time.perf_counter()
                projected = project_greedy_matching(
                    sample.sequence,
                    soft,
                    min_loop=config.min_loop,
                    allow_wobble=model.allow_wobble,
                    allow_pseudoknot=True,
                    min_score=1e-6,
                )
                predictions.append(StructurePrediction(predicted=projected, target=sample.pair_matrix, tier=tier))
                eval_profiler.log(
                    "eval_projection",
                    time.perf_counter() - phase_start,
                    sample_index=sample_event_index,
                    length=len(sample.sequence),
                    tier=tier,
                )
                eval_profiler.log(
                    "eval_sample_total",
                    time.perf_counter() - sample_start,
                    sample_index=sample_event_index,
                    length=len(sample.sequence),
                    tier=tier,
                    flush=True,
                )
                sample_event_index += 1
            react_start = time.perf_counter()
            head = samples[0]
            head_features = _diagnostic_features(head.sequence, adapter, frozen)
            head_marginals = model.forward(head.sequence, head_features).marginals
            a_values, c_values = _reactivity_coefficients(operator, head.sequence, head.probe)
            predicted_react = _predicted_reactivity(head_marginals, a_values, c_values)
            reactivity_by_tier[tier] = reactivity_metrics(predicted_react, head.reactivity, head.weights)
            eval_profiler.log(
                "eval_reactivity_head",
                time.perf_counter() - react_start,
                sample_index=sample_event_index,
                length=len(head.sequence),
                tier=tier,
                flush=True,
            )
            eval_profiler.log(
                "eval_tier_total",
                time.perf_counter() - tier_start,
                sample_index=len(samples),
                tier=tier,
                flush=True,
            )

        metric_start = time.perf_counter()
        tier_metrics = structure_metrics_by_tier(predictions)
        distance_metrics = structure_distance_bin_metrics_by_tier(predictions)
        eval_profiler.log("eval_metric_aggregation", time.perf_counter() - metric_start, flush=True)
        cited = {
            "viral_mRNA": (0.73, "10.1126/sciadv.adz4967"),
            "lncRNA": (0.44, "10.1126/sciadv.adz4967"),
        }
        local = {f"local_{tier}": metrics.mean_f1 for tier, metrics in tier_metrics.items()}
        markdown = render_comparison_markdown(build_comparison_table(cited, local))
        output_dir = Path(args.output_dir)
        artifact_start = time.perf_counter()
        output_dir.mkdir(parents=True, exist_ok=True)
        table_path = output_dir / "comparison_table.md"
        table_path.write_text(markdown, encoding="utf-8")
        checkpoint_path = _write_training_checkpoint_artifact(
            output_dir,
            config=config,
            result=result,
            metadata={
                "backend": args.backend,
                "dataset": "efold",
                "eval_tiers": sorted(eval_by_tier),
                "family_balanced_batches": config.family_balanced_batches,
                "mode": "warm_start" if adapter is not None else "base",
                "train_json": list(args.train_json),
                "train_samples": len(train_samples),
                "window_size": args.window_size,
                "window_stride": args.window_stride,
            },
        )
        eval_profiler.log("eval_artifact_write_total", time.perf_counter() - artifact_start, flush=True)
    finally:
        eval_profiler.close()

    summary = {
        "backend": args.backend,
        "dataset": "efold",
        "mode": "warm_start" if adapter is not None else "base",
        "epochs": config.epochs,
        "train_samples": len(train_samples),
        "eval_samples": {tier: len(samples) for tier, samples in eval_by_tier.items()},
        "family_balanced_batches": config.family_balanced_batches,
        "contact_long_range_min_distance": config.contact_long_range_min_distance,
        "contact_long_range_weight": config.contact_long_range_weight,
        "length_buckets": _length_bucket_counts(train_samples, args.bucket_boundaries),
        "window_size": args.window_size,
        "window_stride": args.window_stride,
        "tiers": {
            tier: {
                "count": metrics.count,
                "mean_f1": round(metrics.mean_f1, 4),
                "mean_mcc": round(metrics.mean_mcc, 4),
                "micro_f1": round(metrics.micro_f1, 4),
                "micro_mcc": round(metrics.micro_mcc, 4),
            }
            for tier, metrics in tier_metrics.items()
        },
        "distance_bins": {
            tier: {
                label: {
                    "count": metrics.count,
                    "mean_f1": round(metrics.mean_f1, 4),
                    "mean_mcc": round(metrics.mean_mcc, 4),
                    "micro_f1": round(metrics.micro_f1, 4),
                    "micro_mcc": round(metrics.micro_mcc, 4),
                }
                for label, metrics in bins.items()
            }
            for tier, bins in distance_metrics.items()
        },
        "reactivity": {
            tier: {
                "count": metrics.count,
                "pearson": round(metrics.pearson, 4),
                "spearman": round(metrics.spearman, 4),
                "calibrated_mae": round(metrics.calibrated_mae, 4),
            }
            for tier, metrics in reactivity_by_tier.items()
        },
        "profile": result.profile_summary,
        "comparison_markdown": markdown,
        "artifacts": [str(table_path), str(checkpoint_path)],
        "note": (
            "Local rows are same-protocol recomputes from this eFold JSON run. "
            "Cited eFold rows remain separate and are never merged with local numbers."
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


def _parse_number_grid(raw: str, cast) -> tuple:
    values = tuple(cast(part.strip()) for part in str(raw).split(",") if part.strip())
    if not values:
        raise ValueError("grid must contain at least one value")
    return values


def _load_c0_samples(path: str, limit: Optional[int] = None) -> tuple:
    import heapq

    from reactflow.protocol import stable_sample_key
    from reactflow.train import iter_sample_cache, load_efold_samples

    source = Path(path)
    if source.suffix.lower() != ".jsonl":
        samples = load_efold_samples([source], limit=None, min_length=1, max_length=None)
        return tuple(
            sorted(
                samples,
                key=lambda sample: stable_sample_key(sample.source_id, sample.sequence),
            )[:limit]
            if limit is not None
            else samples
        )
    if limit is None:
        return tuple(iter_sample_cache(source))
    if limit < 0:
        raise ValueError("sample limit must be non-negative")
    if limit == 0:
        return tuple()
    heap = []
    for counter, sample in enumerate(iter_sample_cache(source)):
        key = int(stable_sample_key(sample.source_id, sample.sequence), 16)
        entry = (-key, counter, sample)
        if len(heap) < limit:
            heapq.heappush(heap, entry)
        elif key < -heap[0][0]:
            heapq.heapreplace(heap, entry)
    selected = [(-negative_key, counter, sample) for negative_key, counter, sample in heap]
    selected.sort(key=lambda item: (item[0], item[1]))
    return tuple(item[2] for item in selected)


def _cmd_calibrate_inference(args: argparse.Namespace) -> int:
    """Lock CTMC and decoder settings on a deterministic validation subset."""

    from reactflow.c0_evaluate import (
        aggregate_structure_records,
        code_sha256,
        frozen_feature_provenance,
        sha256_path,
        structure_record_metrics,
    )
    from reactflow.checkpoint import read_training_checkpoint
    from reactflow.constraints import matrix_to_pairs, validate_pair_matrix
    from reactflow.features import load_frozen_features
    from reactflow.inference import (
        DecoderConfig,
        InferenceConfig,
        InferenceMode,
        MatchingPolicy,
        decode_calibrated_marginal,
        predict_structure,
    )
    from reactflow.probing import ProfilePrediction, calibration_manifest, fit_probe_calibration
    from reactflow.protocol import stable_subset
    from reactflow.reactivity import ReactivityForwardOperator

    checkpoint_path = Path(args.checkpoint)
    checkpoint = read_training_checkpoint(checkpoint_path)
    frozen = load_frozen_features(
        Path(args.frozen_dir),
        max_loaded_shards=args.frozen_cache_shards,
    ) if args.frozen_dir else None
    all_samples = _load_c0_samples(args.validation_json, limit=args.validation_count)
    if not all_samples:
        raise ValueError("validation set is empty")
    final_samples = stable_subset(
        all_samples,
        min(args.validation_count, len(all_samples)),
        source_id=lambda sample: sample.source_id,
        sequence=lambda sample: sample.sequence,
        seed=args.seed,
    )
    coarse_samples = final_samples[: min(args.coarse_count, len(final_samples))]
    steps_grid = _parse_number_grid(args.steps_grid, int)
    samples_grid = _parse_number_grid(args.samples_grid, int)
    temperatures = _parse_number_grid(args.temperature_grid, float)
    thresholds = _parse_number_grid(args.threshold_grid, float)
    policies = tuple(
        MatchingPolicy(value)
        for value in (args.matching_policy or ["nested_dp", "pseudoknot_allowed_greedy"])
    )

    coarse_rows = []
    coarse_cache = {}
    for steps in steps_grid:
        for sample_count in samples_grid:
            records = []
            cached = []
            for sample in coarse_samples:
                result = predict_structure(
                    checkpoint,
                    sample.sequence,
                    frozen,
                    InferenceConfig(
                        mode=InferenceMode.CTMC_SAMPLE,
                        seed=args.seed,
                        num_steps=steps,
                        num_samples=sample_count,
                    ),
                    DecoderConfig(matching_policy=MatchingPolicy.NESTED_DP),
                )
                decoded = decode_calibrated_marginal(
                    sample.sequence,
                    result.pair_frequency,
                    result.unpaired_probability,
                    DecoderConfig(matching_policy=MatchingPolicy.NESTED_DP),
                )
                records.append(
                    {
                        "metrics": structure_record_metrics(decoded, sample.pair_matrix),
                        "legal": result.validation.valid,
                        "runtime_seconds": result.runtime_seconds,
                    }
                )
                cached.append((sample, result.pair_frequency, result.unpaired_probability, result.runtime_seconds))
            summary = aggregate_structure_records(records)
            row = {"num_steps": steps, "num_samples": sample_count, **summary}
            coarse_rows.append(row)
            coarse_cache[(steps, sample_count)] = cached
    selected_coarse = max(
        coarse_rows,
        key=lambda row: (
            row.get("mean_exact_f1", 0.0),
            row.get("mean_shifted_f1", 0.0),
            row.get("legality_rate", 0.0),
            -(row["num_steps"] * row["num_samples"]),
            -row.get("runtime_seconds_total", 0.0),
        ),
    )
    selected_key = (selected_coarse["num_steps"], selected_coarse["num_samples"])
    if len(final_samples) == len(coarse_samples):
        marginal_cache = coarse_cache[selected_key]
    else:
        marginal_cache = []
        for sample in final_samples:
            result = predict_structure(
                checkpoint,
                sample.sequence,
                frozen,
                InferenceConfig(
                    mode=InferenceMode.CTMC_SAMPLE,
                    seed=args.seed,
                    num_steps=selected_key[0],
                    num_samples=selected_key[1],
                ),
                DecoderConfig(matching_policy=MatchingPolicy.NESTED_DP),
            )
            marginal_cache.append((sample, result.pair_frequency, result.unpaired_probability, result.runtime_seconds))

    decoder_rows = []
    for temperature in temperatures:
        for threshold in thresholds:
            for policy in policies:
                decoder = DecoderConfig(
                    temperature=temperature,
                    threshold=threshold,
                    matching_policy=policy,
                )
                records = []
                for sample, pair_frequency, unpaired, runtime in marginal_cache:
                    decoded = decode_calibrated_marginal(sample.sequence, pair_frequency, unpaired, decoder)
                    records.append(
                        {
                            "metrics": structure_record_metrics(decoded, sample.pair_matrix),
                            "legal": True,
                            "runtime_seconds": runtime,
                        }
                    )
                decoder_rows.append(
                    {
                        "temperature": temperature,
                        "threshold": threshold,
                        "matching_policy": policy.value,
                        **aggregate_structure_records(records),
                    }
                )
    selected_decoder = max(
        decoder_rows,
        key=lambda row: (
            row.get("mean_exact_f1", 0.0),
            row.get("mean_shifted_f1", 0.0),
            -abs(math.log(max(float(row.get("pair_count_ratio") or 1e-12), 1e-12))),
            row["matching_policy"] == MatchingPolicy.NESTED_DP.value,
        ),
    )

    operator = ReactivityForwardOperator()
    profiles = []
    for sample, _pair_frequency, unpaired, _runtime in marginal_cache:
        predicted = operator.from_expectations(sample.sequence, unpaired, None, sample.probe)
        profiles.append(
            ProfilePrediction(
                source_id=sample.source_id or "",
                probe=sample.probe,
                predicted=predicted,
                target=sample.reactivity,
                weights=sample.weights,
                reactivity_source=sample.reactivity_source,
                length=len(sample.sequence),
                snr=sample.reactivity_snr,
                quality=sample.reactivity_quality,
            )
        )
    probe_calibration = fit_probe_calibration(profiles, split="validation")
    payload = {
        "schema_version": 1,
        "fitted_split": "validation",
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_path(checkpoint_path),
        "code_sha256": code_sha256(),
        "frozen_features": frozen_feature_provenance(Path(args.frozen_dir) if args.frozen_dir else None),
        "validation_path": str(Path(args.validation_json).resolve()),
        "validation_sha256": sha256_path(Path(args.validation_json)),
        "seed": args.seed,
        "validation_sample_ids": [sample.source_id for sample in final_samples],
        "coarse_sample_count": len(coarse_samples),
        "validation_sample_count": len(final_samples),
        "candidate_grid": {
            "num_steps": list(steps_grid),
            "num_samples": list(samples_grid),
            "temperature": list(temperatures),
            "threshold": list(thresholds),
            "matching_policy": [policy.value for policy in policies],
        },
        "selection_protocol": {
            "checkpoint": ["validation_exact_f1", "validation_shifted_f1", "parameter_count"],
            "ctmc": [
                "validation_exact_f1",
                "validation_shifted_f1",
                "legality_rate",
                "fewer_model_calls",
                "runtime_seconds",
            ],
            "decoder": [
                "validation_exact_f1",
                "validation_shifted_f1",
                "pair_count_ratio_closeness",
                "nested_dp",
            ],
        },
        "selected_inference": {
            "num_steps": selected_key[0],
            "num_samples": selected_key[1],
            "selection_metrics": selected_coarse,
        },
        "selected_decoder": {
            key: selected_decoder[key]
            for key in ("temperature", "threshold", "matching_policy")
        },
        "coarse_results": coarse_rows,
        "decoder_results": decoder_rows,
        "probing_calibration": calibration_manifest(probe_calibration),
        "test_override_allowed": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "selected_inference": payload["selected_inference"], "selected_decoder": payload["selected_decoder"]}, sort_keys=True))
    return 0


def _cmd_evaluate_checkpoint(args: argparse.Namespace) -> int:
    """Evaluate a fixed checkpoint using a validation-locked decoder manifest."""

    from reactflow.c0_evaluate import (
        aggregate_structure_records,
        read_decoder_manifest,
        sha256_path,
        structure_record_metrics,
        verify_frozen_feature_provenance,
    )
    from reactflow.checkpoint import read_training_checkpoint
    from reactflow.constraints import matrix_to_pairs, validate_pair_matrix
    from reactflow.features import load_frozen_features
    from reactflow.inference import (
        DecoderConfig,
        InferenceConfig,
        InferenceMode,
        MatchingPolicy,
        decode_calibrated_marginal,
        predict_structure,
    )
    from reactflow.probing import ProbeCalibration, ProfilePrediction, aggregate_full_profiles
    from reactflow.protocol import normalize_tier_label
    from reactflow.reactivity import ReactivityForwardOperator

    checkpoint_path = Path(args.checkpoint)
    checkpoint = read_training_checkpoint(checkpoint_path)
    manifest = read_decoder_manifest(Path(args.decoder_manifest), checkpoint_path=checkpoint_path)
    frozen_provenance = verify_frozen_feature_provenance(
        manifest.get("frozen_features", {}),
        Path(args.frozen_dir) if args.frozen_dir else None,
    )
    selected_inference = manifest["selected_inference"]
    selected_decoder = manifest["selected_decoder"]
    decoder = DecoderConfig(
        temperature=float(selected_decoder["temperature"]),
        threshold=float(selected_decoder["threshold"]),
        matching_policy=MatchingPolicy(selected_decoder["matching_policy"]),
    )
    frozen = load_frozen_features(
        Path(args.frozen_dir),
        max_loaded_shards=args.frozen_cache_shards,
    ) if args.frozen_dir else None
    requested_modes = {InferenceMode(value) for value in (args.mode or [mode.value for mode in InferenceMode])}
    modes = tuple(mode for mode in InferenceMode if mode in requested_modes)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "predictions.jsonl"
    records_by_key = {}
    profiles = []
    operator = ReactivityForwardOperator()
    sample_ids = {}
    data_provenance = {}
    with prediction_path.open("w", encoding="utf-8") as handle:
        for spec in args.eval_json:
            if "=" not in spec:
                raise ValueError("--eval-json must be tier=path")
            raw_tier, raw_path = spec.split("=", 1)
            tier = normalize_tier_label(raw_tier)
            data_provenance[tier] = {
                "path": str(Path(raw_path).resolve()),
                "sha256": sha256_path(Path(raw_path)),
            }
            tier_limit = None if tier in set(args.full_tier) else args.limit_per_tier
            samples = _load_c0_samples(raw_path, limit=tier_limit)
            sample_ids[tier] = [sample.source_id for sample in samples]
            for sample in samples:
                ctmc_result = None
                for mode in modes:
                    key = f"{tier}:{mode.value}"
                    rows = records_by_key.setdefault(key, [])
                    result = ctmc_result if mode is InferenceMode.CALIBRATED_MARGINAL else None
                    if result is not None:
                        decode_started = time.perf_counter()
                        structure = decode_calibrated_marginal(
                            sample.sequence,
                            result.pair_frequency,
                            result.unpaired_probability,
                            decoder,
                        )
                        decode_seconds = time.perf_counter() - decode_started
                        validation = validate_pair_matrix(
                            sample.sequence,
                            structure,
                            min_loop=decoder.min_loop,
                            allow_wobble=True,
                            allow_pseudoknot=(
                                decoder.matching_policy is MatchingPolicy.PSEUDOKNOT_ALLOWED_GREEDY
                            ),
                        )
                        runtime_seconds = result.runtime_seconds + decode_seconds
                        provenance = {
                            **result.provenance,
                            "mode": InferenceMode.CALIBRATED_MARGINAL.value,
                            "shared_ctmc_cache": True,
                            "decoder_seconds": decode_seconds,
                        }
                        unpaired_probability = result.unpaired_probability
                    else:
                        result = predict_structure(
                            checkpoint,
                            sample.sequence,
                            frozen,
                            InferenceConfig(
                                mode=mode,
                                seed=int(manifest["seed"]),
                                num_steps=int(selected_inference["num_steps"]),
                                num_samples=int(selected_inference["num_samples"]),
                            ),
                            decoder,
                        )
                        structure = result.structure
                        validation = result.validation
                        runtime_seconds = result.runtime_seconds
                        provenance = result.provenance
                        unpaired_probability = result.unpaired_probability
                        if mode is InferenceMode.CTMC_SAMPLE:
                            ctmc_result = result
                    metrics = structure_record_metrics(structure, sample.pair_matrix)
                    row = {
                        "tier": tier,
                        "mode": mode.value,
                        "source_id": sample.source_id,
                        "sequence_length": len(sample.sequence),
                        "length_bucket": _c0_length_bucket(len(sample.sequence)),
                        "predicted_pairs": [list(pair) for pair in matrix_to_pairs(structure)],
                        "target_pairs": [list(pair) for pair in matrix_to_pairs(sample.pair_matrix)],
                        "metrics": metrics,
                        "legal": validation.valid,
                        "runtime_seconds": runtime_seconds,
                        "provenance": provenance,
                    }
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                    rows.append(row)
                    if mode is InferenceMode.CALIBRATED_MARGINAL:
                        predicted_reactivity = operator.from_expectations(
                            sample.sequence,
                            unpaired_probability,
                            None,
                            sample.probe,
                        )
                        profiles.append(
                            ProfilePrediction(
                                source_id=sample.source_id or "",
                                probe=sample.probe,
                                predicted=predicted_reactivity,
                                target=sample.reactivity,
                                weights=sample.weights,
                                reactivity_source=sample.reactivity_source,
                                length=len(sample.sequence),
                                snr=sample.reactivity_snr,
                                quality=sample.reactivity_quality,
                            )
                        )
    summary = {
        "schema_version": 1,
        "checkpoint": str(checkpoint_path.resolve()),
        "decoder_manifest": str(Path(args.decoder_manifest).resolve()),
        "frozen_features": frozen_provenance,
        "data_provenance": data_provenance,
        "sample_ids": sample_ids,
        "results": {key: aggregate_structure_records(rows) for key, rows in sorted(records_by_key.items())},
        "length_stratified": {
            key: {
                bucket: aggregate_structure_records(
                    [row for row in rows if row["length_bucket"] == bucket]
                )
                for bucket in sorted({row["length_bucket"] for row in rows})
            }
            for key, rows in sorted(records_by_key.items())
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    calibration_rows = manifest.get("probing_calibration", {}).get("probes", {})
    calibrations = {
        probe: ProbeCalibration(**row)
        for probe, row in calibration_rows.items()
    }
    probing = aggregate_full_profiles(profiles, calibrations)
    (output_dir / "probing_metrics.json").write_text(json.dumps(probing, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": str(output_dir / "metrics.json"), "predictions": str(prediction_path), "probing": str(output_dir / "probing_metrics.json")}, sort_keys=True))
    return 0


def _cmd_preflight_checkpoint(args: argparse.Namespace) -> int:
    """Project fixed-matrix runtime from a small, deterministic CTMC sample."""

    from reactflow.c0_evaluate import read_decoder_manifest, verify_frozen_feature_provenance
    from reactflow.checkpoint import read_training_checkpoint
    from reactflow.features import load_frozen_features
    from reactflow.inference import DecoderConfig, InferenceConfig, InferenceMode, MatchingPolicy, predict_structure
    from reactflow.protocol import normalize_tier_label

    checkpoint_path = Path(args.checkpoint)
    checkpoint = read_training_checkpoint(checkpoint_path)
    manifest = read_decoder_manifest(Path(args.decoder_manifest), checkpoint_path=checkpoint_path)
    frozen_provenance = verify_frozen_feature_provenance(
        manifest.get("frozen_features", {}),
        Path(args.frozen_dir) if args.frozen_dir else None,
    )
    inference = manifest["selected_inference"]
    decoder_row = manifest["selected_decoder"]
    decoder = DecoderConfig(
        temperature=float(decoder_row["temperature"]),
        threshold=float(decoder_row["threshold"]),
        matching_policy=MatchingPolicy(decoder_row["matching_policy"]),
    )
    frozen = load_frozen_features(
        Path(args.frozen_dir), max_loaded_shards=args.frozen_cache_shards
    ) if args.frozen_dir else None
    total_matrix_samples = 0
    preflight_samples = []
    tier_counts = {}
    for spec in args.eval_json:
        if "=" not in spec:
            raise ValueError("--eval-json must be tier=path")
        raw_tier, raw_path = spec.split("=", 1)
        tier = normalize_tier_label(raw_tier)
        source_path = Path(raw_path)
        if source_path.suffix.lower() == ".jsonl":
            with source_path.open(encoding="utf-8") as handle:
                available = sum(1 for line in handle if line.strip())
        else:
            available = len(_load_c0_samples(raw_path, limit=None))
        selected_count = available if tier in set(args.full_tier) else min(available, args.limit_per_tier)
        tier_counts[tier] = selected_count
        total_matrix_samples += selected_count
        preflight_samples.extend(
            _load_c0_samples(raw_path, limit=min(args.preflight_per_tier, selected_count))
        )
    runtimes = []
    for sample in preflight_samples:
        result = predict_structure(
            checkpoint,
            sample.sequence,
            frozen,
            InferenceConfig(
                mode=InferenceMode.CALIBRATED_MARGINAL,
                seed=int(manifest["seed"]),
                num_steps=int(inference["num_steps"]),
                num_samples=int(inference["num_samples"]),
            ),
            decoder,
        )
        runtimes.append(result.runtime_seconds)
    mean_seconds = sum(runtimes) / len(runtimes) if runtimes else 0.0
    projected_hours = mean_seconds * total_matrix_samples / 3600.0
    payload = {
        "schema_version": 1,
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "decoder_manifest": str(Path(args.decoder_manifest).resolve()),
        "frozen_features": frozen_provenance,
        "tier_sample_counts": tier_counts,
        "preflight_sample_count": len(runtimes),
        "mean_ctmc_seconds_per_sample": mean_seconds,
        "projected_unique_ctmc_hours": projected_hours,
        "shared_ctmc_between_modes": True,
        "max_projected_hours": args.max_projected_hours,
        "within_runtime_gate": projected_hours <= args.max_projected_hours,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["within_runtime_gate"] else 3


def _c0_length_bucket(length: int) -> str:
    if length <= 64:
        return "len_le_64"
    if length <= 128:
        return "len_65_128"
    if length <= 256:
        return "len_129_256"
    return "len_gt_256"


def _cmd_sample(args: argparse.Namespace) -> int:
    """Draw a legal 2D structure ensemble from the CTMC sampler.

    Trains the deterministic pilot to obtain a denoiser, draws ``--num-samples``
    structures for ``--sequence`` (or the first pilot sequence when omitted),
    reports the empirical legality rate (guaranteed ``1.0`` by projection) and the
    ensemble pairing-frequency matrix, and writes that matrix as an SVG heatmap.

    Complexity: O(epochs*N*L^2 H^2) training plus O(num_samples*(L^3+L^2 H^2)).
    """

    from reactflow.model import PairwiseDenoiser
    from reactflow.sampling import ensemble_unpaired_probability, pairing_frequency_matrix, sample_structures
    from reactflow.synthetic import make_dataset
    from reactflow.train import TrainConfig, train_pilot

    config = TrainConfig(epochs=args.epochs, seed=args.seed)
    samples = make_dataset(count=args.samples, stem=args.stem, loop=args.loop, probe=args.probe, seed=1)
    result = train_pilot(samples=samples, config=config)
    model = PairwiseDenoiser(result.parameters, min_loop=config.min_loop)

    sequence = args.sequence if args.sequence else samples[0].sequence
    structures = sample_structures(
        model,
        sequence,
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        seed=args.seed,
        allow_pseudoknot=not args.no_pseudoknot,
    )
    legal = sum(1 for structure in structures if structure.validation.valid)
    frequency = pairing_frequency_matrix(structures)
    unpaired = ensemble_unpaired_probability(structures)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    heatmap_path = output_dir / "ensemble_pairing_frequency.svg"
    write_pair_heatmap_svg(frequency, heatmap_path, title=f"Ensemble pairing frequency ({sequence})")

    summary = {
        "sequence": sequence,
        "num_samples": len(structures),
        "num_steps": args.num_steps,
        "legal_count": legal,
        "legality_rate": legal / len(structures),
        "ensemble_unpaired_probability": [round(value, 4) for value in unpaired],
        "artifacts": [str(heatmap_path)],
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if legal == len(structures) else 1


def _cmd_guidance_scan(args: argparse.Namespace) -> int:
    """Sweep inference-time energy guidance ``eta`` and write the tradeoff curve.

    Trains the deterministic pilot, forms the denoiser soft pairing matrix for the
    target sequence, and sweeps ``eta`` over ``energy_guided_scores`` + exact
    max-weight nested projection.  Reports the per-eta legality, pair energy,
    structure energy, pair count and (when a reference dot-bracket is given) F1,
    asserts the monotonicity/legality acceptance criteria, and writes an SVG scan
    curve.  Returns non-zero if the scan is illegal anywhere or non-monotone.

    Complexity: O(epochs*N*L^2 H^2) training plus O(E*L^3) for the scan.
    """

    from reactflow.constraints import dotbracket_to_matrix as _dot
    from reactflow.model import PairwiseDenoiser, marginal_pair_matrix
    from reactflow.synthetic import make_dataset
    from reactflow.thermo import guidance_eta_scan, guidance_scan_is_monotone
    from reactflow.train import TrainConfig, build_features, train_pilot

    config = TrainConfig(epochs=args.epochs, seed=args.seed)
    samples = make_dataset(count=args.samples, stem=args.stem, loop=args.loop, probe=args.probe, seed=1)
    result = train_pilot(samples=samples, config=config)
    model = PairwiseDenoiser(result.parameters, min_loop=config.min_loop)

    if args.sequence:
        sequence = args.sequence
        reference = _dot(args.reference) if args.reference else None
    else:
        sample = samples[0]
        sequence = sample.sequence
        reference = args.reference and _dot(args.reference) or sample.pair_matrix

    features = build_features(sequence, 1.0, [0 for _ in sequence])
    soft = marginal_pair_matrix(model.forward(sequence, features).marginals)
    etas = [float(value) for value in args.etas.split(",")]
    points = guidance_eta_scan(sequence, soft, etas, reference=reference, exact=not args.greedy)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    series = {
        "pair_energy": [point.pair_energy for point in points],
        "pair_count": [float(point.pair_count) for point in points],
    }
    if reference is not None:
        series["f1"] = [point.f1_to_reference for point in points]
    scan_path = output_dir / "guidance_eta_scan.svg"
    write_guidance_scan_svg(etas, series, scan_path, title=f"Guidance eta scan ({sequence})")

    monotone = guidance_scan_is_monotone(points)
    summary = {
        "sequence": sequence,
        "exact_projection": not args.greedy,
        "points": [
            {
                "eta": point.eta,
                "legal": point.legal,
                "pair_count": point.pair_count,
                "pair_energy": round(point.pair_energy, 4),
                "structure_energy": round(point.structure_energy, 4),
                "crossing_count": point.crossing_count,
                "f1_to_reference": None if point.f1_to_reference is None else round(point.f1_to_reference, 4),
            }
            for point in points
        ],
        "legal_throughout": all(point.legal for point in points),
        "pair_energy_monotone_non_increasing": monotone,
        "artifacts": [str(scan_path)],
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if monotone else 1


def build_parser() -> argparse.ArgumentParser:
    """Build the ReactFlow CLI parser.

    Complexity: O(1).
    """

    parser = argparse.ArgumentParser(prog="reactflow")
    sub = parser.add_subparsers(required=True)

    validate = sub.add_parser("validate-csv", help="validate Ribonanza-style CSV profiles")
    validate.add_argument("path")
    validate.add_argument("--limit", type=int, default=5)
    validate.add_argument("--normalization", choices=["p90", "zscore", "minmax"], default="p90")
    validate.set_defaults(func=_cmd_validate_csv)

    symbolic = sub.add_parser("verify-symbolic", help="run SymPy derivation checks")
    symbolic.set_defaults(func=_cmd_verify_symbolic)

    heatmap = sub.add_parser("plot-dotbracket", help="write SVG heatmap from dot-bracket")
    heatmap.add_argument("dotbracket")
    heatmap.add_argument("output")
    heatmap.add_argument("--title", default="Pair score heatmap")
    heatmap.set_defaults(func=_cmd_plot_dotbracket)

    profiles = sub.add_parser("plot-profiles", help="write SVG profile overlay")
    profiles.add_argument("--predicted", required=True)
    profiles.add_argument("--target", required=True)
    profiles.add_argument("--output", required=True)
    profiles.add_argument("--title", default="Reactivity profile overlay")
    profiles.set_defaults(func=_cmd_plot_profiles)

    train = sub.add_parser("train", help="run the synthetic training pilot (base or C5 warm-start) and write SVG diagnostics")
    train.add_argument("--output-dir", default="artifacts/train")
    train.add_argument("--epochs", type=int, default=40)
    train.add_argument("--hidden-size", type=int, default=8)
    train.add_argument("--learning-rate", type=float, default=0.2)
    train.add_argument("--lambda-react", type=float, default=1.0)
    train.add_argument("--lambda-thermo", type=float, default=0.0, help="weight on the Turner thermodynamic semi-supervision term")
    train.add_argument("--lambda-calib", type=float, default=0.0, help="weight on the variance-aware ensemble-calibration NLL term (0 disables)")
    train.add_argument("--calib-beta", type=float, default=1.0, help="structural-variance scale beta for the ensemble-calibration term")
    train.add_argument("--calib-tau-squared", type=float, default=0.05, help="measurement-noise variance floor tau^2 for the ensemble-calibration term")
    train.add_argument("--lambda-contact", type=float, default=0.0, help="weight on the legal contact-map denoising auxiliary (0 disables)")
    train.add_argument("--contact-negative-weight", type=float, default=0.25, help="scale for legal non-pair BCE in the contact auxiliary")
    train.add_argument("--contact-long-range-min-distance", type=int, default=24, help="RF-CF2 span threshold for long-range contact weighting")
    train.add_argument("--contact-long-range-weight", type=float, default=1.0, help="RF-CF2 weight for legal contact candidates at or beyond the long-range threshold")
    train.add_argument("--thermo-mode", choices=["mse", "kl"], default="mse")
    train.add_argument("--adapter-dim", type=int, default=0, help="C5 warm-start adapter output dim (0 disables warm-start)")
    train.add_argument("--adapter-lr", type=float, default=None, help="adapter SGD step (defaults to --learning-rate)")
    train.add_argument("--frozen-dir", default="", help="directory of an exported frozen-feature shard (required when --adapter-dim > 0)")
    train.add_argument("--frozen-cache-shards", type=int, default=4, help="LRU child-shard cache size for sharded frozen features")
    train.add_argument("--profile-path", default="", help="write detailed per-phase training timings as JSONL")
    train.add_argument("--batch-size", type=int, default=None, help="samples per gradient update (default: full batch)")
    train.add_argument("--bucket-boundaries", type=_parse_bucket_boundaries, default=tuple(), help="comma-separated length buckets, e.g. 64,128,256")
    train.add_argument("--family-balanced-batches", action="store_true", help="interleave training samples across family/cluster labels within each length bucket")
    train.add_argument("--backend", choices=["stdlib", "torch"], default="stdlib", help="training backend")
    train.add_argument("--torch-device", default="cpu", help="torch device when --backend torch")
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--samples", type=int, default=6)
    train.add_argument("--stem", type=int, default=4)
    train.add_argument("--loop", type=int, default=4)
    train.add_argument("--probe", choices=["2A3", "DMS"], default="2A3")
    train.set_defaults(func=_cmd_train)

    evaluate = sub.add_parser("evaluate", help="run the C5.4 tiered evaluation protocol (F1/MCC + generalization gap + honest cited-vs-local table)")
    evaluate.add_argument("--output-dir", default="artifacts/evaluate")
    evaluate.add_argument("--epochs", type=int, default=40)
    evaluate.add_argument("--adapter-dim", type=int, default=0, help="C5 warm-start adapter output dim (0 disables warm-start)")
    evaluate.add_argument("--adapter-lr", type=float, default=None, help="adapter SGD step (defaults to the denoiser learning rate)")
    evaluate.add_argument("--frozen-dir", default="", help="directory of an exported frozen-feature shard (required when --adapter-dim > 0)")
    evaluate.add_argument("--frozen-cache-shards", type=int, default=4, help="LRU child-shard cache size for sharded frozen features")
    evaluate.add_argument("--lambda-thermo", type=float, default=0.0, help="weight on the Turner thermodynamic semi-supervision term")
    evaluate.add_argument("--lambda-calib", type=float, default=0.0, help="weight on the variance-aware ensemble-calibration NLL term (0 disables)")
    evaluate.add_argument("--calib-beta", type=float, default=1.0, help="structural-variance scale beta for the ensemble-calibration term")
    evaluate.add_argument("--calib-tau-squared", type=float, default=0.05, help="measurement-noise variance floor tau^2 for the ensemble-calibration term")
    evaluate.add_argument("--lambda-contact", type=float, default=0.0, help="weight on the legal contact-map denoising auxiliary (0 disables)")
    evaluate.add_argument("--contact-negative-weight", type=float, default=0.25, help="scale for legal non-pair BCE in the contact auxiliary")
    evaluate.add_argument("--contact-long-range-min-distance", type=int, default=24, help="RF-CF2 span threshold for long-range contact weighting")
    evaluate.add_argument("--contact-long-range-weight", type=float, default=1.0, help="RF-CF2 weight for legal contact candidates at or beyond the long-range threshold")
    evaluate.add_argument("--thermo-mode", choices=["mse", "kl"], default="mse")
    evaluate.add_argument("--profile-path", default="", help="write detailed per-phase training timings as JSONL")
    evaluate.add_argument("--batch-size", type=int, default=None, help="samples per gradient update (default: full batch)")
    evaluate.add_argument("--bucket-boundaries", type=_parse_bucket_boundaries, default=tuple(), help="comma-separated length buckets, e.g. 64,128,256")
    evaluate.add_argument("--family-balanced-batches", action="store_true", help="interleave training samples across family/cluster labels within each length bucket")
    evaluate.add_argument("--backend", choices=["stdlib", "torch"], default="stdlib", help="training backend")
    evaluate.add_argument("--torch-device", default="cpu", help="torch device when --backend torch")
    evaluate.add_argument("--seed", type=int, default=0)
    evaluate.add_argument("--samples", type=int, default=6, help="samples per generalization tier")
    evaluate.add_argument("--stem", type=int, default=4)
    evaluate.add_argument("--loop", type=int, default=4)
    evaluate.add_argument("--probe", choices=["2A3", "DMS"], default="2A3")
    evaluate.set_defaults(func=_cmd_evaluate)

    cache_efold = sub.add_parser("prepare-efold-cache", help="filter eFold/RNAndria JSON into reusable JSONL sample cache")
    cache_efold.add_argument("json", nargs="+", help="one or more eFold/RNAndria JSON files")
    cache_efold.add_argument("--output", required=True, help="output JSONL cache")
    cache_efold.add_argument("--limit", type=int, default=None, help="accepted sample cap")
    cache_efold.add_argument("--scan-limit", type=int, default=None, help="source record scan cap")
    cache_efold.add_argument("--min-length", type=int, default=1)
    cache_efold.add_argument("--max-length", type=int, default=256)
    cache_efold.add_argument("--window-size", type=int, default=None, help="slice long records into local windows of this length")
    cache_efold.add_argument("--window-stride", type=int, default=None, help="stride for --window-size (defaults to non-overlapping windows)")
    cache_efold.add_argument("--bucket-boundaries", type=_parse_bucket_boundaries, default=tuple(), help="comma-separated length buckets stored in cache metadata")
    cache_efold.add_argument("--min-loop", type=int, default=3)
    cache_efold.add_argument("--probe", choices=["2A3", "DMS"], default="2A3")
    cache_efold.set_defaults(func=_cmd_prepare_efold_cache)

    split_efold = sub.add_parser("split-efold-cache", help="split eFold JSONL caches into clan-disjoint train/val/test/novel files")
    split_efold.add_argument("cache", nargs="+", help="one or more prepared eFold JSONL caches")
    split_efold.add_argument("--output-dir", required=True, help="directory for split JSONL files")
    split_efold.add_argument("--manifest", default="", help="optional manifest path (defaults to output-dir/split_manifest.json)")
    split_efold.add_argument("--metadata-tsv", default="", help="optional TSV: record_id, clan/family, cluster")
    split_efold.add_argument("--bucket-boundaries", type=_parse_bucket_boundaries, default=(64, 128, 256), help="comma-separated length buckets")
    split_efold.add_argument("--novel-clan-fraction", type=float, default=0.15)
    split_efold.add_argument("--train-fraction", type=float, default=0.8)
    split_efold.add_argument("--val-fraction", type=float, default=0.1)
    split_efold.add_argument("--test-fraction", type=float, default=0.1)
    split_efold.add_argument("--seed", type=int, default=0)
    split_efold.set_defaults(func=_cmd_split_efold_cache)

    train_efold = sub.add_parser("train-efold", help="train on real eFold/RNAndria JSON structure records")
    train_efold.add_argument("json", nargs="+", help="one or more eFold/RNAndria JSON files")
    train_efold.add_argument("--output-dir", default="artifacts/train_efold")
    train_efold.add_argument("--limit", type=int, default=8, help="accepted samples after filtering")
    train_efold.add_argument("--min-length", type=int, default=1)
    train_efold.add_argument("--max-length", type=int, default=256)
    train_efold.add_argument("--window-size", type=int, default=None, help="slice raw JSON records into local windows before training")
    train_efold.add_argument("--window-stride", type=int, default=None, help="stride for --window-size")
    train_efold.add_argument("--epochs", type=int, default=10)
    train_efold.add_argument("--hidden-size", type=int, default=8)
    train_efold.add_argument("--learning-rate", type=float, default=0.2)
    train_efold.add_argument("--lambda-react", type=float, default=0.0, help="default 0 avoids pseudo-reactivity supervision on structure-only records")
    train_efold.add_argument("--lambda-thermo", type=float, default=0.0)
    train_efold.add_argument("--lambda-calib", type=float, default=0.0, help="weight on the variance-aware ensemble-calibration NLL term (0 disables)")
    train_efold.add_argument("--calib-beta", type=float, default=1.0, help="structural-variance scale beta for the ensemble-calibration term")
    train_efold.add_argument("--calib-tau-squared", type=float, default=0.05, help="measurement-noise variance floor tau^2 for the ensemble-calibration term")
    train_efold.add_argument("--lambda-contact", type=float, default=0.0, help="weight on the legal contact-map denoising auxiliary (0 disables)")
    train_efold.add_argument("--contact-negative-weight", type=float, default=0.25, help="scale for legal non-pair BCE in the contact auxiliary")
    train_efold.add_argument("--contact-long-range-min-distance", type=int, default=24, help="RF-CF2 span threshold for long-range contact weighting")
    train_efold.add_argument("--contact-long-range-weight", type=float, default=1.0, help="RF-CF2 weight for legal contact candidates at or beyond the long-range threshold")
    train_efold.add_argument("--thermo-mode", choices=["mse", "kl"], default="mse")
    train_efold.add_argument("--adapter-dim", type=int, default=0)
    train_efold.add_argument("--adapter-lr", type=float, default=None)
    train_efold.add_argument("--frozen-dir", default="")
    train_efold.add_argument("--frozen-cache-shards", type=int, default=4)
    train_efold.add_argument("--profile-path", default="", help="write detailed per-phase training timings as JSONL")
    train_efold.add_argument("--batch-size", type=int, default=None, help="samples per gradient update (default: full batch)")
    train_efold.add_argument("--bucket-boundaries", type=_parse_bucket_boundaries, default=tuple(), help="comma-separated length buckets, e.g. 64,128,256")
    train_efold.add_argument("--family-balanced-batches", action="store_true", help="interleave training samples across family/cluster labels within each length bucket")
    train_efold.add_argument("--backend", choices=["stdlib", "torch"], default="stdlib", help="training backend")
    train_efold.add_argument("--torch-device", default="cpu", help="torch device when --backend torch")
    train_efold.add_argument("--seed", type=int, default=0)
    train_efold.add_argument("--min-loop", type=int, default=3)
    train_efold.add_argument("--probe", choices=["2A3", "DMS"], default="2A3")
    train_efold.set_defaults(func=_cmd_train_efold)

    eval_efold = sub.add_parser("evaluate-efold", help="train on eFold JSON and score named eFold/RNAndria tiers")
    eval_efold.add_argument("--train-json", nargs="+", required=True, help="training eFold/RNAndria JSON files")
    eval_efold.add_argument("--eval-json", action="append", default=[], help="named eval tier in the form tier=path; repeatable")
    eval_efold.add_argument("--output-dir", default="artifacts/evaluate_efold")
    eval_efold.add_argument("--train-limit", type=int, default=8)
    eval_efold.add_argument("--eval-limit", type=int, default=8)
    eval_efold.add_argument("--min-length", type=int, default=1)
    eval_efold.add_argument("--max-length", type=int, default=256)
    eval_efold.add_argument("--window-size", type=int, default=None, help="slice raw JSON records into local windows before training/eval")
    eval_efold.add_argument("--window-stride", type=int, default=None, help="stride for --window-size")
    eval_efold.add_argument("--epochs", type=int, default=10)
    eval_efold.add_argument("--hidden-size", type=int, default=8)
    eval_efold.add_argument("--learning-rate", type=float, default=0.2)
    eval_efold.add_argument("--lambda-react", type=float, default=0.0, help="default 0 avoids pseudo-reactivity supervision on structure-only records")
    eval_efold.add_argument("--lambda-thermo", type=float, default=0.0)
    eval_efold.add_argument("--lambda-calib", type=float, default=0.0, help="weight on the variance-aware ensemble-calibration NLL term (0 disables)")
    eval_efold.add_argument("--calib-beta", type=float, default=1.0, help="structural-variance scale beta for the ensemble-calibration term")
    eval_efold.add_argument("--calib-tau-squared", type=float, default=0.05, help="measurement-noise variance floor tau^2 for the ensemble-calibration term")
    eval_efold.add_argument("--lambda-contact", type=float, default=0.0, help="weight on the legal contact-map denoising auxiliary (0 disables)")
    eval_efold.add_argument("--contact-negative-weight", type=float, default=0.25, help="scale for legal non-pair BCE in the contact auxiliary")
    eval_efold.add_argument("--contact-long-range-min-distance", type=int, default=24, help="RF-CF2 span threshold for long-range contact weighting")
    eval_efold.add_argument("--contact-long-range-weight", type=float, default=1.0, help="RF-CF2 weight for legal contact candidates at or beyond the long-range threshold")
    eval_efold.add_argument("--thermo-mode", choices=["mse", "kl"], default="mse")
    eval_efold.add_argument("--adapter-dim", type=int, default=0)
    eval_efold.add_argument("--adapter-lr", type=float, default=None)
    eval_efold.add_argument("--frozen-dir", default="")
    eval_efold.add_argument("--frozen-cache-shards", type=int, default=4)
    eval_efold.add_argument("--profile-path", default="", help="write detailed per-phase training timings as JSONL")
    eval_efold.add_argument("--batch-size", type=int, default=None, help="samples per gradient update (default: full batch)")
    eval_efold.add_argument("--bucket-boundaries", type=_parse_bucket_boundaries, default=tuple(), help="comma-separated length buckets, e.g. 64,128,256")
    eval_efold.add_argument("--family-balanced-batches", action="store_true", help="interleave training samples across family/cluster labels within each length bucket")
    eval_efold.add_argument("--backend", choices=["stdlib", "torch"], default="stdlib", help="training backend")
    eval_efold.add_argument("--torch-device", default="cpu", help="torch device when --backend torch")
    eval_efold.add_argument("--seed", type=int, default=0)
    eval_efold.add_argument("--min-loop", type=int, default=3)
    eval_efold.add_argument("--probe", choices=["2A3", "DMS"], default="2A3")
    eval_efold.add_argument(
        "--inference-mode",
        choices=["legacy_direct", "ctmc_sample", "calibrated_marginal"],
        default="calibrated_marginal",
        help="corrected CTMC inference is default; legacy_direct is regression-only",
    )
    eval_efold.add_argument("--validation-json", default="", help="required for corrected inference calibration")
    eval_efold.add_argument("--inference-seed", type=int, default=20260718)
    eval_efold.add_argument("--inference-coarse-count", type=int, default=128)
    eval_efold.add_argument("--inference-validation-count", type=int, default=512)
    eval_efold.add_argument("--inference-steps-grid", default="8,16,32")
    eval_efold.add_argument("--inference-samples-grid", default="4,8,16")
    eval_efold.add_argument("--inference-temperature-grid", default="0.5,1.0,2.0")
    eval_efold.add_argument("--inference-threshold-grid", default="-2,-1,0,1,2")
    eval_efold.set_defaults(func=_cmd_evaluate_efold)

    calibrate_inference = sub.add_parser(
        "calibrate-inference",
        help="lock CTMC and structured-decoder settings on validation only",
    )
    calibrate_inference.add_argument("--checkpoint", required=True)
    calibrate_inference.add_argument("--validation-json", required=True)
    calibrate_inference.add_argument("--frozen-dir", default="")
    calibrate_inference.add_argument("--frozen-cache-shards", type=int, default=4)
    calibrate_inference.add_argument("--output", required=True)
    calibrate_inference.add_argument("--seed", type=int, default=20260718)
    calibrate_inference.add_argument("--coarse-count", type=int, default=128)
    calibrate_inference.add_argument("--validation-count", type=int, default=512)
    calibrate_inference.add_argument("--steps-grid", default="8,16,32")
    calibrate_inference.add_argument("--samples-grid", default="4,8,16")
    calibrate_inference.add_argument("--temperature-grid", default="0.5,1.0,2.0")
    calibrate_inference.add_argument("--threshold-grid", default="-2,-1,0,1,2")
    calibrate_inference.add_argument(
        "--matching-policy",
        action="append",
        choices=["nested_dp", "pseudoknot_allowed_greedy"],
        default=None,
    )
    calibrate_inference.set_defaults(func=_cmd_calibrate_inference)

    evaluate_checkpoint = sub.add_parser(
        "evaluate-checkpoint",
        help="evaluate a fixed checkpoint using a validation-locked decoder manifest",
    )
    evaluate_checkpoint.add_argument("--checkpoint", required=True)
    evaluate_checkpoint.add_argument("--decoder-manifest", required=True)
    evaluate_checkpoint.add_argument("--eval-json", action="append", required=True, help="tier=path")
    evaluate_checkpoint.add_argument("--frozen-dir", default="")
    evaluate_checkpoint.add_argument("--frozen-cache-shards", type=int, default=4)
    evaluate_checkpoint.add_argument("--output-dir", required=True)
    evaluate_checkpoint.add_argument("--limit-per-tier", type=int, default=1000)
    evaluate_checkpoint.add_argument("--full-tier", action="append", default=["PDB"])
    evaluate_checkpoint.add_argument(
        "--mode",
        action="append",
        choices=["legacy_direct", "ctmc_sample", "calibrated_marginal"],
        default=None,
    )
    evaluate_checkpoint.set_defaults(func=_cmd_evaluate_checkpoint)

    preflight_checkpoint = sub.add_parser(
        "preflight-checkpoint",
        help="estimate fixed-matrix CTMC runtime before the full evaluation",
    )
    preflight_checkpoint.add_argument("--checkpoint", required=True)
    preflight_checkpoint.add_argument("--decoder-manifest", required=True)
    preflight_checkpoint.add_argument("--eval-json", action="append", required=True, help="tier=path")
    preflight_checkpoint.add_argument("--frozen-dir", default="")
    preflight_checkpoint.add_argument("--frozen-cache-shards", type=int, default=4)
    preflight_checkpoint.add_argument("--limit-per-tier", type=int, default=1000)
    preflight_checkpoint.add_argument("--full-tier", action="append", default=["PDB"])
    preflight_checkpoint.add_argument("--preflight-per-tier", type=int, default=11)
    preflight_checkpoint.add_argument("--max-projected-hours", type=float, default=24.0)
    preflight_checkpoint.add_argument("--output", required=True)
    preflight_checkpoint.set_defaults(func=_cmd_preflight_checkpoint)

    sample = sub.add_parser("sample", help="draw a legal 2D structure ensemble from the CTMC sampler")
    sample.add_argument("--sequence", default="", help="target sequence (defaults to the first pilot sequence)")
    sample.add_argument("--num-samples", type=int, default=200)
    sample.add_argument("--num-steps", type=int, default=32)
    sample.add_argument("--no-pseudoknot", action="store_true", help="forbid crossing pairs during projection")
    sample.add_argument("--output-dir", default="artifacts/sample")
    sample.add_argument("--epochs", type=int, default=40)
    sample.add_argument("--seed", type=int, default=0)
    sample.add_argument("--samples", type=int, default=6)
    sample.add_argument("--stem", type=int, default=4)
    sample.add_argument("--loop", type=int, default=4)
    sample.add_argument("--probe", choices=["2A3", "DMS"], default="2A3")
    sample.set_defaults(func=_cmd_sample)

    scan = sub.add_parser("guidance-scan", help="sweep inference-time energy guidance eta and plot the tradeoff")
    scan.add_argument("--sequence", default="", help="target sequence (defaults to the first pilot sequence)")
    scan.add_argument("--reference", default="", help="reference dot-bracket for F1 (defaults to the pilot structure)")
    scan.add_argument("--etas", default="0.0,0.25,0.5,1.0,2.0", help="comma-separated guidance strengths")
    scan.add_argument("--greedy", action="store_true", help="use greedy projection instead of exact nested DP")
    scan.add_argument("--output-dir", default="artifacts/guidance")
    scan.add_argument("--epochs", type=int, default=40)
    scan.add_argument("--seed", type=int, default=0)
    scan.add_argument("--samples", type=int, default=6)
    scan.add_argument("--stem", type=int, default=4)
    scan.add_argument("--loop", type=int, default=4)
    scan.add_argument("--probe", choices=["2A3", "DMS"], default="2A3")
    scan.set_defaults(func=_cmd_guidance_scan)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Execute the ReactFlow command-line interface.

    Complexity: parser construction is O(1); subcommand complexity is documented
    in the corresponding handler.
    """

    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
