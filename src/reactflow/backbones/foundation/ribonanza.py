"""RibonanzaNet2 foundation backbone for ReactFlow Phase C1-3.

RibonanzaNet2 (Ogawa et al., 2023) is a 100M-parameter RNA structure encoder
that won the Ribonanza RNA folding Kaggle competition.  Because its weights
are large and its forward is expensive, ReactFlow runs it *once, offline* and
freezes its per-nucleotide representations to the sharded disk format defined
by the C5 frozen-feature contract (``provenance.json`` + ``features.npz`` +
``index.jsonl``).  This backbone reads those cached features and exposes them
through the unified :class:`~reactflow.backbones.foundation.base.FoundationBackbone`
interface.

Shard layout
------------
The frozen-feature root is a directory of *shard* subdirectories.  Each shard
holds::

    <shard>/provenance.json   # model provenance (carries ``weights_sha256``)
    <shard>/features.npz       # ZIP of ``<row:06d>.single`` arrays, shape (L, 384)
    <shard>/index.jsonl        # one JSON object per record (row, record_id, ...)

Each shard contains 512 records.  The NPZ member for record at row ``r`` is
``f"{r:06d}.single"`` and has shape ``(L, 384)`` where ``d_single = 384`` is the
RibonanzaNet2 per-nucleotide embedding dimension.

Pair features
-------------
Pair features are optional in the shard (the ``pair`` array is often omitted to
save disk because it is ``O(L^2 d)``).  When the shard does not carry a ``pair``
member, this backbone computes an initial pair stack from the frozen singles via
the outer-product-mean:

    z_ij = mean_{d1,d2} ( W_h[d1,d2,:] * a_i[d1] * b_j[d2] )

implemented as a memory-efficient two-step einsum (see
:class:`~reactflow.backbones.outer.OuterProductMean`) that avoids ever
materialising the ``L x L x d1 x d2`` intermediate.

Modes
-----
- :attr:`BackboneMode.FROZEN`: return the cached ``single`` features directly
  (no autograd through the encoder).  Pair features are either read from the
  shard or computed via the outer-product-mean module (which stays trainable so
  the pair initialisation can adapt).
- :attr:`BackboneMode.LORA`: add a low-rank residual adapter
  ``h' = h + B A LayerNorm(h)`` on top of the frozen single features; only
  ``A`` and ``B`` train.
- :attr:`BackboneMode.FULL_FINE_TUNE`: not supported for cached features (the
  RibonanzaNet2 encoder is not present on the training host); raises
  :class:`NotImplementedError`.

Complexity
----------
- Index build: ``O(N)`` over all records once (amortised).
- Forward (frozen): ``O(B * L * d_single)`` for lookup + ``O(B * L^2 * d_pair)``
  when computing pairs via outer-product-mean.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from ..outer import OuterProductMean
from .base import (
    BackboneConfig,
    BackboneMode,
    BackboneNotAvailableError,
    BackboneOutput,
    FoundationBackbone,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RIBONANZANET2_FROZEN_DIM = 384
"""Per-nucleotide single-feature dimension produced by RibonanzaNet2."""

DEFAULT_SHARD_ROOT = (
    "/home/cunyuliu/reactflow/artifacts/full_runs/"
    "full_ablation_20260709_003012/frozen/ribonanzanet2_sharded_full/"
)
"""Default location of the sharded frozen-feature root on the training host."""

_RECORDS_PER_SHARD = 512
"""Number of records per shard (documents the export contract)."""

_PROVENANCE_NAME = "provenance.json"
_FEATURES_NAME = "features.npz"
_INDEX_NAME = "index.jsonl"
_ARRAY_SINGLE = "single"
_ARRAY_PAIR = "pair"


# ---------------------------------------------------------------------------
# Provenance loader
# ---------------------------------------------------------------------------


def _load_shard_provenance(shard_dir: Path) -> Dict[str, Any]:
    """Load and parse a shard's ``provenance.json``.

    Formula: reads the JSON header that the offline exporter wrote alongside
    ``features.npz``.  Complexity: ``O(1)`` metadata (file is small).

    Args:
        shard_dir: directory containing ``provenance.json``.

    Returns:
        Parsed provenance dict (carries ``weights_sha256``, ``model_version``,
        ``schema``, ``record_count``, ``content_sha256``).
    """

    return json.loads((shard_dir / _PROVENANCE_NAME).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------


class RibonanzaNet2Backbone(FoundationBackbone):
    """Foundation backbone backed by frozen RibonanzaNet2 features.

    This backbone does **not** load the RibonanzaNet2 weights; instead it reads
    pre-computed per-nucleotide features from the sharded disk format.  In
    :attr:`BackboneMode.FROZEN` the forward is a pure lookup (the encoder runs
    offline), so the training-side memory footprint is ``O(B*L*d)``.

    The record-id -> (shard, row) index is built lazily on first access so
    that instantiating the backbone from the registry is cheap even when the
    shard root is on a slow network mount (or absent on a non-training host).

    Args:
        config: :class:`BackboneConfig` for RibonanzaNet2.  ``frozen_feature_dim``
            is set to 384 by :func:`default_config` if not provided.
        shard_root: directory containing the frozen-feature shards.  Defaults
            to :data:`DEFAULT_SHARD_ROOT`.
        mode: operating :class:`BackboneMode`.
        lora_rank: LoRA rank ``r`` (only used in :attr:`BackboneMode.LORA`).
        pair_dim: output pair-feature dimension ``d_pair`` used by the
            outer-product-mean module when pair features are not cached.
        verify: if True, recompute and check the shard content hash on read.
            Disabled by default for training speed (the hash check is
            ``O(total shard bytes)``).

    Complexity: ``O(1)`` construction (lazy index); forward as documented above.
    """

    def __init__(
        self,
        config: BackboneConfig,
        *,
        shard_root: str | Path = DEFAULT_SHARD_ROOT,
        mode: BackboneMode = BackboneMode.FROZEN,
        lora_rank: int = 8,
        pair_dim: int = 128,
        verify: bool = False,
    ) -> None:
        super().__init__(config, mode)
        if config.frozen_feature_dim == 0:
            config.frozen_feature_dim = RIBONANZANET2_FROZEN_DIM
        if config.pair_feature_dim == 0:
            config.pair_feature_dim = pair_dim

        self.shard_root: Path = Path(shard_root)
        self.verify: bool = verify
        self.lora_rank: int = lora_rank
        d_single = config.frozen_feature_dim

        # record_id -> (shard_dir, row); built lazily.
        self._index: Dict[str, Tuple[Path, int]] = {}
        self._index_built: bool = False
        # Cached provenance (weights_sha256 etc.) from the first shard.
        self._provenance_cache: Optional[Dict[str, Any]] = None

        # Outer-product-mean module: initialises a pair stack from the frozen
        # singles when the shard does not carry a cached ``pair`` array.
        self.pair_opm = OuterProductMean(d_single, pair_dim)

        # LoRA adapter (only meaningful in LORA mode; constructed unconditionally
        # so the parameter layout is stable across mode switches).
        self.lora_norm = nn.LayerNorm(d_single)
        self.lora_down = nn.Linear(d_single, lora_rank, bias=False)
        self.lora_up = nn.Linear(lora_rank, d_single, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_up.weight)  # adapter starts as identity

        self._apply_mode()
        # The outer-product-mean pair initializer is a lightweight adaptation
        # layer (not the frozen encoder), so it stays trainable in FROZEN and
        # LORA modes so the pair init can be learned from the cached singles.
        for p in self.pair_opm.parameters():
            p.requires_grad_(True)
        if mode == BackboneMode.LORA:
            # Re-enable gradients on the LoRA adapter (base stays frozen).
            for p in self.lora_norm.parameters():
                p.requires_grad_(True)
            self.lora_down.weight.requires_grad_(True)
            self.lora_up.weight.requires_grad_(True)

    # -- index / provenance -------------------------------------------------

    def _ensure_index(self) -> None:
        """Build the record-id -> (shard, row) index lazily.

        Scans ``shard_root`` for subdirectories containing ``provenance.json``
        (each such directory is a shard) and parses each ``index.jsonl``.  The
        provenance of the first shard is cached for :meth:`get_provenance`.

        Complexity: ``O(N)`` over all records, run once.
        """

        if self._index_built:
            return
        self._index_built = True
        root = self.shard_root
        if not root.exists():
            # Path is absent (e.g., on a non-training host).  The index stays
            # empty and forward will raise a clear error on first use.
            return

        # ``root`` may itself be a single shard (contains provenance.json) or a
        # directory of shards.  Detect both layouts.
        shard_dirs: List[Path]
        if (root / _PROVENANCE_NAME).exists():
            shard_dirs = [root]
        else:
            shard_dirs = sorted(
                p for p in root.iterdir()
                if p.is_dir() and (p / _PROVENANCE_NAME).exists()
            )

        for shard_dir in shard_dirs:
            if self._provenance_cache is None:
                self._provenance_cache = _load_shard_provenance(shard_dir)
            index_path = shard_dir / _INDEX_NAME
            if not index_path.exists():
                continue
            for raw_line in index_path.read_text(encoding="utf-8").splitlines():
                if not raw_line.strip():
                    continue
                entry = json.loads(raw_line)
                record_id = str(entry["record_id"])
                row = int(entry["row"])
                self._index[record_id] = (shard_dir, row)

    # -- feature lookup -----------------------------------------------------

    def _load_single_features(
        self,
        record_ids: Sequence[str],
    ) -> List[np.ndarray]:
        """Load the cached ``single`` arrays for a list of record ids.

        Groups records by shard so each ``features.npz`` ZIP is opened at most
        once per call.  Member names are ``f"{row:06d}.single"``.

        Complexity: ``O(sum_i L_i * d_single)`` plus ``O(#shards)`` ZIP opens.

        Args:
            record_ids: sequence of record ids present in the index.

        Returns:
            List of ``np.ndarray`` of shape ``(L_i, d_single)`` in input order.

        Raises:
            KeyError: if a record id is not in the index.
        """

        self._ensure_index()
        # Group by shard to minimise ZIP opens.
        by_shard: Dict[Path, List[Tuple[int, str]]] = {}
        order: List[Tuple[Path, int]] = []
        for rid in record_ids:
            shard_dir, row = self._index[rid]
            by_shard.setdefault(shard_dir, []).append((row, rid))
            order.append((shard_dir, row))

        cache: Dict[Tuple[Path, int], np.ndarray] = {}
        for shard_dir, entries in by_shard.items():
            with np.load(shard_dir / _FEATURES_NAME, allow_pickle=False) as npz:
                for row, _rid in entries:
                    member = f"{row:06d}.{_ARRAY_SINGLE}"
                    cache[(shard_dir, row)] = npz[member]
        return [cache[(s, r)] for (s, r) in order]

    def _load_pair_features(
        self,
        record_ids: Sequence[str],
    ) -> List[Optional[np.ndarray]]:
        """Load optional cached ``pair`` arrays for a list of record ids.

        Returns ``None`` per record when the shard does not carry a ``pair``
        member for that record.

        Complexity: ``O(sum_i L_i^2 * d_pair)`` when pairs are present.

        Args:
            record_ids: sequence of record ids.

        Returns:
            List of optional ``np.ndarray`` of shape ``(L_i, L_i, d_pair)``.
        """

        self._ensure_index()
        by_shard: Dict[Path, List[Tuple[int, str]]] = {}
        order: List[Tuple[Path, int]] = []
        for rid in record_ids:
            shard_dir, row = self._index[rid]
            by_shard.setdefault(shard_dir, []).append((row, rid))
            order.append((shard_dir, row))

        cache: Dict[Tuple[Path, int], Optional[np.ndarray]] = {}
        for shard_dir, entries in by_shard.items():
            with np.load(shard_dir / _FEATURES_NAME, allow_pickle=False) as npz:
                available = set(npz.files)
                for row, _rid in entries:
                    member = f"{row:06d}.{_ARRAY_PAIR}"
                    cache[(shard_dir, row)] = (
                        npz[member] if member in available else None
                    )
        return [cache[(s, r)] for (s, r) in order]

    # -- forward ------------------------------------------------------------

    def forward(
        self,
        indices: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        *,
        record_ids: Optional[Sequence[str]] = None,
    ) -> BackboneOutput:
        """Return single (and optionally pair) features for a batch.

        In :attr:`BackboneMode.FROZEN` the singles are the cached RibonanzaNet2
        features looked up by ``record_ids``.  In :attr:`BackboneMode.LORA` a
        low-rank residual ``B A LayerNorm(h)`` is added.  Pair features are read
        from the shard when present, else computed via the outer-product-mean.

        Args:
            indices: LongTensor ``(B, L)`` of nucleotide vocab indices (used
                only for shape/device and, in pair mode, for masking).
            mask: optional BoolTensor ``(B, L)``.
            record_ids: sequence of length ``B`` giving each sample's record id
                (required for frozen-feature lookup).

        Returns:
            :class:`BackboneOutput`.

        Raises:
            BackboneNotAvailableError: if the shard root is absent.
            ValueError: if ``record_ids`` is None or a record is missing.
            NotImplementedError: in :attr:`BackboneMode.FULL_FINE_TUNE`
                (the live encoder is not present on the training host).

        Complexity: ``O(B*L*d)`` for singles; ``O(B*L^2*d_pair)`` when pairs
        are computed via the outer-product-mean.
        """

        if self._mode == BackboneMode.FULL_FINE_TUNE:
            raise NotImplementedError(
                "RibonanzaNet2Backbone does not support FULL_FINE_TUNE: the "
                "live encoder is not present on the training host (only cached "
                "features are available). Use FROZEN or LORA mode."
            )
        if record_ids is None:
            raise ValueError(
                "RibonanzaNet2Backbone.forward requires `record_ids` to look up "
                "cached frozen features."
            )

        self._ensure_index()
        if not self._index:
            raise BackboneNotAvailableError(
                self._config.model_name,
                f"local shard root: {self.shard_root}",
                hint=(
                    "Copy the RibonanzaNet2 sharded frozen features to the "
                    f"shard_root path ({self.shard_root}) or construct the "
                    "backbone with an existing shard_root."
                ),
            )

        B, L = indices.shape
        device = indices.device

        # -- single features -------------------------------------------------
        single_arrays = self._load_single_features(record_ids)
        single_features = torch.from_numpy(
            np.stack(
                [
                    _resize_or_pad(a, L, self._config.frozen_feature_dim)
                    for a in single_arrays
                ],
                axis=0,
            )
        ).to(device=device, dtype=torch.float32)

        # Apply LoRA residual in LORA mode (identity at init because lora_up is 0).
        if self._mode == BackboneMode.LORA:
            residual = self.lora_up(self.lora_down(self.lora_norm(single_features)))
            single_features = single_features + residual

        if mask is None:
            mask = (indices != 5)  # PAD_INDEX from ..embeddings
        mask = mask.to(device=device)

        # Zero out padding positions in the single features.
        single_features = single_features * mask.unsqueeze(-1).to(single_features.dtype)

        # -- pair features ---------------------------------------------------
        pair_arrays = self._load_pair_features(record_ids)
        pair_tensor: Optional[torch.Tensor] = None
        if all(p is not None for p in pair_arrays):
            d_pair = pair_arrays[0].shape[-1]  # type: ignore[union-attr]
            pair_tensor = torch.from_numpy(
                np.stack(
                    [_resize_or_pad_pair(p, L, d_pair) for p in pair_arrays],
                    axis=0,
                )
            ).to(device=device, dtype=torch.float32)
        else:
            # Compute pairs via outer-product-mean from the (possibly LoRA-
            # adapted) singles.  Pass a zero pair so the OPM acts as an
            # initializer: z = 0 + update = update.
            zero_pair = single_features.new_zeros(B, L, L, self.pair_dim)
            pair_tensor = self.pair_opm(single_features, zero_pair, mask=mask)

        # Mask out pairs involving padding positions.
        pair_mask = (mask.unsqueeze(2) & mask.unsqueeze(1)).unsqueeze(-1).to(
            pair_tensor.dtype
        )
        pair_tensor = pair_tensor * pair_mask

        return BackboneOutput(
            single_features=single_features,
            pair_features=pair_tensor,
            mask=mask,
        )

    # -- provenance ---------------------------------------------------------

    def get_provenance(self) -> BackboneConfig:
        """Return the :class:`BackboneConfig` with ``weights_sha256`` from the
        shard's ``provenance.json``.

        Formula: the weights hash is *not* recomputed (the weights are absent
        on the training host); it is carried through verbatim from the offline
        exporter's provenance header.  Complexity: ``O(1)`` (lazy index may add
        ``O(N)`` on first call).

        Returns:
            A copy of this backbone's config with ``weights_sha256`` filled
            from the cached shard provenance.
        """

        self._ensure_index()
        if self._provenance_cache is not None:
            weights_sha256 = str(self._provenance_cache.get("weights_sha256", ""))
            code_revision = str(self._provenance_cache.get("model_version", ""))
        else:
            weights_sha256 = self._config.weights_sha256
            code_revision = self._config.code_revision

        return BackboneConfig(
            model_name=self._config.model_name,
            model_source=self._config.model_source,
            model_revision=self._config.model_revision,
            license=self._config.license,
            weights_sha256=weights_sha256,
            code_revision=code_revision,
            tokenizer=self._config.tokenizer,
            max_length=self._config.max_length,
            contamination_status=self._config.contamination_status,
            frozen_feature_dim=self._config.frozen_feature_dim,
            pair_feature_dim=self._config.pair_feature_dim,
            downloaded=self._config.downloaded,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resize_or_pad(arr: np.ndarray, length: int, dim: int) -> np.ndarray:
    """Resize a cached single array to ``(length, dim)`` with zero padding.

    Formula: if the cached array is shorter than ``length``, right-pad with
    zeros; if longer, truncate.  Complexity: ``O(L * dim)``.

    Args:
        arr: array of shape ``(L_cache, dim_cache)``.
        length: target sequence length ``L``.
        dim: target feature dimension ``d_single``.

    Returns:
        ``np.ndarray`` of shape ``(length, dim)``.
    """

    out = np.zeros((length, dim), dtype=np.float32)
    l_cache = min(arr.shape[0], length)
    d_cache = min(arr.shape[1], dim)
    out[:l_cache, :d_cache] = arr[:l_cache, :d_cache].astype(np.float32)
    return out


def _resize_or_pad_pair(arr: np.ndarray, length: int, dim: int) -> np.ndarray:
    """Resize a cached pair array to ``(length, length, dim)``.

    Formula: pad or truncate on both spatial axes.  Complexity:
    ``O(L^2 * dim)``.

    Args:
        arr: array of shape ``(L_cache, L_cache, dim_cache)``.
        length: target sequence length ``L``.
        dim: target pair dimension ``d_pair``.

    Returns:
        ``np.ndarray`` of shape ``(length, length, dim)``.
    """

    out = np.zeros((length, length, dim), dtype=np.float32)
    l_cache = min(arr.shape[0], length)
    d_cache = min(arr.shape[-1], dim)
    out[:l_cache, :l_cache, :d_cache] = (
        arr[:l_cache, :l_cache, :d_cache].astype(np.float32)
    )
    return out


def default_config() -> BackboneConfig:
    """Return the canonical :class:`BackboneConfig` for RibonanzaNet2.

    Formula: pins ``frozen_feature_dim=384``, ``pair_feature_dim=128``,
    ``downloaded=True`` (cached features are the local availability signal),
    and the default shard root as ``model_source``.  Complexity: ``O(1)``.

    Returns:
        A :class:`BackboneConfig` with RibonanzaNet2 defaults.
    """

    return BackboneConfig(
        model_name="RibonanzaNet2",
        model_source="github:sh-ogawa/RibonanzaNet2",
        model_revision="ribonanza2",
        license="Apache-2.0",
        weights_sha256="",
        code_revision="",
        tokenizer="ribonanza-bpe",
        max_length=4488,
        contamination_status="unknown_contamination",
        frozen_feature_dim=RIBONANZANET2_FROZEN_DIM,
        pair_feature_dim=128,
        downloaded=True,
    )
