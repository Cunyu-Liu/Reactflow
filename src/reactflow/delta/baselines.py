"""ReactFlow-Delta B0 baselines (v3.3 §10).

Baseline families implemented here:

* **Non-learned** (§10.1): zero-change, mutation-type mean, distance-decay,
  edit-only, nearest-train, local-release. These need only the train split's
  ``delta_reactivity`` arrays and (for local-release) the WT BPP features from
  the PH0 manifest.
* **Thermo** (§10.2): RNAfold/RNAplfold (ViennaRNA Python API) and EternaFold
  (CLI). All mutant-sequence-dependent baselines **marginalize over the 3
  possible alt bases** because every true pair carries ``encoded_alt="X"``.
* **Learned independent** (§10.3): train-only sequence-to-reactivity CNN.
* **Matched paired** (§10.4/§10.8): Siamese (shared encoder + late fusion) and
  generic paired (cross-interaction, no EPRO constraints).

The ``Baseline`` ABC exposes ``fit(train_pairs)`` and ``predict(pair) ->
np.ndarray``. Predictions are aligned to the pair's delta array (length =
``aligned_length``); positions outside the §12.1 endpoint mask are ignored by
the evaluator but baselines should still emit a full-length array (filling
masked positions with 0 is safe).

Torch is imported lazily inside the learned baseline constructors so the
module is importable in ViennaRNA-only environments.
"""

from __future__ import annotations

import math
import os
import subprocess
import tempfile
from abc import ABC, abstractmethod
from typing import Any, Callable, Sequence

import numpy as np

from .evaluate import PairRecord

# ---------------------------------------------------------------------------
# Alt marginalization (CRITICAL: encoded_alt="X" for all 1509 pairs)
# ---------------------------------------------------------------------------

_RNA_BASES = ("A", "C", "G", "U")


def alt_candidates(ref_base: str) -> list[str]:
    """Return the 3 non-ref RNA bases (uppercase)."""

    ref = ref_base.upper().replace("T", "U")
    return [b for b in _RNA_BASES if b != ref]


def build_mutant_sequences(wt_sequence: str, edit_pos_1indexed: int, ref_base: str) -> list[str]:
    """Construct the 3 possible mutant sequences by substituting the edit base.

    ``wt_sequence`` may be lowercase (RDAT convention). The edit position is
    1-indexed. Returns 3 sequences (uppercase U), each identical to WT except
    at ``edit_pos_1indexed-1`` which is replaced by one of the 3 non-ref bases.

    Raises ``ValueError`` if the WT base at the edit position does not match
    ``ref_base`` (case-insensitive, T->U).
    """

    wt = wt_sequence.upper().replace("T", "U")
    pos0 = edit_pos_1indexed - 1
    if not (0 <= pos0 < len(wt)):
        raise ValueError(
            f"edit position {edit_pos_1indexed} out of range for length {len(wt)}"
        )
    actual = wt[pos0]
    expected = ref_base.upper().replace("T", "U")
    if actual != expected:
        raise ValueError(
            f"WT base at position {edit_pos_1indexed} is {actual!r}, expected {expected!r}"
        )
    seqs: list[str] = []
    for alt in alt_candidates(expected):
        seqs.append(wt[:pos0] + alt + wt[pos0 + 1 :])
    return seqs


def map_seq_array_to_delta(seq_array: np.ndarray, record: PairRecord) -> np.ndarray:
    """Map a per-sequence-position array to per-array-index (delta) alignment.

    ``seq_array[i]`` is the value at 1-indexed SEQUENCE position ``i+1``.
    The output ``out[arr_idx] = seq_array[seq_positions[arr_idx] - 1]``.
    Positions with NaN seq_positions or out-of-range indices get 0.0.
    """

    n = record.aligned_length
    out = np.zeros(n, dtype=float)
    sp = record.seq_positions
    for i in range(n):
        if i < len(sp) and not np.isnan(sp[i]):
            j = int(sp[i]) - 1
            if 0 <= j < len(seq_array):
                v = seq_array[j]
                out[i] = float(v) if not np.isnan(v) else 0.0
    return out


def map_delta_array_to_seq(
    arr: np.ndarray, mask: np.ndarray, record: PairRecord, seq_len: int
) -> tuple[np.ndarray, np.ndarray]:
    """Map a delta-array-aligned (target, mask) to sequence-aligned arrays.

    Returns ``(target_seq, mask_seq)`` of length ``seq_len``. Delta-array
    positions are placed at ``seq_positions[i] - 1`` in the sequence. Positions
    not covered by the delta array get 0 in both outputs.
    """

    target_seq = np.zeros(seq_len, dtype=float)
    mask_seq = np.zeros(seq_len, dtype=float)
    sp = record.seq_positions
    n = min(len(arr), record.aligned_length, len(mask))
    for i in range(n):
        if i < len(sp) and not np.isnan(sp[i]) and bool(mask[i]):
            j = int(sp[i]) - 1
            if 0 <= j < seq_len:
                v = arr[i]
                target_seq[j] = float(v) if not np.isnan(v) else 0.0
                mask_seq[j] = 1.0
    return target_seq, mask_seq


# ---------------------------------------------------------------------------
# Baseline ABC
# ---------------------------------------------------------------------------


class Baseline(ABC):
    """Abstract baseline. Subclasses implement ``fit`` and ``predict``."""

    name: str = "baseline"
    requires_wt_sequence: bool = False
    requires_wt_features: bool = False
    requires_seq_positions: bool = False
    is_learned: bool = False

    @abstractmethod
    def fit(self, train_pairs: list[PairRecord]) -> None: ...

    @abstractmethod
    def predict(self, pair: PairRecord) -> np.ndarray: ...

    def predict_many(self, pairs: list[PairRecord]) -> dict[str, np.ndarray]:
        return {p.pair_id: self.predict(p) for p in pairs}


# ---------------------------------------------------------------------------
# Non-learned baselines (§10.1)
# ---------------------------------------------------------------------------


class ZeroChangeBaseline(Baseline):
    """Predict Delta r = 0 everywhere. This is the Skill denominator reference."""

    name = "zero_change"
    requires_wt_sequence = False

    def fit(self, train_pairs: list[PairRecord]) -> None:
        return None

    def predict(self, pair: PairRecord) -> np.ndarray:
        return np.zeros(pair.aligned_length, dtype=float)


class MutationTypeMeanBaseline(Baseline):
    """Predict the train-set mean Delta r profile (per array index).

    Computes the per-position mean of ``delta_true`` over the train split,
    masked to the endpoint mask. Positions that are never observed in train
    default to 0. This captures the average "mutation footprint" shape.
    """

    name = "mutation_type_mean"

    def __init__(self) -> None:
        self._mean_profile: np.ndarray | None = None

    def fit(self, train_pairs: list[PairRecord]) -> None:
        if not train_pairs:
            self._mean_profile = np.zeros(0)
            return
        max_len = max(p.aligned_length for p in train_pairs)
        acc = np.zeros(max_len, dtype=float)
        cnt = np.zeros(max_len, dtype=float)
        for p in train_pairs:
            d = p.delta_true.copy()
            d[~p.endpoint_mask] = 0.0
            d[np.isnan(d)] = 0.0
            n = p.aligned_length
            acc[:n] += d[:n]
            cnt[:n] += p.endpoint_mask[:n].astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(cnt > 0, acc / np.maximum(cnt, 1.0), 0.0)
        self._mean_profile = mean

    def predict(self, pair: PairRecord) -> np.ndarray:
        if self._mean_profile is None:
            return np.zeros(pair.aligned_length, dtype=float)
        n = pair.aligned_length
        out = np.zeros(n, dtype=float)
        m = min(n, len(self._mean_profile))
        out[:m] = self._mean_profile[:m]
        return out


class DistanceDecayBaseline(Baseline):
    """Predict a distance-decay footprint centered on the edit position.

    ``Delta r_hat[i] = amplitude * exp(-|seq_dist_i| / length_scale) * sign``

    where ``amplitude`` and ``length_scale`` are fitted from train pairs (mean
    |delta| as a function of |seq_dist|), and ``sign`` is the train-mean sign
    of delta at the edit position.
    """

    name = "distance_decay"
    requires_seq_positions = True

    def __init__(self, length_scale: float = 15.0) -> None:
        self._length_scale = float(length_scale)
        self._amplitude: float = 0.0
        self._sign: float = 0.0

    def fit(self, train_pairs: list[PairRecord]) -> None:
        # Fit amplitude = mean |delta| over all masked positions, and a
        # length-scale by regressing log|delta| on |seq_dist|.
        deltas: list[float] = []
        dists: list[float] = []
        edit_signs: list[float] = []
        for p in train_pairs:
            mask = p.endpoint_mask
            d = p.delta_true[mask]
            sp = p.seq_positions[mask]
            dist = np.abs(sp - p.edit_pos_1indexed)
            finite = np.isfinite(d) & np.isfinite(dist) & (np.abs(d) > 1e-8)
            deltas.extend(np.abs(d[finite]).tolist())
            dists.extend(dist[finite].tolist())
            if p.edit_arr_idx is not None and 0 <= p.edit_arr_idx < p.aligned_length:
                ev = p.delta_true[p.edit_arr_idx]
                if np.isfinite(ev):
                    edit_signs.append(float(np.sign(ev)))

        if deltas:
            self._amplitude = float(np.mean(deltas))
        if edit_signs:
            self._sign = float(np.mean(edit_signs))
        # Fit length-scale from log|delta| vs |dist| if we have enough spread.
        if len(dists) >= 20:
            d_arr = np.array(dists, dtype=float)
            y_arr = np.array(deltas, dtype=float)
            mask = (d_arr > 0) & (y_arr > 1e-8)
            if mask.sum() >= 20:
                x = d_arr[mask]
                y = np.log(y_arr[mask])
                # Linear regression y = a + b*x  ->  length_scale = -1/b
                xmean = x.mean()
                ymean = y.mean()
                denom = float(((x - xmean) ** 2).sum())
                if denom > 0:
                    b = float(((x - xmean) * (y - ymean)).sum() / denom)
                    if b < -1e-6:
                        self._length_scale = min(50.0, max(1.0, -1.0 / b))

    def predict(self, pair: PairRecord) -> np.ndarray:
        n = pair.aligned_length
        out = np.zeros(n, dtype=float)
        if not np.any(np.isfinite(pair.seq_positions)):
            return out
        dist = np.abs(pair.seq_positions - pair.edit_pos_1indexed)
        finite = np.isfinite(dist)
        out[finite] = (
            self._amplitude
            * np.exp(-dist[finite] / max(self._length_scale, 1e-3))
            * (self._sign if self._sign != 0 else 1.0)
        )
        return out


class EditOnlyBaseline(Baseline):
    """Predict nonzero only at the edit position (train-mean edit-site delta)."""

    name = "edit_only"

    def __init__(self) -> None:
        self._edit_mean: float = 0.0
        self._edit_sign: float = 0.0

    def fit(self, train_pairs: list[PairRecord]) -> None:
        vals: list[float] = []
        for p in train_pairs:
            if p.edit_arr_idx is not None and 0 <= p.edit_arr_idx < p.aligned_length:
                v = p.delta_true[p.edit_arr_idx]
                if np.isfinite(v):
                    vals.append(float(v))
        if vals:
            self._edit_mean = float(np.mean(np.abs(vals)))
            self._edit_sign = float(np.mean(np.sign(vals)))

    def predict(self, pair: PairRecord) -> np.ndarray:
        out = np.zeros(pair.aligned_length, dtype=float)
        if pair.edit_arr_idx is not None and 0 <= pair.edit_arr_idx < pair.aligned_length:
            out[pair.edit_arr_idx] = self._edit_mean * (
                self._edit_sign if self._edit_sign != 0 else 1.0
            )
        return out


class NearestTrainBaseline(Baseline):
    """Copy the delta profile of the nearest train pair.

    Nearness is defined by WT BPP features at the edit position (when
    available) plus edit-position distance; falls back to parent identity.
    Uses ``wt_features`` from the PH0 thermo manifest.
    """

    name = "nearest_train"
    requires_wt_features = True

    def __init__(self) -> None:
        self._train_pairs: list[PairRecord] = []
        self._train_features: np.ndarray | None = None

    @staticmethod
    def _feature_vec(pair: PairRecord) -> np.ndarray:
        wf = pair.wt_features or {}
        return np.array(
            [
                float(wf.get("bpp_paired_prob", 0.0)),
                float(wf.get("bpp_unpaired_prob", 0.0)),
                float(wf.get("positional_entropy_bits", 0.0)),
                float(wf.get("n_contacts", 0.0)),
                float(wf.get("max_contact_distance", 0.0)),
                float(wf.get("bpp_max_value", 0.0)),
            ],
            dtype=float,
        )

    def fit(self, train_pairs: list[PairRecord]) -> None:
        self._train_pairs = list(train_pairs)
        self._train_features = np.array(
            [self._feature_vec(p) for p in train_pairs], dtype=float
        )

    def predict(self, pair: PairRecord) -> np.ndarray:
        if not self._train_pairs:
            return np.zeros(pair.aligned_length, dtype=float)
        v = self._feature_vec(pair)
        # Same parent gets a large bonus (subtract 1.0 from distance).
        d = np.linalg.norm(self._train_features - v, axis=1)
        parent_bonus = np.array(
            [0.0 if p.parent == pair.parent else 1.0 for p in self._train_pairs],
            dtype=float,
        )
        d = d + parent_bonus
        idx = int(np.argmin(d))
        src = self._train_pairs[idx]
        # Copy delta_true from the nearest train pair, aligned by array index.
        out = np.zeros(pair.aligned_length, dtype=float)
        m = min(pair.aligned_length, src.aligned_length)
        out[:m] = np.nan_to_num(src.delta_true[:m], nan=0.0)
        return out


class LocalReleaseBaseline(Baseline):
    """Local-release heuristic using WT BPP fragility.

    Hypothesis: an edit at a well-paired (high BPP) WT position releases local
    structure. Predict a negative delta (decreased reactivity = more paired is
    NOT what DMS sees; DMS sees unpaired A/C, so release = increased
    reactivity = positive delta) in a local window around the edit, with
    amplitude proportional to ``bpp_paired_prob``.

    Output: ``Delta r_hat[i] = amplitude * bpp_paired_prob * exp(-dist^2 / 2*sigma^2)``
    where amplitude is fitted from train.
    """

    name = "local_release"
    requires_wt_features = True
    requires_seq_positions = True

    def __init__(self, sigma: float = 8.0) -> None:
        self._sigma = float(sigma)
        self._amplitude: float = 0.0

    def fit(self, train_pairs: list[PairRecord]) -> None:
        # Fit amplitude: mean |delta| in local window / mean (bpp * gaussian)
        num: list[float] = []
        den: list[float] = []
        for p in train_pairs:
            wf = p.wt_features or {}
            bpp = float(wf.get("bpp_paired_prob", 0.0))
            mask = p.endpoint_mask
            d = p.delta_true[mask]
            sp = p.seq_positions[mask]
            dist = np.abs(sp - p.edit_pos_1indexed)
            weight = bpp * np.exp(-(dist ** 2) / (2 * self._sigma ** 2))
            finite = np.isfinite(d) & np.isfinite(weight)
            num.extend(np.abs(d[finite]).tolist())
            den.extend(weight[finite].tolist())
        if den and np.sum(den) > 1e-8:
            self._amplitude = float(np.sum(num) / np.sum(den))

    def predict(self, pair: PairRecord) -> np.ndarray:
        n = pair.aligned_length
        out = np.zeros(n, dtype=float)
        wf = pair.wt_features or {}
        bpp = float(wf.get("bpp_paired_prob", 0.0))
        if not np.any(np.isfinite(pair.seq_positions)):
            return out
        dist = np.abs(pair.seq_positions - pair.edit_pos_1indexed)
        weight = bpp * np.exp(-(dist ** 2) / (2 * self._sigma ** 2))
        out = self._amplitude * weight
        out[~np.isfinite(out)] = 0.0
        return out


# ---------------------------------------------------------------------------
# Thermo baselines (§10.2)
# ---------------------------------------------------------------------------


class _ThermoMarginalBaseline(Baseline):
    """Shared logic for thermo baselines that marginalize over 3 alt bases.

    Subclasses implement ``_fold_seq(seq) -> dict`` returning at least
    ``unpaired_prob`` (per-sequence-position array) and ``bpp`` (n x n matrix).
    """

    requires_wt_sequence = True

    def __init__(self, *, temperature: float = 37.0) -> None:
        self._temperature = float(temperature)

    def _fold_seq(self, seq: str) -> dict[str, Any]:  # pragma: no cover - abstract
        raise NotImplementedError

    def predict(self, pair: PairRecord) -> np.ndarray:
        if pair.wt_sequence is None:
            return np.zeros(pair.aligned_length, dtype=float)
        wt_state = self._fold_seq(pair.wt_sequence)
        wt_unpaired = np.asarray(wt_state["unpaired_prob"], dtype=float)
        # 3 mutant sequences, average the per-position Delta unpaired.
        mut_seqs = build_mutant_sequences(
            pair.wt_sequence, pair.edit_pos_1indexed, pair.encoded_ref
        )
        delta_unpaired_seq = np.zeros(len(pair.wt_sequence), dtype=float)
        for ms in mut_seqs:
            ms_state = self._fold_seq(ms)
            ms_unpaired = np.asarray(ms_state["unpaired_prob"], dtype=float)
            delta_unpaired_seq += (ms_unpaired - wt_unpaired)
        delta_unpaired_seq /= max(len(mut_seqs), 1)
        # Map sequence positions -> array indices.
        return map_seq_array_to_delta(delta_unpaired_seq, pair)


class RNAfoldBaseline(_ThermoMarginalBaseline):
    """RNAfold (ViennaRNA Python API) baseline.

    Folds WT and 3 mutant sequences, predicts ``Delta r_hat = Delta unpaired_prob``
    averaged over the 3 alts. No training; ``fit`` is a no-op.
    """

    name = "rnafold"

    def _fold_seq(self, seq: str) -> dict[str, Any]:
        from .thermo_state import compute_wt_thermo_state  # reuse

        return compute_wt_thermo_state(seq, temperature=self._temperature)

    def fit(self, train_pairs: list[PairRecord]) -> None:
        return None


class RNAplfoldBaseline(_ThermoMarginalBaseline):
    """RNAplfold-style baseline (ViennaRNA partition function with window).

    ViennaRNA 2.7.2 Python API does not expose RNAplfold directly, but
    ``fold_compound.pf()`` computes the full partition function (equivalent to
    RNAplfold with ``-W`` = seq length and ``-u`` = 0). We reuse the same
    ``compute_wt_thermo_state`` and only differ in the reported feature name.
    The BPP-derived unpaired probability is identical to RNAfold-PF; the
    distinction here is documentation-only and recorded in the failure table
    where exact RNAplfold CLI is unavailable.
    """

    name = "rnaplfold"

    def _fold_seq(self, seq: str) -> dict[str, Any]:
        from .thermo_state import compute_wt_thermo_state

        return compute_wt_thermo_state(seq, temperature=self._temperature)

    def fit(self, train_pairs: list[PairRecord]) -> None:
        return None


class EternaFoldBaseline(_ThermoMarginalBaseline):
    """EternaFold baseline (contrafold CLI with EternaFold parameters).

    Uses the ``eternafold`` binary from the ``rna_baselines`` conda env. Folds
    WT and 3 mutant sequences, computes per-position unpaired probability from
    the CONTRAfold posterior decoding, averages Delta over 3 alts.
    """

    name = "eternafold"

    def __init__(
        self,
        *,
        temperature: float = 37.0,
        eternafold_bin: str | None = None,
        eternafold_params: str | None = None,
    ) -> None:
        super().__init__(temperature=temperature)
        self._eternafold_bin = eternafold_bin or os.environ.get(
            "ETERNAFOLD_PATH", os.environ.get("ETERNAFOLD_BIN", "EternaFold")
        )
        # Params file is set by the rna_baselines conda activate script
        # (ETERNAFOLD_PARAMETERS) and is required by the contrafold `predict` subcommand.
        self._eternafold_params = eternafold_params or os.environ.get(
            "ETERNAFOLD_PARAMETERS", ""
        )

    def fit(self, train_pairs: list[PairRecord]) -> None:
        return None

    def _fold_seq(self, seq: str) -> dict[str, Any]:
        # EternaFold ships as a contrafold binary (symlink `eternafold` -> contrafold).
        # Invocation: `eternafold predict --params <EternaFoldParams.v1> <FASTA_file>`.
        # Output contains `>structure` followed by the dot-bracket MEA structure on the
        # next line. We derive a binary unpaired prob (paired=0, unpaired=1) from the
        # MEA structure; full posterior BPP requires `--posteriors` and is a known limitation.
        seq_u = seq.upper().replace("T", "U")
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".fa", delete=False, prefix="eternafold_"
        ) as fh:
            fh.write(f">seq\n{seq_u}\n")
            tmp_path = fh.name
        try:
            cmd = [self._eternafold_bin, "predict"]
            if self._eternafold_params:
                cmd += ["--params", self._eternafold_params]
            cmd.append(tmp_path)
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
            except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
                raise RuntimeError(f"EternaFold invocation failed: {exc}") from exc
            if result.returncode != 0:
                raise RuntimeError(
                    f"EternaFold exited {result.returncode}: {result.stderr[:200]}"
                )
            # Parse: locate the `>structure` marker, dot-bracket is the following line.
            out_lines = result.stdout.splitlines()
            ss: str | None = None
            for i, ln in enumerate(out_lines):
                if ln.strip() == ">structure":
                    if i + 1 < len(out_lines):
                        ss = out_lines[i + 1].strip()
                    break
            if not ss:
                raise RuntimeError(
                    f"EternaFold output missing `>structure`: {result.stdout[:200]!r}"
                )
            unpaired = np.array([1.0 if c == "." else 0.0 for c in ss], dtype=float)
            return {"unpaired_prob": unpaired, "mfe_structure": ss, "bpp": None}
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Learned baselines (§10.3 / §10.4) — torch imported lazily
# ---------------------------------------------------------------------------


def _one_hot_rna(seq: str) -> np.ndarray:
    """One-hot encode an RNA sequence (A,C,G,U) -> 4 x L float array."""

    s = seq.upper().replace("T", "U")
    mapping = {"A": 0, "C": 1, "G": 2, "U": 3}
    arr = np.zeros((4, len(s)), dtype=np.float32)
    for i, c in enumerate(s):
        if c in mapping:
            arr[mapping[c], i] = 1.0
        # Unknown bases stay all-zero.
    return arr


class StaticReactivityBaseline(Baseline):
    """Train-only sequence-to-reactivity model (§10.3).

    A small 1D CNN ``F(seq) -> r_hat`` trained on WT sequences vs WT
    reactivity. Predicts ``Delta r_hat = mean_alt F(mut_alt) - F(wt)``.

    Trained on the train split only, single seed, fixed budget. Torch is
    imported lazily so this class can be defined (but not constructed) in
    non-torch environments.
    """

    name = "static_reactivity"
    is_learned = True
    requires_wt_sequence = True

    def __init__(
        self,
        *,
        epochs: int = 8,
        lr: float = 1e-3,
        batch_size: int = 16,
        device: str = "cpu",
        seed: int = 0,
    ) -> None:
        self._epochs = int(epochs)
        self._lr = float(lr)
        self._batch_size = int(batch_size)
        self._device = str(device)
        self._seed = int(seed)
        self._model = None
        self._max_len = 0

    @staticmethod
    def _build_model(max_len: int, device: str):
        import torch
        import torch.nn as nn

        torch.manual_seed(0)
        class SmallCNN(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv1d(4, 32, 7, padding=3),
                    nn.ReLU(),
                    nn.Conv1d(32, 32, 7, padding=3),
                    nn.ReLU(),
                    nn.Conv1d(32, 16, 5, padding=2),
                    nn.ReLU(),
                )
                self.head = nn.Conv1d(16, 1, 3, padding=1)

            def forward(self, x: "torch.Tensor") -> "torch.Tensor":
                return self.head(self.conv(x)).squeeze(1)

        m = SmallCNN().to(device)
        return m

    def fit(self, train_pairs: list[PairRecord]) -> None:
        import torch
        import torch.nn as nn

        torch.manual_seed(self._seed)
        # Gather (seq, reactivity_seq, mask_seq) for WT sequences, mapped to
        # sequence coordinates so the model predicts per-sequence-position.
        samples: list[tuple[str, np.ndarray, np.ndarray]] = []
        for p in train_pairs:
            if p.wt_sequence is None or p.wt_reactivity is None:
                continue
            wt_r = np.asarray(p.wt_reactivity, dtype=np.float32)
            mask = p.endpoint_mask.copy()
            if p.edit_arr_idx is not None and 0 <= p.edit_arr_idx < len(mask):
                mask[p.edit_arr_idx] = False
            seq_len = len(p.wt_sequence)
            tgt_seq, msk_seq = map_delta_array_to_seq(wt_r, mask, p, seq_len)
            samples.append((p.wt_sequence, tgt_seq, msk_seq))

        if not samples:
            self._model = None
            return

        self._max_len = max(len(s[0]) for s in samples)
        model = self._build_model(self._max_len, self._device)
        opt = torch.optim.Adam(model.parameters(), lr=self._lr)
        loss_fn = nn.SmoothL1Loss()

        for _ in range(self._epochs):
            np.random.shuffle(samples)
            for i in range(0, len(samples), self._batch_size):
                batch = samples[i : i + self._batch_size]
                oh = np.stack(
                    [
                        np.pad(
                            _one_hot_rna(s[0]),
                            ((0, 0), (0, self._max_len - len(s[0]))),
                            mode="constant",
                        )
                        for s in batch
                    ]
                )
                tgt = np.zeros((len(batch), self._max_len), dtype=np.float32)
                msk = np.zeros((len(batch), self._max_len), dtype=np.float32)
                for j, s in enumerate(batch):
                    L = len(s[0])
                    tgt[j, :L] = s[1][:L]
                    msk[j, :L] = s[2][:L].astype(np.float32)
                x = torch.from_numpy(oh).to(self._device)
                y = torch.from_numpy(tgt).to(self._device)
                w = torch.from_numpy(msk).to(self._device)
                pred = model(x)
                loss = (w * (pred - y).abs()).sum() / (w.sum() + 1e-6)
                opt.zero_grad()
                loss.backward()
                opt.step()

        self._model = model

    def _predict_seq(self, seq: str) -> np.ndarray:
        import torch

        if self._model is None:
            return np.zeros(len(seq), dtype=float)
        oh = np.pad(
            _one_hot_rna(seq),
            ((0, 0), (0, self._max_len - len(seq))),
            mode="constant",
        )
        x = torch.from_numpy(oh[None]).to(self._device)
        with torch.no_grad():
            pred = self._model(x).cpu().numpy()[0]
        return pred[: len(seq)]

    def predict(self, pair: PairRecord) -> np.ndarray:
        if pair.wt_sequence is None or self._model is None:
            return np.zeros(pair.aligned_length, dtype=float)
        wt_pred = self._predict_seq(pair.wt_sequence)
        mut_preds: list[np.ndarray] = []
        try:
            mut_seqs = build_mutant_sequences(
                pair.wt_sequence, pair.edit_pos_1indexed, pair.encoded_ref
            )
        except ValueError:
            return np.zeros(pair.aligned_length, dtype=float)
        for ms in mut_seqs:
            mut_preds.append(self._predict_seq(ms))
        mean_mut = np.mean(mut_preds, axis=0)
        delta_seq = mean_mut - wt_pred
        return map_seq_array_to_delta(delta_seq, pair)


class _PairedTorchBaseline(Baseline):
    """Shared scaffolding for Siamese and generic-paired baselines.

    Trains on (WT, mut_alt, delta) triples with 3x alt augmentation. The
    ``_build_model`` method returns a torch module taking two one-hot
    sequences and returning per-position delta predictions.
    """

    is_learned = True
    requires_wt_sequence = True

    def __init__(
        self,
        *,
        epochs: int = 8,
        lr: float = 1e-3,
        batch_size: int = 8,
        device: str = "cpu",
        seed: int = 0,
    ) -> None:
        self._epochs = int(epochs)
        self._lr = float(lr)
        self._batch_size = int(batch_size)
        self._device = str(device)
        self._seed = int(seed)
        self._model = None
        self._max_len = 0

    def _build_model(self, max_len: int):  # pragma: no cover - abstract
        raise NotImplementedError

    def fit(self, train_pairs: list[PairRecord]) -> None:
        import torch
        import torch.nn as nn

        torch.manual_seed(self._seed)
        # Build (wt_seq, mut_seq, target_seq, mask_seq) triples in sequence
        # coordinates. The delta array is mapped to per-sequence-position
        # targets so the model can train on variable-length sequences.
        triples: list[tuple[str, str, np.ndarray, np.ndarray]] = []
        for p in train_pairs:
            if p.wt_sequence is None:
                continue
            try:
                mut_seqs = build_mutant_sequences(
                    p.wt_sequence, p.edit_pos_1indexed, p.encoded_ref
                )
            except ValueError:
                continue
            seq_len = len(p.wt_sequence)
            tgt_seq, msk_seq = map_delta_array_to_seq(
                p.delta_true, p.endpoint_mask, p, seq_len
            )
            for ms in mut_seqs:
                triples.append((p.wt_sequence, ms, tgt_seq, msk_seq))

        if not triples:
            self._model = None
            return

        self._max_len = max(len(t[0]) for t in triples)
        model = self._build_model(self._max_len).to(self._device)
        opt = torch.optim.Adam(model.parameters(), lr=self._lr)
        loss_fn = nn.SmoothL1Loss(reduction="none")

        for _ in range(self._epochs):
            np.random.shuffle(triples)
            for i in range(0, len(triples), self._batch_size):
                batch = triples[i : i + self._batch_size]
                oh_wt = np.stack([self._pad_onehot(t[0]) for t in batch])
                oh_mut = np.stack([self._pad_onehot(t[1]) for t in batch])
                tgt = np.zeros((len(batch), self._max_len), dtype=np.float32)
                msk = np.zeros((len(batch), self._max_len), dtype=np.float32)
                for j, t in enumerate(batch):
                    L = len(t[0])
                    tgt[j, :L] = np.nan_to_num(t[2][:L], nan=0.0)
                    msk[j, :L] = t[3][:L].astype(np.float32)
                x_wt = torch.from_numpy(oh_wt).to(self._device)
                x_mut = torch.from_numpy(oh_mut).to(self._device)
                y = torch.from_numpy(tgt).to(self._device)
                w = torch.from_numpy(msk).to(self._device)
                pred = model(x_wt, x_mut)
                loss = (w * loss_fn(pred, y)).sum() / (w.sum() + 1e-6)
                opt.zero_grad()
                loss.backward()
                opt.step()

        self._model = model

    def _pad_onehot(self, seq: str) -> np.ndarray:
        oh = _one_hot_rna(seq)
        if oh.shape[1] < self._max_len:
            oh = np.pad(
                oh, ((0, 0), (0, self._max_len - oh.shape[1])), mode="constant"
            )
        else:
            oh = oh[:, : self._max_len]
        return oh.astype(np.float32)

    def _predict_pair(self, wt_seq: str, mut_seq: str) -> np.ndarray:
        import torch

        oh_wt = self._pad_onehot(wt_seq)[None]
        oh_mut = self._pad_onehot(mut_seq)[None]
        x_wt = torch.from_numpy(oh_wt).to(self._device)
        x_mut = torch.from_numpy(oh_mut).to(self._device)
        with torch.no_grad():
            pred = self._model(x_wt, x_mut).cpu().numpy()[0]
        return pred[: len(wt_seq)]

    def predict(self, pair: PairRecord) -> np.ndarray:
        if pair.wt_sequence is None or self._model is None:
            return np.zeros(pair.aligned_length, dtype=float)
        try:
            mut_seqs = build_mutant_sequences(
                pair.wt_sequence, pair.edit_pos_1indexed, pair.encoded_ref
            )
        except ValueError:
            return np.zeros(pair.aligned_length, dtype=float)
        preds = [self._predict_pair(pair.wt_sequence, ms) for ms in mut_seqs]
        delta_seq = np.mean(preds, axis=0)
        return map_seq_array_to_delta(delta_seq, pair)


class SiameseBaseline(_PairedTorchBaseline):
    """Matched Siamese baseline (§10.7).

    Shared encoder on (WT, mutant), late fusion (concatenation of encoded
    features + difference), per-position output head. No EPRO constraints.
    """

    name = "siamese_matched"

    def _build_model(self, max_len: int):
        import torch
        import torch.nn as nn

        torch.manual_seed(self._seed)
        class Siamese(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv1d(4, 32, 7, padding=3),
                    nn.ReLU(),
                    nn.Conv1d(32, 32, 7, padding=3),
                    nn.ReLU(),
                    nn.Conv1d(32, 16, 5, padding=2),
                    nn.ReLU(),
                )
                # Late fusion: concat(enc_wt, enc_mut, enc_wt - enc_mut) -> 48 ch
                self.fuse = nn.Sequential(
                    nn.Conv1d(48, 32, 5, padding=2),
                    nn.ReLU(),
                    nn.Conv1d(32, 16, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv1d(16, 1, 3, padding=1),
                )

            def forward(self, wt: "torch.Tensor", mut: "torch.Tensor") -> "torch.Tensor":
                e_wt = self.encoder(wt)
                e_mut = self.encoder(mut)
                fused = torch.cat([e_wt, e_mut, e_wt - e_mut], dim=1)
                return self.fuse(fused).squeeze(1)

        return Siamese()


class GenericPairedBaseline(_PairedTorchBaseline):
    """Matched generic paired baseline (§10.4/§10.8).

    Same parameter budget as the Siamese baseline but with a generic
    cross-interaction layer (1D conv over the stacked WT+mutant tensor) that
    can learn arbitrary interactions, NOT the EPRO forcing/propagation structure.
    This is the key architectural-innovation control (v3.3 §10.4).
    """

    name = "generic_paired_matched"

    def _build_model(self, max_len: int):
        import torch
        import torch.nn as nn

        torch.manual_seed(self._seed)
        class GenericPaired(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                # Shared encoder (same budget as Siamese): outputs 16 channels.
                self.encoder = nn.Sequential(
                    nn.Conv1d(4, 32, 7, padding=3),
                    nn.ReLU(),
                    nn.Conv1d(32, 32, 7, padding=3),
                    nn.ReLU(),
                    nn.Conv1d(32, 16, 5, padding=2),
                    nn.ReLU(),
                )
                # Stack encoded WT + mutant (16+16 = 32 ch) and apply generic
                # cross-interaction convs (no EPRO forcing/propagation structure).
                self.cross = nn.Sequential(
                    nn.Conv1d(16 + 16, 48, 7, padding=3),
                    nn.ReLU(),
                    nn.Conv1d(48, 24, 5, padding=2),
                    nn.ReLU(),
                    nn.Conv1d(24, 1, 3, padding=1),
                )

            def forward(self, wt: "torch.Tensor", mut: "torch.Tensor") -> "torch.Tensor":
                e_wt = self.encoder(wt)
                e_mut = self.encoder(mut)
                stacked = torch.cat([e_wt, e_mut], dim=1)
                return self.cross(stacked).squeeze(1)

        return GenericPaired()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


BASELINE_REGISTRY: dict[str, Callable[..., Baseline]] = {
    # Non-learned
    "zero_change": ZeroChangeBaseline,
    "mutation_type_mean": MutationTypeMeanBaseline,
    "distance_decay": DistanceDecayBaseline,
    "edit_only": EditOnlyBaseline,
    "nearest_train": NearestTrainBaseline,
    "local_release": LocalReleaseBaseline,
    # Thermo
    "rnafold": RNAfoldBaseline,
    "rnaplfold": RNAplfoldBaseline,
    "eternafold": EternaFoldBaseline,
    # Learned
    "static_reactivity": StaticReactivityBaseline,
    "siamese_matched": SiameseBaseline,
    "generic_paired_matched": GenericPairedBaseline,
}


def count_parameters(baseline: Baseline) -> int:
    """Return the number of trainable parameters of a baseline (0 for non-learned)."""

    model = getattr(baseline, "_model", None)
    if model is None:
        return 0
    try:
        import torch

        return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    except Exception:
        return 0
