"""Outcome-blind RiNALMo exact-SNV dependency features for Model Rescue v7."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

from scripts.reactflow_delta.model_rescue_v7_schema import (
    FEATURE_NAMES,
    LOG_ODDS_EPSILON,
    RINALMO_CONFIG_NAME,
    RNA_BASES,
    RNA_BASE_TO_INDEX,
)


def normalize_rna_sequence(sequence: str) -> str:
    value = str(sequence).upper().replace("T", "U")
    if not value or set(value) - set(RNA_BASES):
        raise ValueError("v7 accepts only non-empty A/C/G/U sequences")
    return value


def exact_mutant_sequence(
    wt_sequence: str, source_pos: int, ref: str, alt: str
) -> str:
    wt = normalize_rna_sequence(wt_sequence)
    ref = normalize_rna_sequence(ref)
    alt = normalize_rna_sequence(alt)
    if len(ref) != 1 or len(alt) != 1 or ref == alt:
        raise ValueError("v7 requires a one-base substitution with ref != alt")
    if not 0 <= source_pos < len(wt):
        raise ValueError("v7 mutation source is outside the full construct")
    if wt[source_pos] != ref:
        raise ValueError("v7 mutation reference disagrees with the WT sequence")
    return f"{wt[:source_pos]}{alt}{wt[source_pos + 1:]}"


def dependency_features_from_acgu_logits(
    wt_logits: np.ndarray | torch.Tensor,
    mutant_logits: np.ndarray | torch.Tensor,
    wt_sequence: str,
    source_pos: int,
    *,
    epsilon: float = LOG_ODDS_EPSILON,
) -> np.ndarray:
    """Compute the frozen six-channel directed dependency representation.

    Inputs contain only the four A/C/G/U logits for full-construct positions;
    CLS/EOS and all non-ACGU vocabulary entries must already be removed.
    """

    wt = torch.as_tensor(wt_logits, dtype=torch.float32)
    mutant = torch.as_tensor(mutant_logits, dtype=torch.float32)
    sequence = normalize_rna_sequence(wt_sequence)
    expected_shape = (len(sequence), len(RNA_BASES))
    if tuple(wt.shape) != expected_shape or tuple(mutant.shape) != expected_shape:
        raise ValueError(
            f"v7 ACGU logits must both have shape {expected_shape}, got "
            f"{tuple(wt.shape)} and {tuple(mutant.shape)}"
        )
    if not 0 <= source_pos < len(sequence):
        raise ValueError("v7 source position is outside the construct")
    if not torch.isfinite(wt).all() or not torch.isfinite(mutant).all():
        raise ValueError("v7 received non-finite RiNALMo logits")
    if not 0.0 < epsilon < 0.5:
        raise ValueError("v7 log-odds epsilon must lie strictly between zero and 0.5")

    wt_prob = torch.softmax(wt, dim=-1).clamp(min=epsilon, max=1.0 - epsilon)
    mutant_prob = torch.softmax(mutant, dim=-1).clamp(
        min=epsilon, max=1.0 - epsilon
    )
    log_two = torch.log(torch.tensor(2.0, dtype=torch.float32))
    wt_log_odds = (torch.log(wt_prob) - torch.log1p(-wt_prob)) / log_two
    mutant_log_odds = (torch.log(mutant_prob) - torch.log1p(-mutant_prob)) / log_two
    signed = mutant_log_odds - wt_log_odds
    receiver_indices = torch.tensor(
        [RNA_BASE_TO_INDEX[base] for base in sequence], dtype=torch.long
    )
    receiver_shift = signed.gather(1, receiver_indices[:, None]).squeeze(1)
    max_absolute = signed.abs().amax(dim=1)
    features = torch.cat(
        [signed, receiver_shift[:, None], max_absolute[:, None]], dim=1
    )
    features[source_pos] = 0.0
    result = features.cpu().numpy().astype(np.float32, copy=False)
    if result.shape != (len(sequence), len(FEATURE_NAMES)):
        raise RuntimeError("v7 dependency feature width drifted from the contract")
    if not np.isfinite(result).all():
        raise RuntimeError("v7 dependency features are non-finite")
    return result


class RiNALMoGigaLogitInferer:
    """Thin adapter around the frozen official RiNALMo-Giga implementation."""

    def __init__(
        self,
        *,
        code_root: Path,
        weights_path: Path,
        device: str,
        attention_backend: str = "flash",
    ) -> None:
        code_root = Path(code_root).resolve()
        weights_path = Path(weights_path).resolve()
        if not (code_root / "rinalmo" / "model" / "model.py").is_file():
            raise FileNotFoundError("official RiNALMo code root is incomplete")
        if not weights_path.is_file():
            raise FileNotFoundError("official RiNALMo-Giga weight file is absent")
        if attention_backend != "flash":
            raise ValueError("formal v7 inference requires official FlashAttention")
        if str(code_root) not in sys.path:
            sys.path.insert(0, str(code_root))

        from rinalmo.config import model_config
        from rinalmo.data.alphabet import Alphabet
        from rinalmo.model.model import RiNALMo

        config = model_config(RINALMO_CONFIG_NAME)
        config.model.transformer.use_flash_attn = True
        alphabet = Alphabet(**config["alphabet"])
        model = RiNALMo(config)
        state = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()

        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("formal V7M1 RiNALMo inference requires a CUDA device")
        self.dtype = torch.bfloat16
        self.model = model.to(device=self.device, dtype=self.dtype)
        self.alphabet = alphabet
        # Alphabet.encode maps U to the T token, so the fourth conceptual RNA
        # probability is read from the official T vocabulary entry.
        self.acgu_indices = tuple(
            alphabet.get_idx(token) for token in ("A", "C", "G", "T")
        )
        if len(set(self.acgu_indices)) != 4:
            raise RuntimeError("RiNALMo alphabet does not expose four distinct RNA bases")
        self.attention_backend = attention_backend

    def __call__(
        self, sequences: Sequence[str], *, batch_size: int
    ) -> dict[str, np.ndarray]:
        normalized = [normalize_rna_sequence(sequence) for sequence in sequences]
        if len(set(normalized)) != len(normalized):
            raise ValueError("v7 inferer expects a deduplicated sequence list")
        output: dict[str, np.ndarray] = {}
        for start in range(0, len(normalized), batch_size):
            batch = normalized[start : start + batch_size]
            tokens = torch.tensor(
                self.alphabet.batch_tokenize(batch),
                dtype=torch.int64,
                device=self.device,
            )
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=self.dtype
            ):
                logits = self.model(tokens)["logits"]
            acgu = logits[:, 1:-1, list(self.acgu_indices)].float().cpu().numpy()
            for sequence, values in zip(batch, acgu):
                if values.shape != (len(sequence), 4):
                    raise RuntimeError("RiNALMo output length disagrees with the input")
                output[sequence] = np.asarray(values, dtype=np.float32)
        if set(output) != set(normalized):
            raise RuntimeError("RiNALMo inference did not cover the requested universe")
        return output


LogitInferer = Callable[[Sequence[str]], dict[str, np.ndarray]]


def batched_infer(
    inferer: Any, sequences: Iterable[str], *, batch_size: int
) -> dict[str, np.ndarray]:
    ordered = sorted({normalize_rna_sequence(sequence) for sequence in sequences})
    if hasattr(inferer, "__call__"):
        try:
            return inferer(ordered, batch_size=batch_size)
        except TypeError:
            return inferer(ordered)
    raise TypeError("v7 logit inferer must be callable")
