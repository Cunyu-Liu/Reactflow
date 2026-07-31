"""ViennaRNA thermodynamic state wrapper for PH0 WT-only physics audit.

PH0 is a WT-structure-context identifiability audit. Because all 1509 true_pairs
carry ``encoded_alt="X"`` (M2-seq convention; mutant sequences are not
constructible), mutant thermo states and Δthermo are BLOCKED. This module
computes WT-only states:

  * MFE structure and energy (``RNA.fold_compound.mfe``)
  * Partition function ensemble free energy (``RNA.fold_compound.pf``)
  * Base-pair probability matrix (``RNA.fold_compound.bpp``)
  * Per-position unpaired probability (derived from BPP)
  * Per-position positional entropy (derived from BPP)

ViennaRNA 2.7.2 Python API is used (no CLI binary). RNA and numpy are imported
lazily so the package remains importable in stdlib-only environments.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any

THERMO_STATE_SCHEMA_VERSION = "reactflow-delta-ph0-thermo-state-v1"


def get_tool_version() -> dict[str, Any]:
    """Return tool provenance for ViennaRNA. Imports RNA lazily."""

    try:
        import RNA  # type: ignore
        version = RNA.__version__
    except Exception:
        version = "unavailable"
    return {
        "name": "ViennaRNA",
        "version": version,
        "api": "python module RNA",
        "install_method": "conda environment editflow311",
        "no_cli_binary": True,
    }


def _normalize_rna(seq: str) -> str:
    """Uppercase and convert DNA T to RNA U."""

    return seq.upper().replace("T", "U")


def compute_wt_thermo_state(seq: str, *, temperature: float = 37.0) -> dict[str, Any]:
    """Compute the WT thermodynamic state for an RNA sequence.

    Returns a dict with MFE structure/energy, PF energy, BPP matrix (0-indexed
    n×n), unpaired probability per position, and positional entropy per position.

    Raises ImportError if ViennaRNA is not installed.
    """

    import RNA  # type: ignore  # noqa: F401

    seq_rna = _normalize_rna(seq)
    n = len(seq_rna)
    if n == 0:
        raise ValueError("empty sequence")
    if not re.fullmatch(r"[ACGU]+", seq_rna):
        raise ValueError(f"sequence contains non-RNA characters: {seq_rna[:40]!r}")

    md = RNA.md()
    md.temperature = temperature
    fc = RNA.fold_compound(seq_rna, md)

    # MFE: returns (structure_string, energy_float)
    mfe_structure, mfe_energy = fc.mfe()

    # Partition function: returns (structure_string, pf_energy_float)
    pf_structure, pf_energy = fc.pf()

    # BPP: 1-indexed (n+1)×(n+1) tuple of tuples
    bpp_raw = fc.bpp()
    bpp: list[list[float]] = [[0.0] * n for _ in range(n)]
    for i in range(1, n + 1):
        row = bpp_raw[i]
        for j in range(1, n + 1):
            bpp[i - 1][j - 1] = float(row[j])

    # Unpaired probability: P_unpaired(i) = 1 - sum_j P(i pairs j)
    unpaired_prob: list[float] = []
    for i in range(n):
        paired = sum(bpp[i][j] for j in range(n) if j != i)
        unpaired_prob.append(max(0.0, 1.0 - paired))

    # Positional entropy in bits: S(i) = -sum p*log2(p) over all pairing
    # partners and the unpaired state.
    positional_entropy: list[float] = []
    for i in range(n):
        s = 0.0
        for j in range(n):
            if j == i:
                continue
            p = bpp[i][j]
            if p > 1e-15:
                s -= p * math.log2(p)
        pu = unpaired_prob[i]
        if pu > 1e-15:
            s -= pu * math.log2(pu)
        positional_entropy.append(s)

    return {
        "schema_version": THERMO_STATE_SCHEMA_VERSION,
        "sequence": seq_rna,
        "seq_sha256": hashlib.sha256(seq_rna.encode("ascii")).hexdigest(),
        "length": n,
        "mfe_structure": mfe_structure,
        "mfe_energy_kcal_mol": float(mfe_energy),
        "pf_structure": pf_structure,
        "pf_energy_kcal_mol": float(pf_energy),
        "bpp": bpp,
        "unpaired_prob": unpaired_prob,
        "positional_entropy_bits": positional_entropy,
        "params": {
            "temperature_celsius": temperature,
            "model": "ViennaRNA default (Turner 2004 energy parameters)",
            "unique_MFE": True,
        },
        "tool": get_tool_version(),
    }


def _find_mfe_partner(mfe_ss: str, pos_0: int) -> int | None:
    """Find the MFE base-pair partner of position ``pos_0`` (0-indexed).

    Handles standard dot-bracket notation with ``()`` only (no pseudoknots in
    MFE output for these sequences).
    """

    char = mfe_ss[pos_0]
    if char == ".":
        return None
    stack: list[int] = []
    if char == "(":
        for j in range(pos_0 + 1, len(mfe_ss)):
            if mfe_ss[j] == "(":
                stack.append(j)
            elif mfe_ss[j] == ")":
                if not stack:
                    return j  # 0-indexed partner
                stack.pop()
    elif char == ")":
        for j in range(pos_0 - 1, -1, -1):
            if mfe_ss[j] == ")":
                stack.append(j)
            elif mfe_ss[j] == "(":
                if not stack:
                    return j  # 0-indexed partner
                stack.pop()
    return None


def extract_position_features(
    state: dict[str, Any],
    pos_1indexed: int,
    *,
    contact_bpp_threshold: float = 0.05,
) -> dict[str, Any]:
    """Extract WT structural features at a given edit position.

    ``pos_1indexed`` is 1-indexed into the SEQUENCE (matching the registry's
    ``encoded_position_1indexed``).

    Returns MFE pairing, BPP-derived paired probability, max BPP partner,
    unpaired probability, positional entropy, and contact positions (positions
    with BPP > threshold to the edit position).
    """

    n = state["length"]
    pos_0 = pos_1indexed - 1
    if not (0 <= pos_0 < n):
        raise ValueError(f"position {pos_1indexed} out of range for length {n}")

    bpp = state["bpp"]

    # Paired probability at edit position
    paired_prob = sum(bpp[pos_0][j] for j in range(n) if j != pos_0)

    # Max BPP partner
    max_bpp = 0.0
    max_partner_0: int | None = None
    for j in range(n):
        if j != pos_0 and bpp[pos_0][j] > max_bpp:
            max_bpp = bpp[pos_0][j]
            max_partner_0 = j

    # MFE pairing
    mfe_ss = state["mfe_structure"]
    mfe_paired = mfe_ss[pos_0] not in (".",)
    mfe_partner_0 = _find_mfe_partner(mfe_ss, pos_0) if mfe_paired else None

    # Contact positions (BPP > threshold)
    contacts: list[dict[str, Any]] = []
    for j in range(n):
        if j != pos_0 and bpp[pos_0][j] > contact_bpp_threshold:
            contacts.append({"position_1indexed": j + 1, "bpp": float(bpp[pos_0][j])})
    contacts.sort(key=lambda c: c["bpp"], reverse=True)

    # Distance to furthest contact (sequence distance)
    max_contact_distance = max(
        abs(c["position_1indexed"] - pos_1indexed) for c in contacts
    ) if contacts else 0

    return {
        "edit_position_1indexed": pos_1indexed,
        "mfe_paired": bool(mfe_paired),
        "mfe_partner_1indexed": (mfe_partner_0 + 1) if mfe_partner_0 is not None else None,
        "bpp_paired_prob": float(paired_prob),
        "bpp_unpaired_prob": float(state["unpaired_prob"][pos_0]),
        "bpp_max_partner_1indexed": (max_partner_0 + 1) if max_partner_0 is not None else None,
        "bpp_max_value": float(max_bpp),
        "positional_entropy_bits": float(state["positional_entropy_bits"][pos_0]),
        "contact_positions": contacts,
        "n_contacts": len(contacts),
        "max_contact_distance": int(max_contact_distance),
        "contact_bpp_threshold": contact_bpp_threshold,
    }


# ---------------------------------------------------------------------------
# SEQPOS coordinate helpers (reusable, fixes seqpos_to_indices lowercase bug)
# ---------------------------------------------------------------------------

_SEQPOS_TOKEN_RE = re.compile(r"^[ACGUNXacgunx](.+)$")


def parse_seqpos_token(token: str) -> int | None:
    """Parse a SEQPOS token like ``g-18``, ``A1``, ``c14`` to its integer value.

    Strips a single leading nucleotide letter (case-insensitive) and parses the
    remainder as a signed integer. Returns ``None`` if parsing fails.
    """

    m = _SEQPOS_TOKEN_RE.match(token)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def seqpos_to_sequence_positions(seqpos_tokens: list[str], offset: int) -> list[int | None]:
    """Map SEQPOS tokens to 1-indexed SEQUENCE positions.

    Formula: ``sequence_position = token_value - offset``.

    This is a case-insensitive replacement for ``rdat.seqpos_to_indices`` which
    only handles uppercase tokens and cannot parse negative positions.
    """

    result: list[int | None] = []
    for token in seqpos_tokens:
        value = parse_seqpos_token(token)
        if value is None:
            result.append(None)
        else:
            result.append(value - offset)
    return result


def find_array_index_for_sequence_position(
    seqpos_tokens: list[str], offset: int, target_seq_pos: int
) -> int | None:
    """Find the delta-array index that corresponds to a 1-indexed SEQUENCE position.

    Returns the 0-based array index, or ``None`` if not found.
    """

    seq_positions = seqpos_to_sequence_positions(seqpos_tokens, offset)
    for i, sp in enumerate(seq_positions):
        if sp == target_seq_pos:
            return i
    return None
