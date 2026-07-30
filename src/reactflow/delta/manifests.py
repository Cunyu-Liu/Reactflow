"""Small, dependency-free provenance helpers used by ReactFlow-Delta stages."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Union


PathLike = Union[str, Path]


@dataclass(frozen=True)
class ContractFingerprint:
    """Immutable identity of a contract file recorded in an audit artifact."""

    path: str
    sha256: str
    size_bytes: int


def sha256_file(path: PathLike, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 of a regular file without loading it all into memory."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(file_path)

    digest = sha256()
    with file_path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint_contract(path: PathLike) -> ContractFingerprint:
    """Build the serializable contract fingerprint required by stage manifests."""

    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(file_path)

    return ContractFingerprint(
        path=str(file_path),
        sha256=sha256_file(file_path),
        size_bytes=file_path.stat().st_size,
    )
