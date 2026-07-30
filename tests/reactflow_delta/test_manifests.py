from pathlib import Path

import pytest

from reactflow.delta import ContractFingerprint, fingerprint_contract, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[2]
V3_CONTRACT = PROJECT_ROOT / "docs" / "contracts" / "ReactFlow分阶段执行提示词_ReactFlowDelta科研合同_v3_EPRO_20260729.md"
V3_SHA256 = "3efcc1504208d8089236dfe4e7d41553741441d3b86b6174c8b5af52d614ec10"


def test_v3_contract_fingerprint_is_frozen() -> None:
    fingerprint = fingerprint_contract(V3_CONTRACT)

    assert fingerprint == ContractFingerprint(
        path=str(V3_CONTRACT),
        sha256=V3_SHA256,
        size_bytes=V3_CONTRACT.stat().st_size,
    )
    assert sha256_file(V3_CONTRACT) == V3_SHA256


def test_fingerprint_rejects_missing_contract(tmp_path: Path) -> None:
    missing_contract = tmp_path / "missing-contract.md"

    with pytest.raises(FileNotFoundError):
        fingerprint_contract(missing_contract)


def test_chunk_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        sha256_file(V3_CONTRACT, chunk_size=0)
