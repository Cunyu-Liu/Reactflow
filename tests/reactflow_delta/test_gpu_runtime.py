from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from scripts.reactflow_delta import gpu_runtime


def test_gpu_runtime_rejects_explicit_cpu(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(RuntimeError, match="CUDA_REQUIRED.*is not CUDA"):
        gpu_runtime.require_cuda_device("cpu")


def test_gpu_runtime_rejects_unavailable_cuda(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(
        RuntimeError,
        match="CUDA_REQUIRED.*cuda_available=False",
    ):
        gpu_runtime.require_cuda_device("cuda:0")


def test_gpu_runtime_returns_and_records_the_allocated_cuda_device(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch,
        "empty",
        lambda *_args, **_kwargs: SimpleNamespace(device=torch.device("cuda:3")),
    )
    assert gpu_runtime.require_cuda_device("cuda:3") == "cuda:3"
    evidence = capsys.readouterr().out
    assert "CUDA_RUNTIME_OK" in evidence
    assert "requested_device='cuda:3'" in evidence
    assert "actual_device='cuda:3'" in evidence
