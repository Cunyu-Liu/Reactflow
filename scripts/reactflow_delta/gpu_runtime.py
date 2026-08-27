"""GPU-only runtime checks shared by active ReactFlow-Delta runners."""

from __future__ import annotations

import torch


def require_cuda_device(requested_device: str) -> str:
    """Resolve a real CUDA device or fail before a runner writes artifacts."""
    try:
        parsed = torch.device(requested_device)
    except (RuntimeError, ValueError, TypeError) as error:
        raise RuntimeError(
            f"CUDA_REQUIRED: invalid requested_device={requested_device!r}"
        ) from error
    if parsed.type != "cuda":
        raise RuntimeError(
            f"CUDA_REQUIRED: requested_device={requested_device!r} is not CUDA"
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA_REQUIRED: "
            f"requested_device={requested_device!r} "
            f"cuda_available={torch.cuda.is_available()} "
            f"torch_cuda={torch.version.cuda!r}"
        )
    try:
        probe = torch.empty(1, device=parsed)
    except Exception as error:
        raise RuntimeError(
            "CUDA_REQUIRED: failed to allocate on "
            f"requested_device={requested_device!r} "
            f"cuda_available={torch.cuda.is_available()} "
            f"torch_cuda={torch.version.cuda!r}"
        ) from error
    if probe.device.type != "cuda":
        raise RuntimeError(
            "CUDA_REQUIRED: device allocation did not produce a CUDA tensor; "
            f"actual_device={probe.device}"
        )
    print(
        "CUDA_RUNTIME_OK: "
        f"requested_device={requested_device!r} "
        f"actual_device={str(probe.device)!r} "
        f"torch_cuda={torch.version.cuda!r}",
        flush=True,
    )
    return str(probe.device)
