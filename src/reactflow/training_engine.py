"""Training engine with DDP/FSDP, bf16, gradient checkpointing, OOM retry, and NaN/Inf guard.

Spec reference: ReactFlow分阶段执行提示词.md lines 529-542 (C1-3 training engineering).

This module provides the production training loop used for the C1-3 full-scale
PairFormer training.  It implements the engineering checklist from the spec:

- DDP / FSDP process-group setup (:meth:`TrainingEngine.setup_distributed`).
- bf16 mixed precision via ``torch.autocast`` (CPU or CUDA).
- Gradient checkpointing (``model.gradient_checkpointing_enable``).
- FlashAttention toggle (config flag, applied by the model when supported).
- Sharded / exact-resume checkpoints (model + optimizer + scheduler + RNG).
- Validation selection + patience-based early stopping.
- Structured JSON-lines logging.
- OOM retry that halves the batch on ``torch.cuda.OutOfMemoryError``.
- Non-finite (NaN/Inf) guard on loss and gradients.

The loss function ``loss_fn(model_output, batch)`` may return either a scalar
``torch.Tensor`` or a ``dict`` with a ``"total"`` key (as produced by
:func:`reactflow.losses.pairformer_loss`).

Complexity
----------
- ``train_epoch``: ``O(N_batches * forward_backward_cost)``.
- ``evaluate``: ``O(N_val_batches * forward_cost)``.
- ``NonFiniteGuard.check_loss`` / ``check_grads``: ``O(1)`` and ``O(P)``
  respectively, where ``P`` is the number of model parameters.
- ``OOMRetryHandler.run``: ``O(retries * fn_cost)`` in the worst case.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn
from torch.utils.data import DataLoader

# ``torch.cuda.OutOfMemoryError`` exists on CPU-only builds too (it is just an
# exception class), but guard the access for very old torch versions.
try:  # pragma: no cover - dependent on torch build
    _CUDA_OOM_ERROR: type = torch.cuda.OutOfMemoryError
except AttributeError:  # pragma: no cover
    _CUDA_OOM_ERROR = RuntimeError


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class OOMBatchSkipped(Exception):
    """Raised by :class:`OOMRetryHandler` when a batch cannot be reduced further.

    The caller should catch this, log it, and continue to the next batch.
    """


# ---------------------------------------------------------------------------
# TrainingConfig
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Configuration for :class:`TrainingEngine`.

    Attributes:
        device: target device string (e.g. ``"cuda:0"``, ``"cpu"``).
        world_size: number of processes for distributed training (1 = single).
        use_ddp: wrap the model with ``DistributedDataParallel``.
        use_fsdp: wrap the model with ``FullyShardedDataParallel``.
        use_bf16: enable bf16 autocast.
        use_grad_checkpoint: call ``model.gradient_checkpointing_enable()``.
        use_flash_attention: hint for the model to use FlashAttention (applied
            by the model, not the engine).
        max_grad_norm: max gradient norm for clipping (``<= 0`` disables).
        oom_retry_max: max OOM retries per batch before skipping.
        nan_inf_guard: enable the NaN/Inf guard.
        log_interval: log every N steps.
        seed: base RNG seed.
        optimizer: optimizer name (currently only ``"adamw"``).
        lr: peak learning rate.
        weight_decay: AdamW weight decay (applied to decay params only).
        warmup_steps: linear warmup steps.
        cosine_decay_steps: total steps for cosine schedule (``0`` = no decay).
        checkpoint_dir: directory for checkpoints.
        resume_from: optional checkpoint path to resume from.
        save_every: save a checkpoint every N epochs.
        eval_every: run validation every N epochs.
        early_stop_patience: stop after N evaluations without improvement (``0``
            disables early stopping).
        early_stop_min_delta: minimum decrease to count as improvement.
    """

    # Device / distributed
    device: str = "cpu"
    world_size: int = 1
    use_ddp: bool = False
    use_fsdp: bool = False
    use_bf16: bool = False
    use_grad_checkpoint: bool = False
    use_flash_attention: bool = False

    # Numerics
    max_grad_norm: float = 1.0
    oom_retry_max: int = 3
    nan_inf_guard: bool = True
    log_interval: int = 50
    seed: int = 42

    # Optimizer / schedule
    optimizer: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    cosine_decay_steps: int = 0

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    resume_from: Optional[str] = None
    save_every: int = 1
    eval_every: int = 1

    # Early stopping
    early_stop_patience: int = 5
    early_stop_min_delta: float = 0.0

    def __post_init__(self) -> None:
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {self.warmup_steps}")
        if self.cosine_decay_steps < 0:
            raise ValueError(
                f"cosine_decay_steps must be >= 0, got {self.cosine_decay_steps}"
            )
        if self.oom_retry_max < 0:
            raise ValueError(f"oom_retry_max must be >= 0, got {self.oom_retry_max}")
        if self.optimizer != "adamw":
            raise ValueError(
                f"only 'adamw' optimizer is supported, got '{self.optimizer}'"
            )
        if self.use_ddp and self.use_fsdp:
            raise ValueError("use_ddp and use_fsdp are mutually exclusive")
        if self.world_size < 1:
            raise ValueError(f"world_size must be >= 1, got {self.world_size}")


# ---------------------------------------------------------------------------
# NonFiniteGuard
# ---------------------------------------------------------------------------


class NonFiniteGuard:
    """Context manager guarding forward/backward against NaN/Inf.

    Usage::

        with NonFiniteGuard(enabled=True, params=model.parameters()) as guard:
            loss = compute_loss(...)
            if guard.check_loss(loss):
                continue  # loss is non-finite
            loss.backward()
        if guard.skip_step:
            optimizer.zero_grad(set_to_none=True)
            continue

    On exit, if the loss was finite and backward ran, the guard inspects the
    gradients of ``params`` for NaN/Inf.  If either the loss or any gradient is
    non-finite, ``skip_step`` is set to ``True`` and a warning is logged so the
    caller can skip the optimizer step.

    Args:
        enabled: if ``False``, the guard is a no-op.
        params: iterable of parameters whose ``.grad`` is inspected after
            backward.  May be ``None`` to skip gradient checking.
        logger: optional :class:`logging.Logger` for skip warnings.

    Complexity: ``check_loss`` is ``O(1)`` (single reduction);
    ``check_grads`` is ``O(P)`` where ``P`` is the number of parameters.
    """

    def __init__(
        self,
        enabled: bool = True,
        params: Optional[Iterable[nn.Parameter]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.enabled = enabled
        self._params: List[nn.Parameter] = list(params) if params is not None else []
        self.logger = logger
        self.skip_step: bool = False
        self.loss: Optional[torch.Tensor] = None
        self._reason: Optional[str] = None

    def __enter__(self) -> "NonFiniteGuard":
        self.skip_step = False
        self.loss = None
        self._reason = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        # Propagate any real exception; do not swallow.
        if exc_type is not None:
            return False
        if not self.enabled:
            return False
        # If the loss itself was non-finite, backward never ran; grads are
        # stale/None, so just log and leave skip_step as-is.
        if not self.skip_step:
            self.check_grads()
        if self.skip_step:
            self._log_skip()
        return False

    def check_loss(self, loss: torch.Tensor) -> bool:
        """Check a loss tensor for NaN/Inf.

        Returns ``True`` if the loss is non-finite (and sets ``skip_step``).
        The caller should skip ``backward()`` in that case.

        Complexity: ``O(1)``.
        """
        self.loss = loss
        if not self.enabled:
            return False
        if not torch.isfinite(loss).all():
            self.skip_step = True
            try:
                value = float(loss.detach())
            except (ValueError, OverflowError):
                value = float("nan")
            self._reason = f"non-finite loss: {value}"
            return True
        return False

    def check_grads(self) -> bool:
        """Inspect parameter gradients for NaN/Inf.

        Returns ``True`` if any inspected gradient is non-finite (and sets
        ``skip_step``).  Parameters with ``None`` gradients are skipped.

        Complexity: ``O(P)`` over inspected parameters.
        """
        if not self.enabled or self.skip_step:
            return self.skip_step
        for i, p in enumerate(self._params):
            if p.grad is None:
                continue
            if not torch.isfinite(p.grad).all():
                self.skip_step = True
                self._reason = f"non-finite gradient at param index {i}"
                return True
        return False

    def _log_skip(self) -> None:
        msg = f"NonFiniteGuard skipping optimizer step: {self._reason}"
        if self.logger is not None:
            self.logger.warning(msg)
        else:  # pragma: no cover - fallback
            print(f"[WARN] {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# OOMRetryHandler
# ---------------------------------------------------------------------------


class OOMRetryHandler:
    """Handles CUDA OOM by halving the batch and retrying.

    Wraps a forward/backward callable.  On ``torch.cuda.OutOfMemoryError`` (or a
    ``RuntimeError`` whose message mentions "out of memory"), the handler
    reduces the batch to its first half and retries, up to ``max_retries``
    times.  If the batch cannot be reduced further (batch size 1), the batch is
    skipped by raising :class:`OOMBatchSkipped`.

    Args:
        max_retries: maximum number of OOM retries per batch.
        logger: optional logger for OOM events.

    Complexity: ``O((1 + max_retries) * fn_cost)`` worst case.
    """

    def __init__(
        self,
        max_retries: int = 3,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.max_retries = max_retries
        self.logger = logger
        self.oom_count: int = 0

    def run(self, fn: Callable[[Any], Any], batch: Any, device: Optional[str] = None) -> Any:
        """Execute ``fn(batch)`` with OOM retry.

        Args:
            fn: callable performing forward+backward on a (possibly reduced)
                batch and returning any result.
            batch: the input batch (tensor, dict, or tuple/list of tensors).
            device: optional device string for memory diagnostics.

        Returns:
            The return value of ``fn`` on the (possibly reduced) batch.

        Raises:
            OOMBatchSkipped: if the batch cannot be reduced further after OOM.
        """
        current = batch
        for attempt in range(self.max_retries + 1):
            try:
                return fn(current)
            except (_CUDA_OOM_ERROR, RuntimeError) as exc:  # noqa: PERF203
                if not self._is_oom(exc):
                    raise
                self.oom_count += 1
                self._log_oom(attempt, current, device, exc)
                reduced = self._reduce_batch(current)
                if reduced is None:
                    raise OOMBatchSkipped(
                        f"batch unreducible (size 1) after {attempt + 1} OOM(s)"
                    ) from exc
                current = reduced
        raise OOMBatchSkipped(
            f"exhausted {self.max_retries + 1} OOM retries"
        )

    @staticmethod
    def _is_oom(exc: BaseException) -> bool:
        """Return True if ``exc`` is a CUDA out-of-memory error.

        Complexity: ``O(len(message))``.
        """
        if isinstance(exc, _CUDA_OOM_ERROR):
            return True
        if isinstance(exc, RuntimeError):
            msg = str(exc).lower()
            return "out of memory" in msg or "cuda" in msg and "memory" in msg
        return False

    @staticmethod
    def _batch_size(batch: Any) -> int:
        """Return the leading-dimension size of ``batch``.

        Complexity: ``O(1)``.
        """
        if isinstance(batch, torch.Tensor):
            return batch.shape[0] if batch.dim() > 0 else 1
        if isinstance(batch, dict):
            for v in batch.values():
                if isinstance(v, torch.Tensor) and v.dim() > 0:
                    return v.shape[0]
            return 1
        if isinstance(batch, (list, tuple)):
            for v in batch:
                if isinstance(v, torch.Tensor) and v.dim() > 0:
                    return v.shape[0]
            return 1
        return 1

    @classmethod
    def _reduce_batch(cls, batch: Any) -> Optional[Any]:
        """Halve the batch along dimension 0.

        Returns ``None`` if the batch size is already 1 (cannot reduce).

        Complexity: ``O(batch_elements)``.
        """
        size = cls._batch_size(batch)
        if size <= 1:
            return None
        half = size // 2
        if half < 1:
            return None
        return cls._slice_batch(batch, half)

    @staticmethod
    def _slice_batch(batch: Any, count: int) -> Any:
        """Return the first ``count`` rows of ``batch``.

        Complexity: ``O(batch_elements)``.
        """
        if isinstance(batch, torch.Tensor):
            return batch[:count]
        if isinstance(batch, dict):
            return {
                k: (v[:count] if isinstance(v, torch.Tensor) and v.dim() > 0 else v)
                for k, v in batch.items()
            }
        if isinstance(batch, list):
            return [
                (v[:count] if isinstance(v, torch.Tensor) and v.dim() > 0 else v)
                for v in batch
            ]
        if isinstance(batch, tuple):
            return tuple(
                (v[:count] if isinstance(v, torch.Tensor) and v.dim() > 0 else v)
                for v in batch
            )
        return None

    def _log_oom(self, attempt: int, batch: Any, device: Optional[str], exc: BaseException) -> None:
        size = self._batch_size(batch)
        mem_info = ""
        if torch.cuda.is_available():
            try:
                allocated = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()
                mem_info = f" allocated={allocated} reserved={reserved}"
            except Exception:  # pragma: no cover
                mem_info = ""
        msg = (
            f"OOM on attempt {attempt + 1}/{self.max_retries + 1} "
            f"(batch_size={size}, device={device or 'n/a'}{mem_info}): {exc}"
        )
        if self.logger is not None:
            self.logger.warning(msg)
        else:  # pragma: no cover
            print(f"[WARN] {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# OptimizerFactory
# ---------------------------------------------------------------------------


class OptimizerFactory:
    """Factory for optimizers and learning-rate schedulers.

    Complexity: constructing the optimizer is ``O(P)`` over parameters; the
    scheduler is ``O(1)``.
    """

    @staticmethod
    def create_adamw(
        model: nn.Module,
        lr: float,
        weight_decay: float,
        warmup_steps: int,
        total_steps: int,
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
        """Create an AdamW optimizer with linear-warmup + cosine-decay schedule.

        Weight decay is applied only to parameters with ``ndim >= 2`` (matrices
        and embeddings); biases and 1-D parameters (LayerNorm scales) are
        excluded, following the standard transformer recipe.

        Args:
            model: the model to optimize.
            lr: peak learning rate.
            weight_decay: weight decay for decay params (0 for no-decay params).
            warmup_steps: linear warmup steps (``0`` = no warmup).
            total_steps: total optimizer steps for cosine decay (``0`` = no
                decay, constant LR after warmup).

        Returns:
            ``(optimizer, scheduler)`` where ``scheduler`` is a
            :class:`torch.optim.lr_scheduler.LambdaLR`.

        Complexity: ``O(P)``.
        """
        decay_params: List[nn.Parameter] = []
        no_decay_params: List[nn.Parameter] = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            # 1-D params (biases, LayerNorm) get no weight decay.
            if p.ndim <= 1 or name.endswith(".bias"):
                no_decay_params.append(p)
            else:
                decay_params.append(p)
        param_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(param_groups, lr=lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: OptimizerFactory._lr_lambda(
                step, warmup_steps, total_steps
            ),
        )
        return optimizer, scheduler

    @staticmethod
    def _lr_lambda(step: int, warmup_steps: int, total_steps: int) -> float:
        """Linear warmup then cosine decay to 0.

        - ``step < warmup_steps``: ``step / max(1, warmup_steps)``.
        - After warmup, if ``total_steps > warmup_steps``: cosine decay from
          1.0 to 0.0 over ``total_steps - warmup_steps``.
        - If ``total_steps <= warmup_steps``: constant 1.0 after warmup.

        Complexity: ``O(1)``.
        """
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        denom = total_steps - warmup_steps
        if denom <= 0:
            return 1.0
        progress = float(step - warmup_steps) / float(denom)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# TrainingEngine
# ---------------------------------------------------------------------------


class TrainingEngine:
    """Full training loop with AMP, grad clipping, OOM retry, and NaN guard.

    Args:
        model: the model to train (``nn.Module``).
        train_loader: training :class:`DataLoader`.
        val_loader: validation :class:`DataLoader` (may be ``None``).
        config: :class:`TrainingConfig`.
        loss_fn: callable ``loss_fn(model_output, batch) -> Tensor | dict``.  If
            a dict is returned, the ``"total"`` key is used as the scalar loss.
        device: optional device override (defaults to ``config.device``).  The
            model is moved to this device.

    The engine calls ``model(batch)`` for the forward pass and
    ``loss_fn(outputs, batch)`` for the loss, so the model is expected to accept
    whatever the loader yields (e.g. a dict or tensor).

    Complexity: see module docstring.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: TrainingConfig,
        loss_fn: Callable[[Any, Any], Union[torch.Tensor, Dict[str, torch.Tensor]]],
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.config = config
        self.loss_fn = loss_fn
        self.device = torch.device(device if device is not None else config.device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.logger = self._make_logger()

        # Optimizer / scheduler (created lazily so total_steps can be derived).
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None
        self._total_steps: int = 0

        # State.
        self.epoch: int = 0
        self.global_step: int = 0
        self._is_distributed: bool = False

        # Helpers.
        self.oom_handler = OOMRetryHandler(
            max_retries=config.oom_retry_max, logger=self.logger
        )

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup_optimizer(self, total_steps: Optional[int] = None) -> None:
        """Create the AdamW optimizer and LR scheduler.

        Args:
            total_steps: total optimizer steps for cosine decay.  If ``None``,
                defaults to ``config.cosine_decay_steps``.

        Complexity: ``O(P)``.
        """
        if total_steps is None:
            total_steps = self.config.cosine_decay_steps
        self._total_steps = total_steps
        self.optimizer, self.scheduler = OptimizerFactory.create_adamw(
            self.model,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            total_steps=total_steps,
        )

    def setup_distributed(self) -> None:
        """Set up the DDP / FSDP process group and wrap the model.

        Reads ``RANK`` / ``WORLD_SIZE`` / ``LOCAL_RANK`` from the environment.
        When ``world_size <= 1`` or neither DDP nor FSDP is enabled, this is a
        no-op (single-process training).

        Uses the ``nccl`` backend when CUDA is available, else ``gloo``.

        Complexity: ``O(1)`` (process-group init is constant in framework code).
        """
        if self.config.world_size <= 1:
            return
        if not (self.config.use_ddp or self.config.use_fsdp):
            return
        if not torch.distributed.is_available():  # pragma: no cover
            raise RuntimeError("torch.distributed is not available in this build")

        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", str(self.config.world_size)))

        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend=backend, rank=rank, world_size=world_size,
            )
        self._is_distributed = True

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            self.device = torch.device(f"cuda:{local_rank}")
            self.model = self.model.to(self.device)

        if self.config.use_fsdp:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            self.model = FSDP(
                self.model,
                device_id=local_rank if torch.cuda.is_available() else None,
                use_orig_params=True,
            )
        elif self.config.use_ddp:
            from torch.nn.parallel import DistributedDataParallel as DDP

            self.model = DDP(
                self.model,
                device_ids=[local_rank] if torch.cuda.is_available() else None,
                find_unused_parameters=True,
            )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one training epoch.

        Per batch:
        1. Move to device.
        2. Forward + backward under OOM retry and the non-finite guard.
        3. If the guard or OOM handler signals a skip, skip the optimizer step.
        4. Clip gradients to ``max_grad_norm``.
        5. ``optimizer.step()`` + ``scheduler.step()``.

        Args:
            epoch: the epoch index (0-based).

        Returns:
            Dict with averaged metrics: ``loss``, ``lr``, ``skipped_nan``,
            ``skipped_oom``, ``num_batches``.

        Complexity: ``O(N * forward_backward_cost)`` where ``N`` is the number
        of training batches.
        """
        if self.optimizer is None:
            self.setup_optimizer()
        self.model.train()
        self._maybe_enable_grad_checkpoint()
        self.epoch = epoch

        running_loss = 0.0
        num_batches = 0
        skipped_nan = 0
        skipped_oom = 0

        for step, batch in enumerate(self.train_loader):
            batch = self._to_device(batch)

            def forward_backward(sub_batch: Any) -> Tuple[torch.Tensor, bool]:
                assert self.optimizer is not None
                self.optimizer.zero_grad(set_to_none=True)
                with self._autocast():
                    outputs = self.model(sub_batch)
                    loss_out = self.loss_fn(outputs, sub_batch)
                    loss = self._extract_loss(loss_out)
                guard = NonFiniteGuard(
                    enabled=self.config.nan_inf_guard,
                    params=list(self.model.parameters()),
                    logger=self.logger,
                )
                with guard:
                    if guard.check_loss(loss):
                        return loss.detach(), True
                    loss.backward()
                return loss.detach(), guard.skip_step

            try:
                loss_tensor, skip_step = self.oom_handler.run(
                    forward_backward, batch, device=str(self.device)
                )
            except OOMBatchSkipped:
                skipped_oom += 1
                continue

            if skip_step:
                skipped_nan += 1
                # Clear any stale/poisoned gradients before the next batch.
                assert self.optimizer is not None
                self.optimizer.zero_grad(set_to_none=True)
            else:
                self._clip_grads()
                assert self.optimizer is not None
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.global_step += 1

            running_loss += float(loss_tensor)
            num_batches += 1

            if step % max(1, self.config.log_interval) == 0:
                self._log_metric(
                    "train_step",
                    {
                        "epoch": epoch,
                        "step": step,
                        "global_step": self.global_step,
                        "loss": float(loss_tensor),
                        "lr": self._current_lr(),
                        "skip_nan": skip_step,
                    },
                )

        avg_loss = running_loss / max(1, num_batches)
        metrics = {
            "loss": avg_loss,
            "lr": self._current_lr(),
            "skipped_nan": float(skipped_nan),
            "skipped_oom": float(skipped_oom),
            "num_batches": float(num_batches),
        }
        self._log_metric("train_epoch", {"epoch": epoch, **metrics})
        return metrics

    def evaluate(self) -> Dict[str, float]:
        """Run validation evaluation (no grad, no OOM retry).

        Returns:
            Dict with ``loss`` (mean over val batches) and ``num_batches``.

        Complexity: ``O(N_val * forward_cost)``.
        """
        if self.val_loader is None:
            return {"loss": float("nan"), "num_batches": 0.0}
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        with torch.no_grad():
            for batch in self.val_loader:
                batch = self._to_device(batch)
                with self._autocast():
                    outputs = self.model(batch)
                    loss_out = self.loss_fn(outputs, batch)
                    loss = self._extract_loss(loss_out)
                total_loss += float(loss.detach())
                num_batches += 1
        avg_loss = total_loss / max(1, num_batches)
        metrics = {"loss": avg_loss, "num_batches": float(num_batches)}
        self._log_metric("eval", {"epoch": self.epoch, **metrics})
        return metrics

    def train(self, num_epochs: int) -> Dict[str, List[float]]:
        """Full training loop with validation selection and early stopping.

        Resumes from ``config.resume_from`` if set.  Saves a checkpoint every
        ``config.save_every`` epochs and keeps a ``best.pt`` at the lowest
        validation loss.

        Args:
            num_epochs: total number of epochs to train.

        Returns:
            Dict with ``train_loss`` and ``val_loss`` history lists.

        Complexity: ``O(num_epochs * train_epoch_cost + eval_cost)``.
        """
        if self.optimizer is None:
            self.setup_optimizer()
        if self.config.resume_from is not None and self.global_step == 0:
            self.load_checkpoint(self.config.resume_from)

        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        val_history: List[float] = []
        best_val = float("inf")
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(self.epoch, num_epochs):
            train_metrics = self.train_epoch(epoch)
            history["train_loss"].append(train_metrics["loss"])

            if self.val_loader is not None and (epoch % max(1, self.config.eval_every) == 0):
                val_metrics = self.evaluate()
                val_loss = val_metrics["loss"]
                val_history.append(val_loss)
                history["val_loss"].append(val_loss)

                if val_loss < best_val - self.config.early_stop_min_delta:
                    best_val = val_loss
                    self.save_checkpoint(checkpoint_dir / "best.pt")

                if self._should_stop_early(val_history):
                    self._log_metric(
                        "early_stop",
                        {"epoch": epoch, "best_val": best_val, "patience": self.config.early_stop_patience},
                    )
                    break

            if epoch % max(1, self.config.save_every) == 0:
                self.save_checkpoint(checkpoint_dir / "latest.pt")

        return history

    def _should_stop_early(self, val_history: Sequence[float]) -> bool:
        """Patience-based early stopping.

        Returns ``True`` if the most recent improvement was more than
        ``early_stop_patience`` evaluations ago.

        Complexity: ``O(len(val_history))``.
        """
        patience = self.config.early_stop_patience
        if patience <= 0 or len(val_history) <= patience:
            return False
        best = val_history[0]
        last_improve = 0
        for i, v in enumerate(val_history[1:], start=1):
            if v < best - self.config.early_stop_min_delta:
                best = v
                last_improve = i
        return (len(val_history) - 1 - last_improve) >= patience

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """Save model, optimizer, scheduler, epoch, step, and RNG state.

        For DDP/FSDP-wrapped models, the underlying module is unwrapped so the
        checkpoint is portable. For FSDP, only rank 0 saves (other ranks get
        an empty state dict from ``_model_state_dict`` and skip saving).

        Complexity: ``O(model_size)``.
        """
        # For distributed training, only rank 0 saves the checkpoint
        is_distributed = self.config.use_ddp or self.config.use_fsdp
        if is_distributed and torch.distributed.is_initialized():
            if torch.distributed.get_rank() != 0:
                return  # Non-rank0 processes skip saving

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rng_state: Dict[str, Any] = {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
        }
        try:
            import numpy as np

            rng_state["numpy"] = np.random.get_state()
        except ImportError:  # pragma: no cover
            pass
        if torch.cuda.is_available():
            rng_state["cuda"] = torch.cuda.get_rng_state_all()

        payload = {
            "model_state_dict": self._model_state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "epoch": self.epoch,
            "global_step": self.global_step,
            "rng_state": rng_state,
            "config": self.config.__dict__,
        }
        torch.save(payload, path)

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """Restore model, optimizer, scheduler, epoch, step, and RNG state.

        Provides exact resume: the RNG states are restored so subsequent
        sampling is identical to a non-interrupted run.

        Complexity: ``O(model_size)``.
        """
        payload = torch.load(Path(path), map_location=str(self.device), weights_only=False)
        self._load_model_state(payload["model_state_dict"])
        if self.optimizer is not None and payload.get("optimizer_state_dict") is not None:
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        if self.scheduler is not None and payload.get("scheduler_state_dict") is not None:
            self.scheduler.load_state_dict(payload["scheduler_state_dict"])
        self.epoch = payload["epoch"]
        self.global_step = payload["global_step"]

        rng_state = payload.get("rng_state", {})
        if "python" in rng_state:
            random.setstate(rng_state["python"])
        if "torch" in rng_state:
            # RNG state must be a CPU ByteTensor; map_location may have moved it.
            ts = rng_state["torch"]
            if isinstance(ts, torch.Tensor):
                ts = ts.cpu().to(torch.uint8)
            torch.set_rng_state(ts)
        if "numpy" in rng_state:
            try:
                import numpy as np

                np.random.set_state(rng_state["numpy"])
            except ImportError:  # pragma: no cover
                pass
        if "cuda" in rng_state and torch.cuda.is_available():
            cs = rng_state["cuda"]
            if isinstance(cs, list):
                cs = [c.cpu().to(torch.uint8) if isinstance(c, torch.Tensor) else c for c in cs]
            torch.cuda.set_rng_state_all(cs)

    def _model_state_dict(self) -> Dict[str, torch.Tensor]:
        """Return the state dict, unwrapping DDP/FSDP if present.

        For FSDP, uses ``FULL_STATE_DICT`` to gather all parameters on rank 0
        (other ranks get an empty dict). For DDP, unwraps ``.module``.

        Complexity: ``O(model_size)``.
        """
        model = self.model

        # Check for FSDP first (isinstance is more reliable than hasattr)
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import StateDictType, FullStateDictConfig

            if isinstance(model, FSDP):
                full_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                with FSDP.state_dict_type(
                    model, StateDictType.FULL_STATE_DICT, full_config
                ):
                    return model.state_dict()
        except ImportError:
            pass

        # DDP wrap
        if hasattr(model, "module"):
            return model.module.state_dict()
        return model.state_dict()

    def _load_model_state(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load a state dict into the (possibly unwrapped) model.

        Complexity: ``O(model_size)``.
        """
        model = self.model
        if hasattr(model, "module"):
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _maybe_enable_grad_checkpoint(self) -> None:
        """Enable gradient checkpointing if configured and supported.

        Complexity: ``O(1)``.
        """
        if not self.config.use_grad_checkpoint:
            return
        method = getattr(self.model, "gradient_checkpointing_enable", None)
        if callable(method):
            method()

    def _autocast(self):
        """Return an autocast context manager (bf16 if enabled, else no-op).

        On CPU-only builds with ``use_bf16``, CPU bf16 autocast is used.

        Complexity: ``O(1)``.
        """
        if self.config.use_bf16:
            device_type = "cuda" if self.device.type == "cuda" else "cpu"
            return torch.autocast(device_type=device_type, dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _clip_grads(self) -> None:
        """Clip gradients to ``config.max_grad_norm`` (if > 0).

        Complexity: ``O(P)``.
        """
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )

    def _current_lr(self) -> float:
        """Return the current learning rate from the scheduler/optimizer.

        Complexity: ``O(1)``.
        """
        if self.scheduler is not None:
            return float(self.scheduler.get_last_lr()[0])
        if self.optimizer is not None:
            return float(self.optimizer.param_groups[0]["lr"])
        return self.config.lr

    def _to_device(self, batch: Any) -> Any:
        """Move a batch (tensor / dict / tuple) to ``self.device``.

        Complexity: ``O(batch_elements)``.
        """
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        if isinstance(batch, dict):
            return {
                k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
        if isinstance(batch, (list, tuple)):
            moved = [
                (v.to(self.device) if isinstance(v, torch.Tensor) else v) for v in batch
            ]
            return type(batch)(moved)
        return batch

    @staticmethod
    def _extract_loss(
        loss_out: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """Extract a scalar loss from ``loss_fn`` output.

        Accepts a scalar tensor or a dict with a ``"total"`` key.

        Complexity: ``O(1)``.
        """
        if isinstance(loss_out, dict):
            return loss_out["total"]
        return loss_out

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _make_logger(self) -> logging.Logger:
        """Create a JSON-lines logger writing to stderr.

        Complexity: ``O(1)``.
        """
        logger = logging.getLogger(f"reactflow.training.{id(self)}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(_JsonFormatter())
            logger.addHandler(handler)
        return logger

    def _log_metric(self, event: str, payload: Dict[str, Any]) -> None:
        """Emit a structured JSON-lines log record.

        Complexity: ``O(1)``.
        """
        record_payload = {"event": event, **payload}
        self.logger.info(json.dumps(record_payload, default=str))


class _JsonFormatter(logging.Formatter):
    """Logging formatter that emits each record as a JSON object on one line.

    Complexity: ``O(len(record))``.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": record.created,
            "level": record.levelname,
            "logger": record.name,
        }
        message = record.getMessage()
        # If the message is itself JSON (from _log_metric), parse and merge.
        try:
            parsed = json.loads(message)
            if isinstance(parsed, dict):
                payload.update(parsed)
            else:
                payload["msg"] = message
        except (json.JSONDecodeError, TypeError):
            payload["msg"] = message
        return json.dumps(payload, default=str)
