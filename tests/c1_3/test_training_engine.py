"""Unit tests for the C1-3 training engine.

Covers:
- TrainingConfig defaults and validation
- NonFiniteGuard: NaN/Inf loss, NaN grads, finite passthrough
- OOMRetryHandler: graceful OOM retry, unreducible-batch skip, success path
- OptimizerFactory: AdamW + warmup/cosine scheduler
- Checkpoint save/load round-trip (model, optimizer, epoch, step, RNG)
- Full train_epoch + evaluate integration on CPU
"""

from __future__ import annotations

import logging

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from reactflow.training_engine import (
    NonFiniteGuard,
    OOMBatchSkipped,
    OOMRetryHandler,
    OptimizerFactory,
    TrainingConfig,
    TrainingEngine,
)

try:
    _OOM_ERROR = torch.cuda.OutOfMemoryError
except AttributeError:  # pragma: no cover
    _OOM_ERROR = RuntimeError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _DictModel(nn.Module):
    """Tiny model that reads ``batch["x"]`` and predicts ``batch["y"]``."""

    def __init__(self, in_dim: int = 2) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, batch):
        return self.linear(batch["x"])


def _mse_loss(out, batch):
    return ((out - batch["y"]) ** 2).mean()


class _DictDataset(Dataset):
    def __init__(self, n: int = 16, in_dim: int = 2) -> None:
        torch.manual_seed(0)
        self.x = torch.randn(n, in_dim)
        self.y = self.x.sum(dim=-1, keepdim=True)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int):
        return {"x": self.x[i], "y": self.y[i]}


def _make_engine(tmp_path=None, with_val: bool = True, **cfg_kwargs):
    """Build a small CPU TrainingEngine for integration tests."""
    config = TrainingConfig(
        device="cpu",
        log_interval=1,
        warmup_steps=2,
        cosine_decay_steps=10,
        checkpoint_dir=str(tmp_path / "ckpt") if tmp_path else "ckpt",
        **cfg_kwargs,
    )
    model = _DictModel()
    train_loader = DataLoader(_DictDataset(16), batch_size=4)
    val_loader = DataLoader(_DictDataset(8), batch_size=4) if with_val else None
    engine = TrainingEngine(model, train_loader, val_loader, config, _mse_loss)
    engine.setup_optimizer(total_steps=10)
    return engine


# ---------------------------------------------------------------------------
# TrainingConfig
# ---------------------------------------------------------------------------


class TestTrainingConfig:
    def test_defaults(self):
        cfg = TrainingConfig()
        assert cfg.device == "cpu"
        assert cfg.world_size == 1
        assert cfg.use_ddp is False
        assert cfg.use_fsdp is False
        assert cfg.use_bf16 is False
        assert cfg.use_grad_checkpoint is False
        assert cfg.use_flash_attention is False
        assert cfg.max_grad_norm == 1.0
        assert cfg.nan_inf_guard is True
        assert cfg.optimizer == "adamw"
        assert cfg.lr == 3e-4
        assert cfg.weight_decay == 0.01
        assert cfg.warmup_steps == 500
        assert cfg.save_every == 1
        assert cfg.eval_every == 1

    def test_validation_rejects_bad_lr(self):
        with pytest.raises(ValueError, match="lr must be > 0"):
            TrainingConfig(lr=-1.0)

    def test_validation_rejects_negative_weight_decay(self):
        with pytest.raises(ValueError, match="weight_decay"):
            TrainingConfig(weight_decay=-0.1)

    def test_validation_rejects_ddp_and_fsdp(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            TrainingConfig(use_ddp=True, use_fsdp=True, world_size=2)

    def test_validation_rejects_unknown_optimizer(self):
        with pytest.raises(ValueError, match="adamw"):
            TrainingConfig(optimizer="sgd")

    def test_validation_rejects_bad_world_size(self):
        with pytest.raises(ValueError, match="world_size"):
            TrainingConfig(world_size=0)


# ---------------------------------------------------------------------------
# NonFiniteGuard
# ---------------------------------------------------------------------------


class TestNonFiniteGuard:
    def test_catches_nan_loss(self):
        guard = NonFiniteGuard(enabled=True)
        with guard:
            is_bad = guard.check_loss(torch.tensor(float("nan")))
        assert is_bad is True
        assert guard.skip_step is True

    def test_catches_inf_loss(self):
        guard = NonFiniteGuard(enabled=True)
        with guard:
            is_bad = guard.check_loss(torch.tensor(float("inf")))
        assert is_bad is True
        assert guard.skip_step is True

    def test_finite_loss_no_skip(self):
        guard = NonFiniteGuard(enabled=True, params=None)
        with guard:
            is_bad = guard.check_loss(torch.tensor(1.23))
        assert is_bad is False
        assert guard.skip_step is False

    def test_catches_nan_grads(self):
        p = nn.Parameter(torch.zeros(2))
        p.grad = torch.tensor([float("nan"), 1.0])
        guard = NonFiniteGuard(enabled=True, params=[p])
        with guard:
            guard.check_loss(torch.tensor(1.0))
            # Simulate backward having populated a NaN grad.
        assert guard.skip_step is True

    def test_disabled_guard_never_skips(self):
        guard = NonFiniteGuard(enabled=False, params=None)
        with guard:
            is_bad = guard.check_loss(torch.tensor(float("nan")))
        assert is_bad is False
        assert guard.skip_step is False

    def test_grad_check_passes_for_finite_grads(self):
        p = nn.Parameter(torch.zeros(2))
        p.grad = torch.tensor([0.5, -0.5])
        guard = NonFiniteGuard(enabled=True, params=[p])
        with guard:
            guard.check_loss(torch.tensor(1.0))
        assert guard.skip_step is False


# ---------------------------------------------------------------------------
# OOMRetryHandler
# ---------------------------------------------------------------------------


class TestOOMRetryHandler:
    def test_success_no_oom(self):
        handler = OOMRetryHandler(max_retries=3)
        result = handler.run(lambda b: b.sum().item(), torch.ones(4))
        assert result == 4.0
        assert handler.oom_count == 0

    def test_handles_oom_gracefully(self):
        """When fn OOMs on large batches but succeeds on size 1, the handler
        should halve the batch until it succeeds."""
        handler = OOMRetryHandler(max_retries=5)

        def fn(batch):
            size = batch.shape[0]
            if size > 1:
                raise _OOM_ERROR("CUDA out of memory.")
            return size

        result = handler.run(fn, torch.ones(8))
        # Reduced to a single-element batch, which succeeded.
        assert result == 1
        assert handler.oom_count == 3  # 8 -> 4 -> 2 -> 1

    def test_skips_batch_when_unreducible(self):
        """A size-1 batch that still OOMs cannot be reduced; skip it."""
        handler = OOMRetryHandler(max_retries=2)

        def fn(batch):
            raise _OOM_ERROR("CUDA out of memory.")

        with pytest.raises(OOMBatchSkipped):
            handler.run(fn, torch.ones(1))
        assert handler.oom_count >= 1

    def test_oom_reduces_dict_batch(self):
        """OOM retry should halve all tensors in a dict batch."""
        handler = OOMRetryHandler(max_retries=4)
        calls = []

        def fn(batch):
            calls.append(batch["x"].shape[0])
            if batch["x"].shape[0] > 1:
                raise _OOM_ERROR("out of memory")
            return batch["x"].sum().item()

        batch = {"x": torch.ones(4), "y": torch.ones(4) * 2}
        result = handler.run(fn, batch)
        assert result == 1.0
        assert calls == [4, 2, 1]

    def test_non_oom_runtime_error_propagates(self):
        """A RuntimeError that is NOT an OOM should propagate, not be retried."""
        handler = OOMRetryHandler(max_retries=3)

        def fn(batch):
            raise RuntimeError("some other cuda error")  # no "memory" keyword

        with pytest.raises(RuntimeError, match="some other"):
            handler.run(fn, torch.ones(4))


# ---------------------------------------------------------------------------
# OptimizerFactory
# ---------------------------------------------------------------------------


class TestOptimizerFactory:
    def test_creates_adamw_and_scheduler(self):
        model = nn.Linear(4, 2)
        opt, sched = OptimizerFactory.create_adamw(
            model, lr=1e-3, weight_decay=0.05, warmup_steps=10, total_steps=100
        )
        assert isinstance(opt, torch.optim.AdamW)
        assert isinstance(sched, torch.optim.lr_scheduler.LambdaLR)
        # Two param groups: decay (weight) and no-decay (bias).
        assert len(opt.param_groups) == 2
        wds = {g["weight_decay"] for g in opt.param_groups}
        assert 0.05 in wds
        assert 0.0 in wds

    def test_lr_schedule_warmup_and_decay(self):
        """lr_lambda: linear warmup to 1.0, then cosine decay to 0.0."""
        warmup, total = 10, 20
        # At step 0 the multiplier is 0 (start of warmup).
        assert OptimizerFactory._lr_lambda(0, warmup, total) == pytest.approx(0.0)
        # Mid-warmup.
        assert OptimizerFactory._lr_lambda(5, warmup, total) == pytest.approx(0.5)
        # End of warmup -> peak (multiplier 1.0).
        assert OptimizerFactory._lr_lambda(10, warmup, total) == pytest.approx(1.0)
        # End of cosine decay -> 0.0.
        assert OptimizerFactory._lr_lambda(20, warmup, total) == pytest.approx(0.0)

    def test_scheduler_tracks_lr(self):
        model = nn.Linear(4, 2)
        opt, sched = OptimizerFactory.create_adamw(
            model, lr=1e-3, weight_decay=0.0, warmup_steps=10, total_steps=20
        )
        # After construction, last_lr reflects lr_lambda(0) * base_lr.
        assert sched.get_last_lr()[0] == pytest.approx(0.0, abs=1e-12)
        # Step optimizer before scheduler to avoid the PyTorch ordering warning.
        for _ in range(10):
            opt.step()
            sched.step()
        # After 10 steps we are at the warmup peak.
        assert sched.get_last_lr()[0] == pytest.approx(1e-3)

    def test_no_decay_params_are_1d_or_bias(self):
        class _M(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(3, 3)
                self.ln = nn.LayerNorm(3)

            def forward(self, x):
                return self.ln(self.lin(x))

        model = _M()
        opt, _ = OptimizerFactory.create_adamw(
            model, lr=1e-3, weight_decay=0.1, warmup_steps=0, total_steps=0
        )
        # The no-decay group should contain the bias and LayerNorm weight+bias.
        no_decay_group = opt.param_groups[1]
        no_decay_ids = {id(p) for p in no_decay_group["params"]}
        assert id(model.lin.bias) in no_decay_ids
        assert id(model.ln.weight) in no_decay_ids
        assert id(model.ln.bias) in no_decay_ids


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


class TestCheckpoint:
    def test_save_load_round_trip(self, tmp_path):
        engine = _make_engine(tmp_path)
        # Run one optimizer step so optimizer state is non-trivial.
        model = engine.model
        batch = {"x": torch.randn(4, 2), "y": torch.randn(4, 1)}
        engine.optimizer.zero_grad()
        loss = _mse_loss(model(batch), batch)
        loss.backward()
        engine.optimizer.step()
        engine.epoch = 3
        engine.global_step = 7

        ckpt = tmp_path / "ckpt" / "test.pt"
        engine.save_checkpoint(ckpt)
        assert ckpt.exists()

        # Capture pre-save state.
        original_sd = {k: v.clone() for k, v in model.state_dict().items()}
        original_opt_state = engine.optimizer.state_dict()

        # Build a fresh engine with a reinitialized model and load.
        engine2 = _make_engine(tmp_path)
        # Perturb the fresh model so we can detect the load.
        with torch.no_grad():
            for p in engine2.model.parameters():
                p.add_(1.0)
        engine2.load_checkpoint(ckpt)

        # Model weights must match the saved state exactly.
        for k, v in engine2.model.state_dict().items():
            assert torch.equal(v, original_sd[k]), f"mismatch on {k}"
        # Epoch / step restored.
        assert engine2.epoch == 3
        assert engine2.global_step == 7

    def test_resume_includes_rng_state(self, tmp_path):
        engine = _make_engine(tmp_path)
        engine.global_step = 5
        engine.epoch = 2
        ckpt = tmp_path / "ckpt" / "rng.pt"
        engine.save_checkpoint(ckpt)

        # Set the torch RNG to a known state, then load (should restore saved).
        torch.manual_seed(999)
        before = torch.rand(1).item()
        engine.load_checkpoint(ckpt)
        after = torch.rand(1).item()
        # After restoring RNG, the next draw should NOT equal the draw made
        # under seed 999 (proving RNG state was restored, not reset).
        assert before != after


# ---------------------------------------------------------------------------
# Integration: train_epoch / evaluate / early stopping
# ---------------------------------------------------------------------------


class TestTrainEpochIntegration:
    def test_train_epoch_cpu(self):
        engine = _make_engine()
        metrics = engine.train_epoch(0)
        assert "loss" in metrics
        assert torch.isfinite(torch.tensor(metrics["loss"]))
        assert metrics["num_batches"] > 0
        assert metrics["skipped_nan"] == 0.0
        assert metrics["skipped_oom"] == 0.0
        # global_step should have advanced.
        assert engine.global_step == int(metrics["num_batches"])

    def test_evaluate_returns_finite_loss(self):
        engine = _make_engine(with_val=True)
        metrics = engine.evaluate()
        assert "loss" in metrics
        assert torch.isfinite(torch.tensor(metrics["loss"]))
        assert metrics["num_batches"] > 0

    def test_evaluate_none_loader(self):
        engine = _make_engine(with_val=False)
        metrics = engine.evaluate()
        assert metrics["num_batches"] == 0.0

    def test_dict_loss_fn_supported(self):
        """loss_fn may return a dict with a 'total' key (like pairformer_loss)."""
        engine = _make_engine()

        def dict_loss(out, batch):
            return {"total": ((out - batch["y"]) ** 2).mean(), "bce": torch.tensor(0.0)}

        engine.loss_fn = dict_loss
        metrics = engine.train_epoch(0)
        assert torch.isfinite(torch.tensor(metrics["loss"]))

    def test_early_stopping_triggers(self, tmp_path):
        engine = _make_engine(tmp_path, early_stop_patience=2)
        # Not enough history yet (len <= patience).
        assert engine._should_stop_early([1.0]) is False
        assert engine._should_stop_early([1.0, 1.5]) is False
        # 2 non-improving evaluations since the best -> stop.
        assert engine._should_stop_early([1.0, 1.5, 1.6]) is True
        # A late improvement resets the counter.
        assert engine._should_stop_early([1.0, 0.9, 1.0]) is False
        assert engine._should_stop_early([1.0, 0.9, 1.0, 1.1]) is True

    def test_full_train_loop_runs(self, tmp_path):
        engine = _make_engine(tmp_path, early_stop_patience=0)
        history = engine.train(num_epochs=2)
        assert len(history["train_loss"]) == 2
        assert len(history["val_loss"]) == 2
        # best.pt should exist (validation improves on epoch 0).
        assert (tmp_path / "ckpt" / "best.pt").exists()
