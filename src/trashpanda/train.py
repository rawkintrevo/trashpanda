"""Token-budgeted training with accumulation, finite checks, signals, and resume."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
import time
import random
import signal
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore
try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore

from .checkpoint import TrainingState, checkpoint_path, load_checkpoint, restore_rng_state, save_checkpoint
from .config import Config, TrainingConfig
from .data import MMapTokenDataset, PCG64Sampler
from .model import build_model, causal_loss
from .tokenizer import DeterministicTokenizer, tokenizer_snapshot


@dataclass(frozen=True)
class TokenBudgetSchedule:
    context_length: int
    effective_batch_size: int
    token_budget: int

    @classmethod
    def from_config(cls, config: Config) -> "TokenBudgetSchedule":
        return cls(
            context_length=config.model.context_length,
            effective_batch_size=config.training.effective_batch_size,
            token_budget=config.training.token_budget,
        )

    @property
    def tokens_per_sequence(self) -> int:
        return self.context_length - 1

    @property
    def tokens_per_update(self) -> int:
        return self.effective_batch_size * self.tokens_per_sequence

    @property
    def updates(self) -> int:
        return math.ceil(self.token_budget / self.tokens_per_update)

    @property
    def accounted_tokens(self) -> int:
        return self.updates * self.tokens_per_update


@dataclass
class TrainResult:
    state: TrainingState
    checkpoint: Path | None
    device: str
    schedule: TokenBudgetSchedule
    model: Any | None = None
    optimizer: Any | None = None
    sampler: PCG64Sampler | None = None
    tokenizer: Any | None = None


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("training requires PyTorch")


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    _require_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def resolve_device(training: TrainingConfig) -> Any:
    _require_torch()
    if training.device == "cpu":
        return torch.device("cpu")
    if training.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("recipe requests CUDA but no CUDA device is available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def optimizer_param_groups(model: Any, weight_decay: float) -> list[dict[str, Any]]:
    """Separate matrix parameters (decay) from vector parameters (no decay)."""
    _require_torch()
    decay: list[Any] = []
    no_decay: list[Any] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim >= 2:
            decay.append(parameter)
        else:
            no_decay.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _finite_tensor(value: Any) -> bool:
    return bool(torch.isfinite(value).all().item())


class _BoundarySignal:
    def __init__(self) -> None:
        self.requested = False
        self.signal_number: int | None = None

    def handler(self, signum: int, _frame: Any) -> None:
        # Do not interrupt a backward pass.  The loop observes this after the
        # current optimizer boundary and writes a resumable checkpoint.
        self.requested = True
        self.signal_number = signum

    def __enter__(self) -> "_BoundarySignal":
        self.previous: dict[int, Any] = {}
        for signum in (signal.SIGINT, signal.SIGTERM):
            self.previous[signum] = signal.getsignal(signum)
            signal.signal(signum, self.handler)
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        for signum, previous in self.previous.items():
            signal.signal(signum, previous)


def _latest_checkpoint(path: str | Path) -> Path:
    target = Path(path)
    if target.is_file():
        return target
    candidates = sorted(target.glob("checkpoint-*.pt"))
    if not candidates:
        raise FileNotFoundError(f"no checkpoint found in {target}")
    return candidates[-1]


def train(
    config: Config,
    dataset: MMapTokenDataset,
    *,
    model: Any | None = None,
    tokenizer: Any | None = None,
    resume: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    max_steps: int | None = None,
) -> TrainResult:
    """Run updates until the shifted-token budget or max_steps is reached."""
    _require_torch()
    if len(dataset) == 0:
        raise ValueError("training dataset is empty")
    if dataset.context_length != config.model.context_length:
        raise ValueError("prepared dataset context_length does not match model context_length")
    schedule = TokenBudgetSchedule(
        context_length=dataset.context_length,
        effective_batch_size=config.training.effective_batch_size,
        token_budget=config.training.token_budget,
    )
    if schedule.tokens_per_update <= 0:
        raise ValueError("invalid token schedule")
    if model is None:
        seed_everything(config.training.seed, deterministic=config.runtime.deterministic)
        model = build_model(config.model, seed=config.training.seed)
    device = resolve_device(config.training)
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        optimizer_param_groups(model, config.training.weight_decay),
        lr=config.training.learning_rate,
        betas=(config.training.beta1, config.training.beta2),
    )
    sampler = PCG64Sampler(len(dataset), config.training.seed, repeat_data=config.training.repeat_data)
    state = TrainingState()
    resume_path: Path | None = None
    if resume:
        resume_path = _latest_checkpoint(resume)
        payload = load_checkpoint(resume_path, model=model, optimizer=optimizer, map_location=str(device))
        state = TrainingState.from_dict(payload["state"])
        sampler.load_state_dict(payload["sampler"])
        if payload.get("rng"):
            restore_rng_state(payload["rng"])
        if int(payload.get("config", {}).get("model", {}).get("vocab_size", config.model.vocab_size)) != config.model.vocab_size:
            raise ValueError("checkpoint model configuration does not match current recipe")

    destination = Path(checkpoint_dir or config.runtime.checkpoint_dir or config.runtime.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    metrics_path = destination / "metrics.jsonl"
    run_started = time.perf_counter()
    effective_max_steps = max_steps or config.runtime.max_steps
    remaining_tokens = max(0, config.training.token_budget - state.consumed_tokens)
    if not config.training.repeat_data and schedule.updates * config.training.effective_batch_size > len(dataset):
        raise ValueError("--no-repeat-data cannot satisfy the rounded token budget with available sequences")
    updates_remaining = math.ceil(remaining_tokens / schedule.tokens_per_update) if remaining_tokens else 0
    if effective_max_steps is not None:
        updates_remaining = min(updates_remaining, max(0, effective_max_steps - state.update_step))
    if not config.training.repeat_data:
        available = len(dataset) - (sampler.position if sampler._permutation else 0)
        updates_remaining = min(updates_remaining, max(0, available // config.training.batch_size))
    last_checkpoint = resume_path
    scaler = torch.cuda.amp.GradScaler(enabled=config.training.precision == "fp16" and device.type == "cuda")
    autocast_enabled = config.training.precision in {"bf16", "fp16"} and device.type == "cuda"
    autocast_dtype = torch.bfloat16 if config.training.precision == "bf16" else torch.float16

    with _BoundarySignal() as boundary:
        for _ in range(updates_remaining):
            optimizer.zero_grad(set_to_none=True)
            update_loss = 0.0
            for _micro in range(config.training.grad_accumulation_steps):
                indices = sampler.next_indices(config.training.microbatch_size)
                inputs, targets = dataset.batch(indices)
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_enabled):
                    loss = causal_loss(model, inputs, targets)
                if not _finite_tensor(loss):
                    raise FloatingPointError("non-finite loss encountered")
                update_loss += float(loss.detach().float().item())
                scaled_loss = loss / config.training.grad_accumulation_steps
                if scaler.is_enabled():
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                state.micro_steps += 1
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            for parameter in model.parameters():
                if parameter.grad is not None and not _finite_tensor(parameter.grad):
                    raise FloatingPointError("non-finite gradient encountered")
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip, error_if_nonfinite=True)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            state.update_step += 1
            state.consumed_tokens += schedule.tokens_per_update
            should_save = state.update_step % config.training.checkpoint_interval == 0
            should_stop = boundary.requested
            if should_save or should_stop:
                last_checkpoint = _save_training_checkpoint(
                    destination, model, optimizer, state, config, sampler, tokenizer
                )
            with metrics_path.open("a", encoding="utf-8") as metrics:
                metrics.write(json.dumps({
                    "step": state.update_step,
                    "targets": state.consumed_tokens,
                    "train_loss": update_loss / config.training.grad_accumulation_steps,
                    "learning_rate": config.training.learning_rate,
                    "elapsed_seconds": time.perf_counter() - run_started,
                }, sort_keys=True) + "\n")
            if should_stop:
                break

    # Always leave a checkpoint at the actual boundary, including a tiny run
    # shorter than the configured interval.
    if last_checkpoint is None or state.update_step > 0:
        last_checkpoint = _save_training_checkpoint(destination, model, optimizer, state, config, sampler, tokenizer)
    return TrainResult(
        state=state,
        checkpoint=last_checkpoint,
        device=str(device),
        schedule=schedule,
        model=model,
        optimizer=optimizer,
        sampler=sampler,
        tokenizer=tokenizer,
    )


def _save_training_checkpoint(
    destination: Path,
    model: Any,
    optimizer: Any,
    state: TrainingState,
    config: Config,
    sampler: PCG64Sampler,
    tokenizer: Any | None,
) -> Path:
    if tokenizer is None:
        if config.model.vocab_size != 32:
            raise ValueError("production training requires its pinned tokenizer object for checkpoint provenance")
        tokenizer = DeterministicTokenizer()
    return save_checkpoint(
        checkpoint_path(destination, state.update_step),
        model=model,
        optimizer=optimizer,
        state=state,
        config=config.to_dict(),
        sampler_state=sampler.state_dict(),
        tokenizer_snapshot=tokenizer_snapshot(tokenizer).to_dict(),
    )


__all__ = [
    "TokenBudgetSchedule",
    "TrainResult",
    "optimizer_param_groups",
    "resolve_device",
    "seed_everything",
    "train",
]
