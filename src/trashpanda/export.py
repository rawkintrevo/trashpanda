"""Native Hugging Face Llama export without custom code or remote execution."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore

from .checkpoint import load_checkpoint
from .config import Config, config_from_dict
from .model import build_model
from .tokenizer import TinyWordLevelTokenizer, load_tokenizer


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("export requires PyTorch")


def _config_from_dict(value: Mapping[str, Any]) -> Config:
    return config_from_dict(value)


def export_hf(model: Any, output_dir: str | Path, *, config: Config, tokenizer: Any | None = None) -> Path:
    """Call the real HF ``save_pretrained`` path for a LlamaForCausalLM."""
    _require_torch()
    if not hasattr(model, "save_pretrained") or model.__class__.__name__ != "LlamaForCausalLM":
        raise TypeError("export_hf requires transformers.LlamaForCausalLM")
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"export output {output} already exists; refusing to overwrite")
    output.mkdir(parents=True, exist_ok=False)
    model.config.use_cache = True
    model.save_pretrained(output, safe_serialization=True)
    if tokenizer is None:
        tokenizer = TinyWordLevelTokenizer() if config.model.vocab_size == 32 else None
    if tokenizer is None or not hasattr(tokenizer, "save_pretrained"):
        raise ValueError("production export requires the locally cached Mistral tokenizer")
    tokenizer.save_pretrained(output)
    return output


def _safe_load(path: str | Path) -> dict[str, Any]:
    _require_torch()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint is not a mapping")
    return payload


def export_checkpoint(
    checkpoint: str | Path,
    output_dir: str | Path,
    *,
    tokenizer: Any | None = None,
    tokenizer_name: str | None = None,
    local_files_only: bool = True,
) -> Path:
    raw = _safe_load(checkpoint)
    config = config_from_dict(raw["config"])
    model = build_model(config.model, seed=config.training.seed)
    load_checkpoint(checkpoint, model=model, map_location="cpu")
    if tokenizer is None and tokenizer_name:
        tokenizer = load_tokenizer(tokenizer_name, revision=config.data.tokenizer_revision, vocab_size=config.model.vocab_size, local_files_only=local_files_only)
    return export_hf(model, output_dir, config=config, tokenizer=tokenizer)


__all__ = ["export_checkpoint", "export_hf"]
