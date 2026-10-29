"""Hugging Face Llama construction used by both recipes."""
from __future__ import annotations

from contextlib import nullcontext
from typing import Any

from .config import ModelConfig

try:  # Keep config/prepare usable in a minimal source checkout.
    import torch
    import torch.nn.functional as F
    from transformers import LlamaConfig, LlamaForCausalLM
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    F = None  # type: ignore
    LlamaConfig = None  # type: ignore
    LlamaForCausalLM = None  # type: ignore


def _require_model_dependencies() -> None:
    if torch is None or LlamaConfig is None or LlamaForCausalLM is None:
        raise RuntimeError("model construction requires torch and transformers")


def make_hf_config(config: ModelConfig) -> Any:
    """Build a standard LlamaConfig with SDPA and tied input/output embeddings."""
    _require_model_dependencies()
    kwargs = dict(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.kv_heads,
        max_position_embeddings=config.context_length,
        rms_norm_eps=config.rms_norm_eps,
        tie_word_embeddings=config.tie_embeddings,
        attention_bias=False,
        mlp_bias=False,
        rope_theta=10_000.0,
    )
    try:
        hf_config = LlamaConfig(attn_implementation="sdpa", **kwargs)
    except TypeError:  # transformers releases before the public constructor kwarg
        hf_config = LlamaConfig(**kwargs)
    # Transformers selects the attention class from this private field in some
    # releases and from the public field in others; set both deliberately.
    hf_config._attn_implementation = "sdpa"
    hf_config.attn_implementation = "sdpa"
    hf_config.use_cache = False
    return hf_config


def build_model(config: ModelConfig, *, seed: int | None = None, device: str | None = None) -> Any:
    """Construct an actual ``transformers.LlamaForCausalLM`` instance."""
    _require_model_dependencies()
    if seed is not None:
        torch.manual_seed(seed)
    construction = torch.device(device) if device else None
    with torch.device(construction) if construction is not None else nullcontext():
        model = LlamaForCausalLM(make_hf_config(config))
    if config.tie_embeddings:
        model.tie_weights()
    actual = count_parameters(model)
    expected = config.parameter_count()
    if actual != expected:
        raise AssertionError(f"Llama parameter count mismatch: expected {expected}, got {actual}")
    return model


def count_parameters(model: Any) -> int:
    _require_model_dependencies()
    # named_parameters removes the duplicated lm_head alias when embeddings are tied.
    return sum(parameter.numel() for _, parameter in model.named_parameters())


def causal_loss(model: Any, input_ids: Any, labels: Any) -> Any:
    """Compute loss for already-shifted ``SequenceReader`` targets.

    HF's built-in ``labels=`` path shifts once internally; the reader has
    already returned ``row[:-1]`` and ``row[1:]``, so compute this explicit
    cross entropy to account for exactly L-1 targets.
    """
    outputs = model(input_ids=input_ids)
    logits = outputs.logits
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))


TransformerLM = LlamaForCausalLM
TinyLanguageModel = LlamaForCausalLM

__all__ = ["LlamaConfig", "LlamaForCausalLM", "TinyLanguageModel", "TransformerLM", "build_model", "causal_loss", "count_parameters", "make_hf_config"]
