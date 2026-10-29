"""trashpanda: deterministic small causal-LM training from JSONL to native HF."""
from .config import (
    Config,
    DataConfig,
    ModelConfig,
    RuntimeConfig,
    TrainingConfig,
    config_from_dict,
    get_recipe,
    load_config,
    resolve_training_config,
)

__version__ = "0.1.0"

__all__ = [
    "Config",
    "DataConfig",
    "ModelConfig",
    "RuntimeConfig",
    "TrainingConfig",
    "config_from_dict",
    "get_recipe",
    "load_config",
    "resolve_training_config",
]
