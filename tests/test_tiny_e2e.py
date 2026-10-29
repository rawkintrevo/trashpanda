import importlib.util
import json

import pytest

if importlib.util.find_spec("torch") is None or importlib.util.find_spec("transformers") is None:
    pytest.skip("PyTorch/transformers are not installed in this environment", allow_module_level=True)

from trashpanda.checkpoint import list_checkpoints
from trashpanda.config import get_recipe
from trashpanda.data import SequenceReader
from trashpanda.evaluate import evaluate
from trashpanda.export import export_checkpoint
from trashpanda.prepare import prepare_splits
from trashpanda.tokenizer import TinyWordLevelTokenizer
from trashpanda.train import train
from trashpanda.model import build_model


def _dataset(tmp_path):
    text = " ".join(f"w{i % 29}" for i in range(256))
    source = tmp_path / "train.jsonl"
    source.write_text(json.dumps({"text": text}) + "\n", encoding="utf-8")
    validation = tmp_path / "validation.jsonl"
    validation.write_text(json.dumps({"text": " ".join(f"w{i % 29}" for i in range(64))}) + "\n", encoding="utf-8")
    result = prepare_splits(tmp_path / "prepared", train_path=source, validation_path=validation, tokenizer=TinyWordLevelTokenizer(), context_length=32)
    return SequenceReader(result["train"].directory, split="train"), SequenceReader(result["train"].directory, split="validation")


def test_tiny_train_resume_evaluate_export(tmp_path):
    config = get_recipe("tiny")
    dataset, validation = _dataset(tmp_path)
    first = train(config, dataset, checkpoint_dir=tmp_path / "checkpoints", max_steps=3)
    assert first.state.consumed_tokens == 372
    assert first.checkpoint is not None
    resumed = train(config, dataset, resume=first.checkpoint, checkpoint_dir=tmp_path / "checkpoints")
    assert resumed.state.consumed_tokens == 744
    assert len(list_checkpoints(tmp_path / "checkpoints")) >= 2
    result = evaluate(build_model(config.model, seed=0), validation, max_batches=1)
    assert result.tokens == 31
    output = export_checkpoint(resumed.checkpoint, tmp_path / "export")
    assert (output / "config.json").exists()
    assert (output / "pytorch_model.bin").exists() or (output / "model.safetensors").exists()
