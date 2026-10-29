"""Stream JSONL splits into little-endian uint16 fixed-length token shards."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import struct
from typing import Any, Iterable, Mapping, Sequence

from .tokenizer import TinyWordLevelTokenizer, tokenizer_snapshot


@dataclass(frozen=True)
class PreparedDatasetInfo:
    directory: str
    split: str
    shards: tuple[str, ...]
    context_length: int
    sequence_count: int
    token_count: int
    dropped_tokens: int
    vocab_size: int
    sha256: str

    @property
    def training_token_count(self) -> int:
        return self.sequence_count * (self.context_length - 1)

    @property
    def token_file(self) -> str:
        return self.shards[0] if self.shards else ""

    @property
    def manifest_file(self) -> str:
        return str(Path(self.directory) / "manifest.json")

    def to_dict(self) -> dict[str, Any]:
        return {
            "directory": self.directory,
            "split": self.split,
            "shards": list(self.shards),
            "context_length": self.context_length,
            "sequence_count": self.sequence_count,
            "token_count": self.token_count,
            "dropped_tokens": self.dropped_tokens,
            "training_token_count": self.training_token_count,
            "vocab_size": self.vocab_size,
            "sha256": self.sha256,
        }


def _text_from_record(record: Any, text_field: str) -> str:
    if isinstance(record, str):
        return record
    if not isinstance(record, Mapping) or not isinstance(record.get(text_field), str):
        raise ValueError(f"JSONL record must contain string field {text_field!r}")
    return record[text_field]


def _iter_jsonl(path: str | Path, text_field: str) -> Iterable[str]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"input {source}, line {line_number}: blank lines are not allowed")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"input {source}, line {line_number}: invalid JSON at column {exc.colno}") from exc
            if not isinstance(record, Mapping):
                raise ValueError(f"input {source}, line {line_number}: expected a JSON object")
            value = record.get(text_field)
            if not isinstance(value, str) or value == "":
                raise ValueError(f"input {source}, line {line_number}: {text_field!r} must be a nonempty string")
            yield value


def _write_split(
    input_path: str | Path | Sequence[str | Path],
    output: Path,
    split: str,
    tokenizer: Any,
    *,
    context_length: int,
    vocab_size: int,
    text_field: str,
    shard_tokens: int,
) -> tuple[dict[str, Any], PreparedDatasetInfo]:
    max_sequences = max(1, shard_tokens // context_length)
    carry: list[int] = []
    shards: list[dict[str, Any]] = []
    token_count = 0
    sequence_count = 0
    current: list[int] = []

    def flush() -> None:
        nonlocal current
        if not current:
            return
        shard_index = len(shards)
        name = f"{split}-{shard_index:05d}.bin"
        path = output / name
        raw = struct.pack(f"<{len(current)}H", *current)
        path.write_bytes(raw)
        digest = hashlib.sha256(raw).hexdigest()
        shards.append({"path": name, "dtype": "<u2", "sequence_count": len(current) // context_length, "token_count": len(current), "sha256": digest})
        current = []

    input_paths = [input_path] if isinstance(input_path, (str, Path)) else list(input_path)
    for source_path in input_paths:
        for text in _iter_jsonl(source_path, text_field):
            encoded = [int(token) for token in tokenizer.encode(text, add_special_tokens=False)]
            eos_id = getattr(tokenizer, "eos_token_id", 2)
            if not isinstance(eos_id, int):
                raise ValueError("tokenizer must expose an integer eos_token_id")
            encoded.append(eos_id)
            if any(token < 0 or token >= vocab_size or token > 65535 for token in encoded):
                raise ValueError("tokenizer produced a token outside uint16 vocabulary")
            token_count += len(encoded)
            carry.extend(encoded)
            while len(carry) >= context_length:
                current.extend(carry[:context_length])
                del carry[:context_length]
                sequence_count += 1
                if len(current) >= max_sequences * context_length:
                    flush()
    dropped = len(carry)
    flush()
    split_manifest = {
        "shards": shards,
        "token_count": token_count,
        "sequence_count": sequence_count,
        "dropped_tokens": dropped,
        "training_token_count": sequence_count * (context_length - 1),
    }
    all_bytes = b"".join((output / shard["path"]).read_bytes() for shard in shards)
    info = PreparedDatasetInfo(
        directory=str(output),
        split=split,
        shards=tuple(str(output / shard["path"]) for shard in shards),
        context_length=context_length,
        sequence_count=sequence_count,
        token_count=token_count,
        dropped_tokens=dropped,
        vocab_size=vocab_size,
        sha256=hashlib.sha256(all_bytes).hexdigest(),
    )
    return split_manifest, info


def prepare_splits(
    output_dir: str | Path,
    *,
    train_path: str | Path | Sequence[str | Path],
    validation_path: str | Path | Sequence[str | Path] | None = None,
    tokenizer: Any | None = None,
    context_length: int = 32,
    vocab_size: int | None = None,
    text_field: str = "text",
    shard_tokens: int = 1 << 20,
) -> dict[str, PreparedDatasetInfo]:
    """Prepare train and optional validation without adding EOS or padding."""
    if context_length < 2 or shard_tokens <= 0:
        raise ValueError("context_length must be at least two and shard_tokens positive")
    tokenizer = tokenizer or TinyWordLevelTokenizer()
    actual_vocab = int(vocab_size or getattr(tokenizer, "vocab_size", len(tokenizer)))
    if actual_vocab > 65_536:
        raise ValueError("uint16 shards require vocab_size <= 65536")
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"output {output} already exists; refusing to overwrite")
    if validation_path is not None:
        train_inputs = [train_path] if isinstance(train_path, (str, Path)) else list(train_path)
        validation_inputs = [validation_path] if isinstance(validation_path, (str, Path)) else list(validation_path)
        if {Path(item).resolve() for item in train_inputs} & {Path(item).resolve() for item in validation_inputs}:
            raise ValueError("train and validation inputs must be distinct paths")
    output.mkdir(parents=True, exist_ok=False)
    split_paths = {"train": train_path}
    if validation_path is not None:
        split_paths["validation"] = validation_path
    tokenizer_output = output / "tokenizer"
    tokenizer_output.mkdir()
    tokenizer.save_pretrained(tokenizer_output)
    file_hashes = {}
    for file in sorted(tokenizer_output.rglob("*")):
        if file.is_file():
            file_hashes[file.relative_to(tokenizer_output).as_posix()] = hashlib.sha256(file.read_bytes()).hexdigest()
    snapshot = tokenizer_snapshot(tokenizer).to_dict()
    snapshot["file_hashes"] = file_hashes
    manifest: dict[str, Any] = {
        "format": "trashpanda.shards.v2",
        "dtype": "uint16",
        "byte_order": "little",
        "context_length": context_length,
        "vocab_size": actual_vocab,
        "tokenizer": snapshot,
        "tokenizer_sha256": snapshot["tokenizer_sha256"],
        "tokenizer_hash": snapshot["tokenizer_sha256"],
        "tokenizer_file_hashes": snapshot.get("file_hashes", {}),
        "splits": {},
    }
    result: dict[str, PreparedDatasetInfo] = {}
    for split, path in split_paths.items():
        split_manifest, info = _write_split(path, output, split, tokenizer, context_length=context_length, vocab_size=actual_vocab, text_field=text_field, shard_tokens=shard_tokens)
        if info.sequence_count < 1:
            raise ValueError(f"{split} split has no complete sequence of length {context_length}")
        manifest["splits"][split] = split_manifest
        manifest[split] = split_manifest
        result[split] = info
    temporary = output / "manifest.json.tmp"
    manifest["identity"] = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
    temporary.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output / "manifest.json")
    return result


def prepare_jsonl(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    tokenizer: Any | None = None,
    context_length: int = 32,
    vocab_size: int | None = None,
    text_field: str = "text",
    shard_tokens: int = 1 << 20,
    split: str = "train",
    **_ignored: Any,
) -> PreparedDatasetInfo:
    """Single-split compatibility entry point; tails are always dropped exactly."""
    result = prepare_splits(output_dir, train_path=input_path, tokenizer=tokenizer, context_length=context_length, vocab_size=vocab_size, text_field=text_field, shard_tokens=shard_tokens)
    info = result["train"]
    if split != "train":
        info = PreparedDatasetInfo(**{**info.__dict__, "split": split})
    return info


def read_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if path.is_dir():
        path /= "manifest.json"
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if value.get("format") not in {"trashpanda.shards.v2", "trashpanda.tokens.v1"}:
        raise ValueError("unsupported prepared dataset format")
    return value


def prepare_dataset(
    train_path: str | Path,
    validation_path: str | Path | None,
    output_dir: str | Path,
    **kwargs: Any,
) -> dict[str, PreparedDatasetInfo]:
    return prepare_splits(output_dir, train_path=train_path, validation_path=validation_path, **kwargs)


prepare = prepare_jsonl

__all__ = ["PreparedDatasetInfo", "prepare", "prepare_dataset", "prepare_jsonl", "prepare_splits", "read_manifest"]
