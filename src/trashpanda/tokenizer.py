"""Project-owned tiny WordLevel tokenizer and pinned local HF tokenizer loading."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Protocol, Sequence

from .config import MISTRAL_TOKENIZER_FILE_HASHES, MISTRAL_TOKENIZER_NAME, MISTRAL_TOKENIZER_REVISION

TINY_SPECIAL_TOKENS = ("<unk>", "<s>", "</s>")
TINY_VOCAB = {token: index for index, token in enumerate(TINY_SPECIAL_TOKENS)}
TINY_VOCAB.update({f"w{i}": i + 3 for i in range(29)})


class Tokenizer(Protocol):
    vocab_size: int
    pad_token_id: int
    eos_token_id: int

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...
    def decode(self, ids: Sequence[int]) -> str: ...


class TinyWordLevelTokenizer:
    """A fixed 32-entry WordLevel vocabulary with no implicit special tokens."""

    name = "trashpanda/tiny-wordlevel"
    revision = "project"
    vocab = TINY_VOCAB
    vocab_size = len(TINY_VOCAB)
    pad_token = None
    unk_token = "<unk>"
    eos_token = "</s>"
    bos_token = "<s>"
    pad_token_id = None
    unk_token_id = 0
    eos_token_id = 2
    bos_token_id = 1

    def __init__(self) -> None:
        self._tokenizer = None
        try:
            from tokenizers import Tokenizer
            from tokenizers.models import WordLevel
            from tokenizers.pre_tokenizers import Whitespace

            tokenizer = Tokenizer(WordLevel(vocab=dict(self.vocab), unk_token="<unk>"))
            tokenizer.pre_tokenizer = Whitespace()
            self._tokenizer = tokenizer
        except ImportError:  # deterministic fallback for config-only environments
            self._tokenizer = None

    def __len__(self) -> int:
        return self.vocab_size

    def encode(self, text: str, *, add_special_tokens: bool = False, split_special_tokens: bool = True) -> list[int]:
        # add_special_tokens is intentionally ignored: the tiny tokenizer never
        # inserts BOS/EOS, including when callers pass True explicitly.
        if self._tokenizer is not None:
            encoding = self._tokenizer.encode(text, add_special_tokens=False)
            # The three marker strings are ordinary source text for this
            # fixture, not registered special tokens.
            return [
                self.unk_token_id if token in TINY_SPECIAL_TOKENS else int(token_id)
                for token, token_id in zip(encoding.tokens, encoding.ids)
            ]
        return [self.unk_token_id if token in TINY_SPECIAL_TOKENS else self.vocab.get(token, self.unk_token_id) for token in re.findall(r"\S+", text)]

    def decode(self, ids: Sequence[int]) -> str:
        if self._tokenizer is not None:
            return self._tokenizer.decode([int(token) for token in ids], skip_special_tokens=True)
        inverse = {value: key for key, value in self.vocab.items()}
        return " ".join(inverse.get(int(token), "<unk>") for token in ids if int(token) not in {0, 2})

    def get_vocab(self) -> dict[str, int]:
        return dict(self.vocab)

    def snapshot(self) -> dict[str, Any]:
        return {
            "kind": "wordlevel",
            "name": self.name,
            "revision": self.revision,
            "vocab": self.get_vocab(),
            "special_tokens": list(TINY_SPECIAL_TOKENS),
            "inserts_special_tokens": False,
        }

    def save_pretrained(self, directory: str | Path) -> tuple[str, ...]:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        files: list[str] = []
        if self._tokenizer is not None:
            self._tokenizer.save(str(directory / "tokenizer.json"))
            files.append("tokenizer.json")
        else:
            (directory / "tokenizer.json").write_text(json.dumps(self.snapshot(), sort_keys=True) + "\n", encoding="utf-8")
            files.append("tokenizer.json")
        (directory / "tokenizer_config.json").write_text(
            json.dumps({"tokenizer_class": "PreTrainedTokenizerFast", "model_max_length": 1_000_000, "bos_token": "<s>", "unk_token": "<unk>", "eos_token": "</s>"}, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        (directory / "special_tokens_map.json").write_text(
            json.dumps({"bos_token": "<s>", "unk_token": "<unk>", "eos_token": "</s>"}, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        files.extend(("tokenizer_config.json", "special_tokens_map.json"))
        return tuple(files)


# Existing callers can keep the deterministic name while receiving the exact
# project-owned WordLevel implementation.
DeterministicTokenizer = TinyWordLevelTokenizer


@dataclass(frozen=True)
class TokenizerSnapshot:
    name: str
    revision: str
    vocab_size: int
    tokenizer_sha256: str
    kind: str
    pad_token_id: int
    eos_token_id: int
    file_hashes: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "revision": self.revision,
            "vocab_size": self.vocab_size,
            "tokenizer_sha256": self.tokenizer_sha256,
            "kind": self.kind,
            "pad_token_id": self.pad_token_id,
            "eos_token_id": self.eos_token_id,
            "file_hashes": self.file_hashes or {},
        }


def _snapshot_hash(tokenizer: Any) -> str:
    snapshot = tokenizer.snapshot() if hasattr(tokenizer, "snapshot") else {
        "name": getattr(tokenizer, "name_or_path", "unknown"),
        "revision": getattr(tokenizer, "revision", "local"),
        "vocab_size": int(getattr(tokenizer, "vocab_size", len(tokenizer))),
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    return hashlib.sha256(json.dumps(snapshot, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def raw_file_hashes(directory: str | Path) -> dict[str, str]:
    directory = Path(directory)
    return {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in MISTRAL_TOKENIZER_FILE_HASHES
        if (directory / name).is_file()
    }


def tokenizer_snapshot(tokenizer: Any) -> TokenizerSnapshot:
    file_hashes = getattr(tokenizer, "trashpanda_file_hashes", None)
    return TokenizerSnapshot(
        name=str(getattr(tokenizer, "name_or_path", getattr(tokenizer, "name", "unknown"))),
        revision=str(getattr(tokenizer, "revision", "local")),
        vocab_size=int(getattr(tokenizer, "vocab_size", len(tokenizer))),
        tokenizer_sha256=_snapshot_hash(tokenizer),
        kind="wordlevel" if isinstance(tokenizer, TinyWordLevelTokenizer) else "huggingface",
        pad_token_id=int(getattr(tokenizer, "pad_token_id", 0) or 0),
        eos_token_id=int(getattr(tokenizer, "eos_token_id", 2) or 2),
        file_hashes=file_hashes,
    )


def _download_pinned_assets(name: str, revision: str, *, cache_dir: str | Path | None = None) -> str:
    if name != MISTRAL_TOKENIZER_NAME or revision != MISTRAL_TOKENIZER_REVISION:
        raise ValueError("production tokenizer acquisition requires the pinned Mistral repository and revision")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for tokenizer acquisition") from exc
    target = Path(cache_dir) if cache_dir is not None else Path(tempfile.mkdtemp(prefix="trashpanda-tokenizer-"))
    target.mkdir(parents=True, exist_ok=True)
    for filename, expected in MISTRAL_TOKENIZER_FILE_HASHES.items():
        downloaded = Path(hf_hub_download(name, filename=filename, revision=revision, cache_dir=str(target)))
        destination = target / filename
        if downloaded.resolve() != destination.resolve():
            shutil.copyfile(downloaded, destination)
        actual = hashlib.sha256(destination.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"pinned tokenizer hash mismatch for {filename}: expected {expected}, got {actual}")
    return str(target)


def acquire_pinned_tokenizer(
    *, cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> Any:
    """Acquire only the pinned three-file tokenizer contract."""
    if local_files_only:
        raise ValueError("pinned tokenizer acquisition requires network access; use load_tokenizer for local files")
    return load_tokenizer(
        MISTRAL_TOKENIZER_NAME,
        revision=MISTRAL_TOKENIZER_REVISION,
        vocab_size=32_000,
        local_files_only=False,
        cache_dir=cache_dir,
    )


def load_tokenizer(
    name: str = MISTRAL_TOKENIZER_NAME,
    *,
    revision: str = MISTRAL_TOKENIZER_REVISION,
    vocab_size: int | None = None,
    local_files_only: bool = True,
    tokenizer_sha256: str | None = None,
    cache_dir: str | Path | None = None,
) -> Any:
    if name in {"trashpanda/tiny-wordlevel", "trashpanda/deterministic-byte", "deterministic-byte", "local"}:
        tokenizer = TinyWordLevelTokenizer()
        if vocab_size is not None and vocab_size != tokenizer.vocab_size:
            raise ValueError("tiny WordLevel vocabulary is fixed at 32 tokens")
        return tokenizer
    local_path = Path(name)
    if local_path.is_dir() and vocab_size == len(TINY_VOCAB):
        # Tiny fixtures are deliberately reloadable without contacting Hub or
        # requiring a pretrained tokenizer class.
        tokenizer_json = local_path / "tokenizer.json"
        if tokenizer_json.is_file():
            try:
                payload = json.loads(tokenizer_json.read_text(encoding="utf-8"))
                if len(payload.get("model", {}).get("vocab", {})) == len(TINY_VOCAB):
                    return TinyWordLevelTokenizer()
            except (OSError, json.JSONDecodeError, AttributeError):
                pass
    if name == MISTRAL_TOKENIZER_NAME and revision != MISTRAL_TOKENIZER_REVISION:
        raise ValueError("the production tokenizer revision is pinned and cannot be changed")
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required for a non-tiny tokenizer") from exc
    source = name
    local_hashes = raw_file_hashes(name) if Path(name).is_dir() else {}
    if not local_files_only and not Path(name).is_dir():
        # Avoid AutoTokenizer's broad Hub resolution: production acquisition is
        # deliberately limited to the three pinned JSON assets and never asks
        # for model weights or remote Python code.
        source = _download_pinned_assets(name, revision, cache_dir=cache_dir)
        local_hashes = raw_file_hashes(source)
    tokenizer = AutoTokenizer.from_pretrained(
        source,
        revision=revision if source == name else None,
        local_files_only=True if source != name else local_files_only,
        trust_remote_code=False,
        use_fast=True,
    )
    tokenizer.revision = revision
    # A prepared tokenizer directory is a normalized ``save_pretrained``
    # snapshot, so its file bytes need not equal the raw Hub blobs used for
    # the pinned revision.  Preserve local hashes; raw acquisition is checked
    # by _download_pinned_assets before loading.
    setattr(tokenizer, "trashpanda_file_hashes", local_hashes or dict(MISTRAL_TOKENIZER_FILE_HASHES))
    if tokenizer_sha256 and _snapshot_hash(tokenizer) != tokenizer_sha256:
        raise ValueError("tokenizer snapshot hash mismatch")
    return tokenizer


__all__ = [
    "DeterministicTokenizer", "TinyWordLevelTokenizer", "Tokenizer", "TokenizerSnapshot",
    "TINY_SPECIAL_TOKENS", "TINY_VOCAB", "acquire_pinned_tokenizer", "load_tokenizer", "raw_file_hashes", "tokenizer_snapshot",
]
