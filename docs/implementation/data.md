# Current data preparation

[Implementation map](README.md) · [Partial provenance](provenance.md)

## Local contract

The core CLI consumes local JSONL files, not a mandatory hosted dataset. Supply
separate train and validation paths. Each nonblank line must be a JSON object
containing a nonempty string `text` (or the selected text field). Identical input
paths across splits are rejected; duplicated content is not detected.

[`prepare_splits()`](../../src/trashpanda/prepare.py) encodes each document without
implicit special tokens, explicitly appends one EOS, and packs the token stream
into complete context-length sequences. The incomplete final tail is counted and
dropped separately for each split. It does not clean, deduplicate, classify,
license-check, approve, or anonymize the data. Whitespace-only strings are preserved.

It writes little-endian uint16 shards, a tokenizer directory, and `manifest.json`.
The manifest records token/sequence counts, dropped tails, context/vocabulary,
tokenizer metadata, saved tokenizer-file hashes, shard hashes, and an identity
hash of the manifest content before the identity field is added.

[`SequenceReader`](../../src/trashpanda/data.py) checks shard sizes and SHA-256s by
default and memory-maps sequences. A length-L row yields `row[:-1]` inputs and
`row[1:]` targets: **L−1 prediction targets**, not L. Sequences can cross document
boundaries; no padding is introduced.

## Tokenizers and recipes

- `tiny`: project-owned 32-entry WordLevel tokenizer, generated locally; 32-token
  sequences. Preparation still appends EOS even though the tokenizer adds none.
- `300m-a100`: 32K Mistral tokenizer, revision
  `27d67f1b5f57dc0953326b2601d68371d40ea8da`; 2048-token sequences. Loading requests
  no remote code. A reused tokenizer does not mean pretrained model weights are loaded.

The code records hashes of locally saved tokenizer files. Production acquisition
is restricted to the three pinned JSON assets and verifies their upstream raw-file
hash constants before loading. The CLI uses the recipe's pinned revision; a
production revision different from the contract is rejected. Prepared normalized
files are not expected to have the raw Hub hashes.

## Hugging Face example, not the product boundary

[`prepare_fineweb_edu.py`](../../scripts/prepare_fineweb_edu.py) streams
`HuggingFaceFW/fineweb-edu`, config `sample-10BT`, into temporary JSONL and calls
the same preparation path. Defaults select the first 30,000 training documents
and the next 1,000 validation documents. This is a bounded experiment, not a data
approval decision or a representative production corpus guarantee.

The adapter currently does not pin the dataset revision or retain a complete source
record. A Hugging Face listing alone does not establish rights to every underlying
record. Use your own review process before adopting this or any other dataset.

## Current limitations

Manifest identity is written but not recomputed by `read_manifest()`, and checkpoint
resume does not bind the prepared-data identity to the run. Source JSONL hashes,
dataset terms/approvals, immutable dataset versions, and acquisition timestamps
are not captured as an end-to-end record. Preparation also reads all shard bytes
again to compute a split-wide digest, so it is not fully bounded-memory at that stage.

See the [provenance proposal](../roadmap/provenance.md) for the intended extension.
