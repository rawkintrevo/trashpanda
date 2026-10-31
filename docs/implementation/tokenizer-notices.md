# Tokenizer provenance and notices

The production recipe uses the fast tokenizer assets from
`mistralai/Mistral-7B-v0.1` at revision
`27d67f1b5f57dc0953326b2601d68371d40ea8da`. Acquisition is intentionally
restricted to `tokenizer.json`, `tokenizer_config.json`, and
`special_tokens_map.json`; each raw download is checked against the SHA-256
constants shipped in `src/trashpanda/config.py` before it is loaded.

This is an artifact-specific integrity record, not a license grant. The
Mistral repository/model-card license and any applicable attribution or
redistribution terms must be reviewed for the exact files and intended
artifact before publishing a prepared dataset, checkpoint, or export. If that
review is unavailable or uncertain, publication is blocked rather than silently
substituting another tokenizer or treating the model-card label as sufficient.

The `tiny` tokenizer is project-owned and generated at runtime. It has no
external download and is covered by the project Apache-2.0 policy. Its marker
strings (`<unk>`, `<s>`, `</s>`) are ordinary source text and are not implicitly
recognized as special tokens; preparation appends exactly one EOS ID per
record.
