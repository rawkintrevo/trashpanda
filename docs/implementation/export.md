# Current evaluation and export

[Implementation map](README.md) · [Quickstart](../getting-started.md)

## Evaluation and samples

[`evaluate.py`](../../src/trashpanda/evaluate.py) runs inference-mode loss evaluation
without gradients, counts shifted prediction targets, and reports mean loss and
perplexity. The CLI accepts batch size and an optional maximum number of batches.
Sampling supports a seed, temperature, top-k, and generation length with an
isolated Torch generator.

These checks are useful diagnostics, not tests of legal compliance, absence of
memorization, factual reliability, privacy, or readiness for an enterprise task.
Always supply a real validation split: the current CLI falls back to training data
if the requested split is missing. The current perplexity implementation caps the
exponent input at 50; extremely high-loss results should not be interpreted as an
uncapped exact perplexity.

The core training loop does not automatically run the periodic evaluation described
in #1. The optional production driver evaluates and samples after training returns.

## Native Hugging Face export

[`export_hf()`](../../src/trashpanda/export.py) uses the model's standard
`save_pretrained(..., safe_serialization=True)` and the tokenizer's
`save_pretrained()`. It enables generation caching and refuses an existing export
directory. Resulting native model/tokenizer files do not require custom remote
Python classes. [The quickstart](../getting-started.md) includes local-only loading.

For a production checkpoint, supply the matching saved tokenizer with the CLI's
`--tokenizer` argument. A normal checkpoint stores tokenizer metadata, not a full
portable tokenizer snapshot. The export contains no optimizer state and currently
adds no automatic provenance record, model card, license inventory, or file-hash
completion manifest. These are gaps, not automatic consequences of the HF format.

The existing sampling/evaluation CLI may also need the configured production
tokenizer in its local cache; it does not reconstruct tokenizer files from a `.pt`
checkpoint. Preserve those assets separately until the self-contained artifact
contract is implemented.

An exported model can be used for inference; it is not by itself a faithful
training-resume checkpoint. See [training/recovery](training.md) and
[model-bound provenance](../roadmap/provenance.md).
