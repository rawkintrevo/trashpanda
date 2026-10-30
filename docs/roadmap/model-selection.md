# Proposal: select supported models through the CLI

**Status: proposed.** Current `--recipe tiny` and `--recipe 300m-a100` both select
Llama configurations; they are not a general architecture-selection facility.

[Roadmap](README.md) · [Comparisons](../comparisons.md)

## Outcome

An organization selects a supported model recipe without rewriting the training
workflow. Dataset approval, provenance, checkpoint recovery, evaluation, and
artifact inspection remain consistent across supported models.

## Proposed implementation

- Introduce an explicit allowlist of model adapters/recipes. A future `--model`
  selector is a possible interface, **not a currently available flag**. Each adapter
  declares architecture, initialization mode, tokenizer/encoding contract, context,
  parameter count, resource requirements, licensing references, and export format.
- Keep data preparation, training orchestration, provenance, and recovery boundaries
  independent of a particular architecture. Reuse upstream model implementations
  rather than building another tensor framework.
- Separate model identity from execution settings. Resolve the selection before data
  preparation and store immutable model/tokenizer identity in run and artifact records.
- Treat scratch initialization and continuation from existing weights as distinct
  modes. Record parent-weight hashes and upstream lineage for the latter. Training
  on approved data cannot retroactively certify unknown parent data.
- Reject unsupported adapters and incompatible prepared data/checkpoints rather than
  silently switching architectures, tokenizers, or precision.

OLMo is a candidate to investigate, not a committed first addition. Selection must
account for upstream maintenance, tokenizer rights, resource needs, export support,
and the [provenance contract](provenance.md).

## Decisions still needed

Choose the first additional model family and decide whether the initial interface
is an expanded recipe catalogue or a distinct model flag. Define the adapter
contract and permitted configuration overrides before adding third-party plugins;
there is no arbitrary remote-code/plugin execution planned by this proposal.

## Acceptance evidence

For at least two explicitly supported architectures: prepare compatible inputs,
train a bounded run, interrupt/reload, perform a real resumed update, export, and
load offline in a fresh environment. Verify recorded model/tokenizer identity and
reject cross-model or incompatible-data resume. Existing tiny CLI behavior must
remain documented and tested.
