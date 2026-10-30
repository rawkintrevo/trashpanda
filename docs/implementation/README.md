# Current implementation

[Executive summary](../../README.md) · [Proposed capabilities](../roadmap/README.md)

This describes the checked-in code, not every acceptance criterion in the First
Model issues. Implementation exists; completing the production run and providing
organizational supply-chain assurances are separate claims.

| Boundary | Source | Details |
| --- | --- | --- |
| CLI and recipes | [`cli.py`](../../src/trashpanda/cli.py), [`config.py`](../../src/trashpanda/config.py) | [Quickstart](../getting-started.md), [training](training.md) |
| Tokenizer and input preparation | [`tokenizer.py`](../../src/trashpanda/tokenizer.py), [`prepare.py`](../../src/trashpanda/prepare.py) | [Data](data.md) |
| Dataset reader and sampler | [`data.py`](../../src/trashpanda/data.py) | [Data](data.md), [recovery](training.md) |
| Model and training loop | [`model.py`](../../src/trashpanda/model.py), [`train.py`](../../src/trashpanda/train.py) | [Training](training.md) |
| State persistence | [`checkpoint.py`](../../src/trashpanda/checkpoint.py) | [Recovery and limitations](training.md) |
| Evaluation and native HF files | [`evaluate.py`](../../src/trashpanda/evaluate.py), [`export.py`](../../src/trashpanda/export.py) | [Evaluation/export](export.md) |
| Metadata across these boundaries | Manifests, checkpoint payloads, metrics | [Partial provenance](provenance.md) |

## Validation boundaries

The existing [tests](../../tests) cover recipe/config validation, tiny model
construction, preparation, token accounting, and a tiny train/resume/export path.
They are not evidence of a complete production run, policy enforcement, federated
privacy, or end-to-end artifact provenance.

In particular, the current tiny end-to-end test evaluates a fresh model and checks
export file existence; it does not compare exported logits with the resumed model
or establish bitwise split-run equivalence. Historical artifact verification has
additional limitations documented in [experiment evidence](../experiments.md).

Documentation of these gaps is not a fix for them. The [first-model completion
plan](../roadmap/first-model.md) identifies the next validation and recovery work.
