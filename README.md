# Trashpanda

**Open-weight models with an accountable supply chain.**

Trashpanda is being built for organizations that need models trained on known,
organization-approved data—and evidence of what went into them. The goal is a
practical workflow for selecting a model, training it, and delivering weights
with inspectable provenance, not another PyTorch or Hugging Face replacement.

**Today:** an experimental single-machine Llama training foundation. The broader
supply-chain assurance workflow is a roadmap, not a delivered certification or
privacy guarantee. [Read the vision and boundaries →](docs/vision.md)

## What is implemented

| Capability | Current scope and implementation details |
| --- | --- |
| Local data preparation | Separate train/validation JSONL → token shards, counts, and integrity hashes. [How data flows](docs/implementation/data.md) |
| Training and checkpoint state | Two fixed Llama recipes, token-budgeted PyTorch training, model/optimizer/RNG/sampler saves and resume support. [How training and recovery work—and their limits](docs/implementation/training.md) |
| Evaluation and model export | Loss/perplexity, sampling, and native Hugging Face model/tokenizer files. [How evaluation and export work](docs/implementation/export.md) |
| Partial provenance metadata | Prepared-data and tokenizer hashes, configuration, training counters, and logs. These are **not yet bound into a complete model provenance record**. [What is recorded today](docs/implementation/provenance.md) |

[First Model (#1)](https://github.com/rawkintrevo/trashpanda/issues/1) is the initial
milestone, not the whole project. Historical A100 records describe bounded
experiments; **the configured 6B-target training and final-artifact delivery are
not established as complete**. [Evidence and artifact availability →](docs/experiments.md)

## What is next

| Direction | Proposed implementation |
| --- | --- |
| Provenance that travels with the model | Bind datasets, training times, configuration, lineage, and approval evidence to the exported artifact. [Proposal](docs/roadmap/provenance.md) |
| CLI-selectable models | Support additional model architectures through explicit adapters/recipes, preserving evidence and recovery contracts. [Proposal](docs/roadmap/model-selection.md) |
| Federated training | Enable organizations to collaborate while retaining proprietary raw data locally, with an explicit privacy/threat model. [Proposal](docs/roadmap/federated-training.md) |

These are proposed capabilities, not working CLI options. The
[roadmap](docs/roadmap/README.md) also tracks the remaining first-model work.

## Where Trashpanda fits

- **Apache Sourcelume Sample** demonstrates training-data provenance records and
  registry exchange. Trashpanda aims to apply such evidence to an organizational
  model-production workflow; Sourcelume compatibility is a proposal, not an integration.
- **OLMo / OLMo-core** provide open models and research training infrastructure.
  Trashpanda's intended focus is the organization's approved inputs and evidence
  for *its* model, rather than another model family or general training framework.

[Detailed comparison, current gaps, and upstream sources →](docs/comparisons.md)

## Start here

- [CPU quickstart](docs/getting-started.md): exercise preparation, training, resume,
  evaluation, and offline export without a GPU or downloaded training data.
- [Implementation map](docs/implementation/README.md): source modules and current limitations.
- [Data and compute examples](docs/experiments.md): optional FineWeb-Edu and A100/Colab workflows.

Project code is [Apache-2.0 licensed](LICENSE). Data, tokenizer, and weight rights
require separate review. “Approved data” means approved under an organization's
policy—not automatically safe, legally cleared, or certified by Trashpanda.

For contributions, start with the [open issues](https://github.com/rawkintrevo/trashpanda/issues)
and distinguish fixes to current behavior from changes to the proposed designs.
