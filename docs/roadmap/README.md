# Roadmap and proposed implementations

[Executive summary](../../README.md) · [Current implementation](../implementation/README.md)

These are directions confirmed by the maintainer, with proposed technical designs.
They are not a release schedule, shipped capabilities, or approved final protocols.

| Workstream | Starting point | Proposed next step |
| --- | --- | --- |
| [Complete First Model](first-model.md) | Local Llama pipeline and bounded experiment records | Harden recovery/validation and deliver the actual full-budget run with independent durable artifacts |
| [Model-bound provenance](provenance.md) | Shard/tokenizer hashes, checkpoint state, and partial logs | Carry dataset/run lineage, training dates, and approval evidence with model files; evaluate Sourcelume compatibility |
| [Selectable models](model-selection.md) | Two fixed Llama recipes | Add explicit model adapters/selection without abandoning provenance and recovery contracts |
| [Federated training](federated-training.md) | Single-machine training only | Design cross-organization training with local proprietary data, explicit threat model, and validated protections |

## Sequencing and decisions

Recovery and truthful artifact identity are foundations for model-bound provenance.
Additional model adapters should preserve that contract. Federation needs both,
plus separate security/privacy and governance design; keeping data local is not
by itself a confidentiality guarantee.

The proposed designs deliberately list open decisions and acceptance evidence.
Details such as CLI spelling, adapter API, record schema, aggregation algorithm,
and signing system remain proposals. Implementation should be scoped into reviewed
issues rather than silently turning these write-ups into advertised functionality.

First Model's exclusions keep that milestone small. They do not define permanent
exclusions for Trashpanda's broader mission.
