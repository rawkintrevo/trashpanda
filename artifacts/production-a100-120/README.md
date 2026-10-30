# Historical A100 and offline recovery records

These are **experiment records**, not a completed 6B-target model release or a
validated organizational provenance bundle. See the
[evidence and artifact availability guide](../../docs/experiments.md).

- `export/`: tracked HF configuration/tokenizer files; the large model payload is
  ignored by Git and is not included in a fresh clone.
- `checkpoint/verification.json`: metadata from a separate step-1 offline recovery
  experiment initialized from reported A100-exported weights. It does not contain
  the original step-120 optimizer history or prove a real post-reload training update.
- `report/`: retained evaluation and sampling results, not independently regenerated
  by the documentation changes.
- `run-metadata.json`: curated experiment/provenance description, not a record
  automatically produced by the current export path.
- `artifact-manifest.json`: expected sizes/hashes for the large payloads; no public
  download location or independent durability guarantee.

The recorded A100-source run used FineWeb-Edu `sample-10BT`, 30,000 training and
1,000 validation documents, and 120 updates / 31,441,920 prediction targets.
The checkpoint record describes a different state: one offline update / 2,047
targets. Do not treat this checkpoint and the export as an equivalent pair.

Workspace paths may hold deferred large-file pointers. Materialization and actual
payload verification are required before loading; see the linked guide. A saved
verification summary alone is not evidence of machine-loss recovery or completion
of First Model.
