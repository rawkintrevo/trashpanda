# What provenance exists today?

[Implementation map](README.md) · [Model-bound provenance proposal](../roadmap/provenance.md)

**Partially implemented:** Trashpanda records useful integrity and training metadata.
It does not yet deliver the model supply-chain evidence described in the vision.

| Location | Automatically recorded | Not established by that record |
| --- | --- | --- |
| Prepared `manifest.json` | Shard hashes/counts, tokenizer metadata and saved-file hashes, sequence format, manifest identity | Original source/version, usage rights, approval, acquisition time |
| Normal checkpoint `.pt` | Model/optimizer/scheduler/RNG/sampler state, counters, resolved config, prepared-data identity, tokenizer metadata and captured tokenizer files | UTC training dates, code/environment identity, approvals |
| `metrics.jsonl` | Step, targets, loss, learning rate, elapsed duration | Wall-clock timestamps, verifiable run identity, signed audit trail |
| Production driver `run-metadata.json` | Counters, elapsed duration, evaluation/samples, paths, recipe and dataset labels | Independently verified dataset lineage; dataset labels are hard-coded even when existing prepared data are used |
| Native HF export | Model config, weights, tokenizer/generation files, and `trashpanda-provenance.json` linking config/data identity to the source checkpoint | Timestamps, signatures, source rights or approval evidence |

Sources: [`prepare.py`](../../src/trashpanda/prepare.py),
[`checkpoint.py`](../../src/trashpanda/checkpoint.py),
[`train.py`](../../src/trashpanda/train.py),
[`run_a100_production.py`](../../scripts/run_a100_production.py),
[`export.py`](../../src/trashpanda/export.py).

## The important missing link

Prepared-data identity currently stops at preparation. Training does not bind that
identity into its checkpoints, and export does not carry it forward. The current
non-tiny `tokenizer_sha256` is a digest of selected metadata; separate file hashes
identify tokenizer bytes. Do not confuse these two kinds of hash.

Thus the checkpoint/export now bind prepared bytes and recovery state, but do not claim
complete source lineage, approvals, timestamps, or a signed audit record. Elapsed seconds are not
a training date. The HF dataset adapter does not pin an immutable dataset revision.

## Historical records are not an automatic provenance system

The files in [`reports/`](../../reports) and
[`artifacts/production-a100-120/`](../../artifacts/production-a100-120) are retained
experiment reports and curated manifests. No checked-in generator reproduces all
of those historical schemas. Treat their claims according to the
[evidence notes](../experiments.md), not as proof that every new run automatically
gets equivalent provenance or that the artifacts are downloadable from Git.

## Next boundary

The [proposal](../roadmap/provenance.md) makes dataset identity, training times,
parentage, software/configuration, and organization approval claims travel with
specific model files. Optional standards/registry integration and attestations
must remain distinct from legal approval and independent verification.
