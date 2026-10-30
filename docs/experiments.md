# Experiments, compute examples, and artifact availability

[Executive summary](../README.md) · [First-model completion plan](roadmap/first-model.md)

Colab and FineWeb-Edu are examples used while developing the first milestone.
They are not Trashpanda's product definition or required providers.

## Historical evidence, not a completed production release

| Retained record | What it reports | Interpretation |
| --- | --- | --- |
| [`first-model-a100-benchmark.json`](../reports/first-model-a100-benchmark.json) | A100 80 GB; 120 updates; 31,441,920 targets; 39,213.5 measured targets/sec; validation loss 6.5309 on 262,016 targets | A bounded historical benchmark. This report explicitly says its remote artifacts were not retained. |
| [`production-a100-120/run-metadata.json`](../artifacts/production-a100-120/run-metadata.json) and [`report/`](../artifacts/production-a100-120/report) | Another recorded A100-source result: 120 updates / 31,441,920 targets, loss 6.10694 / perplexity 448.963 on 1,119,709 validation targets | Keep this separate from the first report: different evaluation scope and recorded result. No quality comparison is established. |
| [`checkpoint/verification.json`](../artifacts/production-a100-120/checkpoint/verification.json) | Step 1 / 2,047 targets, 200 optimizer-state entries, sampler position after loading | Metadata for a separately initialized offline recovery experiment, not the source run's step-120 optimizer history. It records no post-reload loss, gradients, parameter delta, or incremented training step. |

These are curated records from earlier experiments, not independently rerun
measurements in this documentation change. Their presence does not prove automatic
provenance capture, faithful GPU recovery, or completion of the configured 6B targets.
The [saved samples](../artifacts/production-a100-120/report/samples.json) are diagnostic
outputs, not a model capability benchmark.

## What a fresh clone contains

Git tracks code, small tokenizer/configuration files, reports, and the
[artifact manifest](../artifacts/production-a100-120/artifact-manifest.json).
It does **not** deliver the ignored large payloads:

| Workspace-relative path | Recorded size |
| --- | --- |
| `artifacts/production-a100-120/export/model.safetensors` | 1,192,437,904 bytes |
| `artifacts/production-a100-120/checkpoint/checkpoint-00000001.pt` | 1,788,917,565 bytes |

The manifest records their expected hashes, not a public download location or a
backup guarantee. Obtain the actual payloads from an authorized artifact store and
verify their hashes before loading. No public retrieval URL is currently documented.

In a Mapache workspace these paths may contain `mapache-large-file-v1` deferred
pointers rather than model bytes. Where the backing blob is available, materialize
it with `mapache files pull <path>` before loading, then verify the manifest hash.
That platform operation is not available to a normal Git clone and is not a
Trashpanda storage implementation. Do not hash a pointer and mistake it for the
model payload.

The `.pt` metadata and the source HF export describe different training states.
Do not use them as an export/checkpoint equivalence pair. Read historical
verification summaries within these explicit evidence limits.

## Optional FineWeb-Edu preparation

With the project's dependencies installed, add the optional dataset dependency:

```bash
python -m pip install -e '.[datasets]'
python scripts/prepare_fineweb_edu.py fineweb-prepared \
  --train-documents 30000 --validation-documents 1000 --allow-download
```

This explicitly downloads tokenizer assets and streams a dataset. If following the
offline quickstart in the same shell, unset its offline environment flags first.
Alternatively, provide a local tokenizer using `--tokenizer-cache` and omit
`--allow-download`; the dataset stream still needs network access. The adapter does
not download pretrained model weights.

The prepared sample contains far fewer than 6B unique tokens; repeat-enabled
training reuses sequences. Dataset approval, immutable source pinning, and
license/privacy review are not supplied by this example.

## Optional A100/Colab driver

The driver expects an already provisioned CUDA/BF16 environment and checked-out
source. It does not authenticate, provision, preserve a VM, or copy artifacts to
independent storage. For a deliberately bounded experiment:

```bash
python scripts/run_a100_production.py \
  --root /content/fineweb-benchmark --prepare --max-steps 120
```

Omitting `--max-steps` requests the 6B-target recipe; it is not evidence that the
run has completed. Before a long run, address the
[training/recovery limitations](implementation/training.md), plan independent
checkpoint copies, and verify the actual CUDA environment and applicable dependency
licenses. CPU tests do not validate a GPU installation or guarantee runtime/cost.

The driver writes checkpoints and metrics under `checkpoints/`, creates `export/`,
and embeds evaluation and samples in `run-metadata.json`. It does not generate the
historical `report/` layout above. Its dataset labels are hard-coded even when
reusing prepared data, and it has no resume argument. Use the core CLI with explicit
matching settings for resume; do not restart this driver and assume automatic recovery.
