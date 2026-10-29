# trashpanda

Deterministic tiny/production Llama training from JSONL to native Hugging Face
files. Source contains no model or dataset assets.

## Offline tiny quickstart

Install dependencies in an environment with CPU PyTorch:

```bash
python -m pip install -e '.[test]'
python - <<'PY'
import json
text = " ".join(f"w{i % 29}" for i in range(128))
for name in ("train.jsonl", "validation.jsonl"):
    with open(name, "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"text": text}) + "\\n")
PY
python -m trashpanda prepare --train train.jsonl --validation validation.jsonl --output prepared --recipe tiny
python -m trashpanda train --data prepared --recipe tiny --output run
python -m trashpanda evaluate --checkpoint run/checkpoint-00000006.pt --data prepared
python -m trashpanda export --checkpoint run/checkpoint-00000006.pt --output exported
```

The tiny contract is vocab 32, 2 Llama layers, hidden 32, intermediate 64,
context 32, effective batch 4 (microbatch 2), FP32 CPU, and 744 shifted target
tokens. The project-owned WordLevel tokenizer has IDs 0/1/2 followed by `w0`
through `w28`; it never inserts BOS or EOS. Preparation writes little-endian
uint16 shards, never pads, and records the exact dropped tail per split.

The production `300m-a100` recipe constructs `transformers.LlamaForCausalLM`
from `LlamaConfig` with SDPA and tied embeddings. It has exactly **298,103,808**
parameters, vocab 32000, 22 layers, hidden 1024, intermediate 2560, context
2048, batch 128/microbatch 8, BF16 CUDA, 6B target tokens, and checkpoint
interval 500. Full production training is intentionally not part of the tiny
smoke test. A verified 120-step A100 benchmark and FineWeb-Edu run are recorded
in `reports/first-model-a100-benchmark.json`; the report explicitly distinguishes
remote verification from durable artifact retention.

## FineWeb-Edu adapter

With optional `datasets` and a locally cached pinned Mistral tokenizer:

```bash
python scripts/prepare_fineweb_edu.py fineweb-prepared \
  --tokenizer-cache /path/to/local/mistral-tokenizer \
  --train-documents 30000 --validation-documents 1000
```

The adapter streams FineWeb-Edu and writes only requested prepared output.
Tokenizer loading is local-only and uses `trust_remote_code=False`. Native HF
export uses the model's standard `save_pretrained` format and does not require
custom code.

## Strict overrides

Recipe defaults can be overridden only with a flat JSON object containing any
of `token_budget`, `microbatch_size`, `learning_rate`, and `repeat_data`:

```bash
python -m trashpanda train --data prepared --recipe tiny --output run \
  --tokens 744 --microbatch-size 2 --no-repeat-data
```

Unknown or nested keys are rejected. `--resume` resumes an atomic update-boundary
checkpoint containing model, optimizer, PCG64 sampler, config, RNG, and tokenizer
provenance. A replacement Colab run must copy the completed checkpoint/export to
an operator-provided durable destination before claiming model delivery.
The retained workspace artifact manifest is at
`artifacts/production-a100-120/artifact-manifest.json`; its 1.79 GB checkpoint
and 1.19 GB safetensors file are intentionally excluded from GitHub commits
because GitHub rejects files over 100 MB, but remain in the workspace artifact
directory with SHA-256 provenance.

## Provisioned A100 production driver

After the Colab CLI has provisioned an A100 and the repository is available on
that VM, run the complete configured target with:

```bash
python scripts/run_a100_production.py --root /content/fineweb-run --prepare
```

The driver defaults to the full 6,000,000,000-target `300m-a100` recipe and
writes the resumable checkpoint, metrics, evaluation, samples, and native HF
export beneath the selected root. Use `--max-steps` only for a bounded
benchmark.
