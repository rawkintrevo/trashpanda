# A100 production and recovery artifacts

This directory contains the verified native Hugging Face export and a complete
atomic recovery checkpoint:

- `export/`: native HF Llama export and tokenizer
- `checkpoint/checkpoint-00000001.pt`: model, AdamW optimizer state, RNG
  state, PCG64 sampler/data position, tokenizer snapshot, and full config
- `checkpoint/verification.json`: independent offline reload/resume probe
- `report/`: CUDA evaluation and sampling outputs
- `run-metadata.json`: dataset, hardware, hashes, and provenance

The source A100 run used `HuggingFaceFW/fineweb-edu`, config `sample-10BT`,
30,000 training documents, 1,000 validation documents, BF16 A100 hardware,
120 updates, and 31,441,920 target tokens. Its model has exactly 298,103,808
parameters and validation perplexity 448.963151142471.

The retained checkpoint is a complete recovery checkpoint built from those
verified A100 weights and advanced through one real offline optimizer update.
The verification reloads it, restores optimizer/RNG/sampler/tokenizer/config
state, and successfully advances the sampler and optimizer after reload.
