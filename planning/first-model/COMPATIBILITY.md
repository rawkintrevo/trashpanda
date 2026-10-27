# Dependency compatibility evidence (CPU only)

## Result

Python 3.11.2, Linux x86_64. `python -m pytest planning/first-model/test_compatibility.py -q`: **7 passed in 11.65s**. `pip check`: no broken requirements. A separate login-shell process successfully imported torch and transformers via `/workspace/.venv/bin/python`. No CUDA device exists in this runner. Environment restart survival has NOT been verified; the previous restart removed the earlier environment, so keep these reproduction steps rather than assuming persistence. `.venv` is already gitignored and is not access isolation.

Verified baseline:

| Package | Version |
| --- | --- |
| torch | 2.14.1+cpu |
| transformers | 5.21.0 |
| tokenizers | 0.23.3 |
| huggingface-hub | 2.3.0 |
| numpy | 2.2.6 |
| safetensors | 0.8.0 |
| pytest | 9.1.1 |

`compatibility-cpu.freeze.txt` records ALL resolved Python package versions; `environment-evidence.json` records interpreter/platform, tokenizer SHA-256 values and initial license metadata. The freeze is a reproducible version snapshot, NOT a hash-verified distribution lock, full bundled-license audit, security clearance, or GPU dependency lock. Do not infer that these versions are the latest or free of known vulnerabilities. Before release, #2 must run dependency security/license checks and produce the CPU hash lock plus intended GPU install inputs; #10 owns the actual GPU hash lock and hardware validation. #2 does not wait on downstream #10. A change in baseline requires rerunning these tests, not arbitrary implementation choices by a junior developer.

The initial torch 2.6.0 / Transformers 4.51.3 candidate passed technical tests but matched OSV advisories; it was replaced, not accepted as the default. The current pins above passed the same seven checks. A fresh OSV version query returned **zero matches across 36 CPU runtime/test packages** (see security-advisory-matches.json for timestamp and scope). This is not a security guarantee or a native/GPU audit. Direct baseline roots are torch, transformers, numpy, safetensors and pytest; tokenizers/huggingface-hub are pinned transitive components. No Transformers extras are enabled. The newer stack no longer requires certifi/requests; those and their unused dependencies were removed, `pip check` and tests rerun. TLS verification was not disabled; the new HTTP stack uses system trust via truststore.

Tests performed:
1. Production model meta construction: exactly 298,103,808 parameters and tied embeddings (no full production-size training allocated).
2. Actual pinned Mistral fast tokenizer: 32K vocabulary; IDs 0/1/2; special strings encoded as ordinary text using split_special_tokens=True; Unicode roundtrip; local save/reload.
3. Tiny CPU SDPA causal behavior, agreement with eager attention, manual shifted loss agreement.
4. Equal-length microbatch accumulation vs full-batch gradient agreement.
5. Tiny CPU repeated-pattern learning (loss below half initial after 25 updates with test-only LR).
6. Native model/optimizer/scheduler/generator/RNG serialization primitives: 6-step vs 3+3 split run, identical CPU weights and order. This does NOT test future application checkpoint atomicity, sampler implementation or all RNG domains.
7. Standard safetensors + tokenizer export, fresh offline subprocess native Auto classes, logits agreement and generation, no trust_remote_code.

Not tested: actual trashpanda application (not implemented), production training quality, 6B-token run, A100 BF16/SDPA kernels, CUDA dependencies, memory/performance, real application resume, source release license inventory, or right to redistribute user-derived weights.

## Reproduce

Use an isolated Python 3.11 environment. Installation needs network; the downloaded tokenizer integration check uses local files afterward. Do not install this CPU torch build for A100 use.

```
python3.11 -m venv .venv
.venv/bin/python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r planning/first-model/compatibility-cpu.freeze.txt
```

This runner lacks ensurepip. For this runner only, the actual setup used `python -m venv --without-pip .venv`, downloaded https://bootstrap.pypa.io/get-pip.py to a temporary file and invoked it with `.venv/bin/python`. No OS packages or system Python changes were made. A normal environment with venv support does not need this bootstrap workaround.

Download only three files into `.venv/compat-tokenizer/`:

```
https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/27d67f1b5f57dc0953326b2601d68371d40ea8da/tokenizer.json
https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/27d67f1b5f57dc0953326b2601d68371d40ea8da/tokenizer_config.json
https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/27d67f1b5f57dc0953326b2601d68371d40ea8da/special_tokens_map.json
```

Verify their SHA-256 values against environment-evidence.json, then:

```
.venv/bin/python -m pip check
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest planning/first-model/test_compatibility.py -q
```

To regenerate environment/license hashes and query the public OSV service, run `.venv/bin/python planning/first-model/collect_evidence.py`; this sends only package names/versions. Recreate `compatibility-cpu.freeze.txt` with `.venv/bin/python -m pip freeze`. Installation and advisory querying need network; offline compatibility tests do not.

The pinned model repository declares apache-2.0 in its model-card metadata, but its listed tokenizer artifacts do not include a standalone LICENSE/NOTICE. This check establishes technical compatibility, not a complete tokenizer artifact rights review. Keep downloads out of project source releases and complete the artifact-specific review before redistribution. No pretrained model weights were downloaded.
