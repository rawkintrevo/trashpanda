# First Model: implementation contract

Status: product/design choices approved; dependency spike verified on CPU. This document specifies future implementation, not an existing application. See TEST-CONTRACT.md, COMPATIBILITY.md, and ASF-LICENSE-REVIEW.md for validation and release gates. Epic: https://github.com/rawkintrevo/trashpanda/issues/1.

## Scope and ownership

Build one local JSONL -> scratch base-model training -> native Hugging Face export path. Own a small PyTorch loop; use Transformers' existing Llama implementation. No OLMo dependency, instruction tuning, distributed execution, cleaning/filtering/deduplication, custom tokenizer training, provenance registry, plugin framework, or automatic publication. Keep separate data preparation, sequence reader, model construction, training orchestration, checkpoint, and export modules so future distributed work does not require replacing the data contract. CPU is for tests; production target is one A100 80GB. No capability or four-day performance guarantee.

Suggested package files (not created by this planning work): src/trashpanda/{cli,config,model,prepare,data,train,checkpoint,evaluate,export}.py; tests/; recipes/{300m-a100,tiny}.json. Use argparse and JSON configuration to avoid new framework dependencies. Python 3.11, Linux x86_64 baseline. Use the recorded compatibility pins; GPU packaging is a separately verified variant, not the CPU freeze installed on an A100.

## CLI and configuration

```
trashpanda prepare --train train.jsonl --validation validation.jsonl --output prepared/
trashpanda train --data prepared/ --recipe 300m-a100 --output runs/first-model/
trashpanda train --resume runs/first-model/
trashpanda evaluate --checkpoint runs/first-model/checkpoints/step-00000500 --data prepared/
trashpanda export --checkpoint runs/first-model/checkpoints/step-00000500 --output models/first-model/
```

`prepare` accepts `--recipe {300m-a100,tiny}` (default 300m-a100) to select the fixed sequence length/tokenizer contract. It also accepts repeated `--train` and `--validation` arguments; preserve argument order, then record order within each file. Paths are local files, not URLs. For production, `--tokenizer-dir` allows a local copy of the three pinned raw tokenizer files, validated against the fixed SHA-256 constants below (ship them in the package; no planning-file runtime dependency); otherwise retrieve only those pinned assets and verify the same hashes. Reject `--tokenizer-dir` together with tiny, which always uses the owned builder. `--recipe tiny` generates the project-owned fixture tokenizer locally and uses the short sequence recipe. `train --config FILE` reads a JSON override object and accepts explicit `--tokens`, `--microbatch-size`, and `--no-repeat-data`. Precedence: recipe defaults < config file < explicitly supplied flags. Reject unknown keys, wrong types, nonpositive sizes/budgets, unsupported device/precision combinations, incompatible data/model/tokenizer, and invalid microbatch divisibility. Persist the fully resolved configuration before training. Never silently ignore a flag.

Allowed training JSON override keys (flat object; no others):

| JSON key | Type/constraint | 300m-a100 default | tiny default | CLI override |
| --- | --- | --- | --- | --- |
| token_budget | integer >0, bool rejected | 6000000000 | 744 (6 steps) | --tokens |
| microbatch_size | integer >0 dividing fixed effective batch | 8 | 2 | --microbatch-size |
| learning_rate | finite float >0 | 0.0003 | 0.001 | config only |
| repeat_data | bool | true | true | --no-repeat-data sets false |

Effective batch is fixed at 128 production / 4 tiny; context 2048 / 32; precision/device BF16-CUDA / FP32-CPU; evaluation/checkpoint interval 500 / 3 steps; generation limit 128 / 8. All other recipe constants, including seed=42, are immutable public settings in v1. Initialize Python, NumPy and torch RNGs from seed 42; sampler permutations use PCG64(seed+epoch). No public model architecture overrides. Unit tests may construct internal configurations with small B/L, but these are not accepted public JSON keys. `prepare` accepts no training config. `evaluate`/`export` resolve settings from their checkpoint, not recipe flags. New train requires data/output, defaults recipe to 300m-a100, and accepts the four fields above; reject resume combined with new-run options.

Complete offline tiny example (each JSONL contains enough repeated w0/w1/w2/w3 words for >=32 tokens, and train/validation are distinct files):

```
trashpanda prepare --recipe tiny --train tiny-train.jsonl --validation tiny-validation.jsonl --output tiny-data/
trashpanda train --recipe tiny --data tiny-data/ --output tiny-run/
```

Resume mode accepts only `--resume` and operational logging verbosity; model, tokenizer, data, token budget, schedule, seed, precision and batch configuration cannot change. A changed run requires an explicit future continuation feature, not v1 resume. New output directories must not exist; fail without modifying existing contents. On failure during preparation, leave only a clearly named temporary sibling that is not a valid dataset and explain cleanup. Exit 0 success, 2 usage/data/config errors, 1 runtime failures, 130 graceful SIGINT, 143 graceful SIGTERM. Errors name the path/record/setting and a remedy. Signal handlers only record stop intent; if a signal arrives before/during step-zero evaluation, finish that evaluation then publish step-zero checkpoint and exit; during an optimizer step finish its accumulation and update then checkpoint. If completion and signal coincide, coalesce checkpoint/evaluation and use the signal exit status. Do not promise recovery from SIGKILL.

## Model and tokenizer

Use LlamaForCausalLM(LlamaConfig(...)), random initialization only:

| Setting | Value |
| --- | --- |
| vocabulary | 32000 |
| layers | 22 |
| hidden / intermediate | 1024 / 2560 |
| attention / KV heads | 16 / 16 |
| tied embeddings | true |
| attention / MLP bias | false / false |
| activation | silu (SwiGLU feed-forward) |
| norm | RMSNorm epsilon 1e-5 |
| attention dropout | 0 |
| initialization std | 0.02, Transformers implementation |
| max positions / context | 2048 |
| RoPE theta / scaling | 10000 / none |
| train use_cache | false |
| BOS / EOS / PAD | 1 / 2 / unset |

Actual unique trainable parameter count MUST be 298,103,808 including tied embeddings (verified with meta construction). Use SDPA, not custom kernels. Do not silently fall back to a different architecture. Enforce the context limit in our application; RoPE/HF configuration alone is not a hard input-length guard.

Tokenizer: mistralai/Mistral-7B-v0.1 at revision `27d67f1b5f57dc0953326b2601d68371d40ea8da`. Retrieve tokenizer.json, tokenizer_config.json and special_tokens_map.json; no model weights. Use AutoTokenizer fast loading. Vocabulary 32000, UNK=0, BOS=1, EOS=2. Encode documents with `add_special_tokens=False, split_special_tokens=True`, then append exactly one EOS ID. This makes literal `<s>`, `</s>`, `<unk>` in source ordinary text. No BOS insertion, no vocabulary addition. Save tokenizer assets in prepared data and export. Padding is absent in training. For generation only, pass pad_token_id=EOS with an explicit attention mask; do not mutate the saved tokenizer's vocabulary.

Ship these raw-download integrity constants in `model.py` (or a package resource), not a dependency on this planning directory:

| File | SHA-256 |
| --- | --- |
| tokenizer.json | 11c08db21487c885d8c792180f0be237f6a261b89a46f128a6a80a3aa4bd1720 |
| tokenizer_config.json | ddb008229511e51607002ffe28925001c4a9ca4177dc4de3a655d085cc610b99 |
| special_tokens_map.json | 6fa06efa2785e450051989a6f8fb4416b10149ded485ddd3f127a40734f5cfd0 |

These hash upstream raw inputs; saved normalized tokenizer files get their own manifest hashes.

Tiny test recipe: vocab=32, layers=2, hidden=32, intermediate=64, heads=KV_heads=4, context=32; other model semantics unchanged. Ship a project-owned deterministic tokenizer builder, not generated JSON assets. Build WordLevel with Whitespace pretokenizer: IDs 0/1/2 are <unk>/<s>/</s>, IDs 3..31 are w0..w28 in order. Register the three special tokens; no postprocessor. Generate tokenizer files only into runtime/test output directories. Literal markers with splitting enabled can map to UNK in this deliberately limited vocabulary; assert no special recognition rather than lossless arbitrary-text coverage for tiny. No downloaded artifacts in routine CI. Tests use explicit tiny LR/budget overrides; tiny learning speed is not production tuning evidence.

## Data format v1

Train and validation are mandatory separate inputs. Reject identical input paths across splits; do not claim to detect duplicated content. Each nonblank line must be a JSON object with a nonempty string `text`; ignore other fields. Blank lines, malformed JSON, missing/non-string/empty text fail with file and 1-based line number. Preserve whitespace-only strings and all source text. No cleanup/filtering. Encode bounded groups of records (maximum 256 records per group; process a single long document individually) and write incrementally; memory scales with shards/index and largest document, not the whole corpus.

Outputs: `manifest.json`, `tokenizer/`, `train/000000.bin`, `validation/000000.bin`, etc. Shards are raw little-endian uint16 arrays, at most 64,000,000 tokens each. Stream document tokens + EOS continuously within each split. Shard boundaries have no semantic meaning and must not introduce EOS, padding, or discarded tokens.

Manifest required fields:
- format_version=1; token_dtype="<u2"; sequence_length; tokenizer source/revision, vocab size, special IDs, encoding options, and SHA-256 of saved tokenizer files.
- Per split: total_tokens (before final sequence truncation), usable_sequences=floor(total_tokens/L), dropped_tail_tokens=total_tokens mod L; ordered shards with relative path, token count, byte length and SHA-256.
- Reject path traversal, missing files, byte-length/hash mismatches, unsupported version/dtype, and token IDs outside vocabulary. Validate hashes once at train/evaluate startup. Identity is a SHA-256 of canonical sorted-key JSON manifest bytes; do not treat this as legal provenance.

Reader boundary: immutable `len(dataset)` complete sequences and `dataset[index] -> int64 tensor[L]`. Memory-map shards and gather across boundaries. Packing concatenates documents; attention can cross EOS/document boundaries. Drop and report only the incomplete final sequence of each split. Require at least one complete sequence in both splits. Never pad training sequences. With standard HF causal labels equal to input_ids, L tokens yield L-1 prediction targets; first token of each sequence has no target and no target crosses a sequence boundary. This is the deliberate token accounting convention, not all corpus tokens contributing losses.

Order: NumPy PCG64 permutation of sequence indices with seed `42 + epoch` (epoch starts at zero). Batch sampler consumes that order, spanning an epoch boundary if needed; save epoch and cursor at the next unconsumed sequence. Repetition defaults on. If requested training exceeds one pass, print a prominent warning before starting showing available sequences, planned consumption, approximate passes, and `--no-repeat-data`; log each repeat. With repetition disabled, reject an insufficient sequence count for the rounded full-step budget before any training. Validation never repeats and uses the first min(128, N_validation) sequences in stored order.

## Training and accounting

Production: CUDA device 0, BF16 autocast, FP32 parameters/optimizer state; no GradScaler for BF16. Require CUDA and BF16 support for production recipe. Tiny CPU uses FP32. AdamW: lr=3e-4, betas=(0.9,0.95), eps=1e-8, weight_decay=0.1 for parameters with ndim>=2 (including the single tied embedding parameter), zero otherwise. Include each parameter once. Clip global gradient norm to 1.0 after accumulation and before optimizer step. Use zero_grad(set_to_none=True). Keep tensors/model device handling in orchestration, not hard-coded into data preparation.

Effective batch B=128 sequences; initial microbatch m=8; accumulation=B/m=16. Require m divides B. All sequences equal length, so divide each mean microbatch loss by accumulation. Optimizer advances only after all microbatches. Budget T defaults to 6,000,000,000 prediction targets. Targets per optimizer step K=B*(L-1)=262,016. Planned steps S=ceil(T/K); planned targets=S*K. Print requested and rounded totals. Counters update only after a successful optimizer step; discarded partial accumulation on crash is replayed and never double-counted as completed training. No implicit unbounded run.

Schedule is keyed to 1-based optimizer step s. W=min(S, max(1,ceil(0.01*S))). Set the LR for the upcoming step before its update:
- s<=W: peak_lr*s/W.
- s>W: min_lr + 0.5*(peak_lr-min_lr)*(1+cos(pi*(s-W)/(S-W))), min_lr=0.1*peak_lr.
For S=W there is no decay branch. Save schedule parameters and completed step; resume computes the next LR from the same formula. Reject attempts to change S on resume.

Fail on nonfinite loss/gradients/gradient norm. Do not skip optimizer updates invisibly. Log JSONL each optimizer step with step, targets, epoch/cursor, mean training loss, LR, preclip norm, elapsed time, interval targets/sec and cumulative end-to-end targets/sec. Synchronize CUDA around timing measurements. Throughput denominator must include checkpoint/evaluation time for end-to-end reports. No external logging account.

## Run layout and checkpoint contract

Run directory: resolved-config.json, data-identity.json, metrics.jsonl, samples.jsonl, run-state.json, checkpoints/. Retain immutable configuration and tokenizer/data identity; checkpoint payloads identify these with hashes. Checkpoint at the recipe interval (500 production / 3 tiny optimizer steps), normal completion, and graceful interruption at the next optimizer boundary. Step zero has evaluation but need not have a checkpoint.

Checkpoint `checkpoints/step-00000500/` contains a self-contained `tokenizer/` snapshot copied from prepared data, standard model config + safetensors, optimizer/schedule state, Python/NumPy/torch CPU/CUDA RNG state, sampler epoch/cursor, completed steps/targets, run/config/data/tokenizer identity and a versioned completion manifest with file hashes. Use primitive/tensor encodings compatible with `torch.load(weights_only=True)` for state (convert NumPy arrays to supported representations); never require unsafe arbitrary-object deserialization. Load only locally trusted training checkpoints. Write a unique temporary sibling, flush/fsync files as supported, write completion metadata last, then same-filesystem rename. Document local POSIX filesystem requirement; object-store atomicity is out of scope. Refuse incomplete/corrupt checkpoints. Resume run directory selects highest completed valid step; report skipped unfinished temporary directories, but corruption in a completed checkpoint is an error, not a silent rollback.

A completed checkpoint is independently relocatable for offline export: it contains tokenizer assets and all needed configuration/state; the original prepared-data directory is not required for export. Training resume still requires the original data identity and accessible recorded data path.

Coalesce interval/final/signal checkpoint requests per completed step. Never overwrite an existing completed step directory; validate and reuse it if its identities/counters match, otherwise fail. Completed checkpoint metadata is recovery authority; run-state.json is only a reconstructible summary. A crash after rename and before summary update resumes the published checkpoint. On resume of an already completed budget, validate artifacts and exit 0 without another training step/checkpoint or duplicated metrics/evaluation. Event order at a boundary is successful update/counter -> due evaluation/sampling -> single checkpoint -> summary -> stop/continue. Only resume-authoritative counters determine committed progress; evaluation failure cannot falsely publish a completed artifact.

Keep latest 3 completed checkpoints. Delete only older recognized checkpoint directories after new checkpoint succeeds, never unrelated files or in-flight directories. Restored state includes data cursor and RNG, not merely weights. Log a resume event; truncate or logically exclude metrics after restored step to avoid duplicate successful-step records. No promises of bitwise BF16 cross-device reproducibility.

## Evaluation and export

Evaluate at step zero, at the recipe interval (500 production / 3 tiny steps), and completion; if final step equals an interval evaluate once. Use no_grad/inference mode, model.eval, fixed validation sequences, and no training-data fallback. Accumulate sum(loss * prediction_targets)/sum(prediction_targets); padding targets (if future fixtures exercise them) use -100 and are excluded. Report loss and exp(loss); on overflow represent perplexity as null with an overflow flag, not invalid JSON Infinity. Save and restore RNG and model mode. Prompt list: "The small animal", "Once upon a time", "Today I learned", "The purpose of this experiment is", "In the morning". Greedy generation, max_new_tokens=128, EOS stops, seed=42 recorded, explicit attention mask; truncate prompt from the left if needed to keep prompt+generation within context. Tiny tests override max_new_tokens=8. Record prompts, IDs, settings, outputs, checkpoint step.

`evaluate --checkpoint PATH --data DIR` uses the same bounded checks and prints JSON; no instruction/agent benchmark suite. `export` accepts only a completed compatible checkpoint, refuses an existing output, writes standard config.json, model.safetensors (HF sharding allowed if needed), generation_config.json and tokenizer files, plus applicable license/notice material and a brief model card with base-model status, architecture, context and actual training counters. No optimizer state, remote Python classes, or automated upload. Verify tied embeddings, token IDs, vocabulary and context. Final HF generation may enable caching even though training disables it.

## Private signal-test seam

The trainer's internal Python entry accepts an optional `on_microbatch(step_1based, microbatch_1based, should_stop)` callback, invoked after backward and before starting the next microbatch/update. Public CLI never exposes it and production passes None. A test-only subprocess wrapper invokes the real CLI parser/orchestration with this dependency injected; it uses inherited pipe file descriptors (not environment flags in production) for synchronization. At step 3, microbatch 1 of 2, callback writes READY, polls the supplied zero-argument `should_stop()` predicate until the parent's real SIGINT/SIGTERM sets the normal stop flag, writes ACK and returns. Parent uses select with a 30-second timeout and kills/reaps on failure. Trainer then completes microbatch 2, updates step 3, checkpoints, exits 130/143; planned steps remain 6. This seam is an injectable callback, not a separate training loop or public early-stop option. Also test signal before first update and completion/interval collision. Runtime RNG/sampler state is unmodified by the callback.

## A100 benchmark and run gate

Use the fixed production architecture and tokenized input. Benchmark invocation: `trashpanda train --data prepared/ --recipe 300m-a100 --tokens 31441920 --output runs/benchmark/` (120*K; production default remains 6B). A benchmark harness using an internal optimizer-boundary timing callback warms up 20 full optimizer steps, synchronizes CUDA and resets the wall timer/peak-memory stats immediately after step 20; measure steps 21..120. Stop timing only after the normal completion evaluation/sampling, checkpoint and summary succeed. Thus measured numerator is exactly 100*K=26,201,600 prediction targets, with final checkpoint/evaluation overhead included once. Step-zero evaluation and startup/preparation are excluded and reported separately. This 120-step schedule is for throughput calibration, not a quality run; use the full-budget schedule for actual training. The timing seam is optional `on_event(event, completed_step, completed_targets)`, with events `step_complete` (after boundary work) and `run_complete` (after final summary); default None. Benchmark harness resets on step_complete(20), stops on run_complete(120), and never alters data/training behavior. Record GPU model/memory, driver/CUDA/PyTorch versions, batch/accumulation, SDPA backend, peak allocated/reserved memory, storage characteristics and wall time. If m=8 OOMs, restart cleanly with m=4, then 2, then 1, keeping B=128. Never catch an OOM and continue with potentially partial accumulation. If all fail, stop and escalate; do not silently reduce model, context or effective batch.

Compute 6B / measured end-to-end targets/sec and achievable targets in 2/3/4 days. Four days needs about 17,400 targets/sec; two days about 34,700. Verify interrupted/resumed GPU smoke and independent export load. If measured projection exceeds four days, request a user choice of smaller budget or longer runtime before the full run. User provides GPU access, train/validation files and durable artifact destination. No paid infrastructure provisioning or public upload is authorized. A successful code implementation without an actual trained/exported model does not close First Model.

## Dependencies and release boundaries

Do not vendor dependency code or model/tokenizer assets into the project source release. Tokenizer acquisition is explicit; project-owned fixtures keep CPU CI offline. Use the verified CPU freeze as exact reproduction evidence, not an audited GPU lock or a claim that older pins are security-current. Issue #2 owns the reproducible CPU hash lock, intended GPU install inputs, vulnerability checks, third-party license inventory, and source-archive inspection before release. Issue #10 owns resolution/hash locking and verification of the actual CUDA variant; #2 does not wait on its downstream benchmark. Follow the separate ASF licensing review; unresolved classifications or bundling rights require maintainer review. Apache-2.0 project code does not establish rights to user data or generated weights. No claim of ASF affiliation, acceptance or compliance certification.
