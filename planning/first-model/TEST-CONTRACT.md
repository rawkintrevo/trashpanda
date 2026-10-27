# First Model test contract

These are acceptance tests to IMPLEMENT with the application. Only the seven dependency-spike tests in test_compatibility.py have run so far. Routine CI is CPU-only, offline, single-threaded, FP32, seed 42, using a locally generated project-owned WordLevel tokenizer (32 IDs) and synthetic JSONL. Do not download Mistral assets in routine CI; the separate integration check uses pinned assets.

## Numerical conventions

- CPU deterministic split-run checkpoint/resume: identical consumed sequence indices, counters, schedule and parameter tensors (`rtol=0, atol=0`) under the same environment, threads, device and seed. Restore all RNG domains and compare their next draws.
- CPU export logits / SDPA-vs-eager / unchanged causal prefix: `atol=1e-6, rtol=1e-5`.
- FP32 gradient accumulation vs full batch: `atol=1e-6, rtol=1e-4`.
- GPU BF16 export-vs-checkpoint logits, on the same GPU/dtype/backend: `atol=2e-2, rtol=2e-2`. Record max absolute error. GPU resume guarantees identical data/step/schedule state; compare same-step validation losses with `atol=1e-2, rtol=1e-2`, not cross-device bitwise weights. If tolerances fail, investigate/report rather than relax silently.
- Scalar schedule values: `abs(actual-expected) <= 1e-12` in Python float calculations. Token counts, labels, IDs, sampler state, manifests and hashes are exact integers/bytes.

## Per-issue acceptance matrix

### #2 Package, CLI, environment and licensing

1. Install package in a fresh Python 3.11 environment; console script and `python -m trashpanda --help` both exit 0. All subcommands have help, unknown flags/config keys exit 2.
2. Tests enumerate all four allowed JSON keys (token_budget, microbatch_size, learning_rate, repeat_data), reject unknown/nested/model override keys, and prove precedence defaults < file < explicit flags; omitted bool flags do not overwrite file values. Tiny defaults are B=4, m=2, L=32, budget=744 (6 steps), LR=.001, interval=3, max_new_tokens=8; production defaults match SPEC. Invalid microbatch divisibility, zero/negative/bool budgets, nonfinite LR and unsupported dtype/device fail before outputs are created.
3. CPU/CUDA installation instructions select their intended wheels without silent CPU-on-GPU substitution. Run `pip check`. Hash-lock the resolved CPU runtime/test dependencies here and define the intended GPU install inputs. Issue #10 owns resolution, hash locking and verification of the actual CUDA variant so this issue does not depend on a downstream benchmark. Do not declare CUDA verified from CPU resolution.
4. Build source archive and wheel of our code; inspect member lists for absence of `.venv`, downloads, weights, tokenizer assets, user data, credentials and planning runtime outputs. Check Apache license, applicable notices and project-owned source headers. Require shipping the deterministic tiny tokenizer builder/vocabulary, not generated tokenizer JSON assets. The installed-wheel prepare/train offline integration is owned by #9 after #3-#8 exist; it is not a prerequisite for closing scaffold #2. Dependency license report includes exact versions, underlying license text locations, bundled components, distribution boundary, ASF categorization/evidence and review status.
5. Dependency security scan before release; unresolved vulnerabilities get documented maintainer disposition, not automatic acceptance of this spike's versions.

### #3 Model and tokenizer

1. Meta-device production construction yields exactly 298,103,808 parameters and the full SPEC configuration; tied embedding weights are the same parameter object and occur once in optimizer groups.
2. Tiny forward/backward finite; changed suffix leaves earlier logits unchanged within causal tolerance. Wrapper rejects input length > configured context.
3. Local tokenizer fixture tests no automatic BOS/EOS, exactly one appended EOS per record, ordinary encoding of literal special-token strings, Unicode preservation where fixture supports it, and no vocabulary growth.
4. Separate pinned Mistral integration test verifies 32000 entries, IDs 0/1/2, `split_special_tokens=True` on `<s> hello </s> <unk> café 熊`, exact decoded text and absence of special IDs in encoded source before appended EOS. Local save/reload agrees. Download tests mock network and assert no weight filenames requested. Run mocked production acquisition from an installed wheel outside the repo: packaged SHA-256 constants match SPEC, corrupted raw downloads fail, and no planning/evidence file is read at runtime.

### #4 Preparation and sequence reader

1. Two training files and one validation file preserve specified file/line order; extra JSON fields are ignored. Fixtures cover invalid JSON, blank line, list instead of object, missing text, null/number/empty text, whitespace-only text, UTF-8 and extremely long documents. Assert path and 1-based line diagnostics. Train/validation identical path fails; no semantic deduplication test is promised.
2. Test shard_size=7, sequence_length=4 with a hand-computed token stream of 19 tokens. Expect shard lengths [7,7,5], four exact sequences, final tail=3. No tokens disappear at shard boundaries and no padding is introduced. Unit-level test override only; production shard size remains fixed.
3. File lengths equal token counts*2; explicit little-endian roundtrip includes IDs >255. Verify tokenizer hashes, shard hashes and canonical manifest identity. Missing file, modified byte, wrong dtype/version, out-of-range token and `../` path are rejected.
4. Repeat prepare in two fresh directories: normalized manifest and shard bytes match (no timestamps/absolute paths in identity). Existing output refused. Inject a write failure and ensure no directory with a valid completion manifest is exposed.
5. Reader covers random indexing across files; no code path materializes all corpus tokens. Linux memory regression test uses two fresh subprocesses with identical local tiny tokenizer and 1,024-character records, separately preparing approximately 16 MiB and 128 MiB of source text. Parent streams fixture generation to disk before spawning each child. Measure child peak RSS with resource.getrusage(RUSAGE_SELF).ru_maxrss in KiB; require large-run peak minus small-run peak <=64 MiB. Require at least one shard boundary using an internal test shard-size override, and record both absolute peaks. Keep this separate from tokenizer/model loading tests; no whole-corpus allocation in the parent/child.

### #5 Training

1. Manual cross-entropy on logits[:, :-1] and labels[:, 1:] equals HF loss; targets per sequence=L-1. On L=4, B=2 and requested T=7, K=6, S=2, actual targets=12. Verify printed rounding and recorded counters.
2. With fixed equal-length examples, m=B and m=B/2 accumulation produce matching gradients. Optimizer advances once per full accumulation; norm parameters have no decay; matrix/tied parameters appear exactly once.
3. Schedule test S=1000 => W=10: s=1 LR=peak/10, s=10 LR=peak, s=1000 LR=peak/10. Test S=1, S=2, resume at warmup boundary, and end-of-budget stop.
4. N=5 sequences, B=4: test expected PCG64 permutations seed=42+epoch, epoch-spanning batches, cursor and next order. Default repetition emits startup and per-epoch warnings. `--no-repeat-data` rejects requests needing more than available whole-step sequences before an optimizer step.
5. Tiny repeated-pattern fixture, 25 updates with test override LR=.01: final loss <50% initial loss; not a production convergence guarantee.
6. Inject NaN loss and Inf gradients: nonzero exit, no completed-step counter advancement or misleading success checkpoint. CPU recipe works; production recipe rejects absent CUDA/BF16 capability clearly.
7. JSONL logs contain finite serialized values, monotonic successful counters and reproducible resolved configuration. Mock timing includes checkpoint/evaluation in cumulative end-to-end throughput.

### #6 Checkpoints/resume

1. Train CPU fixture 6 steps uninterrupted versus 3 + save + new-process resume + 3. Match exact sample order, RNG next draws (Python, NumPy, torch), parameters, optimizer tensors, schedule, counters, data epoch/cursor and final losses.
2. Use SPEC's private on_microbatch callback and inherited pipes: pause after step 3/microbatch 1 of 2, parent receives READY then sends real SIGINT; callback waits for the normal handler flag and ACKs before continuing. Verify completed checkpoint is step 3 of an unchanged 6-step plan and exit=130. Repeat with SIGTERM/exit=143; timeout=30 seconds and always reap subprocess. Test pre-first-update signal -> step-zero checkpoint. Verify checkpoint only after the next optimizer boundary for in-step signals. Simulate hard crash: replay from last complete checkpoint with no duplicate completed-step accounting.
3. Inject failure before completion manifest, during state write, and before rename: previous checkpoint still loads, temporary files are not selected. Mutate a completed checkpoint's bytes: refuse corruption instead of silently selecting an older checkpoint.
4. Save 4 checkpoints with retention=3: only oldest recognized complete checkpoint removed, unrelated sibling and in-flight directory preserved. Never delete before new complete checkpoint exists. Coalesce interval+final+signal at one step into one checkpoint; test idempotent reuse of matching same-step artifact and rejection of mismatch. Simulate crash after checkpoint rename but before run-state update: completed checkpoint wins. Resume completed budget exits 0 without duplicate update/evaluation/checkpoint.
5. Changed data hash, tokenizer, architecture, batch, budget, schedule or unsupported checkpoint schema fails with the offending field. Metrics beyond restored state are excluded and resume event documented.
6. Load saved state using weights_only=True. No pickled custom classes required. Test actual same-filesystem atomic-directory behavior on the supported local POSIX test environment.

### #7 Evaluation/sampling

1. Two synthetic batches with different valid target counts produce sum(loss_i*n_i)/sum(n_i), not an unweighted batch mean. Exclude labels=-100; reject zero valid validation targets.
2. Use first min(128,N) validation sequences, never training data; evaluation doesn't move sampler or training RNG and restores prior train/eval mode even on error.
3. Step-zero, interval and final scheduling calls evaluation exactly once when boundaries coincide. Overflowing exp(loss) gives null + flag, not nonstandard Infinity.
4. Fixed prompts and greedy settings yield matching token IDs in a fixed CPU environment; prompt truncation respects context. Tiny generation max_new_tokens=8. Verify outputs record IDs/settings/step and no instruction-chat template is introduced.

### #8 Export

1. Fresh offline subprocess loads AutoModelForCausalLM + AutoTokenizer with local_files_only=True and trust_remote_code=False, compares logits with the checkpoint, and generates at least one token with a fixed non-EOS-producing fixture.
2. Preserve tied weights, IDs, vocab size and context. Copy one completed checkpoint outside its run, hide the original run/prepared data, and export offline using its tokenizer snapshot. No optimizer state or Python model code in export. Tokenizer saved files load without upstream access.
3. Existing output, incomplete checkpoint, incompatible config/tokenizer and missing required assets fail without partial valid export. Checkpoint files remain unchanged.
4. Model card counters come from checkpoint, not CLI guesses. License/notice material follows artifact-specific review; a user data warning is not evidence of redistribution rights.

### #9 End-to-end CPU CI

Run real subprocess CLI: prepare tiny fixture -> train 3 of a planned 6 steps using SPEC's private synchronized microbatch callback and real SIGINT -> resume to 6 -> evaluate -> export -> offline native HF load/generate. Compare uninterrupted reference. CI installs from the CPU lock; after installation, tests require no network or external accounts. CI must not rely on the spike's already installed `.venv`. Include package archive inspection, unit matrix above, and timeout-bounded fixtures.

### #10 GPU benchmark

Follow SPEC's explicit 120-step invocation (--tokens 31441920): synchronize/reset timer and peak memory after step 20; include steps 21..120 and the normal final checkpoint/evaluation/sampling/summary before stopping timer. Assert numerator=26,201,600 and use a mock clock to verify final overhead counted once; report startup/step-zero costs separately. Record peak memory/end-to-end throughput. No claims from isolated kernels. A100 smoke: stop/resume with identical cursor/counters/LR, finite loss, BF16 logit comparison for export, fixed effective batch across OOM retries. Report measured vs projected distinctly. Four-day miss is a user decision gate. GPU lock and license inventory include actual CUDA-selected dependencies.

### #11 Documentation

Exercise every documented command against installed CLI; tiny version is executable without downloading data/models. Explain full recipe data/validation requirements, repetition warning/opt-out, 2047-target accounting, rounded budget, checkpoint recovery, no cleaning, license exceptions and separately installed GPU platform. Full performance estimates cite #10, not the CPU spike.

### #12 Actual model delivery

Requires user-provided GPU/data/durable destination and an agreed budget. Report actual targets, repeated passes, parameters, hardware/runtime, validation loss and fixed samples. Verify export in an independent process and retain a final resumable checkpoint. No public upload or paid provisioning without separate authorization. Missing resources means blocked, not done. This test completes First Model; merely passing CI does not.
