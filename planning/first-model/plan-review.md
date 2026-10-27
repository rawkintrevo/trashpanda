## Verdict

**NO-GO for junior-developer handoff until P1 decisions below are recorded.** The approved architecture and training scope need not change. No P0 found. Review only: no implementation, installation, specification edits, or delegation performed.

Evidence reviewed: `SPEC.md` (102 lines), `TEST-CONTRACT.md` (85), `COMPATIBILITY.md` (59), and `test_compatibility.py` (156), all under `planning/first-model/`.

The **seven passed tests are dependency-spike evidence**, as reported in `COMPATIBILITY.md:5,21-29`; application acceptance tests remain unimplemented (`TEST-CONTRACT.md:3`). I did not rerun tests. In particular, the spike's checkpoint test uses torch permutations and a LambdaLR, not the specified PCG64 sampler or production schedule, and does not validate application atomicity or all RNG domains (`test_compatibility.py:100-128`).

## Gaps

### P1 — G1: Configuration interface is not an implementable closed contract

**Locations:** SPEC “CLI and configuration,” lines 21-23; “Model and tokenizer,” line 50; “Training and accounting,” lines 69-77; TEST-CONTRACT #2.2, #5.5, #9.

Unknown config keys must be rejected, but the allowed key names, types and override limits are not enumerated. Tiny defines model dimensions but not its complete runnable defaults (budget, effective batch, microbatch, LR and sampling limit); tests require LR/budget overrides without defining their JSON keys. Preparation must write sequence_length, but its recipe/default resolution is only implicit. Seed is described as immutable on resume while sampler order is fixed to 42+epoch; whether initial seed overrides exist is unspecified.

**Bounded fix:** Add a compact schema table with exact JSON names, types, defaults per recipe, allowed overrides, subcommand applicability and CLI mappings. Keep production architecture fixed; distinguish internal test overrides from public config. Explicitly fix seed=42 or define how a supported seed affects all RNG domains and the sampler. Provide one complete offline tiny command/config example. Acceptance: parsing tests exercise every accepted field, reject every unsupported field and demonstrate a fresh tiny run without inherited production-scale budget.

### P1 — G2: Export has no defined source for required tokenizer assets

**Locations:** SPEC “Model and tokenizer,” line 48; “Run layout and checkpoint contract,” lines 82-84; “Evaluation and export,” line 92; TEST-CONTRACT #8.1-4.

Tokenizer files are stored in prepared data and export, but checkpoint/run layouts only require tokenizer identity. Export accepts only a checkpoint and output path; neither locating the prepared dataset nor retaining tokenizer files with the checkpoint/run is specified. Hashes alone cannot reconstruct assets. A developer must choose an undocumented external dependency or silently download files.

**Bounded fix:** Choose one offline ownership rule, preferably snapshot tokenizer assets with each completed checkpoint and hash them in its completion manifest. Alternatively define a mandatory run-local snapshot and explicitly require checkpoint access to that run. State the supported relocation boundary. Acceptance: export works after the original prepared-data directory is unavailable, within that documented boundary, without upstream access.

### P1 — G3: Same-step checkpoint publication and recovery authority are undefined

**Locations:** SPEC “Run layout and checkpoint contract,” lines 82-86; TEST-CONTRACT #6.2-6.

Interval, completion and interruption can request `step-N` publication at the same optimizer boundary. Unlike evaluation, checkpoint triggers have no deduplication rule. Atomic rename of a new nonempty directory over an existing nonempty `step-N` is not a valid replacement strategy on the required POSIX environment. `run-state.json` is also listed without stating whether it or the completed checkpoint is authoritative after a crash between writes.

**Bounded fix:** Coalesce checkpoint requests per completed step; never overwrite a completed directory. Specify validation/reuse of an existing same-step checkpoint, and make checkpoint metadata authoritative with run-state reconstructible. Define completion/interruption ordering and behavior when resume finds the budget already completed. Acceptance: interval+final+signal collision publishes one checkpoint; a crash after rename but before run-state update resumes the published step; resuming a completed run does not attempt another publication.

### P1 — G4: CLI interruption acceptance depends on an unspecified hook

**Locations:** TEST-CONTRACT #6.1-2 and #9 (lines 50-51,73); SPEC “CLI and configuration,” line 23.

The end-to-end test must stop exactly after step 3 of an immutable six-step plan, but no hook interface or synchronization protocol is defined. Sending a signal after reading a metric can race with another optimizer step. A boundary-only stop hook also does not prove the separately required mid-accumulation signal behavior. SIGTERM's exit code is unspecified.

**Bounded fix:** Define a private test-only synchronization mechanism, not a budget-changing public flag: for example, a test runner injects a callback at a specified microbatch, signals readiness and waits for the parent to send SIGINT. Keep planned S=6, acknowledge the signal before continuing to the boundary, then exit 130 at step 3. Specify SIGTERM status and whether stop before the first step completes that step or saves step zero. Acceptance: timeout-bounded fresh-process tests cover real SIGINT mid-accumulation and SIGTERM without changing the persisted schedule.

## Risks

### P2 — R1: Tiny tokenizer packaging conflicts with blanket archive exclusions

**Locations:** SPEC lines 21,50,102; TEST-CONTRACT #2.4 (line 20), #11.

The tiny path refers to a checked-in project-owned tokenizer, whereas source/wheel inspection bans “tokenizer assets” without an exception. Installed tiny commands must work offline, so these instructions admit incompatible implementations.

**Bounded fix:** Choose generated fixtures: include a project-owned deterministic tokenizer builder and vocabulary specification in source/wheel, generate assets only into temporary/runtime outputs, and clarify that upstream/generated artifact files remain excluded. Alternatively explicitly exempt the project-owned tiny fixture from the archive ban. Test the installed wheel's offline tiny path.

### P2 — R2: Benchmark invocation and accounting window need one concrete recipe

**Locations:** SPEC “A100 benchmark and run gate,” lines 96-98, versus checkpoint/evaluation cadence at lines 82,90; TEST-CONTRACT #10.

The required 20+100-step window is shorter than the 500-step interval. Completion can supply the required checkpoint/evaluation, but neither a 120-step benchmark budget nor a forced benchmark event is selected. Timing and peak-memory reset boundaries are not specified, allowing materially different reported projections.

**Bounded fix:** Document a benchmark-only invocation with T=120*K, 20 warmup steps, timer/peak-memory reset immediately afterward, and completion checkpoint/evaluation/sampling included before stopping the timer. State whether final checkpoint overhead is included once and that the production default budget remains unchanged. Assert measured numerator=100*K and verify timing boundaries with a mock clock.

### P2 — R3: Memory acceptance is still an instruction to design a test

**Locations:** TEST-CONTRACT #4.5 (line 36); SPEC “Data format v1,” line 54.

The contract explicitly asks for a “documented test threshold” but supplies neither threshold nor fixture sizes/measurement method. “Larger than baseline memory” alone does not define pass/fail and ignores the explicitly allowed shard/index overhead.

**Bounded fix:** Fix two deterministic corpus sizes, maximum record size, subprocess RSS measurement method and a generous explicit growth bound accounting for shard metadata. State platform/CI conditions. The fixture generator must stream its input so the test does not measure its own corpus allocation.

### Reviewed items that do not require redesign

- Production count 298,103,808 and tied embeddings have explicit dependency evidence (`test_compatibility.py:39-45`).
- Shifted HF loss, L-1 targets, rounded full-step budget and accumulation math are consistent; the numerical and small-budget tests are measurable.
- Shard size, dtype, cross-shard gathering, tail dropping and special-string Mistral encoding are explicit and backed by concrete tests or spike evidence. No document-boundary masking change is needed.
- PCG64 epoch-spanning order and next-unconsumed cursor, fixed-budget warmup/cosine schedule (including S=W), weights-only state encodings, corruption rejection and retention have sufficient baseline rules; G1/G3 address remaining integration decisions.
- Evaluation weighting, offline native HF export checks, GPU-only performance gates, and actual-model-delivery gate correctly distinguish implementation from delivery.
- Source releases and model exports have distinct licensing gates. The licensing report is still being consolidated; its absence is **not** a blocker in this review. No new provenance, distributed, instruction-training or architecture work is requested.

### Suggested next steps

Resolve G1-G4 in the specification and acceptance contract before updating implementation issues #1-12. Resolve R1-R3 alongside those bounded edits, then re-review only the changed contracts. Preserve the explicit distinction between seven dependency tests passed and all future application acceptance tests.


---

# Re-review after bounded contract fixes

## Verdict

**GO with notes. No unresolved P0/P1 prevents code implementation.** This verdict supersedes the initial handoff NO-GO above, not the runtime, release or actual-model-delivery gates.

Read the revised SPEC.md, TEST-CONTRACT.md and ASF-LICENSE-REVIEW.md in full. No tests were run, no implementation was reviewed, and only this report was appended. The evidence remains seven previously passed dependency-spike tests; application acceptance tests are still to be implemented (TEST-CONTRACT.md:3).

### Resolution of prior findings

- **G1 closed:** SPEC “CLI and configuration,” lines 21-42, specifies the four-field schema, precedence, fixed seed, recipe defaults and offline tiny invocation. TEST-CONTRACT #2.2 tests this interface. Tiny accounting is consistent: 4*(32-1)=124 targets/step; 744 targets means six steps.
- **G2 closed:** SPEC “Run layout and checkpoint contract,” lines 102-104, requires a self-contained tokenizer snapshot and relocatable offline export. TEST-CONTRACT #8.2 (line 67) hides both original run and data to verify this boundary.
- **G3 closed:** SPEC line 106 makes completed checkpoints authoritative, coalesces publication requests and defines completed-run resume. TEST-CONTRACT #6.4 (line 53) explicitly tests collisions and crash-after-rename recovery.
- **G4 closed:** SPEC “Private signal-test seam,” lines 116-118, defines the injected callable predicate, real signals, READY/ACK synchronization and timeout. SPEC line 42 and TEST-CONTRACT #6.2 define signal statuses, step-zero handling and unchanged six-step budget. This is a test seam, not a new public continuation feature.
- **R1 closed:** SPEC line 68 and TEST-CONTRACT #2.4 ship builder/vocabulary code, not generated tokenizer assets; installed-wheel offline tests are required. Tiny's limited UNK behavior is explicitly distinguished from the pinned Mistral roundtrip.
- **R2 closed:** SPEC line 122 and TEST-CONTRACT #10 define the timing callback/window and included final overhead. Arithmetic is correct: 120*262016=31441920; measured 100*262016=26201600. The benchmark schedule is explicitly not the full training schedule.
- **R3 closed:** TEST-CONTRACT #4.5 (line 36) specifies fresh-process Linux peak RSS, corpus sizes, streamed generation and <=64 MiB growth. This is now an executable acceptance criterion rather than an unspecified threshold.

## Gaps

No blocking gap found in the changed contracts. Licensing and GPU verification remain explicit downstream gates, not claims already established by CPU tests. ASF-LICENSE-REVIEW “Acceptance gates and ownership,” lines 39-47, assigns inventory, tokenizer grant, export notices and GPU artifact review to named issues; local model delivery is explicitly distinct from ASF distribution readiness. This review checks that boundary's consistency, not independent legal clearance of every installed component.

## Risks

Two nonblocking P2 editorial/packaging notes remain:

1. **Recipe cadence wording:** SPEC line 32 explicitly fixes tiny interval=3, but “Run layout” line 100 and “Evaluation and export” line 112 still say every 500 steps without qualification. TEST-CONTRACT #2.2 removes practical doubt. Minimal cleanup: say “at the recipe interval (500 production / 3 tiny)” in both sections so a junior does not hard-code the production cadence.
2. **Installed production hash reference:** SPEC line 21 requires pinned raw tokenizer hashes from environment-evidence.json, while TEST-CONTRACT #2.4 excludes planning runtime outputs from packages. Carry the three approved hashes into a small shipped constant/resource rather than requiring the planning evidence file at runtime. Add an installed-wheel mocked production acquisition check proving validation works outside the repository. The hashes and source of truth are already selected; this is packaging guidance, not a new design decision.

**Next step:** update implementation issues with these revised contracts and proceed with code/tests. Fold in the two small notes without reopening scope. Keep GPU performance, dependency/release review and actual local trained-model delivery explicitly pending until their respective evidence exists.


## Baseline-update addendum

**Verdict remains GO with notes; no new P1 contract blocker.** Read updated COMPATIBILITY.md and ASF-LICENSE-REVIEW.md. COMPATIBILITY “Result” records torch 2.14.1+cpu, Transformers 5.21.0, tokenizers 0.23.3, huggingface-hub 2.3.0, safetensors 0.8.0 and pytest 9.1.1; NumPy remains 2.2.6. The same seven dependency tests reportedly passed offline in 11.65s, with pip check passing. This supersedes earlier baseline evidence, not application acceptance tests. I did not independently rerun tests or advisory queries.

The recorded OSV result is zero version matches for 36 CPU runtime/test packages, not general security clearance or native/CUDA validation. Removed certifi/requests dependencies are distinguished from retained runtime exceptions, without disabling TLS. The revised license register retains NumPy LGPL/GCC, tqdm MPL and regex CNRI review, and explicitly records ISC allowlist extension and LLVM-exception review. Those gates remain assigned to dependency/release work; they do not require architecture or application-contract redesign.

Implementation should use the newly recorded baseline and produce the still-required hash locks; do not reuse the superseded dependency candidate or treat a CPU freeze as a CUDA lock. The prior two nonblocking contract/packaging notes remain unchanged. No files other than this report were modified.
