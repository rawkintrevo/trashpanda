# First Model handoff

This directory is the approved implementation contract and CPU dependency-spike evidence for [First Model #1](https://github.com/rawkintrevo/trashpanda/issues/1). **It is not the application implementation.** No full model has been trained.

Read in order:

1. [SPEC.md](SPEC.md): exact architecture, tokenizer hashes, JSONL/shard format, CLI/config defaults, training schedule/accounting, sampler/resume, evaluation/export and benchmark procedure.
2. [TEST-CONTRACT.md](TEST-CONTRACT.md): concrete application acceptance tests mapped to child issues, including tolerances and interruption/failure cases.
3. [COMPATIBILITY.md](COMPATIBILITY.md): seven CPU dependency checks passed on the selected pinned stack; reproduction instructions and limits.
4. [ASF-LICENSE-REVIEW.md](ASF-LICENSE-REVIEW.md): authoritative exception register, packaging boundary and unresolved release/legal gates.
5. [plan-review.md](plan-review.md): independent review history. Initial gaps were fixed; follow-up verdict GO with notes. Main-agent cleanup addressed the remaining cadence/hash-packaging notes.

Evidence: test_compatibility.py (dependency spike only), compatibility-results.txt, compatibility-cpu.freeze.txt, environment-evidence.json, security-advisory-matches.json. collect_evidence.py regenerates environment/license metadata and queries OSV using public package/version names. asf-review.md retains supporting policy research; its historical baseline is superseded by the authoritative documents above.

## Current status

- Technical design choices fixed; no OLMo dependency or app feature implementation in this change.
- CPU dependency tests: 7 passed; pip check passed. Exact production architecture counted on meta device, not trained at production size.
- Selected baseline: torch 2.14.1+cpu / Transformers 5.21.0; exact transitive version snapshot included. Initial older candidate was replaced after advisory matches.
- OSV version query: zero matches for 36 installed CPU runtime/test packages at recorded timestamp. Not a native/GPU audit or guarantee of security.
- Removed obsolete certifi/requests dependencies; TLS verification remains enabled. Retained MPL/CNRI/native/GPU license variances are documented, not declared ASF-approved.
- Remaining execution gates: real application implementation/tests, CPU/CUDA hash locks and component review, A100 benchmark, user-provided data/GPU/destination, actual training and offline export validation.
- Source/project-wheel release is code-only. Public weight/tokenizer/data release and any bundled runtime require separate rights/policy review and authorization.

Implement children in dependency order; later issues consume earlier contracts rather than inventing alternatives. If an explicit safety/licensing/hardware gate cannot be satisfied, report it to the maintainer; do not silently change the model, token budget, runtime or release claims. The epic closes only after an actual local trained model is delivered, not after these planning checks or CI alone.
