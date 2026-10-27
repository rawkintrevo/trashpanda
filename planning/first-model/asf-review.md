# Research: ASF compatibility of trashpanda’s first-model stack

> Historical research notes from the initial dependency candidate, retained for source evidence. The selected baseline was subsequently upgraded and certifi removed. **[ASF-LICENSE-REVIEW.md](ASF-LICENSE-REVIEW.md) is the current project-specific exception register; [COMPATIBILITY.md](COMPATIBILITY.md) gives the tested versions.** Later addenda here do not override those documents.

## Summary
Keep the ~298M from-scratch Llama POC an Apache-2.0 project-code-only distribution, with independently installed PyTorch/Transformers dependencies and separately acquired tokenizer assets. This minimizes redistribution risk but does **not** make the runtime strictly allowlist-only: MPL-2.0 (certifi/tqdm), regex’s CNRI terms, and the NVIDIA platform remain explicit review items. This is a policy assessment, not ASF approval or legal certification.

Scope: owned PyTorch training loop, JSONL input, HF export, no OLMo dependency. Research covered three angles: ASF inclusion/release rules; upstream license evidence; runtime/data/model boundaries. Current official pages were retrieved directly over public HTTP; existing local policy extracts were also read. No packages were installed or project/issue changes made.

## Findings

1. **Verified policy — A/B/X are not interchangeable with the project allowlist.** [ASF third-party policy][1] permits Category A licenses, including Apache 1.1/2.0, BSD without advertising clause, MIT/X11, W3C, PSF and Boost; public-domain treatment requires evidence and attribution. Category B includes MPL-2.0: prominent labeling (e.g. README identifying dependency, license and homepage), license obligations, and **binary-only** inclusion in convenience binaries; Category B works must not be included in source releases. The narrow exception for small, unmodified runtime source such as standardized DTDs is not general permission to bundle Python libraries. Category X components must not be distributed in source or binaries. ASF’s Category A is broader than the project’s allowlist; a project-approved exception is not an ASF exception or approval.

2. **Verified policy + application — a wheel is not a license workaround.** ASF requires a source package sufficient to build/test with the appropriate platform/tools, and every distributed artifact must satisfy licensing policy ([release policy][2]). A Python sdist containing MPL dependency sources fails the Category B source-inclusion rule. A `.whl` containing those same `.py` files still contains source: the packaging name “binary distribution” does not establish the policy’s “binary-only” condition. The consulted policy gives no blanket Python-wheel exemption. Do not ship dependency wheels, virtualenvs, vendored modules, certificate bundles, tokenizer files, downloaded datasets or weights inside project source archives. For this POC, also avoid bundled-runtime wheels/containers. This no-third-party-bundling boundary is a conservative project choice, not a claim that ASF forbids all third-party source (Category A inclusion is possible).

   LICENSE/NOTICE must describe **exactly the shipped contents**, not separately downloaded dependencies; transitive bundled dependencies count too ([licensing how-to][3], [release policy][2]). Keep external runtime license disclosures in ordinary dependency/runtime documentation, not an inflated project NOTICE. Recheck wheel and sdist archive contents independently.

3. **Package evidence — inspect the pinned distributions, not only metadata.** Upstream project licenses support PyTorch’s BSD-3-Clause-style terms and Transformers’ Apache-2.0 terms ([4], [5]); neither establishes that every wheel or native/transitive component uses only that license. Exact target versions were not supplied for this review, and current upstream branches are not evidence for a specific installed release.

   | Component | Evidence and interpretation | POC disposition |
   |---|---|---|
   | certifi | Upstream LICENSE identifies the Mozilla-derived CA certificate bundle as MPL-2.0 ([6]); current PyPI metadata also says MPL-2.0 ([10]). This is Category B, outside the project allowlist. | Separately installed runtime dependency; do not copy its CA bundle into releases. Record the exception rather than relabeling it permissive. Do not disable TLS verification to remove it. |
   | tqdm | v4.67.1 metadata says `MPL-2.0 AND MIT`; LICENCE applies MPL to Casper da Costa-Luis’s contributions (`files: *`), with MIT contributions including original README/code ([7]). **AND is not OR**: MIT cannot simply be selected for the whole package. A simplistic “code MPL / all documentation MIT” partition was **not established** by these files; README has MIT-origin contributions but the wildcard MPL notice still requires exact-file review. | Treat runtime code as involving Category B. Avoid a direct dependency in the owned loop if plain logging suffices, but check whether Transformers/HF dependencies still require it. Keep any remaining runtime exception explicit. |
   | regex | Upstream LICENSE says derived CPython code is under CNRI Python 1.6 and additions/alterations Apache-2.0 ([8]); current PyPI metadata says `Apache-2.0 AND CNRI-Python` ([10]). | Review the exact pinned artifact’s full license text and inherited notices. Do not select Apache alone or equate CNRI with PSF. Keep unbundled pending review. |

   **CNRI status:** a direct text search of the current ASF resolved policy returned **zero `CNRI` occurrences**; PSF is expressly listed, CNRI-Python is not. SPDX has a distinct CNRI-Python entry ([9]); SPDX identification or OSI recognition is not evidence of explicit ASF Category A listing. Absence from the list is not itself proof of Category X either. Resolve the precise CNRI variant and obligations, then obtain guidance through ASF Legal if ASF inclusion is contemplated. Do not mark this review “approved” from package metadata.

4. **Verified platform guidance; NVIDIA conclusion remains conditional.** ASF explicitly allows software targeting a platform whose terms do not affect the Apache product’s licensing, while warning that this does **not** permit redistributing platform code ([1], platform FAQ). Separately installed NVIDIA driver/CUDA runtime is the appropriate review framing for realistic GPU training—not a nominal optional feature. ASF’s optional-feature test asks whether the **majority of users would want the product without it**, and whether standard use/desirable quality is possible; a CPU smoke test alone does not establish that test for ~298M training.

   Document NVIDIA GPU training as the expected practical platform and CPU as the tested compatibility/smoke path, unless measurements justify more. Review the actual driver, CUDA libraries and other NVIDIA packages selected by the PyTorch install against their exact terms; the CUDA EULA is a separate upstream agreement, not an ASF Category A grant ([11]). No definitive platform determination for this stack was located in the consulted ASF policy. Keep NVIDIA code outside project artifacts, provide separate installation instructions, and escalate uncertainty about effects on Apache code to ASF Legal. A GPU PyTorch wheel or container can bundle vendor libraries or install `nvidia-*` dependencies: “user installed PyTorch” is not proof that a subsequently published image/wheel is clean. User-installed runtime components still have their own license obligations.

5. **Artifact/data boundaries — software licenses do not establish release rights to models.** The Apache license of the HF `tokenizers` library ([12]), or an Apache tag on a tokenizer/model repository, does not prove rights to every vocabulary, merges file, tokenizer JSON, chat template, custom Python file or embedded dataset. Before using a concrete tokenizer revision, check its included license, notices, origin and individual artifact coverage. Prefer an unbundled, explicitly licensed tokenizer with no remote custom code; do not download pretrained weights merely to obtain configuration/tokenization. The exact tokenizer was not identified here, so clearance remains open.

   Train from random initialization without importing Meta weights or third-party implementations into the owned loop. HF export format does not import a model license automatically, nor does it confer rights. User-provided JSONL stays outside source control/releases; require users to have rights for the intended training use and observe privacy/confidentiality restrictions. Permission to train is not necessarily permission to redistribute the raw corpus or publish resulting weights. Use only owned/synthetic fixtures in tests. Any later weight, tokenizer or dataset publication needs its own documented rights/license review and release decision; do not automatically label trained weights Apache-2.0. The consulted ASF software policy does not settle all training-data, memorization or model-weight copyright questions.

## Practical issue acceptance gates

These are proposed checklist items, not changes to issues or assertions of ASF approval.

- **Dependency gate:** attach the actual CPU compatibility stack’s pinned direct/transitive versions, artifact names, license metadata and links to the corresponding license files. Review native/bundled components as needed; do not substitute latest PyPI metadata. Record each non-allowlist item as removed, replaced, or an expressly accepted *project runtime exception*, with ASF review status separate. No new provenance framework is needed: a small dependency table suffices.
- **Source/export gate:** inspect built sdist and project wheel file lists; only project-owned code/config/docs and owned/synthetic fixtures ship. No dependencies, cached HF assets, CA bundles, corpora, checkpoints, NVIDIA libraries or borrowed implementation snippets. LICENSE/NOTICE match each actual archive. Run owned-loop CPU training/checkpoint and HF export/reload smoke tests without requiring pretrained weights.
- **MPL/CNRI gate:** record certifi/tqdm as externally installed MPL exceptions; verify tqdm’s actual required dependency path. Attach regex’s exact inherited CNRI terms and a named unresolved-review item; no CNRI/PSF equivalence or invented ASF clearance. If future bundling is proposed, stop and resolve the ASF inclusion question first, including `.py` source in wheels.
- **GPU gate:** document separate driver/CUDA/PyTorch installation and the realistic GPU requirement. Inventory the selected runtime packages and licenses, confirm nothing is redistributed, and record why platform terms are believed not to alter project-code licensing. Seek ASF Legal guidance for unsettled terms before claiming ASF readiness; do not close this gate merely by declaring CUDA optional.
- **Assets/rights gate:** identify tokenizer repository plus immutable revision and files used; verify applicable licenses/notices and no remote custom code. Keep user JSONL and resulting checkpoints local by default. Release only code for the POC; require a separate affirmative rights decision before publishing weights/data/tokenizer assets.

## Sources

Kept (primary sources; live branch/latest metadata links must be replaced with pinned-version evidence at acceptance):

[1]: https://www.apache.org/legal/resolved.html
[2]: https://www.apache.org/legal/release-policy.html
[3]: https://www.apache.org/dev/licensing-howto.html
[4]: https://raw.githubusercontent.com/pytorch/pytorch/main/LICENSE
[5]: https://raw.githubusercontent.com/huggingface/transformers/main/LICENSE
[6]: https://raw.githubusercontent.com/certifi/python-certifi/master/LICENSE
[7]: https://raw.githubusercontent.com/tqdm/tqdm/v4.67.1/LICENCE
[8]: https://raw.githubusercontent.com/mrabarnett/mrab-regex/hg/LICENSE.txt
[9]: https://spdx.org/licenses/CNRI-Python.html
[10]: https://pypi.org/pypi/regex/json
[11]: https://docs.nvidia.com/cuda/eula/index.html
[12]: https://raw.githubusercontent.com/huggingface/tokenizers/main/LICENSE

- [ASF 3rd Party License Policy][1] — authoritative A/B/X, platform, optional-feature and escalation rules.
- [ASF Release Policy][2] and [Assembling LICENSE and NOTICE][3] — source release and exact-distribution accounting; how-to currently redirects to ASF Infrastructure.
- [PyTorch LICENSE][4], [Transformers LICENSE][5], [certifi LICENSE][6], [regex LICENSE][8] — upstream grant/inherited-license evidence, not complete wheel audits.
- [tqdm v4.67.1 LICENCE][7] and [pyproject.toml](https://raw.githubusercontent.com/tqdm/tqdm/v4.67.1/pyproject.toml) — contribution-specific licenses and explicit AND expression; example version, not an assumed final pin.
- [SPDX CNRI-Python][9] — distinct license text/identifier, not ASF approval.
- PyPI JSON for [regex][10], [certifi](https://pypi.org/pypi/certifi/json), [tqdm](https://pypi.org/pypi/tqdm/json), [torch](https://pypi.org/pypi/torch/json), [transformers](https://pypi.org/pypi/transformers/json) — metadata cross-checks only. Retrieved latest versions included regex 2026.9.29, certifi 2026.10.23, tqdm 4.70.1; these are **not** recommended compatibility pins.
- [NVIDIA CUDA EULA][11] — vendor terms requiring version/component-specific review; not a driver-license substitute.
- [HF tokenizers LICENSE][12] — library license only, not a tokenizer asset/model license.

Discarded as proof: license badges alone, latest PyPI metadata as a substitute for pinned artifact review, a blanket tqdm documentation-MIT inference, and SPDX/OSI status as an ASF category determination. No secondary compliance summaries were relied on.

## Gaps / next steps

The rebuilt dependency lock, exact wheel contents, chosen tokenizer revision and GPU runtime versions are not available to this independent review. Feed those into the small checklist above; keep POC distribution code-only while CNRI, MPL Python-source inclusion (if ever proposed), and NVIDIA platform details remain tracked. User acceptance of runtime exceptions permits an honest POC boundary, not a claim of future ASF acceptance. Seek targeted ASF Legal guidance on unresolved inclusion/platform questions rather than requesting blanket certification.


## Addendum: verified baseline and native-wheel variance

The baseline is now recorded in [`environment-evidence.json`](environment-evidence.json): Python 3.11.2, Linux x86_64, NumPy 2.2.6, PyTorch 2.6.0+cpu and Transformers 4.51.3; CUDA is unavailable. This supersedes the earlier statement that baseline versions were unavailable, **not** the outstanding artifact-level review. The main reviewer reports direct inspection of NumPy’s installed `LICENSE.txt` and `numpy.libs`: bundled libgfortran is GPL-3.0 with the GCC runtime exception, and bundled libquadmath is LGPL-2.1-or-later, with corresponding `.so` files present. These observations are supplied inspection evidence; the JSON’s BSD metadata alone does not capture them.

- **libgfortran: conditional exception, not a plain permissive license.** [ASF’s GCC Runtime Library exception guidance][1] permits incorporation under the relevant GCC exception (2.0 or 3.1), with PMC review that ASF product modules are being combined with the runtime module; otherwise guidance should be sought. Verify the exact exception text/version and combination/build circumstances before claiming this permission applies. A GPL-with-exception label does not automatically approve the entire NumPy wheel, and remains outside the project’s narrow allowlist.
- **libquadmath: distinct Category X risk.** ASF lists LGPL 2/2.1/3 in Category X ([1]); the GCC runtime exception for libgfortran does **not** extend to a separately LGPL-licensed libquadmath. Do not redistribute this wheel, its native libraries, or an environment/container containing them as an ASF product without resolving/removing the prohibited inclusion. An upstream LGPL redistribution permission is not ASF distribution-policy permission. Separate user installation avoids project bundling, but does not by itself establish that a required dependency meets ASF reliance/platform guidance; retain this as an explicit runtime variance and unresolved review, not a nominal optional feature.
- **PyTorch CPU is not a license-cleanliness certificate.** The main reviewer reports approximately 486 KB of bundled notices in the installed torch 2.6.0+cpu license. This establishes the need for component-level review beyond its BSD metadata, not that every notice represents a shipped binary or that all components are approved. CPU-only avoids NVIDIA components in this baseline; it does not eliminate other bundled-component risks. No complete NumPy or PyTorch wheel audit is certified here.

**Additional acceptance evidence:** extend the existing dependency table with these two NumPy runtime components, their inspected license/library paths and exact artifact identity; record torch’s bundled-notice review as incomplete. Keep the code-only/no-wheel-or-environment-redistribution boundary, explicitly disclose these runtime variances, and require targeted review before any bundled distribution. No decision on other specifications is made by this addendum.
