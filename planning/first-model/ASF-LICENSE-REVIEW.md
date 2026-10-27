# ASF-oriented licensing boundary and exception register

## Decision and limits

Project code: Apache-2.0. Prefer Apache 1.1/2.0, BSD 2-/3-clause, MIT/X11, W3C, PSF, Boost and substantiated public domain. The owner permits **documented external dependency/runtime exceptions**, not arbitrary redistribution or an ASF exemption. Minimize deviations; no claim of ASF membership, acceptance, trademark entitlement, legal certification or complete compliance audit.

For v1 distribute only our code/config/docs and project-owned synthetic fixture builder. Do not bundle dependencies, virtualenvs, third-party tokenizer assets, CA bundles, user data, pretrained weights, generated checkpoints, CUDA libraries or runtime container images into source/project wheel artifacts. Users install dependencies and acquire the pinned tokenizer separately. Local model export is a distinct output boundary; it is not automatically authorized for public redistribution. No instruction/provenance/registry feature is added by this policy.

Separately installed dependencies still have their own obligations. Keeping them unbundled minimizes redistribution risk; it does NOT automatically settle ASF policy for a required dependency. Source-only project packaging is an engineering boundary, not a universal license workaround.

## Verified policy

- [ASF third-party policy](https://www.apache.org/legal/resolved.html): Category A may be included; Category B inclusion is conditional, labeled, and binary-only in convenience binaries, not source releases. MPL-2.0 is Category B. Category X must not be redistributed in source or binary form. LGPL is listed among prohibited licenses.
- Python `.whl` naming does not make contained `.py` files object-only code. Do not assume a wheel meets Category B's binary-only condition.
- GCC Runtime Library Exception has specific ASF guidance and review conditions. It is not permission to bundle every GCC-related or LGPL library.
- ASF's platform FAQ permits targeting platforms whose terms do not change Apache product licensing, but does not authorize redistribution of platform code. Practical A100 training needs NVIDIA's platform; do not call CUDA optional solely because tiny CPU tests run.
- [ASF release policy](https://www.apache.org/legal/release-policy.html) and [LICENSE/NOTICE guidance](https://www.apache.org/dev/licensing-howto.html): inspect actual shipped contents and provide their required licenses/notices. Keep external-dependency disclosures in ordinary documentation rather than inventing required NOTICE entries. A future formal ASF release has additional release/process requirements; this POC does not claim to meet them.

## Exceptions and unresolved reviews

Installed CPU baseline is fully version-listed in compatibility-cpu.freeze.txt. Metadata, license-file hashes and observed NumPy native filenames are in environment-evidence.json. The current baseline is torch 2.14.1+cpu / Transformers 5.21.0; an earlier candidate was replaced after advisory matches. The following uses inspected installed license files and the recorded current metadata; historical findings are marked. This is a targeted review, not a complete component inventory.

| Component | Evidence / variance | v1 disposition and remaining gate |
| --- | --- | --- |
| certifi 2026.10.23 (removed) | Historical candidate's LICENSE described Mozilla CA bundle under MPL-2.0, outside original allowlist; ASF Category B. | **Eliminated from current dependency closure.** Updated HF HTTP stack uses system trust via truststore; no TLS verification bypass. certifi/requests and unused dependencies were uninstalled and pip check/offline tests passed. Recheck if extras reintroduce it. |
| tqdm 4.70.1 | Installed LICENCE has MPL-2.0 wildcard contributor grant and MIT contributions. Metadata is MPL-2.0 AND MIT, not a choice to select MIT for all code. | Avoid direct use in our loop; Transformers/HF still require it. External runtime exception, Category B involvement. No vendored Python sources. |
| regex 2026.9.29 | Installed LICENSE identifies inherited CNRI Python 1.6 material plus Apache-2.0 additions. | External runtime exception pending precise inherited-term review. CNRI is not explicitly listed in retrieved ASF resolved policy, nor equivalent to PSF. Absence does not prove Category X. Seek targeted ASF Legal guidance if formal inclusion is contemplated. |
| NumPy 2.2.6 wheel: libgfortran | Wheel LICENSE identifies GPL-3.0-with-GCC-exception; observed numpy.libs/libgfortran-040039e1-0352e75f.so.5.0.0. | Native runtime variance; ASF GCC exception has conditions that must be reviewed, not blanket clearance. Do not redistribute this wheel/native library with trashpanda. |
| NumPy 2.2.6 wheel: libquadmath | Wheel LICENSE identifies LGPL-2.1-or-later; observed numpy.libs/libquadmath-96973f99-934c22de.so.0.0.0. | **Category X redistribution risk.** Unbundled installation is not a formal finding that a required dependency is ASF-approved. Maintainer/ASF review must resolve whether a compliant artifact/platform treatment or replacement build is needed before an ASF distribution claim. Do not publish bundled environments/containers. |
| PyTorch 2.14.1+cpu native/bundled components | Current metadata declares Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT; numerous separate license files are hashed in evidence. Earlier 2.6 CPU candidate had ~486KB aggregate notices, but is no longer selected. | External runtime; component-by-component inventory remains open. LLVM-exception form is outside the original literal allowlist and not explicitly named in the retrieved ASF policy; record precise terms/review rather than assuming blanket approval. CPU audit does not cover CUDA. |
| shellingham 1.5.4 (Typer transitive dependency) | Current metadata identifies ISC. ISC is explicitly included under ASF Category A's MIT/X11 family, but was not in the owner's literal original list. | Documented project allowlist extension; not an identified ASF Category B/X variance. Verify shipped license text during inventory; no vendoring needed. |
| NVIDIA driver/CUDA and GPU wheels | Not installed or tested here; vendor/platform terms and nvidia-* transitive packages depend on selected GPU installation. | Separate documented platform installation, never bundled by us in v1. Hardware issue must inventory exact selected artifacts and escalate terms affecting project-code licensing. No pretend optional-feature exemption. |
| Mistral tokenizer at pinned revision | HF model card declares apache-2.0; three tokenizer JSON artifacts downloaded. Repo listing lacks standalone LICENSE/NOTICE. No pretrained weights downloaded. | Technical compatibility verified, artifact-specific license/notice coverage still needs review before redistribution. Keep upstream assets out of project source archives. Local HF export snapshots them; public export release requires affirmative rights review. |
| User data and resulting model weights | JSONL content/license unknown; software licensing does not establish training or publication rights. | User supplies lawful data; retain locally by default. Record public publication as a separate authorized decision, not automatically Apache-2.0 because our code is. No dataset curation/provenance subsystem required. |

The aggregate NumPy LICENSE also documents BSD-licensed OpenBLAS. Do not mislabel the entire NumPy distribution LGPL; licenses apply to their respective components. Likewise, do not treat SPDX AND expressions as selectable OR alternatives.

## Acceptance gates and ownership

**Issue #2:** retain exact dependency versions; produce the CPU hash lock, intended GPU install inputs and a concise direct/transitive/bundled-component table; list exceptions, source evidence, disposition and unresolved reviews. Inspect built source/project-wheel file lists. LICENSE/NOTICE describes shipped content; runtime docs disclose non-allowlist components prominently. Run a security scan and record maintainer disposition for findings. No automatic bundling or silent dependency changes to evade policy.

**Issue #3:** verify exact tokenizer artifact grant/required notices; pinned immutable files and no pretrained weights/remote model code. If coverage cannot be established, bring a narrowly scoped replacement proposal to the maintainer rather than guessing rights.

**Issue #8:** verify export contents and include applicable artifact notices. Export is local by default; do not apply a blanket Apache weight license without data/model rights review.

**Issue #10:** resolve/hash-lock the actual GPU variant (not a prerequisite for closing #2); inspect actual A100 runtime versions, vendor/native dependencies, measured behavior and separate installation instructions. CPU package inventory is not GPU clearance.

**Issues #11/#12:** disclose limitations and runtime exceptions in README; require explicit public release authorization and rights review for weights/tokenizer/data. First Model can deliver a local HF model without being an ASF release; claiming ASF distribution readiness remains blocked on unresolved items above.

For uncertainty, seek targeted ASF Legal guidance with exact versions/artifacts and the proposed distribution boundary. This is not authority to bypass policy simply because the project owner accepted an exception.

## Supporting research

`asf-review.md` is the independent policy research brief, including source links and an addendum on native libraries. Main-agent verification re-read the official Category B/GCC/platform/ISC guidance and installed certifi (prior candidate), tqdm/regex/NumPy/PyTorch license files, then recorded the project-specific register above. Current baseline passed seven CPU tests and an OSV version query returned zero matches for 36 runtime/test packages; that does not discharge licensing review. No platform code or third-party source was copied into project deliverables.
