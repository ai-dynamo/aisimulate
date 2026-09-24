# Proposed tail runtime evidence contract

No accepted summary is included and runtime admission remains closed. The standalone `collector.glm53flash_tail_qualification.validate_tail_qualification` checks a future `admission-summary.json` against its caller-supplied immutable SHA. It does not change the existing runtime registry.

The public runtime-identity functions route the exact candidate version to this
reader only when a reviewed summary SHA is explicitly added to the registry.
The registry is currently empty: planning, source closure, worker closure and
unaligned-prefill admission still reject the candidate. The tail-only reference
is never a production candidate. No environment variable opens admission.

Once admitted, its worker closure binds the new actual build and wheel, both
patched paths, the 31 qualified source files, two additional unchanged V2 worker
files, and all 19 original native binaries (52 package files). The V2 source
manifest remains the existing immutable upstream manifest; its historical
directory name does not authorize the old runtime. Original Dynamo observer
source pins stay unchanged. Workers read actual file bytes before timing;
calibration and holdout must bind the same runtime closure. This routing does
not replace per-cell capacity qualification.

`expected-runtime.json` preserves the exact original task-owned `kpool-tail-engine-qualification-v2` input (SHA cf7ab8b815ef16e41e178a17cb24effe1f31f54785f0c61b1931d6e3a134e910). Its inherited `build_receipt_sha256` identifies the historical old kpool build and is explicitly unused for new tail admission. New per-kind build and wheel hashes are mandatory in the reader. No source or native implementation is copied in this JSON.

The future summary must contain all four FP8/NVFP4 TP2/TP4 three-profile functional comparisons, actual16-case cache oracle and the exact original NVFP4 TP2 131072-token ordinary-request regression. The long regression is one cell and one request. Every deployment still needs its separate native nine-point hardware/configuration/capacity qualification, followed by complete FPM/Ops validation and separate HTTP measurements.

Each functional cell references original and freshly revalidated comparisons; each original profile supplies preflight, native receipt, worker and cache-group small files plus the complete external raw inventory. A distinct revalidation receipt binds all input hashes, original/derived result hashes, immutable validator sources and assembler source. The consumer audits this chain; the assembler must actually reread original token/metadata arrays. Historical receipts are never rewritten or presented as having originally passed a newer check.

Build/install evidence retains ARM613836 and the actual packaged-verifier execution614605. Its executed-input snapshot predates later packaging-lineage metadata; only the original operative input hashes establish what executed. The new reader keeps this temporal distinction.

All positive unit fixtures are temporary TEST_ONLY artifacts. No fixture, partial matrix, CPU-only install success, quarantined old runtime, or reference-wheel identity grants candidate admission. Accuracy, formal eight-cell coverage and HTTP acceptance remain NOT_EVALUATED.
