# Completed B200 vLLM DSA replacement attempts

The full vLLM 0.24.0 campaign attempted all 34,546 frozen cases. The private
replacement contains 48,585 actual successful full/reuse rows. Its data checks
passed and the seven reviewed systems changes were applied. The
[apply receipt](applied.json) and [post-apply hash check](post-apply-check.json)
confirm the exact change set and unchanged six DSV4 metadata entries. See [summary.json](summary.json) for identities and exact hashes.

| Outcome | Calls | Output variants |
| --- | ---: | ---: |
| Successful native measurement | 32,306 | 48,585 |
| Recorded native failure | 2,240 | 3,360 excluded |
| Unattempted or unresolved | 0 | 0 |

All failed calls are context/prefill. There are 560 failures per projection
precision and 1,120 per architecture. The native executor still classifies
them as `unexpected`; the source diagnosis does not turn them into passes,
xfails or a future skip rule. No partial failed row or historical latency
enters the new tables.

- [actual-failed-cases.csv](actual-failed-cases.csv): all 2,240 exact cases,
  attempts, shapes, precisions, excluded variants and original log hashes.
- [old-coordinates-without-new-measurement.csv](old-coordinates-without-new-measurement.csv):
  the 1,680 original row obligations with no successful replacement. The
  complete 25,639-row before/after mapping remains in the private archive.
- [root-failure-review.json](root-failure-review.json): exact reviewed failure
  manifest SHA and limits of the valid-subset publication decision.

GLM-5.2 is the explicit new GLM architecture reference, including full and
index-reuse variants. Original GLM-5 model identities are retained in the CSV;
this does not assert identical model inputs across those checkpoints. The
collection is scoped to B200; other GPU data has not been recollected.

## Data and provenance checks

The new tables contain 26,005 context and 22,580 generation rows. The
[collected-only sidecar](../new-dsa/collection_meta.yaml) and
[table hashes](../new-dsa/table-sha256.json) identify the actual runtime,
collector closure, dirty-source reference, case plans and GPU UUIDs. No
upstream source code is copied into these records.

The colocated six legacy DSV4 entries and their minimal shared runtime remain
unchanged. The shared sidecar retains its legacy representation; the official
fresh-update validator's refusal of that representation is retained in
[metadata-strategy.json](metadata-strategy.json). This operation is full table
replacement. The collected-only sidecar passed the official full provenance
validator. Table status `complete` means the collection lifecycle completed;
the context table separately records 2,240 classified failures.

- [Official R1–R7 checks](official-data-check.log): all passed.
- [Official backend facts check](official-backend-facts-check.log): all 2,186
  slices match; 48 of those slices describe the new DSA data.
- [Native label audit](native-backend-label-audit.json): all 48,585 raw labels
  equal the passive actual native-backend receipt. Existing FlashMLA/FlashInfer
  label translations remain unchanged; these are whole-module timings.
- [Curated facts update](backend-facts-source-update.json): add the required
  per-tensor FP8/reuse slices and retire the obsolete 20 pilot entries; all
  other registry lines are preserved.
- [Source checks](source-verification-summary.json): current reads the new
  0.24.0 tables directly, shared current/next use that same fresh donor, and
  10,236 non-target source reports remain equal. Next with sharing disabled
  preserves its preexisting primary-only missing-data behavior.

The official full manifest was regenerated from all 862 remaining data files.
Only the two B200 0.24.0 DSA tables, their shared metadata, the obsolete 0.25.1
pilot tables/metadata and generated manifest change in the systems tree;
1,590 other files remain identical.

## Execution health and controls

[allocation-summary.csv](allocation-summary.csv) lists all 16 successful final
allocations and READY hashes. The exhaustive private READY inventories were
rechecked after the final mirror. All 2,971 native worker observations were
zero exits and no longer alive. Every successful production/control timing
uses the original 10 warmups and 30 CUDA graph runs, without eager fallback;
the [timing audit](timing-health-audit.json) records no throttled measurement.
That audit's source-review field reflects its earlier timestamp; the separate
source attestation below records the subsequent disposition.

All 512 control calls completed, producing 768 timings and 384 before/after
pairs. [The complete paired-control report](../control-repeatability/README.md)
retains median drift −6.219% and largest decrease (signed change −17.674%). There was no
predeclared numeric threshold and no repeatability-pass claim. The successful
measurements are retained without scaling or correction.

The wrapper was verified within the original 853-file bundle before job
submission, but the original qualification exercised its runtime dependency
and exact cases through a different wrapper. The separately reviewed source
attestation says `wrapper_prequalified=false`; no qualification record was
rewritten. See the paired-control source-review records.

The [final SG/TRT isolation](sg-trt-final-isolation.json) passed with zero
differences: 5,040 source comparisons, 105 native DSA-view comparisons and
3,360 prediction configurations. Each side actually returned 8,000 of 13,440
requested points. The original missing-data and missing-system-FLOPS errors
remain identical and are not counted as successful predictions. The
[full generated-manifest diff](generated-manifest-diff.json) includes 18 refreshed
SG derived groups; these did not alter the measured SDK sources, views or
prediction results. The SDK source and loaded native binary hashes were
verified unchanged across the comparison.

Historical model-accuracy comparisons are complete in the companion engine
PR #407; see the [parent report](../../README.md). Remote CI remains separate.
Raw native logs, full source inventories and complete case ledgers remain in
the private locations and SHA ledgers recorded by `summary.json`; this report
does not duplicate them.
