# B200 vLLM DSA replacement validation

B200 replacement collection completed **all 34,546 planned native calls**:
32,306 succeeded and 2,240 recorded native failures. The applied replacement
contains **48,585 successful rows**: 26,005 prefill and 22,580 decode. Failed
calls contribute no rows; 3,360 planned variants and 1,680 old-table coordinates
have explicit missing-measurement records.

Official data rules R1–R7, all 2,186 backend-fact slices, full reuse-manifest
regeneration, and 10,236 non-target source comparisons passed. The two B200
0.24.0 DSA tables have been replaced and the obsolete 0.25.1 pilot removed.
Final SG/TRT source/native-view/prediction isolation passed; historical
prediction comparisons are complete and CI remains a separate gate. See the [data report](full-replacement/data-validation/README.md)
and [machine-readable summary](full-replacement/data-validation/summary.json).
The bounded GPU source qualification remains in
[qualified-source-summary.json](full-replacement/qualified-source-summary.json).

Eight DeepSeek-V3.2/B200/vLLM golden records were refreshed for the prediction
changes caused solely by replacing the two DSA tables. The official golden
maintenance tool pinned the values after the original baseline native engine
and final engine reproduced identical new-table results; all eight targeted
checks passed ([attribution and values](full-replacement/golden-refresh.json)).

The previous 0.25.1 pilot and its report are superseded because the collector
omitted native `slot_mapping`, preventing current-query MLA KV writes. Its
receipts remain in this directory solely as historical diagnostic evidence;
see [the superseded report](superseded-0251-README.md). The replacement uses
actual vLLM 0.24.0 and replaces the two original B200 DSA tables directly.

The revised collector initializes FP8 and packed NVFP4 projection weights
through native quantization, forwards current-query KV slots, uses the
canonical GLM-5.2 model configuration, includes full and native index-reuse
variants, and fixes standalone CLI dispatch. Independent single-rank workers
use native file rendezvous instead of a shared fixed TCP port. The explicit
CUDA graph cleanup is enabled only by the vLLM MLA/DSA caller.

The attempted domain covers both GLM DSA and DeepSeek-V3.2, all original
consumer shapes, BF16/block-FP8/per-tensor-FP8/NVFP4 projection requests,
BF16/FP8 KV, and GLM full/reuse variants. GLM-5.2 is the explicit canonical
reference for the GLM architecture; the old GLM-5 model identity is retained in
the coverage ledger, and this is not a same-checkpoint claim. Every failure
keeps its original failed outcome. No old latency fills a missing measurement.

All 16 allocations exited zero; 2,971 native worker-exit observations were
zero and no longer alive. All 512 before/after control calls completed. Their
384 paired timings shifted by a median **−6.219%**, with largest decrease
(signed change **−17.674%**). No numeric threshold was declared beforehand, so this is
not a repeatability-pass claim. See the [paired controls and source review](full-replacement/control-repeatability/README.md).
The standalone control wrapper was frozen and verified before submission but
was not run in the original GPU qualification; its separate after-run source
attestation preserves that distinction and leaves the qualification unchanged.

## Completed source qualification

Job **4778662** ran the frozen source on one B200 with actual vLLM **0.24.0**,
Torch **2.11.0+cu130**, CUDA **13.0**, and FlashInfer **0.6.12**. It exited zero.

Earlier source-header references, including job 4757223, are historical API
checks only; qualification of the current frozen source is recorded by
job **4778662** above.

| Check | Observed result |
| --- | --- |
| Canonical precision/output cases | 32/32 passed; 56 full/reuse graph measurements |
| Native outputs | All finite and nonzero, including prefix-zero single-query and long-prefix prefill |
| GEMM precision | Native BF16, block-FP8, per-tensor FP8, and packed NVFP4 methods/weights verified; NVFP4 scales finite and positive |
| Native executor controls | 16/16 passed; resume skipped all 16 and ran zero new workers |
| Failure-observation protocol | One success and two actual native failures; both failures remain `FAILED`/`unexpected` |
| Native child teardown | All three observed native workers exited zero and stopped |
| Frozen inputs | 427 payload files verified; actual collector and four probed native source hashes matched the bound sources |

Measurements retained the original 10 warmups / 30 graph runs, with graph
fallback disabled. Numeric output observations ran outside the timed region.
FD-level capture preserves native stdout/stderr; a returned timing containing
a CUDA/BLAS error is a failed observation and cannot be published.

The slot-mapping check combines frozen source and execution evidence: the
collector supplies `common_attn_metadata.slot_mapping` under both native
attention/indexer layer keys, and the exercised forwards produced finite,
nonzero output. Slot tensor values were not separately serialized. This does
not establish an independent numerical oracle, matched historical latency,
whole-model accuracy, or concurrent-GPU qualification.

The cached SQSH artifact SHA-256 is
`e34a40cb957ea15986ccbc2597ad9d16dbc707dd0aaefadf510aed0fde0cf12f`.
Actual runtime/source bytes were verified. The retained OCI reference is a
declared registry identity; a cryptographic SQSH-to-OCI association was not
newly established. The node's `REBOOT_REQUESTED` flag is preserved in the raw
evidence; postflight reported no GPU recovery action, active hardware thermal
or power-brake slowdown, or uncorrectable ECC.

## SGLang / TRTLLM isolation

The complete collector unit run passed **3,870 tests**, with **57 skipped**
and **39 deselected**; skips and deselections are not counted as passes.
The receipt records the completed run's log hash. Full GPU attempts and the
official data/source checks above are complete. Final generated-data SG/TRT
isolation also passed: 5,040 source comparisons, 105 native DSA-view comparisons,
and 3,360 prediction configurations. Each side actually returned 8,000 of
13,440 requested prediction points, with identical values and identical
existing missing-data/compute-capability errors. See the
[final isolation receipt](full-replacement/data-validation/sg-trt-final-isolation.json).
Remote CI and historical silicon comparisons remain separate from this isolation check.

CI-pinned Ruff 0.14.1 passes the actual `fast-ci.yml` check/format targets and
all 14 changed Python files. The broader package check finds one pre-existing
`I001` at `tests/unit/test_support_plan.py:749`; that file is byte-identical
to baseline `e45612e1` and `origin/main`, and checking the original git blob
reproduces the same error. Broad formatting passes. The unrelated test was
left unchanged; the broader lint check is not claimed green.

The shared helper keeps its previous default behavior. The targeted collector
suite, including helper isolation, passed 63 tests. Another 36 comparisons
executed the actual helper functions: 18 default
scenarios matched the original source, and 18 explicit-cleanup scenarios
matched the pre-isolation vLLM implementation. These are CPU control-flow
checks, not GPU timing comparisons.

Across SGLang/TRTLLM on SM90/100/103/120, 200 ordered case-list comparisons
matched exactly, covering 468,812 list entries and the recorded model filters.
Shared helper/case YAML bytes still change their truthful provenance hashes;
this does not mean case-generation or default helper behavior changed. No
hash was hidden or reused. These checks do not qualify untested framework
builds; the separate final generated-data isolation above covers the actual
published manifest and recorded SDK query matrix.

## Observed large-query native failure

With the native `FLASHINFER_MLA_SPARSE` path, the pinned vLLM adapter presents
the flattened query-token count as FlashInfer's batch dimension. In the pinned
FlashInfer launch, that dimension becomes `grid.z`. A 65,536-token query
therefore exceeds the native launch dimension limit. The two exercised
`B=2, Q=32768, H=8` cases each emitted 12 actual
`CUDA_ERROR_INVALID_VALUE` messages at `fmhaKernels.cuh:328`; the later graph
exception includes `_v_up_proj` / `CUBLAS_STATUS_EXECUTION_FAILED`.

This is an observed failure of the selected native path, not a reason to
pre-skip shapes or substitute eager timings. The pinned error handler prints
the launch error without throwing; eager timing alone can therefore appear
successful. The collector driver must preserve the native log and reject such
rows. Every formal case was attempted. All 2,240 actual failures were individually
reviewed against their native logs, backend, graph request and source identity;
the successful subset is eligible for data publication while the failures and
missing coordinates remain explicit. The four GEMM modes each had 560 failed
calls; decode had none.

Immutable upstream references (links and hashes only; no external source is
copied into this report):

- [vLLM sparse adapter, commit ee0da84](https://github.com/vllm-project/vllm/blob/ee0da84ab9e04ac7610e28580af62c365e898389/vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py).
- [FlashInfer MLA wrapper, commit d768c14](https://github.com/flashinfer-ai/flashinfer/blob/d768c14e7cf5dd5df45a8a1de78ae815879f108a/flashinfer/mla/_core.py).
- [FlashInfer launcher](https://github.com/flashinfer-ai/flashinfer/blob/d768c14e7cf5dd5df45a8a1de78ae815879f108a/csrc/trtllm_fmha_kernel_launcher.cu),
  [launch dimensions](https://github.com/flashinfer-ai/flashinfer/blob/d768c14e7cf5dd5df45a8a1de78ae815879f108a/include/flashinfer/trtllm/fmha/fmhaKernels.cuh), and
  [native error reporting](https://github.com/flashinfer-ai/flashinfer/blob/d768c14e7cf5dd5df45a8a1de78ae815879f108a/include/flashinfer/trtllm/common.h).

The JSON receipts record actual-image SHA-256 values, qualification evidence
hashes, the frozen collector/runtime identities and final table hashes. Full
raw logs remain in the private campaign archive; compact per-case failure and
old-coordinate CSVs are included in the [data report](full-replacement/data-validation/README.md).
Historical model accuracy and CI acceptance are separate from these data checks.

## Historical prediction comparison

The [companion engine PR #407](https://github.com/ai-dynamo/aisimulate/pull/407)
contains `docs/perf_database/validation/aic-2004-vllm-engine/full-replacement/accuracy/`
with the full 276-output comparison and figures. For the fixed historical
46-row `current` cohort, MAPE changes from **425.80%** with original data/code
to **82.20%** with new data and **32.57%** with new data plus engine fixes.
Worst error changes from **1118.23%** to **218.32%** to **90.73%**.

The final arm improves every point relative to the original arm, while
24/46 points are worse than the data-only arm; the complete report retains
those cases. The worst final point remains a large underprediction. Exact
SDK table keys do not prove identical historical per-sequence prefix lengths,
and the silicon and recollection runtimes differ. These comparisons therefore
do not establish a same-runtime numerical or latency oracle.
