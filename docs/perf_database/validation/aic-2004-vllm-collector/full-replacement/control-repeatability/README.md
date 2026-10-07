# Paired fixed-control measurements

The 384 timing pairs in [paired-control-points.csv](paired-control-points.csv)
are real before/after measurements on the same 16 B200 GPUs. Each side contains
256 native control calls and 384 full/reuse CUDA-graph timings. The
[summary](summary.json) retains all aggregates, the largest changes and source
hashes. These controls do not enter the performance tables.

| After vs before | Change |
| --- | ---: |
| Median | -6.219% |
| Mean | -6.276% |
| Largest decrease | -17.674% |
| Largest increase | +1.406% |
| 95th percentile absolute change | 14.079% |

A negative value means the after measurement was faster. This is a systematic
change, not evidence that timings were identical throughout collection. No
numeric repeatability threshold was declared beforehand; this report does not
invent one or claim numeric repeatability acceptance.

All retained pairs request the original ten warmups and thirty graph replays,
report graph execution, retain the same native attention backend, and have
clean per-case CUDA/BLAS/traceback log checks. No helper result reports
throttling. Both sets of postflight endpoints report 1965 MHz SM, 3996 MHz
memory, and a 1000 W power limit; endpoint temperatures are similar. These
samples do not observe every timing interval and do not prove a warmup,
cache, CPU-resource or GPU-clock explanation. No correction factor is applied.

Controls use batch 1, eight local heads, FP8 KV, and BF16 attention compute:
prefill Q128/P0 and decode with 8192 history tokens, across DeepSeek/GLM and
four GEMM modes. Their repeatability does not characterize all large-prefix,
high-batch or native-failing shapes.

The original qualification did not execute this control wrapper or include it
in its evidence inventory. That limitation is retained: **wrapper_prequalified
is false**. A separately reviewed, explicit post-run source attestation now
traces the wrapper through the unchanged frozen bundle, original prelaunch
verification/submission events, sixteen actual launcher pairs and all control
receipts. See the [root source review](root-source-review.json) and
[independent validator review](independent-validator-review.json).

The original qualification, binding and measurement receipts were not
rewritten. The default strict source gate remains intact; the separately
reviewed route requires an explicit attestation and its exact reviewed SHA.
Its 44 validator tests passed. This is a source-chain disposition only:
it does not accept numeric repeatability drift, native failures, final
publication, model accuracy or merge readiness. Full raw source receipts
remain retained outside the PR, with paths and hashes in the reviews.

The repository CSV copies use LF line endings. The [format receipt](../artifact-format.json)
records their original CRLF source hashes and LF copy hashes, and verifies
identical ordered rows and cells. Original private evidence bytes are unchanged.
