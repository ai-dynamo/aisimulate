# AIC-2004: bounded B200 SGLang DSA collection

These are fresh measurements from the corrected SGLang 0.5.14 attention-module collector. They are an **explicit validation overlay**: this PR does not replace the packaged default SGLang DSA tables. The existing 95,203 context and 6,048 generation rows have legacy provenance, and there is no older B200 SGLang DSA table to supply missing coordinates. Replacing that domain with this bounded matrix would create an unvalidated extrapolation change.

| Artifact | Original observations | Unique published rows | Qualified execution |
|---|---:|---:|---|
| `dsa_context_module_perf.parquet` | 28 | 28 | Native eager prefill above the 2,048 new-token graph ceiling |
| `dsa_generation_module_perf.parquet` | 22 | 18 | Native CUDA graph decode, 50 timed iterations |

All 50 planned attempts passed. Four repeated decode keys retain the original priority-cohort observation in the published table; every observation remains in `raw/`. The native schema and actual latency values are preserved. Full-indexer decode repeats were 0.0913 versus 0.0781 ms at history 8,192 and 0.1608 versus 0.1371 ms at history 131,072 (max/min spreads 16.9% and 17.3%). All four reuse observations round to 0.0390 ms in the native parquet. The priority values were retained by a predetermined source order, not selected for speed; no timing-stability claim is made. `plan.json` also lists the 68 graph-covered prefill attempts deliberately excluded from this qualification scope.

The matrix covers GLM-5.2 NVFP4's BF16 attention projections at eight local heads, GLM-5.2 FP8 block projections at eight heads with FP8/BF16 KV, and DeepSeek-V3.2 FP8 block projections at 16 heads with FP8/BF16 KV. GLM points include full-indexer and real producer/reuse layers. Decode histories include 8,192 and 131,072, plus 1,048,575 for the GLM BF16/FP8-KV pair. Eager prefill covers dense and sparse dispatch and a subset of the original cached-prefix cohort. This is synthetic module timing, not checkpoint inference or full-model accuracy acceptance.

## What changed

The collector now performs the native QKV projection on every forward, initializes historical latent and indexer KV through SGLang's native writers, prepares attention metadata once per batch, and materializes native scheduler input IDs and token counts before dispatch. Reuse layers consume actual producer top-k indices. Generation coordinates are `isl=1`, `step=history`, with native total length `history+1`.

For graph-covered prefill, the adapter uses the exact runtime's compiler, padding, metadata and capture path. The default stock chunk size remains 8,192. Per-model replay contexts and logits-output wrapping have been moved outside layer timing. However, the remaining native Dynamo entry is still visible in the measured module loop. **TC prefill timings are diagnostic and are not in the qualified tables above.** No constant has been subtracted, and native piecewise execution has not been replaced with a bare CUDA graph.

## Runtime and measurement receipts

Job `4764836` ran one B200 on `umbriel-b200-079`, completed with exit `0:0` in 7m21s, and released the allocation. The runtime was official SGLang 0.5.14 at commit `49e384ce9d304648e9959666ecb8ce8cd98d0deb`, amd64 image digest `sha256:9611bd4c5624b0e9e17829506188a12f17205f2083de0dd44d6c521733553a50`. Collector source is commit `6a3f8fc6`; the frozen payload manifest records individual file hashes. Test-only changes after that commit do not alter the collected source.

Each point uses ten warmups and 50 timed forwards. Prefill uses CUDA events around the back-to-back native eager module calls, including their host submission gaps; batch construction, cache population and metadata preparation are outside that interval. Decode uses the collector's native CUDA graph benchmark, with `used_cuda_graph=true` recorded for every observation. The driver only observes batch construction and retains benchmark results; it does not replace the measured forward.

`runtime/case-receipts.json` records actual input-ID counts, positions, cache locations, module configuration and generation graph evidence. The per-group clock logs contain 56 read-only snapshots: 54 show SM/memory clocks 1,965/3,996 MHz and event bits zero; the initial idle snapshot shows 120 MHz/event `0x1`; one post-case snapshot for GLM FP8/BF16-KV q4096 reuse shows event `0x4` at 1,965/3,996 MHz, 36°C and 459 W. Uncorrected volatile ECC counts are zero throughout. Node state had no DRAIN/reboot flag at both job endpoints. A later node-level `pegged clocks` DRAIN was recorded at 18:07:10 PDT, after both this collection and the following prototype had finished. `runtime/node-later-observation.json` preserves the later raw scheduler output and SHA256. Its onset and affected GPU are unknown, so neither retroactive invalidation nor continuous health is inferred. Three task-used nodes later received this alert; no cause has been established. These snapshots are observations, not continuous health certification; disabled power sampling's `throttled=false` is not used as health evidence.

## Diagnostic and incomplete coverage

The earlier 342-attempt matrix completed successfully but its TC timings included more model-entry overhead. Its prefill results are not promoted. Its node later acquired a clock-related DRAIN flag during the final part of that job, so this publication uses the subsequent bounded run on node 079. One intermediate allocation failed before measurement because the previous node-local image had been cleaned up; the failure was retained and the exact image was imported again.

A separate old/new q1 profile on node 079 found identical kernel counts and finite, exact numerical equality of native outer and inner outputs. The comparison used zero tolerance, not bytewise equality. Clean timer observations improved full/reuse from 0.6446/0.3213 ms to 0.5949/0.2874 ms; remaining Dynamo lookup and native custom-op submission costs are visible. These diagnostic observations are not full-model predictions or MAPE results.

Forty repeated q1 captures kept native compile-hook counts at zero after each case. Reserved memory reached a plateau after 32 captures. Allocated memory still grew by about 49 KiB per later capture, so complete elimination of retained allocations is not claimed.

No full SGLang whole-model truth matching this runtime was available. Other head counts, unsampled batch/prefix ranges, CP-specific MQA/top-k tables, and other GPU architectures remain outside qualification. `accuracy_acceptance=NOT_EVALUATED` is explicit in the manifest.

## Reproduction

Use the pinned image, a single B200, the source commit above and the repository's cached model configurations. Set `PYTHONPATH` to the checkout's `python/aisimulate` directory and run `collect_target.py --plan plan.json --group <group-id> --out <directory>` for each group in the plan. The original launcher logic and environment settings are included in `launch.sh` and `stage-job.sh`, with SPDX header comments added to the published copies. The frozen payload hashes identify the executed originals; `manifest.json` records both original and published script hashes. Their `/task`, `/output`, remote scheduler and node-local paths describe the recorded allocation and must be mapped to the new environment. Keep compile caches private to the allocation. The native input/cache seed and iteration settings are in the retained driver.

Collector unit validation: 3,808 passed, 51 skipped, 39 deselected. The context-lifetime test added subsequently passed with the eight existing adapter tests. All data here require an explicit consumer validation overlay; defaults remain the legacy tables.
