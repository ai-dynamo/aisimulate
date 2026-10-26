# B200 vLLM 0.31 NVFP4 full collection validation

- Parent pipeline **76765385**, child **76765436**, job **510097837** succeeded. Collector source `8a411395d3161f1320a9de0e8670deacb40b314c`; runner `3715eaf11590df65a6d4ec7c67d56f45ff76fd19`.
- **37,518 planned / 37,518 measured / 0 failed / 0 unattempted / 0 duplicate identities / 0 invalid latencies**. Every row reports `FlashInferCuteDslNvFp4LinearKernel` on NVIDIA B200, vLLM 0.31.0.
- Independently expanded the exact source getter on SM100: 153,032 cases across four dtypes; 38,258 NVFP4 cases before the approved 740 GLM exclusions. The remaining 37,518 exact identities equal both checkpoint successes and Parquet keys. Excluded cases are deferred work, not successful or failed callbacks.
- Rebuilt the authored collector hash, selected case-plan hash and runtime manifest hash from immutable source. The published artifact and raw result Parquet agree; the pinned official image and framework revision match the request.
- NVFP4 Parquet SHA256: `8b5b51b84c23923f598d0025ba438e5a35326446aee067a9a9cd1cf7bf171df4`. No canonicalization or latency replacement is needed for this event.
- Historical exact-key baseline: all 37,518 keys exist in vLLM 0.25 with zero matching historical failures; 35,890 exist in 0.27.1, with 1,628 absent there. Older values remain diagnostic only. Median signed delta is -12.33% versus 0.25 and -7.87% versus 0.27.1; historical clocks/runtime are not controlled, so these are not regression or accuracy verdicts.
- Prior native numerical gate **2384885** passed 20 bounded quantized-operand cases in eager and CUDA graph modes. Its GEMM collector file is byte-identical to the full-run source and its image/runtime manifest is the same. This does not claim numerical testing of all 37,518 shapes.
- Append all NVFP4 rows to the existing 74,433-row canonical event without changing any earlier row. The resulting **111,951-row** GEMM table has unique dtype/M/N/K identities and two schema-2 collection events. The **636 pre/post-BMM rows remain unchanged**.
- All **111,951 GEMM rows** pass Python orchestration and native consumer queries with shared-layer reuse disabled. NVFP4 uses the existing 9/16-byte memory model and FP4 tensor-core rate; none of its rows need a SOL floor. The existing 39 BF16 SOL floors remain explicit.
- Publication remains **partial**: FP8 M>=2 and explicitly deferred GLM-specific cases remain outside this table; 1,110 known block-FP8 failures remain recorded. No defaults, merge or end-to-end accuracy claim.

Detailed pins, hashes, counts, filters and historical references: `b200-nvfp4-full-validation.json`. The data PR stores the same report as `nvfp4-full-validation.json`.
