# Hecate VR200 GLM-5.2 NVFP4 pilot

This data supports one opt-in SGLang graph-prefill profile, `sglang_glm52_nvfp4_vr200_tp4_graph_v1`, on a four-GPU Hecate VR200 node (SM107). Its immutable profile ID is `829a83e1629ba546dd4bd90e75a2e2496b7fb24ddc8b60dfbf076ba02312cbce`. Use the canonical [`RustForwardPassPerfModel.best_available`](../../../../../../../docs/core-api.md#vera-rubin-glm-52-graph-prefill-pilot) constructor and its direct `predict_prefill_latency` method.

The exact runtime is SGLang `0.5.18+nvinternal.rubin.0.8full.66997102`, source commit `02c5a855aceb968c310e6fbc6632270e26edc84b`, OCI image index `sha256:53299500a280c8de34bd484507a45b2f83b4d5e7c999b77284fa31930f7e63ab`. The checkpoint is `nvidia/GLM-5.2-NVFP4`, with config SHA-256 `d3783a603e5aa9cb58eff7a5d8ac9c42d83156b229efe185c72e9e2dc8444923` and quantization-config SHA-256 `212394aabaa6a4e7823668e7a03d6295422aa2ca2cbdbfbfc519eab602ad20e6`. The profile binds TP4/MoETP4/EP1, PP1, attention DP1, FP8 KV, BF16 projections, deterministic FlashInfer top-k, forced DSA and breakable prefill graphs. Full source/runtime identity, native launch arguments, qualifications and external evidence hashes are retained in the identical profile sidecars beside both composite tables.

The bundle has 1,023 rows across seven parquet tables: 518 GEMM, 88 MoE, 304 DSA context, 84 DSA generation, 14 all-reduce, seven joint attention sequences and eight communication/norm boundaries. The original five tables and hardware YAML remain byte-identical to the accepted input bundle. The packaged [publication receipt](../../prefill_graph_publications/sglang_glm52_nvfp4_vr200_tp4_graph_v1.json) records the output hashes; the two profile sidecars bind every composite row's exact Float64 payload and the original table/YAML hashes.

The attention sequence measures the complete 78-layer attention pattern once (21 full indexers and 57 index-reuse layers). Communication/norm rows measure separate post-attention and following-MLP boundaries, applied 78 and 77 times respectively. These are isolated measured proxies with declared limitations; the full profile documents the synthetic attention state and repeated communication-block methodology. The consumer uses exact keys, rejects incomplete or altered profiles, and applies no full-forward correction or fitted scale.

| Batch | New tokens/request | Cached tokens/request | Predicted ms | Native graph mean ms | Relative error |
| --- | --- | --- | --- | --- | --- |
| 1 | 1,024 | 0 | 34.9049 | 36.8324 | −5.23% |
| 2 | 1,024 | 0 | 48.3867 | 50.9990 | −5.12% |
| 1 | 1,024 | 1,024 | 37.8595 | 39.5441 | −4.26% |
| 1 | 8,192 | 0 | 196.0992 | 202.3784 | −3.10% |
| 2 | 8,192 | 0 | 364.9135 | 374.7311 | −2.62% |
| 1 | 16,384 | 0 | 378.5329 | 387.1901 | −2.24% |
| 1 | 16,384 | 16,384 | 442.6695 | 440.9783 | +0.38% |

The independent comparison SHA-256 is `5e12de81a7b80628efb8513d76d268369b48d1b62424864edf7aff7604326930`; its separate root acceptance is `681d67f2284321be10b9e28a66880e9bc7889ba24d64c4e8beedbe3818f392ca`. All seven mean forward latencies satisfy the fixed 15% criterion. Native references are comparison evidence and never inputs to the prediction. The direct API's `isl` includes the cached prefix; use `new + cached` from the table.

This qualification does not cover scheduler TTFT, decode, model quality, arbitrary shapes or general Vera Rubin support. The hardware profile also contains assumptions: its HBM and network peaks are inferred, and empirical memory coefficients fit seven standalone graph RMSNorm points with about 33% maximum residual. See [`profile-evidence.json`](../../profile-evidence.json) for observed inputs, published source references and fit limitations. Raw GPU exports and internal source archives remain external evidence, not runtime dependencies. The [dedicated collector](../../../../../collector/sglang_rubin/README.md) documents collection and offline publication.
