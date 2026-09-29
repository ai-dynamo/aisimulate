<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# GLM-5.2 NVFP4 on Hecate VR200: forward-step accuracy

The bounded pilot supports both prefill and decode forward-step prediction on the pinned SGLang runtime. All **7/7 prefill cases** meet the original ±15% criterion, with **5.23%** worst absolute error. **17/18 decode cases** meet that criterion; the remaining batch-1 case is **−17.09%**. On 2026-09-22, the pilot owner accepted this observed decode residual for the initial release. This is a documented acceptance of 17.0892%, not a claim that every case meets 15% or a strict 17.00% bound.

These results compare model forward steps at matching batch, new-token and past-KV lengths. They do not validate scheduler TTFT, end-to-end TPOT, throughput or general Vera Rubin support. The native measurements are comparison references, never prediction inputs; no full-forward fitted correction or sample selection is applied.

## Configuration and measurement

| Setting | Validated value |
| --- | --- |
| Hardware / topology | Four Hecate VR200 GPUs, SM107; TP4, MoE TP4/EP1, PP1, attention DP1/CP1 |
| Checkpoint | `nvidia/GLM-5.2-NVFP4`, cached snapshot `hf-aec724e_orig` |
| SGLang | `0.5.18+nvinternal.rubin.0.8full.66997102`, source `02c5a855aceb968c310e6fbc6632270e26edc84b` |
| Dynamo CI image index | `sha256:53299500a280c8de34bd484507a45b2f83b4d5e7c999b77284fa31930f7e63ab` |
| ARM64 image manifest | `sha256:1c7ffbccde1dd9a3ec894db29a1aa3721e3ba393fc7337b2148b2f184aa0bcab` |
| Compute / KV | NVFP4 experts, BF16 projections, FP8 E4M3 KV; TRTLLM DSA and FlashInfer TRTLLM MoE |
| Serving policy | Maximum 32 requests; 40,000-token context limit; 16,384-token prefill chunks; radix cache disabled |
| Graphs | Breakable prefill graphs with token buckets `[1024, 2048, 8192, 16384]`; full decode graph with exact measured batch buckets |
| Predictor | Canonical Rust-backed `op_level` estimator, `fallback_policy="deny"`, `database_mode="SILICON"`, online correction disabled |

Both acquisitions use `SGLANG_ENABLE_MOE_DEFERRED_FINALIZE=0`, `SGLANG_ENABLE_PCG_DSV2_DUAL_STREAM=0`, `SGLANG_FLASHINFER_AUTOTUNE_CACHE=1`, `SGLANG_FLASHINFER_AUTOTUNE_EXTEND=0`, deterministic FlashInfer DSA top-k and the `small` tie break. Prefill is forced onto the matched DSA path. The [collector documentation](../python/aisimulate/collector/sglang_rubin/README.md#frozen-inputs) and [packaged profile](../python/aisimulate/src/aisimulate_core/systems/data/vr200_hecate/README.md) retain the checkpoint hashes and complete runtime identity.

Native time is measured with CUDA events around `ModelRunner.forward` on its forward stream. ForwardBatch construction, prefix materialization, sampling, correctness checks and validation output copies are outside the timed boundary. For each aligned step, take the maximum of the four rank-local durations, then the arithmetic mean of **all 15 steps** in three windows of five. No outliers are discarded and no median is substituted. Signed error is `100 * (predicted_ms / native_mean_ms - 1)`; negative values mean underprediction. Tables round milliseconds to four decimals and errors to two.

Graph execution and graph/eager numerical controls passed independent review for both acquisitions. The fixed controls use absolute tolerance 0.25, relative tolerance 0.01, relative L2 at most 0.01 and cosine similarity at least 0.999. These are execution checks, not model-quality validation. Prefill compares matched DSA paths; small default-MHA versus forced-DSA fixtures remain numerically different.

## Prefill results

Prefill uses the opt-in `sglang_glm52_nvfp4_vr200_tp4_graph_v1` profile, whose immutable SHA-256 is `829a83e1629ba546dd4bd90e75a2e2496b7fb24ddc8b60dfbf076ba02312cbce`. The seven homogeneous contexts below are its complete admitted scope. Each request processes the stated new tokens after a native extend materializes any nonzero prefix.

| Batch | New tokens/request | Past KV/request | Predicted ms | Native mean ms | Signed error |
| --- | --- | --- | --- | --- | --- |
| 1 | 1,024 | 0 | 34.9049 | 36.8324 | -5.23% |
| 2 | 1,024 | 0 | 48.3867 | 50.9990 | -5.12% |
| 1 | 1,024 | 1,024 | 37.8595 | 39.5441 | -4.26% |
| 1 | 8,192 | 0 | 196.0992 | 202.3784 | -3.10% |
| 2 | 8,192 | 0 | 364.9135 | 374.7311 | -2.62% |
| 1 | 16,384 | 0 | 378.5329 | 387.1901 | -2.24% |
| 1 | 16,384 | 16,384 | 442.6695 | 440.9783 | +0.38% |

All seven cases pass ±15%; worst absolute error is 5.233313717%. `predict_prefill_latency(bs, isl, prefix)` takes total input length in `isl`, so use `isl = new_tokens + past_kv` and `prefix = past_kv`.

## Decode results

Decode uses the default op-level path with no prefill graph profile and no observed-MoE distribution override. Each row measures exactly one current token per request with `K` past KV tokens, giving attention length `K + 1`. A native extend materializes the prefix before measurement. Repeated forwards reuse the same prepared state, rather than advancing an end-to-end generation loop.

The native decode capture list is `[1, 2, 3, 4, 8, 12, 16, 24, 29, 31, 32]`. Every tested batch has an exact graph bucket; this comparison does not qualify padding to a larger bucket. Each timed forward replays one decode graph with no eager fallback.

| Batch | Past KV/request | Predicted ms | Native mean ms | Signed error | Original ±15% criterion |
| --- | --- | --- | --- | --- | --- |
| 1 | 1,024 | 5.7822 | 6.9740 | -17.09% | Outside; pilot accepted |
| 3 | 1,024 | 6.5475 | 7.1976 | -9.03% | Pass |
| 8 | 1,024 | 8.2678 | 8.7500 | -5.51% | Pass |
| 29 | 1,024 | 11.3239 | 11.8190 | -4.19% | Pass |
| 31 | 1,024 | 11.5524 | 11.8522 | -2.53% | Pass |
| 32 | 1,024 | 11.6666 | 12.0179 | -2.92% | Pass |
| 1 | 8,192 | 5.9751 | 6.9281 | -13.76% | Pass |
| 3 | 8,192 | 6.6459 | 7.5474 | -11.94% | Pass |
| 8 | 8,192 | 8.3668 | 8.8210 | -5.15% | Pass |
| 29 | 8,192 | 11.6197 | 12.2791 | -5.37% | Pass |
| 31 | 8,192 | 11.8360 | 12.3696 | -4.31% | Pass |
| 32 | 8,192 | 11.9441 | 12.4937 | -4.40% | Pass |
| 1 | 32,768 | 6.1329 | 7.1904 | -14.71% | Pass |
| 3 | 32,768 | 6.7889 | 7.8107 | -13.08% | Pass |
| 8 | 32,768 | 8.5756 | 9.1542 | -6.32% | Pass |
| 29 | 32,768 | 13.7716 | 12.6421 | +8.93% | Pass |
| 31 | 32,768 | 14.1897 | 12.7762 | +11.06% | Pass |
| 32 | 32,768 | 14.3983 | 12.9684 | +11.03% | Pass |

The batch-1, K=1,024 residual is 1.1918 ms: 5.7822 ms predicted versus 6.9740 ms measured. Batch 1 at K=32,768 passes narrowly at −14.71%. These results come from **one complete, independently reviewed decode acquisition**. Its three timing windows are descriptive subdivisions, not independent runs or confidence intervals; repeatability across independent acquisitions remains unestablished. The pilot accepts the observed residual without changing the original comparison, reducer or threshold fields.

## Predicting the reported steps

Use the packaged tables through the canonical constructor. The default decode configuration is:

```python
from aisimulate_core.sdk import RustForwardPassPerfModel

config = {
    "model": "nvidia/GLM-5.2-NVFP4",
    "system": "vr200_hecate",
    "backend": "sglang",
    "backend_version": "0.5.18+nvinternal.rubin.0.8full.66997102",
    "worker_type": "decode",
    "tp": 4, "pp": 1, "attention_dp": 1,
    "moe_tp_size": 4, "moe_ep_size": 1,
    "gemm_quant_mode": "bfloat16", "moe_quant_mode": "nvfp4",
    "fmha_quant_mode": "bfloat16", "kvcache_quant_mode": "fp8",
    "comm_quant_mode": "half",
    "estimation_mode": "op_level", "fallback_policy": "deny",
    "database_mode": "SILICON", "strict_provenance": True,
    "enable_shared_layer": False, "nextn": 0,
    "estimator_config": {
        "op_level": {},
        "correction": {"enabled": False},
    },
}
decode = RustForwardPassPerfModel.best_available(config)
decode_ms = decode.static_phase_latency(
    batch_size=1, input_tokens=1024, output_tokens=2, prefill=False,
)
```

`static_phase_latency` returns total decode time over `output_tokens - 1` iterations. **Use `output_tokens=2` for one decode step**; `output_tokens=1` requests zero decode iterations. Here `input_tokens=K` matches the table's past-KV length and the measured attention length is `K + 1`. This API does not represent scheduler TTFT.

For prefill, explicitly select the graph profile and use its direct scalar method:

```python
prefill_config = {
    **config,
    "worker_type": "prefill",
    "estimator_config": {
        "op_level": {
            "prefill_graph_profile": "sglang_glm52_nvfp4_vr200_tp4_graph_v1",
        },
        "correction": {"enabled": False},
    },
}
prefill = RustForwardPassPerfModel.best_available(prefill_config)
prefill_ms = prefill.predict_prefill_latency(bs=1, isl=2048, prefix=1024)
```

Save the complete resolved configuration from `diagnostics()["provenance"]["config"]` with predictions. The selected prefill profile rejects decode, mixed steps, unsupported shapes, the generic static API, energy/SOL diagnostics and scheduler/replay selection. Its original seven-context qualification is unchanged. See the [Core API](core-api.md#vera-rubin-glm-52-graph-prefill-pilot) for the profile's complete API restrictions.

## Evidence and limits

The prefill native reference is [GitLab job 448540801](https://gitlab-master.nvidia.com/dl/jet/ci/-/jobs/448540801), Slurm 619010. The decode native reference is [GitLab job 449988661](https://gitlab-master.nvidia.com/dl/jet/ci/-/jobs/449988661), Slurm 626087. Original artifacts retain launch configuration, source identity, graph audits, numerical controls and raw rank-local timings. External evidence is identified by the following SHA-256 values:

| Evidence | SHA-256 |
| --- | --- |
| Prefill seven-row comparison | `5e12de81a7b80628efb8513d76d268369b48d1b62424864edf7aff7604326930` |
| Prefill independent-review root acceptance | `681d67f2284321be10b9e28a66880e9bc7889ba24d64c4e8beedbe3818f392ca` |
| Decode 18-row comparison | `8b63a6245ef493cd227b0cabb3a884b5d910c3abd6f99e598b9c2de12ba2514c` |
| Decode independent-review root acceptance | `c41ee6064f817f27b79b3c15e134a15f13273bcf81b33fdf79f66898a2f599d0` |
| Source/release-wheel prediction equivalence | `015fe78a3282d8040baa1adff82f292d47e38df800bd78ab55e88bc9ebca3ea8` |

The later review acceptances supersede intermediate pending-review fields in the frozen comparison files. The decode acceptance validates acquisition and arithmetic while preserving the original accuracy failure at 17/18. The subsequent pilot-owner acceptance permits this documented residual; it does not rewrite that evidence. Source and release-wheel builds of `f39e50ef1147190ade080ae13da2e124cb3c06f9` produce identical predictions for all 18 decode and seven prefill calls; prefill values agree with the original comparison at the displayed precision.

Coverage is limited to the pinned checkpoint, image, topology, flags, graph buckets and synthetic fixtures above. Other batches, context lengths, capture policies, quantizations and hardware require separate validation. Timing agreement does not establish model quality, peak graph/KV memory, or end-to-end serving accuracy. The [hardware provenance](../python/aisimulate/src/aisimulate_core/systems/profile-evidence.json) records inferred hardware peaks and empirical memory-model limitations. No pending or failed operation-level diagnostic job contributes timing data to this report.
