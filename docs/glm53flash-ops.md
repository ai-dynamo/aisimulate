<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GLM-5.3-Flash operator measurements on GB300

The Ops implementation builds on the SOL model graph and collects actual native serving work for both vLLM and SGLang. Required coverage is text inference, native FP8 and NVIDIA NVFP4 checkpoints, pure TP2/TP4, and inclusive context lengths through 131072 tokens. DP, PP, CP and EP remain one; MTP/speculative decoding, EPLB and CPU offload are disabled. Optional NVFP4 TP1 needs separate capacity qualification. TensorRT-LLM serving is outside this matrix.

**Delivery status: 0/8 required deployments have completed independent Ops accuracy acceptance. The measured lookup and collection software are implemented; complete qualified tables and independent prefill/decode MAPE remain pending.** Source tests, capacity tests, same-calibration diagnostics and individual native completions do not establish full deployment acceptance.

## Model and runtime identities

| Item | Immutable identity |
|---|---|
| FP8 checkpoint | `zai-org/GLM-5.3-Flash@eb9eb208eb0d988989d07a6a12d0fdeb5f52574a` |
| NVFP4 checkpoint | `nvidia/GLM-5.3-Flash-NVFP4@09b04e5e74bca08ca8549fc736d4cdd8624bfde3` |
| vLLM upstream source | `ced6857afa0ea7b2e3f0846a62e1394e90f15607` (`v0.30.0`) |
| vLLM qualified functional repair | `0.30.0+glm53tail.eb4704514fdf`; exact source/binary closure and immutable qualification summary are required |
| SGLang source | `94602c9c2b7cbdb8efd5c52802dac6a1c180089e` (`v0.5.20`) |

The original vLLM kpool candidate is quarantined following circular-tail out-of-bounds evidence. The tail repair has distinct three-profile functional qualification across all four vLLM deployments, a separate 16-case cache oracle, and one original NVFP4 TP2 131072-token ordinary-request regression. That long regression does not establish long-request coverage in the other cells; each deployment still requires its own capacity and performance qualification. Functional eligibility does not qualify a different build, Ops execution policy, operation timing or full matrix. Stock vLLM cached-prefill restrictions remain applicable to stock identities. The reference-only runtime is not an eligible measured producer.

Every deployment binds checkpoint/config, actual module precision, backend build, loaded native libraries, cache/state layout, graph/dispatch policy and requested memory/allocator policy. SGLang's use of a TRT-LLM kernel does not imply TRT-LLM serving support. Unknown historical allocator identity cannot be relabeled as a newly observed default or explicit policy.

## What is measured

- **IndexPool and NoPE sparse MLA:** learned pooling, pooled index-key writes, scoring/selection, pool-to-token expansion and incomplete tails are included in their actual native ownership boundaries. Main MLA KV is not compressed by the pooling ratio.
- **KDA:** native convolution, prefill/recurrent work, gates and normalization retain actual initial-state, chunk, precision and fusion boundaries.
- **mHC:** native pre/post and fused boundaries follow each backend; deferred work cannot be charged twice.
- **Dense/MoE and supporting work:** dense/routed/shared FFNs, routing, projections, quantization, embedding, logits and TP communication preserve native fusion and mixed precision.

The model graph has 277 named vLLM units and 366 SGLang units, plus directly measured setup. Each calibration forward selects one actual slowest whole-forward TP rank and takes all unit observations from that rank. Original per-rank observations and overlap remain available. Independent per-operation rank maxima, whole-forward residuals and fitted correction factors cannot manufacture operation costs.

Eager observations and graph observations are separate execution contracts. Graph ownership uses original-to-executable CUDA/CUPTI mappings without inserted timing nodes. vLLM serving distinguishes observed FULL, PIECEWISE and NONE execution, including padding and initialized dispatch policy. For NONE, the excluded fifth warmup establishes activity ownership and the ten retained repetitions use native interval/event measurements. SGLang prefill retains its native disabled-prefill/FULL-decode policy and native setup allocation. SGLang FULL decode requires actual capture-size and replay ownership evidence.

Each profiled calibration needs a separate unprofiled control with the same actual inputs and state. SGLang native-prefill export enforces its independent five-percent timing control. The vLLM serving and SGLang graph exporters record the original control ratios; export success alone is not timing-equivalence acceptance. Control differences and profiling perturbation remain in the evidence; timings are not rescaled. Fixed finite logit bias in the experimental SGLang decode control uses its native sampling API and still requires measured input equality. It is not a guarantee that a target token will be selected.

SGLang native prefill initializes distinct CUDA event pairs before the whole-model timer and reuses them only after the existing synchronization and all operation, setup and whole-forward interval reads. The first excluded warmup for each geometry records the actual call sequence; changed hook identity, order, arity or incomplete calls reject reuse. This collector policy is recorded separately and leaves native model calls and the five-percent timing gate unchanged. The earlier six-point pilot completed native collection but failed that gate with 26–30% perturbation; its data is excluded. Event-pool source tests do not establish lower GPU overhead or qualify a replacement campaign.

## Measured lookup

Exact physical identity, backend, checkpoint, TP, phase, precision and runtime policy must match. Missing measured data fails in SILICON and HYBRID; neither silently substitutes SOL.

New tables may explicitly opt into bounded workload interpolation. The rules are fixed before holdout latency evaluation: exact points first, then P interpolation at fixed B/Q, or P0 Q interpolation at fixed B. There is no extrapolation, batch interpolation or simultaneous two-axis interpolation. Native ownership, state and dispatch boundaries stay distinct. In vLLM, FULL/PIECEWISE endpoints retain the actual native descriptor and padding, while NONE physical tokens can vary with B*Q. Existing per-unit MLA pool/tail and KDA initial-state partitions remain enforced; a different aligned/tail partition cannot serve as an interpolation endpoint. These vLLM partitions are not imposed on SGLang. SGLang prefill requires complete shared endpoints for all 367 units; vLLM serving requires all 278 units, including setup. Raw kernel/activity fingerprints remain in the evidence and audit rather than imposing a universal requirement that every workload use identical launch geometry.

`EngineHandle.glm53flash_lookup_audit(phase, B, Q, P)` exposes each selected measured endpoint, interpolation weight, latency, activity fingerprint, rank and source/evidence identity. Successful opt-in validation rows preserve these audits and original shard/point identity. Unknown, partial or mixed table contracts are rejected. Legacy tables retain their existing lookup rules; the new SGLang prefill contract does not change the legacy decode contract.

## Campaign and acceptance

The formal geometry set freezes 397 calibration and 221 independent holdout points per deployment, with distinct corpus/request/geometry identities, five warmups and ten retained repetitions. Scheduled prefill is limited to 8192 new tokens. Original point identifiers are local to each phase and are paired with phase in unions and reports. Failed children and missing requested points stay visible; partial failed children cannot be spliced into a passing campaign.

Each of the eight required deployment cells must provide complete qualified measured operation coverage and independent whole-forward **prefill MAPE <=20% and decode MAPE <=20%**. Reports also include WAPE, P95/max APE, long-context groups, all original requested points and failure/missing counts. HTTP TTFT/TPOT is reported separately.

Qualified Parquet data and provenance enter the versioned performance database. Original campaigns and large diagnostics remain in external evidence storage. Delivery also requires exact-point and wrong-identity checks, boundary behavior, legacy regressions, a current native build, installed-wheel prediction and offline cache loading. The PR's deployment matrix is the current record of execution, collection, coverage and error status; historical smoke results are not promoted to current-runtime acceptance.

## Interfaces and reproduction

The current measured table contracts are distinct:

| Table | Schema | Native execution scope |
|---|---|---|
| `glm53flash_graph_perf.parquet` | 1 | SGLang FULL decode |
| `glm53flash_graph_perf.parquet` | 2 | vLLM FULL-only graph measurements |
| `glm53flash_graph_perf.parquet` | 3 | vLLM serving NONE/PIECEWISE/FULL, selected from the actual initialized policy |
| `glm53flash_sglang_prefill_perf.parquet` | 4 | SGLang native prefill, all 366 named units plus setup |

A shared filename does not make schemas interchangeable. Schema3 without an
analysis opt-in retains its prior exact/same-dispatch P-only rule; schema4
without the opt-in remains exact-only.

For bounded lookup, call `glm53flash_vllm_serving_export.export_serving` with
`lookup_contract="vllm_serving_bounded_p_q_v1"`, or
`glm53flash_sglang_prefill_export.export_prefill` with
`lookup_contract="sglang_prefill_bounded_p_q_v1"`. Both accept
`(calibration_root, frozen_calibration_run, new_table_path, *, control_root,
control_run, lookup_contract)`; the output basename must match the table above.
Inputs are the original frozen native run objects and separately executed
controls, not latency-only rows. The opt-in adds analysis metadata and leaves
native policy and original receipts unchanged. The corresponding complete-shard
publication API also accepts that explicit contract; it revalidates every
original child and point map before publication.

Compile the public `EngineHandle` against the bound measured database with
SILICON and denied missing-data fallback. Then call
`engine.glm53flash_lookup_audit("context", B, Q, P)` or
`engine.glm53flash_lookup_audit("generation", B, 1, P)` for the same native
selector used by prediction. The audit uses query length Q directly; the public
prefill prediction interface uses inclusive input length P+Q. Strict holdout
validation preserves per-point `prediction_evidence` and, across shards,
`prediction_evidence_origins`, while retaining failed points as error rows.
Neither a successful self-query nor an endpoint audit substitutes for the
independent accuracy gate.

Use the [collector contract](../python/aisimulate/collector/README.glm53flash.md), [vLLM serving/graph contract](../python/aisimulate/collector/README.glm53flash-vllm-full.md), [SGLang contract](../python/aisimulate/collector/README.glm53flash_sglang.md), [strict Ops validator](../python/aisimulate/collector/glm53flash_validation.py) and [independent forward validator](../python/aisimulate/collector/fpm_forward/glm53flash_validation.py). The [shared model contract](glm53flash-native-contract.md) defines the graph and precision boundaries. A reproducible campaign must include its frozen commands, point maps, runtime/container identities and original evidence; task-local qualification commands alone are not a published data package.
