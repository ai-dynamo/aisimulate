# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
---
title: Migrate from AIConfigurator
subtitle: Translate AIConfigurator CLI commands into AISimulate CLI commands
---

> [!WARNING]
> **Experimental.** The AISimulate recommendation schema and search behavior may change without a
> standard deprecation period.

This guide compares the compatibility `aiconfigurator` CLI with the public AISimulate CLI introduced
by the unified command surface. A legacy command becomes an AISimulate recommendation YAML plus one
`aisimulate recommend --config ...` command.

## Common input mapping

| AIConfigurator CLI | AISimulate recommendation | Notes |
|---|---|---|
| `--model-path` | `engine.model` | Same model identifier |
| `--system` | `engine.hardware` | Same hardware identifier |
| `--backend` | `engine.backend` | One backend or an explicit recommendation domain |
| `--total-gpus` | `optimization.constraints.max_candidate_gpus` | Maximum GPUs per candidate |
| `--isl` | `traffic.source.input_tokens` | Synthetic input length |
| `--osl` | `traffic.source.output_tokens` | Synthetic output length |
| `--ttft` | `evaluation.sla.ttft_ms` | Time-to-first-token bound in milliseconds |
| `--tpot` | `evaluation.sla.itl_ms` | Per-request goodput uses average ITL; strict mode compares aggregate mean TPOT |
| `--strict-sla` | `optimization.strict_sla: true` | Reject before scalar ranking or Pareto dominance |

## Engine and request controls

The additional controls below remain flat fields under `engine`. They are concrete, pinned inputs;
adding them does not create new recommendation dimensions or enlarge the search space.

| AIConfigurator input | AISimulate recommendation | Notes |
|---|---|---|
| `--prefix` | `traffic.source.cached_prefix_tokens` | Exact cached tokens shared by synthetic requests; must not exceed the input length |
| `--max-seq-len` | `engine.context_length` | Search capacity limit; AISimulate also accepts `max`, and the engine runner enforces it directly for vLLM |
| `--nextn` | `engine.nextn` | Concrete MTP draft depth from 0 through 5 |
| `--nextn-accepted` | `engine.nextn_accepted` | Required when `nextn > 0`; measured average accepted draft tokens in `[0, nextn]` |
| `--enable-chunked-prefill` | `engine.enable_chunked_prefill: true` | Enables chunking on aggregated and prefill workers; decode retains its backend default |
| `--free-gpu-memory-fraction` | `engine.free_gpu_memory_fraction` | Shared override for every active worker role |
| `--enable-wideep` / `Task.enable_wideep` | `engine.enable_wideep: true` | Requires an MoE model; the legacy CLI flag is deprecated even though the Task field remains available |
| `Task.enable_eplb` | `engine.enable_eplb: true` | Requires an MoE model |
| `Task.wideep_num_slots` | `engine.wideep_num_slots` | Positive EPLB slot count |
| `--moe-backend` / `Task.moe_backend` | `engine.moe_backend` | `deepep_moe` or `megamoe`; SGLang only |
| `Task.attention_backend` | `engine.attention_backend` | `flashinfer` or `fa3`; SGLang MLA models only |
| `--gemm-quant-mode` | `engine.gemm_quant_mode` | Same AIConfigurator enum name |
| `--moe-quant-mode` | `engine.moe_quant_mode` | Same AIConfigurator enum name; requires an MoE model |
| `--kvcache-quant-mode` | `engine.kvcache_quant_mode` | Same AIConfigurator enum name |
| `--fmha-quant-mode` | `engine.fmha_quant_mode` | Same AIConfigurator enum name |
| `--comm-quant-mode` | `engine.comm_quant_mode` | Same AIConfigurator enum name |

`--nextn auto` does not transfer as the string `auto`. Resolve it with AIConfigurator first, then
copy the resulting integer depth and the measured `nextn_accepted` value. AISimulate does not infer
an acceptance assumption.

### Example with the new controls

Legacy AIConfigurator command:

```bash
aiconfigurator cli default \
  --model-path deepseek-ai/DeepSeek-R1 \
  --system h200_sxm \
  --backend sglang \
  --total-gpus 8 \
  --isl 4096 \
  --osl 1024 \
  --prefix 1024 \
  --max-seq-len 8192 \
  --nextn 3 \
  --nextn-accepted 1.5 \
  --enable-chunked-prefill \
  --free-gpu-memory-fraction 0.85 \
  --moe-backend deepep_moe \
  --gemm-quant-mode fp8 \
  --moe-quant-mode fp8 \
  --kvcache-quant-mode fp8 \
  --fmha-quant-mode fp8 \
  --comm-quant-mode fp8
```

The equivalent pinned controls in `recommendation.yaml` are:

```yaml
traffic:
  source:
    type: synthetic
    input_tokens: 4096
    output_tokens: 1024
    cached_prefix_tokens: 1024
  load:
    type: constant_rate
    requests_per_second: 4
  stop:
    requests_per_load_unit: 10

engine:
  mode: aggregated
  model: deepseek-ai/DeepSeek-R1
  hardware: h200_sxm
  backend: sglang
  context_length: 8192
  nextn: 3
  nextn_accepted: 1.5
  enable_chunked_prefill: true
  free_gpu_memory_fraction: 0.85
  moe_backend: deepep_moe
  gemm_quant_mode: fp8
  moe_quant_mode: fp8
  kvcache_quant_mode: fp8
  fmha_quant_mode: fp8
  comm_quant_mode: fp8
  workers:
    aggregated: {}

evaluation: {}

optimization:
  target: throughput
  constraints:
    max_candidate_gpus: 8
```

Run it with:

```bash
aisimulate recommend --config recommendation.yaml
```

The traffic rate and stop condition are explicit AISimulate choices; the legacy capacity sweep did
not define an equivalent offered load. For disaggregated configurations, the shared engine controls
apply to both roles, except decode retains its backend chunking default. Keep using
`workers.prefill.kv_cache.capacity.memory_fraction` and
`workers.decode.kv_cache.capacity.memory_fraction` when the two roles need different memory limits.

## Illustrative strict SLA translation

The examples below demonstrate how the strict-SLA fields map, but they are not behaviorally
equivalent workloads. The legacy `cli default` command capacity-sweeps aggregated and
disaggregated configurations under the eight-GPU budget without a fixed offered load. The
AISimulate example makes the additional choices of aggregated mode and a constant 4 RPS so its
traffic and replay behavior are explicit.

Legacy command:

```bash
aiconfigurator cli default \
  --model-path meta-llama/Meta-Llama-3.1-8B \
  --system gb200 \
  --backend trtllm \
  --total-gpus 8 \
  --isl 1024 \
  --osl 128 \
  --ttft 800 \
  --tpot 30 \
  --strict-sla
```

Save this AISimulate configuration as `recommendation.yaml`:

```yaml
traffic:
  source:
    type: synthetic
    input_tokens: 1024
    output_tokens: 128
  load:
    type: constant_rate
    requests_per_second: 4
  stop:
    requests_per_load_unit: 10

engine:
  mode: aggregated
  model: meta-llama/Meta-Llama-3.1-8B
  hardware: gb200
  backend: trtllm
  workers:
    aggregated: {}

evaluation:
  sla:
    ttft_ms: 800
    itl_ms: 30

optimization:
  target: throughput
  strict_sla: true
  constraints:
    max_candidate_gpus: 8
```

Run the illustrative AISimulate command:

```bash
aisimulate recommend --config recommendation.yaml
```

Without `optimization.strict_sla`, the configured thresholds classify individual requests for
goodput; a slow request contributes no tokens to goodput, but does not reject the whole candidate.
With `strict_sla: true`, AISimulate additionally compares every configured bound with the
candidate's aggregate mean metrics and removes a violation before ranking or Pareto analysis.
Comparisons are inclusive, and missing/non-finite aggregate metrics or zero qualifying latency
samples reject the candidate.

Request-level goodput and strict filtering both accept `ttft_ms` or `itl_ms` independently; an unset
field is unbounded. `strict_sla` changes only whether the configured bounds additionally reject a
candidate based on its aggregate means.

## Migration limits

- `--stack engine` is the default. Use `--stack dynamo` only when the independently installed Dynamo
  package provides the required runner and configuration adapters.
- Keep using the compatibility CLI for legacy flags that are not listed in this guide until their
  AISimulate mapping is implemented and qualified.
