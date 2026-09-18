<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Speculative decoding cost models

Speculation adds draft operations and widens target verification. The performance model prices each iteration; the caller supplies the average number of accepted draft tokens to project that cost into service metrics. Acceptance is an input assumption, not a prediction from the cost model.

This package supports `ngram`, `eagle3`, `dflash`, `dspark`, `draft_model`, and `mtp`. Draft schemes require their checkpoint configuration; n-gram drafting has no draft network, weights, or KV cache. `dense_draft.py` contains shared draft geometry rather than a separate registered scheme.

## Estimate command

The compatibility command shipped in the `aisimulate` wheel accepts `--spec-method`, `--spec-num-draft-tokens`, `--spec-accepted-tokens`, and, for a separate checkpoint, `--spec-draft-model-path`:

```bash
aiconfigurator cli estimate \
  --model-path Qwen/Qwen3-8B \
  --system h100_sxm --backend vllm --backend-version 0.24.0 \
  --estimate-mode static_gen --isl 64 --osl 128 --batch-size 8 \
  --gemm-quant-mode bfloat16 --kvcache-quant-mode bfloat16 \
  --fmha-quant-mode bfloat16 \
  --spec-method ngram --spec-num-draft-tokens 3 \
  --spec-accepted-tokens 1.5
```

Here `1.5` is an illustrative acceptance assumption. The target verifies four tokens per request and the expected output progress is 2.5 tokens per iteration. N-gram requires `trigger_rate=1.0`; mixed drafted and draft-less rounds are rejected until weighted round costs are supported.

Non-MTP schemes support `agg`, `static`, `static_ctx`, and `static_gen` estimates. MTP retains the existing `nextn`/`nextn_accepted` behavior, including disaggregated estimates. The unified `aisimulate predict` and `aisimulate recommend` commands do not acquire speculative configuration through this migration.

## Python and task configuration

```python
from aisimulate.legacy_cli.api import cli_estimate

result = cli_estimate(
    mode="static_gen",
    model_path="Qwen/Qwen3-8B",
    system_name="h100_sxm",
    backend_name="vllm",
    backend_version="0.24.0",
    isl=64,
    osl=128,
    batch_size=8,
    gemm_quant_mode="bfloat16",
    kvcache_quant_mode="bfloat16",
    fmha_quant_mode="bfloat16",
    speculative={
        "method": "ngram",
        "params": {"num_speculative_tokens": 3},
        "accepted_tokens": 1.5,
    },
)
```

The same `speculative` mapping can be supplied to an aggregate `Task` or its experiment YAML. Non-MTP schemes are rejected for disaggregated, AFD, and EPD tasks. `accepted_tokens` must be finite and between zero and the configured draft count. Explicit MTP uses `params: {depth: N}` and is equivalent to `nextn=N` with the same acceptance assumption. Do not combine an active legacy MTP configuration with a speculative block.

Core SDK callers configure `ModelConfig(speculation=SpeculationConfig(...))`. N-gram, EAGLE-3, and standalone-draft schemes use `num_speculative_tokens`; DFlash and DSpark use `num_draft_tokens`; MTP uses `depth`. The model exposes the resulting scheme and verification width. Model construction snapshots the speculation inputs; changing the caller's configuration requires building a new model. Core timing APIs return iteration cost; applying accepted-token progress is the upper layer's responsibility.

Standalone drafts support registry model families, including MoE, subject to their existing backend and performance-data coverage requirements. Draft TP defaults to target TP and can be set with `params.draft_tp_size`; expert TP follows draft TP, while the existing configuration resolver derives expert parallelism. Each draft step repeats the native generation graph, including composite operations, while reusing the same weights and prefill graph.

## Whole-forward FPM

With `forward_model="fpm"`, target verification uses a whole-model FPM operation and draft operations remain explicit. At concurrency `c` and verification width `w`, the target query uses `c * w` tokens while retaining the total KV of `c` requests. Existing FPM model, system, backend, worker-role, and coverage requirements still apply. Plain MTP remains unsupported with FPM because the collected target curves do not include its draft-head cost.

The operation schema changes to version 18; rebuild the native extension and recompile cached engine plans. Existing autoregressive defaults remain width one.

## Modeling limits

- Schemes that require draft query overrides are rejected before materialization. This includes DeepSeek-V4 DSpark with its sliding-window KV cap; dense DSpark without overrides remains supported.
- Host-side proposal lookup, sampling, and framework overhead are outside the operation graph.
- FPM verification maps onto autoregressive collection rows; it is not a measured wide-verification surface.
- Aggregate scheduling retains the source mean-field approximation. The upstream source documents errors at workloads below one full prefill per round.
- In op-level mixed estimates, draft prefill uses the same full-prefill amortization as target context attention, including draft compute operations. Its cost depends only on the prefill workload. Draft generation runs only for decode requests and uses the draft's own query width.
- Model-registry support does not guarantee backend/system performance coverage: unsupported operation graphs and missing required data fail explicitly, and composite overlap operations retain their existing refusal of a complete SOL decomposition.
- The attention model uses physical bounds without fitted calibration. Source accuracy numbers are historical measurements, not validation of this migration or a guarantee for another model or workload.

## Source

Adapted from [AIConfigurator PR #1563](https://github.com/ai-dynamo/aiconfigurator/pull/1563), pinned at [6290c161a354da5250c391bd43372b2e9c6f4a51](https://github.com/ai-dynamo/aiconfigurator/tree/6290c161a354da5250c391bd43372b2e9c6f4a51/aic-core/src/aiconfigurator_core/sdk/speculation). Original path: `aic-core/src/aisimulate_core/sdk/speculation/`. Copyright NVIDIA CORPORATION & AFFILIATES; Apache-2.0. AISimulate adaptations preserve current engine contracts and strengthen invalid-input handling. See the repository migration ledger for the complete path mapping.
