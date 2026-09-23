# GLM-5.3-Flash native FPM adapter

This adapter is under GPU qualification. CPU lifecycle tests do not establish
hardware state correctness, graph replay, timing accuracy, or dataset acceptance.

The vLLM adapter calls the pinned Dynamo self-benchmark contract with real
same-request prefix histories. All seed forwards complete before suffix or
steady decode measurement. Every coordinate has five real warmups and ten
retained observations. Only their median enters the native rank artifact; all
observations, actual dispatch modes/padding, prompt/output token histories and
completed seed counts remain available to the strict reader. Synthetic prefix
and decode injection raise instead of falling back.

The timing boundary is `vllm_native_scheduler_output_interval`: the native
prefill schedule-to-output interval and second decode output interval. It is
not pure GPU elapsed time. Enable `--cudagraph-metrics` to return the actual
runtime dispatch, independently of the benchmark's expected capture metadata.

Fixed API sources:

- vLLM v0.30.0, commit `ced6857afa0ea7b2e3f0846a62e1394e90f15607`.
- Dynamo instrumentation, commit `54960177085413259859c88bd34ed0734d4c2ea9`.

The exact source-file hashes in `runtime-source-sha256.json` must match the
installed image/overlay. Configure an independently qualified Dynamo overlay
on PYTHONPATH; `runtime-paths.json` declares its default mounted location.
The source-checked lazy bootstrap activates only in scheduler processes.

Required environment: `DYN_FPM_GLM53FLASH_REAL_KV=1`, `DYN_FPM_INPUT_TEXT`, and
`DYN_FPM_TOKENIZER_REVISION` equal to the fixed FP8 or NVFP4 checkpoint revision.
The matching installed AISimulate wheel must provide the GLM model descriptor
and FPM execution identity. The collector stages the content-hashed corpus,
explicit point manifest and adapter. Native global warmup is set to zero because
the adapter performs its own five real warmups per exact geometry.

The formal baseline is TP2/TP4, DP=PP=CP=EP=1, full text inference with no
speculation, EPLB, offload, connectors or ubatching. Maximum context is 131072,
batch is at most 32, and scheduled prefill new-token total is at most 8192.
CUDA graph policy is native; eager-only campaigns require a separate identity.

## Attribution

`glm53flash_scheduler.py` is modified code derived from
[the Dynamo scheduler](https://github.com/ai-dynamo/dynamo/blob/54960177085413259859c88bd34ed0734d4c2ea9/components/src/dynamo/vllm/instrumented_scheduler.py),
using AISimulate's existing DeepSeek same-request adapter as its integration
precedent. Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
Apache-2.0; the upstream license is retained in `LICENSE`. No upstream root
NOTICE exists. The exact derived path is listed in root THIRD_PARTY_NOTICES.md
and its byte-identical packaged copy. vLLM implementation files are not vendored.
