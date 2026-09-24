# GLM-5.3-Flash native FPM adapter

This adapter is under GPU qualification. CPU lifecycle tests do not establish
hardware state correctness, graph replay, timing accuracy, or dataset acceptance.

The vLLM adapter calls the pinned Dynamo self-benchmark contract with real
same-request prefix histories. All seed forwards complete before suffix or
steady decode measurement. Every coordinate has five real warmups and ten
retained observations. Only their median enters the native rank artifact; all
observations, actual dispatch modes/padding, prompt/output token histories and
completed seed counts remain available to the strict reader. Synthetic prefix
and decode injection raise instead of falling back. Completed repetition histories
are appended to the attempt's JSONL after its native timing intervals finish.
Partial raw evidence survives later failures; a fresh attempt refuses to overwrite
an existing history. The reader indexes offsets and validates one repetition at
a time, so full-campaign token arrays are not retained in host memory.

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

## Native IndexPool qualification restriction

Stock vLLM cached prefill with a prefix not divisible by four and at least two
new tokens is unqualified. GB300 split/one-shot probes observed wrong pooled
cache entries at P4097/Q3 (B1/B2) and P4097/Q4 (B1), while aligned controls
and every one-shot oracle passed. This is a conservative start-contract gate,
not a claim that every rejected geometry was independently probed. Q1 uses a
different native dispatch and retains its own qualification requirement.

The producer and reader explicitly reject affected coordinates; frozen requested
points remain in parent coverage and failure reports. A repair requires a new
runtime/source identity and independent native qualification before collection.
Memory feasibility, successful execution, and finite latency do not remove this
gate. See the Ops companion's `docs/glm53flash-kpool-native-gb300.json` for the
source-pinned probe receipt and snapshot digests.

## Attribution

`glm53flash_scheduler.py` is modified code derived from
[the Dynamo scheduler](https://github.com/ai-dynamo/dynamo/blob/54960177085413259859c88bd34ed0734d4c2ea9/components/src/dynamo/vllm/instrumented_scheduler.py),
using AISimulate's existing DeepSeek same-request adapter as its integration
precedent. Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
Apache-2.0; the upstream license is retained in `LICENSE`. No upstream root
NOTICE exists. The exact derived path is listed in root THIRD_PARTY_NOTICES.md
and its byte-identical packaged copy. vLLM implementation files are not vendored.

## Inclusive measured context boundary

The frozen measured limit remains at most 131072. For GLM, the renderer resolves
`--fpm-max-model-len=-1` to that measured limit, then starts native vLLM with seven
additional internal positions (131079 at the maximum). The producer requires
`DYN_FPM_GLM53FLASH_MEASURED_CONTEXT` to agree with the actual native configured
limit; if absent it uses 131072. This reserve allows the real seed and output
lifecycle to reach the measured decode at past-KV 131071 without native request
termination first. It does not enlarge the measured context or prove capacity.

New artifacts bind context-policy version 1, measured/runtime/headroom fields,
native `limits.max_model_len`, input provenance and the hybrid-state bound.
Readers require that policy. The sole legacy exception is producer overlay
`e391db177f53430c4280807fcc0eafdace5310cda6f6549ef7f2fb54e6cad984`, observed in both
prefill/decode short canaries on allocation 603053 with native limit 131072.
That exception preserves historical receipts and does not qualify exact 128K
execution. Frozen existing canaries are unchanged; new boundary probes require
new source and run receipts.
