<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Native FPM collection with separate prefill and decode workers

Use this reference during onboarding stages 2 and 4 to configure a real P/D
serving deployment, then stage 5 to check it before collection. Run your own
workload through your own deployment tools. The main
[end-to-end offline profiling guide](e2e-gbdt.md) covers selection, training and
Replay; [troubleshooting](native-pd-troubleshooting.md) covers the failure modes
encountered while preparing these examples.

Both roles must load the real model and exchange real KV state. A dummy decode
worker, fake prefill or replay of recorded transfer events does not qualify this
path. Each role needs capacity for its model, KV cache, graphs and workspace;
adding prefill and decode GPU memory together does not establish that either fits.

## Tested identities

The completed examples used eight B200 GPUs on one host: four disjoint GPUs per
role, TP4/EP4, one attention data-parallel group. The two backends ran separately
with different models. These are useful starting configurations, not a backend
performance comparison or a qualification of another topology. Apply the
[completion and native-label checks](native-pd-troubleshooting.md) to your own
run and keep prediction accuracy and Replay qualification separate.

| Identity | vLLM | SGLang |
| --- | --- | --- |
| Model | `deepseek-ai/DeepSeek-V4-Flash` | `deepseek-ai/DeepSeek-V4.1-Flash` |
| Model and tokenizer revision | `60d8d70770c6776ff598c94bb586a859a38244f1` | `fb2764a5cf321eaa5070ca8f9e892818f477c16d` |
| Image repository | `nvcr.io/nvidia/ai-dynamo/vllm-runtime` | `nvcr.io/nvidia/ai-dynamo/sglang-runtime` |
| Linux AMD64 digest | `sha256:99b42c3586442132f9873e356d1f411fc6045e60cd7d280671b88de28a819805` | `sha256:b9ddf09444b4c454605b68ab2dbd7e99b6bdbe5fa7a459935edaa78c61592549` |
| Installed engine | `0.26.0` | `0.0.0.dev1+ge087e662b` |
| Installed Dynamo | `1.4.0` | `1.5.0` |
| KV transfer | NIXL `1.3.1` | Mooncake `0.3.13.post1` |

An equivalent instrumented deployment can use another registry or installation
method. Record its actual package versions, producer source and model/tokenizer
revisions. Inspect the chosen image's worker help before translating the flags
below. The SGLang image tag is not its installed Dynamo version.

For the pinned SGLang producer, first materialize and verify the
[native prefill-counter correction](../../../scripts/fpm_self_service/native_patches/README.md#materialize-and-verify).
Mount the corrected module into **both** workers and verify the imported file's
path and hash there. Keep its provenance and license with the capture. The image
digest alone does not identify the corrected producer. A different SGLang version
needs its own source check; do not apply this patch blindly.

## Prepare your deployment and workload

Use an existing instrumented deployment, launch one with your infrastructure's
tools, or import completed native traces. Slurm, Kubernetes, SSH and access to a
particular registry are deployment choices, not collection requirements.

For the tested Dynamo stack, start discovery and messaging services before the
frontend and workers. The recorded configuration used etcd discovery, NATS,
TCP requests and ZMQ events; the images included etcd `3.5.33` and nats-server
`2.12.14`. Set `ETCD_ENDPOINTS`, `NATS_SERVER` and one unique `--namespace` consistently, with `--discovery-backend etcd --request-plane tcp
--event-plane zmq` on the frontend and workers. Adapt addresses for your network;
loopback addresses work only when every relevant process shares that network.

Before launching, reserve distinct role-specific system, FPM, HTTP, bootstrap,
NCCL and transfer ports as applicable. Assign disjoint `CUDA_VISIBLE_DEVICES`.
Set distinct `DYN_SYSTEM_PORT` and `DYN_FORWARDPASS_METRIC_PORT` in each
worker environment for **both backends**. Give each worker a writable persistent
output directory and a short private IPC
directory. Keep model and tokenizer assets accessible inside the runtime. Record
resolved engine settings and measured KV capacity separately for each role.

Your load generator needs to preserve the intended workload semantics: input and
output token budgets, cached prefixes, arrival or concurrency schedule, and any
session history and delays. Pin its version and dependencies, check every input
plus requested output against the target context limit, and save the seed and
preparation logs. Dataset preparation must finish before the timed run starts.
Choose sending and drain deadlines that allow the admitted requests to finish.
Changing the workload or reducing offered load is valid, but record the change
and keep the original request outcomes rather than silently dropping failures.

## Enable native capture on each worker

Set these values in each worker's **actual process environment**. Substitute its
role and a persistent path; `DYN_FPM_OUTPUT_PATH` is a filename prefix, not a
single finished file.

```bash
export FPM_ROLE=prefill  # use decode in the other worker
export DYN_FPM_TRACE=1
export DYN_FPM_MODE=full
export DYN_FPM_OUTPUT_PATH="/persistent/run/traces/$FPM_ROLE/fpm"
export DYN_FPM_MAX_SEGMENTS=64
export DYN_FPM_JSONL_GZ_ROLL_BYTES=134217728
```

The native sink adds producer and segment suffixes. Size retention for the whole
run, inspect rotation, and retain every segment in both role directories. The
values above were used in the tested runs; they do not guarantee enough space
for a longer or busier run. Do not enable `DYN_FPM_BENCHMARK_OUTPUT_PATH` for this
route. Record worker IDs rather than assigning one shared ID to both roles.

## vLLM worker settings

Launch each role with `python3 -m dynamo.vllm` using the shared deployment
settings above and the following engine arguments. These settings describe the
pinned DeepSeek run; reassess model-specific options for your model.

| Setting | Arguments in the tested deployment |
| --- | --- |
| Model and tokenizer | `--model /runtime/model --served-model-name deepseek-ai/DeepSeek-V4-Flash --trust-remote-code --tokenizer-mode deepseek_v4` |
| Parallelism | `--tensor-parallel-size 4 --pipeline-parallel-size 1 --data-parallel-size 1 --enable-expert-parallel` |
| Cache and memory | `--kv-cache-dtype fp8 --block-size 256 --enable-prefix-caching --gpu-memory-utilization 0.90` |
| Context and requests | `--max-model-len 262144 --max-num-seqs 256` |
| Native producer | `--scheduler-cls dynamo.vllm.instrumented_scheduler.InstrumentedScheduler` |
| Real P/D transfer | `--disaggregation-mode prefill` or `decode`, with `--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'` |
| Execution | `--distributed-executor-backend mp --distributed-timeout-seconds 1800 --no-enable-flashinfer-autotune --no-enable-log-requests` |
| Prefill role | `--max-num-batched-tokens 4096 --no-async-scheduling` |
| Decode role | `--max-num-batched-tokens 8192` |
| Resolved settings | `--dump-config-to /persistent/run/traces/ROLE/resolved-config.json` |

The run used `--compilation-config` with `mode: 0`, `cudagraph_mode: "FULL"`,
`max_cudagraph_capture_size: 4096`, and these `cudagraph_capture_sizes`: 1, 2, 4,
8; 16–256 by 8; 272–512 by 16; 576–1024 by 64; 1152–1536 by 128; and
1792–4096 by 256. Preserve the resolved graph policy with your capture; graph
memory and timing change when it changes. The prefill worker also enabled ZMQ
KV events with a unique local endpoint and topic `kv-events`.

Set unique `VLLM_PORT` and `VLLM_NIXL_SIDE_CHANNEL_PORT` for each role, in
addition to the shared system/FPM port settings. Set `VLLM_HOST_IP` and
`VLLM_NIXL_SIDE_CHANNEL_HOST` to addresses reachable by the peer. The tested
single-host run used loopback. `VLLM_RPC_BASE_PATH` and `TMPDIR` used private
short paths; account for the additional UUID socket name when checking the
107-byte Unix socket pathname limit.

## SGLang worker settings

Launch each role with `python3 -m dynamo.sglang`, the shared deployment settings
and the following engine arguments. Verify the producer correction before
accepting any prefill data.

| Setting | Arguments in the tested deployment |
| --- | --- |
| Model | `--model-path /runtime/model --served-model-name deepseek-ai/DeepSeek-V4.1-Flash --trust-remote-code` |
| Parallelism | `--tp-size 4 --ep-size 4 --dp-size 1` |
| Cache and memory | `--page-size 256 --mem-fraction-static 0.8` |
| Context and requests | `--context-length 262144 --max-running-requests 256` |
| Native producer | `--enable-metrics --disable-overlap-schedule` |
| Real P/D transfer | `--disaggregation-mode prefill` or `decode`, with `--disaggregation-transfer-backend mooncake` |
| Reachable bootstrap | `--host 0.0.0.0`, separate `--port`, `--nccl-port` and `--disaggregation-bootstrap-port` values per role |
| Prefill role | `--chunked-prefill-size 4096 --max-prefill-tokens 16384 --load-balance-method round_robin` |
| Decode role | `--prefill-round-robin-balance` (supported but deprecated in this pinned version) |
| Model parsing | `--reasoning-parser auto --dyn-reasoning-parser deepseek_v41 --dyn-tool-call-parser deepseek_v41 --dyn-enable-structural-tag` |
| Watchdog | `--watchdog-timeout 3600` |

Set `MC_TCP_MAX_QUEUED_TRANSFERS_PER_PEER=65535` in **both** workers and
`SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1800` in **decode** for this tested recipe.
These are workload-dependent operational choices, not native-counter fixes.
[Check their meaning and failure symptoms](native-pd-troubleshooting.md) before
changing them. Preserve the transfer backend and queue/deadline settings in the
capture identity.

Inspect the resolved settings: overlap scheduling must be disabled, and both
`enable_encoder_swa_bounded_replay` and `enable_decoder_swa_bounded_replay` must be
false. The selected native five-counter producer does not count encoder
bounded-replay work correctly. Keep mixed chunks, speculation, attention DP and
prefill CP disabled for this qualified recipe; a new mode needs separate
counter/timer qualification. Record the actual child-process environment,
including queue and waiting-timeout values; a planned command does not prove
that a worker received them.

## Check role and transfer behavior before the workload

1. Confirm the frontend advertises the expected model and both workers report
   ready with the intended GPU assignments, parallelism and imported producer.
2. Send two distinct short requests with `ignore_eos: true` and 16 requested
   output tokens. Require one finished choice, integer usage with exactly 16
   completion tokens, and consistent prompt and total counts. HTTP 200 alone
   does not establish successful generation.
3. Confirm native busy records from **both** expected worker IDs. Retain worker
   and transfer evidence that real prefill produced KV and real decode consumed
   it. If transfer success is inferred from successful generation and the pinned
   engine's completion gates, label it as an inference. The tested SGLang run did
   not expose a direct transfer-completion record. Role directories alone do not
   establish disaggregation.
4. Before a longer capture, exercise a cached prefix, a request spanning prefill
   chunks and concurrent requests from your workload. Check labels and counter
   conventions for these shapes; the completed short smoke does not qualify all
   of them. Keep smoke boundaries so these rows can be excluded from training.

Run the planned workload only after this check passes. Retain client outcomes,
measurement and drain boundaries, role logs, raw native envelopes and resolved
configuration. Then follow [completion and retention checks](native-pd-troubleshooting.md)
and the main guide's [data preparation](e2e-gbdt.md#2-check-and-prepare-the-data).
Keep prefill and decode identities separate through export and training.
