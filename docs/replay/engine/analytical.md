<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Analytical AFD and EPD

Attention/FFN disaggregation (AFD) and encoder/prefill/decode disaggregation
(EPD) are estimated analytically. Neither runs request-by-request token replay
for the disaggregated part. Their inputs and reports are therefore narrower
than ordinary aggregated or P/D replay. AFD and EPD cannot be combined. An
encoder pool with `mode: native` is the exception: it replays SGLang's encoder
servers event by event; see [Native encoder pools](#native-encoder-pools).

<a id="afd"></a>
<a id="analytical-afd"></a>

## AFD

AFD places attention (A) and FFN/MoE (F) work in separate pools. Set
`engine.mode: afd` and an `engine.afd` block:

```yaml
traffic:
  source: {type: synthetic, input_tokens: 1024, output_tokens: 128}
  load: {type: concurrency, concurrency: 16}
  stop: {requests: 32}
engine:
  mode: afd
  model: Qwen/Qwen3-30B-A3B
  hardware: h200_sxm
  backend: vllm
  afd:
    phase: both
    combined_with_pd: false
    n_a_nodes: 2
    n_f_nodes: 4
    tp_a: 4
    a_batch_size: 64
```

| Knob | Default | Recommend | Rules |
|---|---|---|---|
| `engine.afd.phase` | Required | fixed | `both` for pure AFD; `prefill` or `decode` with `combined_with_pd: true`. |
| `engine.afd.combined_with_pd` | Required | fixed | `false`: AFD serves both phases. `true`: AFD serves one phase and an ordinary worker serves the other. |
| `engine.afd.n_a_nodes`, `n_f_nodes`, `tp_a` | Required | Enumerated within the GPU budget | Positive. Recommend may constrain `tp_a`. |
| `engine.afd.a_batch_size` | Required | User-supplied domain | Positive attention-worker batch size. |
| `engine.afd.f_moe_ep_size` | `1` | Model-derived domain | Positive; recommend also accepts `n_f_nodes` or `ffn_tp`. Dense models require `1`. |
| `engine.afd.num_microbatches` | `3` | `{choices: [2, 3, 4]}` | Positive. |
| `engine.afd.pipeline_model` | `optimistic` | `{choices: [optimistic, conservative]}` | `optimistic`, `conservative` or `serial`. |
| `engine.afd.comm_overhead_factor` | `1.0` | fixed | Positive factor on A/F communication. |
| `engine.afd.boundary_on_attn` | `true` | fixed | A/F boundary convention. |
| `engine.afd.max_af_ratio`, `max_candidates` | `4.0`, `10000` | Recommend only | Bounds on the generated topology space. |

Workers:

- `combined_with_pd: false` requires `phase: both` and no other worker roles.
- `combined_with_pd: true` requires exactly the opposite-phase worker. For
  example, `phase: decode` requires `workers.prefill`.
- `engine.kv_transfer` is rejected; A↔F transfers come from `engine.afd`.

The pass time is built from measured per-layer A, F, A→F and F→A times:

| `pipeline_model` | Per-microbatch-layer cadence |
| --- | --- |
| `optimistic` | `max(A, F, A→F + F→A)` |
| `conservative` | `max(A + A→F, F + F→A)` |
| `serial` | `A + A→F + F + F→A` |

A pass adds pipeline fill to the cadence of every microbatch and layer. Queueing
contributes to TTFT and E2E latency. TPOT is
`(E2E − TTFT) / (OSL − 1)`, or zero for one-token outputs.

Requirements: fixed synthetic input and output lengths and no trace.
`kv_capacity_fraction` load is rejected because the A/F pools have no
scheduler-visible KV capacity.

Each prediction writes `afd-replay-spec.json` and `afd-qualification.json`
with the model, topology and GPU accounting. They are marked
`native_deployment_supported: false`; no runnable backend deployment is
generated. See [search space](../../sweeper/search-space.md) for recommendation
rules.

<a id="epd"></a>
<a id="analytical-epd"></a>

## EPD

EPD adds an encoder pool for image inputs in front of ordinary language
replay. Add `engine.workers.encoder` and give the synthetic source fixed images,
as in [`examples/cli/epd-predict-aggregated.yaml`](../../../examples/cli/epd-predict-aggregated.yaml):

```yaml
traffic:
  source:
    type: synthetic
    input_tokens: 128
    output_tokens: 32
    images: {height: 448, width: 448, count: 1}
  load: {type: concurrency, concurrency: 8}
  stop: {requests: 16}
engine:
  model: Qwen/Qwen3-VL-8B-Instruct
  hardware: h200_sxm
  backend: sglang
  backend_version: 0.5.14
  mode: aggregated
  workers:
    encoder: {tensor: 1, replicas: 1, batch_size: 2}
    aggregated:
      parallelism: {tensor: 1, replicas: 1}
      scheduler: {max_batched_tokens: 8192, max_sequences: 8}
```

```bash
aisimulate predict --stack engine --config examples/cli/epd-predict-aggregated.yaml \
  --output-dir ./epd-prediction
```

A disaggregated variant is in
[`examples/cli/epd-predict-disaggregated.yaml`](../../../examples/cli/epd-predict-disaggregated.yaml).

| Knob | Default | Recommend | Rules |
|---|---|---|---|
| `engine.workers.encoder.hardware` | `engine.hardware` | fixed | Encoder system identifier. |
| `engine.workers.encoder.backend_version` | Resolved | fixed | Encoder performance data. The backend follows `engine.backend`. |
| `engine.workers.encoder.tensor` | `1` | Scalar or `choices` | Positive. |
| `engine.workers.encoder.replicas` | `1` | Scalar or `choices` | Positive. |
| `engine.workers.encoder.batch_size` | `1` | Scalar or `choices` | 1 to 8. |
| `engine.workers.encoder.latency_correction` | `1.0` | fixed | Positive multiplier on encoder latency. |
| `engine.workers.encoder.rate_degradation` | `0.9` | fixed | In `(0, 1]`. Fraction of ideal encoder throughput that is usable. Unused by `mode: native`. |
| `engine.workers.encoder.mode` | `analytical` | fixed | `analytical` adds the encoder batch latency to mean TTFT. `native` replays SGLang `--encoder-only` servers event by event and gates each request's admission to the language worker; `batch_size` is then the loop's cap (`SGLANG_ENCODER_MAX_BATCH_SIZE`, default 8) and must be a scalar. SGLang only. See [Native encoder pools](#native-encoder-pools). |
| `engine.workers.encoder.host_profile` | Unset | fixed | `mode: native` only, required: `{path, frontend: python}`. The table's `process` stage prices the encoder's CPU preprocessing. |
| `engine.workers.encoder.transfer.bandwidth_gb_per_second` | Unset | fixed | `mode: native` only, required. Encoder-to-language link; embeddings fan out to every tensor-parallel rank. |

Visual tokens are computed from the image size and added to each request's
text input once. Language replay then runs as usual. Afterwards, throughput is
capped at the degraded encoder capacity, and the raw encoder batch latency is
added to mean TTFT and E2E. Encoder queueing does not feed back into the
language schedule.

Results are marked `analytical_epd_overlay` and report means only, without
percentiles, per-request records or telemetry. GPU-hours include encoder GPUs.

Requirements: fixed synthetic text and image inputs, fixed `concurrency` load,
static workers, and default op-level language timing. Rejected: traces,
sessions, variable lengths, prefix sharing, rate or KV-relative load,
whole-forward (FPM) timing, adapters, and `--capture-per-request`. Encoder CPU
time and embedding transfer are not modeled, and no deployment artifacts are
generated. The encoder model follows
[AIC revision f8f2341](https://github.com/ai-dynamo/aiconfigurator/commit/f8f2341cb5761877bda694ab954cb6f5eff78fd4).

<a id="native-encoder-pools"></a>

### Native encoder pools

`mode: native` replaces the overlay with an event-level replay of SGLang's
encoder servers, as in
[`examples/cli/vl-predict-epd-native.yaml`](../../../examples/cli/vl-predict-epd-native.yaml).
Each replica runs one serial loop: it takes the queued requests up to
`batch_size`, prices the image processor from the host table's `process`
stage, one encoder forward over the batch with the canonical timing model, and
the embedding transfer at `transfer.bandwidth_gb_per_second`. A request reaches
the language worker once its last part arrived, and its TTFT counts the wait.
The language worker runs `--language-only`: it rejects `vision`, `frontend`
and `host_profile` but may run `host_loop`.

Native pools lift the fixed-concurrency, aggregate-SLA, per-request, `op_level`
and pixel-budget restrictions above; static worker pools and an aggregated or
P/D SGLang language deployment remain required. The pool's GPUs count in
`gpu_hours`, `encoder_gpus` and `total_gpus`; the report adds
`encoder_latency_ms` and, per request, `encoder_ready_ms`. Mechanics and
approximations are on
[SGLang VL host loop](sglang-vl-host-loop.md#native-encoder-disaggregation).
