<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Analytical AFD and EPD

AFD estimates each A/F pipeline pass analytically inside continuous request replay.
EPD supports an analytical encoder overlay or native event-level SGLang encoder replay;
see [Native encoder pools](#native-encoder-pools). AFD and EPD cannot be combined.

<a id="afd"></a>
<a id="analytical-afd"></a>

## AFD

AFD places attention (A) and FFN/MoE (F) work in separate pools. Set
`workers.<role>.type: afd` with an `afd` block in that role.
`engine.mode` independently selects shared P/D resources (`aggregated`) or
separate P/D resources (`disaggregated`):

```yaml
traffic:
  source:
    type: synthetic
    input_tokens: 1024
    output_tokens: 128
  load:
    type: concurrency
    concurrency: 16
  stop:
    requests: 32
engine:
  mode: aggregated
  model: Qwen/Qwen3-30B-A3B
  hardware: h200_sxm
  backend: sglang
  workers:
    aggregated:
      type: afd
      afd:
        n_a_nodes: 2
        n_f_nodes: 4
        tp_a: 1
        a_batch_size: 64
```

Worker type, context and startup fields are documented in [Workers](workers.md#worker-fields).
The AFD block has these fields:

| Knob | Default | Recommend | Rules |
|---|---|---|---|
| `engine.workers.<role>.afd.n_a_nodes`, `n_f_nodes`, `tp_a` | Required | Node counts within the GPU budget; A TP defaults to 1 | Positive; `tp_a` must divide GPUs per node and each active attention head group. A replica is one TP group. |
| `engine.workers.<role>.afd.a_system`, `f_system` | `engine.hardware` | fixed | Per-pool hardware. Node width and HBM capacity are resolved independently from each selected system. All pools share `engine.backend_version`. |
| `engine.workers.<role>.afd.a_batch_size` | Required | User-supplied domain | Positive batch size per A replica; sets scheduler capacity and the frozen measurement batch. |
| `engine.workers.<role>.afd.f_moe_ep_size` | F GPU count | Pure EP | Omitted values resolve to `n_f_nodes * f_gpus_per_node`. Explicit values are preserved and must give MoE TP1. Recommend accepts positive integers, `f_gpu_count`, or `n_f_nodes`. |
| `engine.workers.<role>.afd.num_microbatches` | `3` | `{choices: [2, 3, 4]}` | Positive, at most `a_batch_size`. Circulating microbatches for routed layers; the dense prefix runs once at the full A-replica batch. Instantaneous decode reduces this count only when an A replica has fewer sequences. |
| `engine.workers.<role>.afd.pipeline_layout` | `four_stage` | fixed | `four_stage`: independent A, A→F, F, F→A stages. `two_stage`: continuous A + A→F and F + F→A groups on two resources. |
| `engine.workers.<role>.afd.chunked_prefill_size` | unset | fixed | Positive chunk size; unset pre-fills the prompt in one pass. Explicit zero is rejected. |
| `engine.workers.<role>.afd.comm_overhead_factor` | `1.0` | fixed | Positive finite factor on A↔F transfers only; local A TP collectives and return-side restore are not scaled. Explicit zero is rejected. |
| `engine.workers.<role>.afd.routing_location` | `a` | fixed | Where MoE routing information is produced. `a` runs routing on A replicas before expert-directed A→F transfer. `f` parses and serializes, but execution reports not implemented. |
| `engine.workers.<role>.afd.max_af_ratio` | `4.0` | Recommend only | Bound on A/F node ratio. Complete candidate limits belong to `optimization.max_topology_candidates`. |
| `engine.attention_backend`, `engine.moe_kernel_source` | unset | fixed | Explicit kernel selectors passed to the per-pool op-level models. See [precision and execution controls](../../perf-model/configuration.md#flat-engine-precision-and-execution-controls). |

### Hardware and versions

A/F pools can select different systems, for example H200 attention (8 GPUs per
node) with GB200 experts (4 per node). Saved topology uses `gpus_per_node` for
A and `f_gpus_per_node` for F. Both widths and HBM capacities come from the
selected hardware specs; there are no public node-width overrides. GPU use is
`n_a_nodes * gpus_per_node + n_f_nodes * f_gpus_per_node`.
An ordinary P/D companion can independently override its `hardware`.

One `engine.backend_version` is shared by every active P/D and A/F pool.
Explicit versions must exist on all hardware. Omitted versions and
`current`/`previous`/`next` resolve per hardware and must agree; otherwise
specify a common literal version. The simulator does not choose an older common
version automatically. Recommendation checks every active role using its
configured data roots and saves the common literal version, so reload cannot
move with an alias.

### Deployment combinations

| `engine.mode` | AFD roles | Resource ownership |
| --- | --- | --- |
| `aggregated` | `aggregated` | One A/F pair serves Prefill and Decode. |
| `disaggregated` | `prefill` | Prefill A/F pair hands off to an ordinary Decode pool. |
| `disaggregated` | `decode` | Ordinary Prefill pool hands off to a Decode A/F pair. |
| `disaggregated` | `prefill`, `decode` | Independent Prefill and Decode A/F pairs, with a P/D handoff. |

Both P/D roles must be explicit. A/F resources are never shared between roles;
each pair is one logical replay worker with physical A/F GPU accounting.
AFD requires SGLang; ordinary workers retain their existing backend support.

`engine.kv_transfer` controls only P→D KV transfer. `bytes_per_token: auto` uses
per-request KV geometry, independent of A-pool batch capacity. Bandwidth and
`full_prompt`/`destination_missing` follow [ordinary P/D handoff](kv-transfer.md).
A↔F traffic is charged inside the AFD pass. Omitting `kv_transfer` retains the
default handoff behavior. For old configurations and saved artifacts, follow
[AFD migration](../../aic-backward-compatibility/migration.md#afd-translation).

### Memory and capacity

`a_batch_size` is the capacity of one A TP replica. Scheduler sequence capacity
is `a_replica_count * a_batch_size`; the A-pool per-GPU KV budget determines
available KV tokens across replicas. Both A and F must pass memory preflight
for each phase they serve; an infeasible pool fails with `out_of_memory`.
Memory fit and KV admission use the same non-KV footprint and selected data roots.

For synthetic traffic without an explicit context window, preflight reserves
`isl` for prefill and `isl + osl` for decode. The scheduler request limit is
`isl + osl`. An explicit `context_length`, or the model window resolved for
trace/agentic traffic, remains a full KV reservation. Aggregated AFD checks both
phases. See [worker context limits](workers.md#context-limits) for inheritance.

Transient prefill activations and staging buffers use a separate bound: known
input lengths are capped by the SGLang chunk, prefill-token and request limits;
trace/session inputs without a declared bound reserve that scheduler window.
The trace provider's placeholder `isl` is never used as its prefill memory bound.
Reducing `afd.chunked_prefill_size` reduces transient buffers while preserving
the complete KV reservation. Fit checks and KV admission share this bound.

### Timing and supported inputs

AFD uses continuous SGLang request replay, with analytical timing for each A/F
pass. Prefill uses the actual batch and uncached suffix. Fixed synthetic decode
uses frozen full-batch, stride-averaged timing; variable traffic uses the current
batch and KV position. See [pipeline and batch pricing](../../sweeper/afd-topology.md#staged-evaluation)
for dense-prefix execution, microbatch splitting and stage timelines.
Queueing contributes to TTFT and E2E; TPOT is `(E2E − TTFT) / (OSL − 1)`, or
zero for one-token outputs.

Supported: node-local attention TP × DP, pure routed F EP, fixed/variable
synthetic traffic, traces and cached prefixes. Source layers must be an optional
contiguous dense prefix followed by routed MoE layers. Dense layers after the
first routed layer (such as Llama 4 Maverick), pure dense models, GDN, Mamba and
unsupported attention/state layouts are rejected. Pure dense models use ordinary
deployments. Missing source composition or required performance data fails explicitly.

F-side routing (`routing_location: f`) parses but fails execution preflight
before data loading, without falling back to `a`. F MoE TP must be 1.
KV-relative `kv_capacity_fraction` load and EPD composition remain unsupported.
Defaults apply only to omitted fields; explicit zero is rejected where a positive
value is required.

`engine.comm_quant_mode` controls pool communication precision and the AFD
transfer/collective/combine plan. Omission keeps AFD communication at `half`
without filling the optional pool override; explicit precision requires matching
operator/data support. Kernel selectors and unsupported EPLB, redundant slots
and nondefault MoE backends are described in
[performance-model controls](../../perf-model/configuration.md#flat-engine-precision-and-execution-controls).
See [speculation](speculation.md#mtp--eagle) for MTP behavior.

### Output artifacts

Each successful AFD prediction writes `afd-replay-spec.json` and
`afd-qualification.json`, recording role-indexed model identity, topology,
provenance, GPU accounting and native A/F memory checks. Qualification links to
the replay spec by SHA-256 with deterministic JSON ordering. Contract
qualification, memory feasibility and measured serving accuracy are separate:
writing a contract without replay leaves memory validation pending.

Artifacts use `native_deployment_supported: false` and `launch.supported: false`;
there is no runnable AFD deployment renderer. For recommendation domains see
[search space](../../sweeper/search-space.md#attention-ffn-disaggregation).

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
| `engine.workers.encoder.hardware` | `engine.hardware` | fixed | Encoder system identifier; may differ from the language workers' `engine.hardware`. |
| `engine.workers.encoder.backend_version` | Resolved | fixed | Encoder performance data. The backend follows `engine.backend`. |
| `engine.workers.encoder.tensor` | `1` | Scalar or `choices` | Positive. |
| `engine.workers.encoder.replicas` | `1` | Scalar or `choices` | Positive. |
| `engine.workers.encoder.batch_size` | `1` for `analytical`, `8` for `native` | Scalar or `choices` | 1 to 8. |
| `engine.workers.encoder.latency_correction` | `1.0` | fixed | Positive multiplier on encoder latency. Analytical pools only; `mode: native` rejects any other value. |
| `engine.workers.encoder.rate_degradation` | `0.9` | fixed | In `(0, 1]`. Fraction of ideal encoder throughput that is usable. Unused by `mode: native`. |
| `engine.workers.encoder.mode` | `analytical` | fixed | `analytical` adds the encoder batch latency to mean TTFT. `native` replays SGLang `--encoder-only` servers event by event and gates each request's admission to the language worker; `batch_size` is then the loop's cap (`SGLANG_ENCODER_MAX_BATCH_SIZE`, default 8) and must be a scalar; encoder servers are modeled at `--dp-size 1`, and extra `replicas` approximate DP ranks only for single-image requests. SGLang only. See [Native encoder pools](#native-encoder-pools). |
| `engine.workers.encoder.host_profile` | Unset | fixed | `mode: native` only: `{path, frontend: python}`. The table's `process` stage prices the encoder's CPU preprocessing. Exclusive with `frontend`; one of the two is required. |
| `engine.workers.encoder.frontend` | Unset | fixed | `mode: native` only: the resolved stages with their `measured_for`, as a saved recommendation carries them in place of `host_profile`; the `process` stage prices the encoder's CPU preprocessing. Exclusive with `host_profile`. A `predict` of such a YAML reads no host table and refuses a workload other than `measured_for`. |
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
generated. Analytical encoder pools build Qwen3-VL and Kimi K3 vision towers;
native pools build Qwen3-VL towers only, refusing Kimi K3 at the image geometry
and Llama 4 at the encoder builder. Other families are rejected, while encoding
on the worker keeps its Llama 4 tile geometry. The encoder model follows
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

<!-- Heterogeneous AFD/fabric behavior adapted and modified from
https://github.com/LPhgh/aisimulate/tree/af56328c4ba7ffa1b8c42c8375244568c2e7bb8c
Original paths and Apache-2.0 attribution: root THIRD_PARTY_NOTICES.md. -->
