<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Sweeper Python SDK

Use the public `Sweeper` execution interface with an explicit runner factory.
`SmartSearchConfig` is the SDK schema. Do not pass its YAML directly to
`aisimulate recommend`, which accepts the [public CLI schema](../reference/configuration.md).

## Execute a built-in engine sweep

The checked-in EPD example is one complete SDK input; it evaluates the existing
analytical encoder integration, with the [limits](../replay/engine/analytical.md)
described by Replay. Run from the repository root:

```python
from aisimulate.runner import EngineReplayRunnerFactory
from aisimulate.sweeper import SmartSearchConfig, Sweeper

if __name__ == "__main__":
    config = SmartSearchConfig.from_yaml("examples/sweeper/epd.yaml")
    result = Sweeper(runner_factory=EngineReplayRunnerFactory()).run(config)
    for candidate_id, candidate in zip(result.selected_candidate_ids, result.selected_candidates, strict=True):
        print(candidate_id, candidate.score, candidate.used_gpus)
```

`run()` returns a `SweepResult`, not a list. `selected_candidate_ids` contains
the stable ledger IDs; `selected_candidates` contains `Candidate` objects in
the same order. The complete records remain available through `result.candidates`.
Worker factories and provider arguments
must be serializable when processes are used. Each run creates fresh search
state and workers. Direct `Sweeper` calls have bounded pool cleanup; host-memory
admission and RSS supervision belong to the public CLI/`run_recommendation`
path described in [local resources](../reference/local-resources.md).

## Top-Level Shape

```yaml
search_space:
  model_name: example/model
  hardware_sku: h200_sxm
  prefill_hardware_sku: h200_sxm
  decode_hardware_sku: gb200
  gpu_budget: 32
  deployment_mode: [disagg, agg]
  backend: [vllm, sglang]
  backend_version:
    vllm: current
    sglang: current
  database_mode: HYBRID
  transfer_policy: balanced
  estimation_mode: auto
  fallback_policy: deny
  estimator_config:
    fpm_regression:
      min_observations: 5
  systems_paths: [default]

workload:
  isl: 1024
  osl: 128
  request_rate: 4
  num_request_ratio: 10

goal:
  target: throughput_per_gpu
  # Optional legacy-compatible aggregate gating. Replay goodput remains
  # per-request; strict_sla filters aggregate means before ranking.
  strict_sla: true
  sla:
    ttft_ms: 800
    itl_ms: 30

sweep:
  max_rounds: 10
  candidates_per_round: 8
  parallel_evals: 4
```


Optional `adapters` maps installed provider names to provider-owned search
spaces. Public config adapters and SDK providers are distinct contracts; see
[configuration ABI](../adapters/configuration-abi.md).

## Backend Fields

| Field | Default | Purpose |
|---|---|---|
| `model_name` | required | model identifier |
| `hardware_sku` | required | AI Configurator system identifier |
| `prefill_hardware_sku` | `None` | optional disaggregated-prefill system override; inherits `hardware_sku` |
| `decode_hardware_sku` | `None` | optional disaggregated-decode system override; inherits `hardware_sku` |
| `deployment_mode` | `[disagg, agg]` | deployment branches to search |
| `backend` | `[vllm]` | engine backends to search |
| `backend_version` | `None` | exact version for one backend, or a per-backend version mapping; omitted backends resolve once to latest |
| `database_mode` | `SILICON` | forward-pass estimator data-source policy; see [performance-model configuration](../perf-model/configuration.md) |
| `transfer_policy` | `None` (all) | Core-owned empirical-transfer preset or tier list; used only by `HYBRID` and `EMPIRICAL` |
| `estimation_mode` | `auto` | search op-level, FPM interpolation, then regression; or select an explicit mode |
| `fallback_policy` | `deny` | constrain explicit estimator selection; auto always searches the full priority order |
| `estimator_config` | `{}` (Core defaults) | runtime tuning controls such as observation limits, regression buckets, correction bounds, and workload-axis capacity |
| `systems_paths` | omitted | preserve configured SDK/environment discovery; an explicit list sets ordered request-scoped roots, with `default` selecting the packaged Core root |
| `gpu_budget` | `32` | maximum GPUs per candidate |
| `min_gpu_budget` | `None` | optional lower bound during enumeration |
| `context_length` | `None` | optional KV-feasibility and runtime prompt-plus-output token limit |
| `prefill_context_length` | `None` | prefill role limit; overrides `context_length` |
| `decode_context_length` | `None` | decode role limit; overrides `context_length` |
| `parallel_configs` | `[]` | optional pinned parallel configurations |
| `startup_time` | `None` | optional simulated worker startup time |
| `aic_nextn` | `None` | optional speculative-decoding depth |

An explicit positive `context_length` is passed as AISimulate's internal
`max_model_len` for vLLM, TRT-LLM, and SGLang in both aggregated and
prefill/decode deployments. Prompts at or above the limit are rejected, and
generation stops when prompt plus output reaches the limit. A role-specific
`prefill_context_length` or `decode_context_length` overrides the shared limit
for that worker. When neither is supplied, Sweeper leaves its runtime limit
unset. See [worker context limits](../replay/engine/workers.md#context-limits)
for fallback and validation rules.

Each engine role also has lists for `max_num_batched_tokens` and `max_num_seqs`, plus pinned block
size, GPU-memory-utilization, prefix-caching, and `<role>_forward_model` fields (`op_level` by default,
or `fpm` for whole-forward timing from a collected FPM cell). Set `<role>_fpm_parquet_path` to the
external parquet for that role (`agg`, `prefill`, or `decode`); the adjacent same-stem
`.metadata.json` sidecar is required. The path requires default timing with `forward_model: fpm`
and is preserved in deployment metadata, runtime arguments, and candidate YAML.
A one-item list pins a searched field.

`prefill_hardware_sku` and `decode_hardware_sku` apply only to the ordinary `disagg` branch. Either
override may be set independently: an omitted role inherits `hardware_sku`. Both roles still share
the configured model, backend, backend version, and total `gpu_budget`. When `backend_version` is
omitted, the latest performance-data version for both effective SKUs must match; otherwise pin one
version supported by both systems. These overrides are part of the Sweeper YAML/SDK contract; the
public CLI exposes concrete per-role `engine.workers.prefill.hardware` and
`engine.workers.decode.hardware` overrides; those SKUs are not search dimensions.

The current Dynamo Router adapter uses the shared `hardware_sku` for
`prefill_load_model.type: aic`. Sweeper rejects a candidate before replay when its materialized
Router AIC system differs from the effective prefill SKU. This also affects matching overrides:
`hardware_sku: h200_sxm` with both role SKUs set to `gb200` needs a GB200 prefill load model.
Use a Router provider that consumes `prefill_hardware_sku`, or select a non-AIC load model.
Correctly materialized AIC hooks, decode-only overrides, and shared-SKU behavior remain supported.

## Pinned engine and request controls


Engine controls are pinned for a study; they do not add optimizer dimensions.
`SearchSpace` accepts `enable_eplb`, `wideep_num_slots`, `moe_backend`,
`attention_backend`, `gemm_quant_mode`, `moe_quant_mode`, `kvcache_quant_mode`,
`fmha_quant_mode`, and `comm_quant_mode`. They travel in the canonical
`ForwardPassPerfModelConfig` through exact candidate construction, saved
prediction YAML, and native replay. Quantization overrides also participate in
KV feasibility and capacity-cache identity. EPLB, slots, and MoE backend
selection require an MoE model; nondefault MoE backends require SGLang.
Collected FPM interpolation rejects EPLB, slot, and MoE-backend overrides that
its cells cannot represent. Custom timing, AFD, and analytical encoder runs
reject these model controls.

`aic_nextn` is the compute-side MTP draft depth (0–5); zero disables MTP.
For a positive depth, set `nextn_accepted`
explicitly to the expected number of accepted draft tokens, between zero and
that depth. Replay realizes a fractional expected count with guaranteed whole
tokens followed by one Bernoulli token; it does not estimate model acceptance.
`enable_chunked_prefill` is optional and applies to aggregated/prefill roles;
omission preserves the backend default.

`Workload.cached_prefix_tokens` is an exact shared prefix for synthetic traffic.
It must fit the shortest generated input, including a configured random-length
range. It creates shared tokens and does not prewarm the KV cache: the first
request is cold, and subsequent reuse follows backend cache and block rules.
Positive cached prefixes are unsupported for AFD and AFD+PD.

The model controls above and synthetic cached prefixes are supported by
`--stack engine`. Optional runners, including Dynamo, must explicitly advertise
these capabilities before accepting them; older adapters are rejected before
replay. Their downstream schemas and synthetic-input bindings require separate
qualification.

Use `context_length` for the sequence limit and each role's existing memory
fraction controls for KV sizing. The deprecated `enable_wideep` switch is not
exposed; current topology selects the MoE execution regime.

## Workload fields

These fields belong to SDK `Workload`; public CLI traffic fields are documented
in [Replay workloads](../replay/workloads.md). Trace timestamps drive open-loop
replay unless `replay_concurrency` is set; that cap uses closed-loop replay and
ignores trace timestamps. Synthetic traffic requires exactly one of
`request_rate`, `concurrency`, or `kv_load_ratio`.


Every `Workload` field:

| Field | Type | Default | Meaning |
|---|---|---|---|
| `isl` | `int \| None` | `None` | Synthetic input (prompt) sequence length, tokens. Required for synthetic. |
| `osl` | `int \| None` | `None` | Synthetic output sequence length, tokens. Required for synthetic. |
| `concurrency` | `int \| None` | `None` | Fixed positive closed-loop in-flight cap. It is always scalar, including under a Pareto goal. |
| `kv_load_ratio` | `float \| list[float] \| None` | Pareto default: `[0.0, 1.0]` when no other load is set | Candidate-relative closed-loop load. A scalar pins the load for any goal; a two-value `[min, max]` range is a continuous Vizier dimension and is allowed only under a Pareto goal. Values are non-negative; a user may set a maximum above `1` to search oversubscription. |
| `request_rate` | `float \| None` | `None` | Open-loop QPS. Mutually exclusive with `concurrency` and `kv_load_ratio`. |
| `num_request_ratio` | `float \| None` | `None` | Synthetic request count relative to the load: `num_requests = round(num_request_ratio * load)`. Required for synthetic; see [candidate-relative load](search-space.md#num_request_ratio-synthetic-length-scales-with-the-load). |
| `random_range_ratio` | `float` | `1.0` | Uniformly sample each synthetic ISL and OSL from `[int(ratio * configured_length), configured_length]`. Must be in `(0.0, 1.0]`; `1.0` preserves fixed lengths. Single-turn only. |
| `random_seed` | `int` | `0` | Unsigned 64-bit seed for deterministic synthetic ISL/OSL sampling. |
| `shared_prefix_ratio` | `float` | `0.0` | Fraction of shared prefix across requests (cache-locality / prefix sharing). |
| `num_prefix_groups` | `int` | `0` | Number of distinct shared-prefix groups. |
| `turns_per_session` | `int` | `1` | Turns per multi-turn session. |
| `inter_turn_delay_ms` | `float` | `0.0` | Think-time between turns in a multi-turn synthetic session, ms. |
| `source_type` | `str \| None` | `None` | Explicit traffic source kind passed to replay; must be `"trace"` when `agentic_lanes` or `agentic_snapshot` is set. |
| `load_type` | `str \| None` | `None` | Explicit traffic load kind passed to replay; must be `"trace_timestamps"` when `agentic_lanes` or `agentic_snapshot` is set. |
| `trace_path` | `str \| None` | `None` | Path to a replay trace. Its presence selects the trace shape and **forbids** all synthetic fields. |
| `trace_format` | `str` | `"mooncake"` | Replay-ready trace schema. A runner may validate supported formats. |
| `arrival_speedup_ratio` | `float` | `1.0` | Scales the trace's inter-arrival times (open-loop trace only). `>1` speeds arrivals up. |
| `agentic_lanes` | `int \| None` | `None` | Positive number of client play lanes for `weka`, `agentic_mooncake`, or agentic `dynamo` timestamp replay; does not cap concurrent child requests. Requires `source_type: trace`, `trace_path`, and `load_type: trace_timestamps`, with no `replay_concurrency`. |
| `agentic_snapshot` | `AgenticSnapshotOptions \| None` | `None` (unset) | Optional object `{seed: u64}`; required `seed` is an unsigned 64-bit integer (`0` through `2^64 - 1`). Requires positive `agentic_lanes`, `source_type: trace`, `trace_path`, and `load_type: trace_timestamps`, with no `replay_concurrency`; supported formats are `weka`, `agentic_mooncake`, and agentic `dynamo`. Unset preserves turn-zero execution. |
| `agentic_warmup` | `bool` | `False` | Requires `agentic_snapshot` and positive lanes when enabled. Runs primers and ten warmup requests per lane before measured replay, preserving native cache state; see [warmup behavior](../replay/workloads.md#agentic-warmup). Failed preparation is retained as candidate evidence and excluded from ranking. |
| `agentic_profile` | `AgenticProfileOptions \| None` | `None` (unset) | Optional object; `{}` enables continuous lane replenishment with the defaults below. Requires `agentic_snapshot` and its agentic trace/lane controls; cannot be combined with `max_sim_time_ms`. Unset preserves finite replay. See [continuous profile](../replay/workloads.md#agentic-profile) for the complete configuration, results, and limits. |
| `agentic_profile.duration_seconds` | `float` | `3600.0` when enabled | Positive finite admission duration, starting at the preparation barrier or simulation start without warmup. Stops new workload requests and replacement plays at the deadline. |
| `agentic_profile.response_grace_seconds` | `float` | `30.0` when enabled | Nonnegative finite response window after admission closes; remaining client requests are then canceled. |
| `agentic_profile.cancel_drain_seconds` | `float` | `10.0` when enabled | Nonnegative finite upper bound for cancellation acknowledgements. Supported offline runtimes acknowledge synchronously; server/GPU cleanup is not guaranteed by this budget. |
| `agentic_profile.tree_idle_cap_seconds` | `float` | `300.0` when enabled | Positive finite idle cap for advancing a play's pending workload timers when that play has no outstanding requests. |
| `agentic_profile.global_idle_cap_seconds` | `float` | `10.0` when enabled | Positive finite idle cap for advancing pending workload timers when the entire client workload has no outstanding requests. Engine completions and server cleanup keep their actual timestamps. |
| `replay_concurrency` | `int \| None` | `None` | Closed-loop in-flight cap **for a trace**; when set, trace timestamps are ignored. For synthetic closed-loop use `concurrency` instead. |

The synthetic fields are `isl`, `osl`, `request_rate`, `concurrency`, `kv_load_ratio`,
`num_request_ratio`, `random_range_ratio`, and `random_seed`;
`shared_prefix_ratio`, `num_prefix_groups`, `turns_per_session`, `inter_turn_delay_ms` are
shared synthetic knobs carried by `ReplaySpec.workload`.

Profile controls are concrete workload values, not search dimensions. The built-in Engine runner
supports profiles on offline aggregated or P/D vLLM/SGLang replay with HBM-only KV cache.
vLLM also supports local or shared [G2 host offload](../replay/engine/kv-cache.md) on a static single
aggregated worker or 1P1D, with attention DP1 on every role. Speculative decoding and G3
remain unsupported. Warmup is optional; enabling it retains the saved snapshot frontier.
A completed lane takes a new turn-zero play from the shared corpus cursor, wrapping as necessary
until the admission deadline. Snapshot sampling and warmup still differ from the AgentX reference;
see the [profile limits](../replay/workloads.md#agentic-profile). Injected runners must advertise
`supports_agentic_profile`; unsupported runners fail validation.

## Validation (`Workload._validate_workload`)

- **Trace workload** (`trace_path` set): must **not** set any synthetic field
  (`isl`, `osl`, `request_rate`, `concurrency`, `kv_load_ratio`, `num_request_ratio`, or
  non-default random-length options) — error lists the offenders. `replay_concurrency`, if set,
  must be a positive int.
- **Synthetic workload** (no `trace_path`): **exactly one** of `request_rate`,
  `concurrency`, or `kv_load_ratio` (none / multiple -> error); `isl`, `osl`,
  `num_request_ratio` are all
  **required**; `replay_concurrency` is rejected (it is trace-only — use `concurrency`).
  `concurrency` must be one positive int; `request_rate`, `isl`, `osl`, and
  `num_request_ratio` must be positive. KV-load values must be finite and non-negative.
  `random_range_ratio` must be finite and in `(0.0, 1.0]`, `random_seed` must be an
  unsigned 64-bit integer, and randomized lengths currently require a single-turn workload.
- **Ranged KV load only under Pareto** — `[min, max]` must contain exactly two values with
  `min < max`; `SmartSearchConfig._validate_kv_load_ratio_range` rejects it for scalar
  goals. A scalar `kv_load_ratio` is valid for every goal. A synthetic Pareto config that
  omits all three load fields defaults to `kv_load_ratio: [0.0, 1.0]`.

Candidate-relative `kv_load_ratio` and `num_request_ratio` are resolved after
candidate capacity is known; their formulas and grouped-cache field differences
are in [search space](search-space.md). Shared traffic and AgentX lifecycle
semantics belong to [Replay](../replay/workloads.md).

## Goal and sweep control

`goal` is an `OptimizationGoal`; use [optimization goals](optimization-goals.md)
for scalar, strict-SLA and Pareto behavior. With the historical SDK
`sweep.max_trials: null`, `max_rounds` and `candidates_per_round` define barrier
rounds. With a total `max_trials`, `algorithm` chooses the seeded Bayesian or
random sampler. `parallel_evals` bounds replay-worker fan-out;
`max_eval_seconds` bounds one candidate. Public CLI `optimizer` fields compile
into this SDK representation and must not be copied verbatim under `sweep`.

## Provider Preparation

A provider implements two operations:

1. `generate_search_space(search_spec, context)` validates the complete adapter-owned search space
   and returns branch-specific parameters plus reusable prepared state.
2. `materialize_replay(plan, selection, context)` turns one namespaced selection into an
   `AdapterReplaySpec` with concrete configuration and optional runtime hooks.

Sweeper namespaces provider parameters as `adapter::<adapter name>::<local parameter>`. This avoids
collisions without adding feature-specific fields to the core schema.

## Extension ABI

The [runner ABI](../adapters/runner-abi.md) defines capabilities, execution and
result obligations. The separate [SDK provider ABI](../adapters/configuration-abi.md)
defines prepared search dimensions and per-candidate materialization. Refer to
those contracts when integrating an existing framework; only the contract is required to integrate an existing implementation.
