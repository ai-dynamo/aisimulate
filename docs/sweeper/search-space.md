<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Search space and budgets

A recommendation expands deployment choices into concrete candidates, checks
memory and runner capabilities, and evaluates surviving candidates with Replay.
The public CLI YAML and SDK YAML are separate schemas: examples in the first
three sections use public CLI fields; sections marked SDK use `SmartSearchConfig`.

<a id="recommendation-domains"></a>

## Recommendation Domains

A field in a recommendation input is either a concrete value or one explicit domain object.

<a id="choices"></a>

### Choices

```yaml
engine:
  backend:
    choices: [vllm, sglang]
```

`choices` must be nonempty and contain unique values valid for the field. A bare YAML sequence never
means a search domain.

<a id="numeric-range"></a>

### Numeric Range

```yaml
engine:
  workers:
    aggregated:
      parallelism:
        preset: false
        replicas:
          range:
            min: 1
            max: 8
            step: 1
            scale: linear
```

| Range Field | Type | Default | Constraints |
|---|---|---:|---|
| `min` | integer or number | Required | Finite and no greater than `max`. |
| `max` | same as `min` | Required | Finite and no less than `min`. |
| `step` | same as range | None | Positive; allowed only for `linear`. Required for integer linear ranges. |
| `scale` | enum | `linear` | `linear` or `log`. |

For `log`, `min` must be positive and `step` is rejected. Integer-valued fields always materialize
integers, including when sampled from a log range.

<a id="domain-validation"></a>

### Domain Validation

A field accepts at most one domain form. Domains are allowed only where **Default Range** is not `x`:

- Engine mode, backend, `hardware: auto`, parallelism preset and leaves, scheduler, and supported
  backend-specific fields.
- Router policy, load model, and supported policy-specific fields.
- Planner preset sub-items, policy, and supported Planner-specific fields.
- Traffic load intensity and timing fields marked `-` in the table.

The following stay concrete:

- Model, concrete hardware values, backend version, and context length.
- Traffic source, token lengths, session shape, trace contents, and stopping condition.
- Evaluation thresholds.
- Optimization target, hardware selection, constraints, and optimizer controls.

<a id="presets-and-default-ranges"></a>

## Presets and Default Ranges

The component reference tables use five columns:

- **Knob** is the complete YAML path.
- **Default** is the concrete value used by `predict` when the knob is omitted.
- **Default Range** is the recommendation domain used when preset search is disabled. `x` means the
  knob is non-sweepable and rejects any domain. `-` means the knob is sweepable, but its default
  domain is the singleton concrete default. Any displayed `choices` or `range` is searched by
  default. On a `preset` row, this column lists the built-in preset choices; `auto` on the
  parallelism preset means the Sweeper generates its projected default space.
- **Preset** names the smallest configuration object whose preset covers the knob. `-` means no
  preset covers it.
- **Rules** carries the type, conditional availability, and validation that would otherwise require
  repeated prose in each component reference.

A preset is a list of complete mappings. Each mapping must specify every knob belonging to the
smallest preset class shown in the table. Use `null` for an inactive knob only where the component
allows it; see the [Planner interval rules](../replay/dynamo.md).
A mapping is one atomic candidate choice; values inside it are not independently combined.

For any preset-capable object in `recommend`, `preset` has these forms:

```yaml
# Omitted, or written explicitly: use the component's built-in default preset list.
preset: default
```

To replace a default preset list, provide a list of complete atomic mappings under `preset`. The
parallelism section below shows the complete syntax; Planner uses the same list shape.

```yaml
# Disable preset search. These two spellings are equivalent.
preset: false
# preset: {}
```

When `preset` is omitted or `default`, the built-in preset list is the default sweep space. A custom
list replaces it. A list entry missing any covered knob, containing an unknown knob, or containing a
`choices`, `range`, or `auto` domain is rejected.

The built-in list is versioned public configuration data owned by the component provider. It follows
the same complete-mapping validation as a user-provided list; it is not an opaque runtime mode. Names
shown in a preset row's Default Range are public identifiers that each expand to one complete mapping.

When `preset` is `false` or `{}`, every covered sweepable knob becomes an independent sweep dimension.
An explicit concrete value pins the knob; an explicit `choices` or `range` replaces its table-defined
default range. An omitted `-` knob uses the singleton concrete default. An `x` knob stays pinned and
rejects a domain. The Sweeper evaluates the Cartesian product and rejects infeasible concrete
combinations. A preset and independent domains cannot be active on the same object.

Preset controls are recommendation-only. Recommended prediction YAMLs contain only the expanded
concrete knobs.

If the optional `router` or `planner` section is absent, that component stays fixed at its concrete
default. A present Router searches its direct knob ranges. A present Planner activates its default
sub-item preset sweeps.

<a id="parallelism-preset-behavior"></a>

### Parallelism Preset Behavior

`engine.workers.<role>.parallelism` uses the same `preset: default` spelling as every other
preset-capable object:

```yaml
parallelism:
  preset: default
```

The default preset selects complete, feasible parallelism mappings within the GPU budget.
It accounts for the model, backend, memory capacity, and supported worker shapes. Its current
worker-size ladder is `1, 2, 4, 8, 16` GPUs with pipeline parallelism fixed at `1`; replica counts
share the candidate budget. See the [projection algorithm](#parallelism-search-projection)
for the internal search representation.

A user-provided preset is a list of complete parallelism mappings:

```yaml
parallelism:
  preset:
    - {replicas: 1, tensor: 1, pipeline: 1, attention_data: 1, moe_tensor: 1, moe_expert: 1}
    - {replicas: 2, tensor: 2, pipeline: 1, attention_data: 1, moe_tensor: 1, moe_expert: 1}
```

Unlike the built-in default preset, this list is kept flat: each complete mapping is one categorical
choice and the Sweeper does not decompose it. To search independent dimensions, disable the preset
and provide zero or more per-knob domains:

```yaml
parallelism:
  preset: false
  replicas: {range: {min: 1, max: 8, step: 1}}
  tensor: {choices: [1, 2, 4, 8]}
```

Omitted parallelism knobs then use their table-defined default ranges, and the Sweeper evaluates the
Cartesian product before feasibility filtering.

<a id="optimizer-controls"></a>

## Optimizer Controls

```yaml
optimizer:
  algorithm: bayesian
  max_trials: 320
  parallelism: 16
  candidate_timeout_seconds: 600
  seed: 42
```

| Knob | Default | Default Range | Preset | Rules |
|---|---:|---|---|---|
| `optimizer.algorithm` | `bayesian` | `x` | `-` | `bayesian` or `random`. |
| `optimizer.max_trials` | `320` | `x` | `-` | Positive total trial budget. |
| `optimizer.parallelism` | `16` | `x` | `-` | Positive. |
| `optimizer.candidate_timeout_seconds` | `600` | `x` | `-` | Positive wall-clock limit per candidate. |
| `optimizer.seed` | `42` | `x` | `-` | Nonnegative. |


The progress bar counts settled suggestions, including cache hits, unsupported candidates,
failures and timeouts. The summary separately reports actual evaluations and cache hits;
`suggesting` means time is being spent in the optimizer, while `evaluating` means candidates
are being materialized or replayed. Suggestion time is reported at the end.

`random` visits shuffled legal backend/topology pairs before returning to their scheduler
and workload domains. Finite complete configurations are sampled without replacement,
without materializing the Cartesian product. Exhausting a finite space ends the search
before `max_trials`; the summary says so. For `min_gpus`, each coverage pass starts with
the smallest legal GPU counts. This improves coverage within a small budget, but does not
prove global optimality across all scheduler/workload combinations.

Bayesian duplicates reuse cached measurements and still consume the suggestion budget.
Deterministic KV-capacity, SLA and load-constraint failures are also cached. Random search
interleaves host-resource and timeout retries with new configurations within the same trial
budget, including continuous domains. Single-slot asks alternate between pending retries and
new configurations so neither starves; unexpected runtime failures are not cached. The CLI folds selected
scheduler-limit variants only when all other prediction inputs and all reported metrics
match; the complete candidate ledger remains in JSON/CSV.

## Parallelism search projection

The built-in default follows the existing Sweeper projection algorithm below. It does not expose the
six YAML leaves as six independent optimizer parameters.

First, the Sweeper builds the legal configuration pool for each deployment-mode branch. It enumerates
worker sizes from the current `1, 2, 4, 8, 16` GPU ladder, with pipeline parallelism fixed at `1`, then
enumerates legal tensor, attention-data, MoE-tensor, and MoE-expert shapes. It applies model-width,
backend, real-silicon, KV-capacity, GPU-budget, and runner-capability filters. For every surviving
worker shape, it enumerates positive replica counts that fit the budget. A disaggregated pool contains
prefill/decode pairs whose combined GPU count fits the same budget. Aggregated and disaggregated modes
use separate optimizer studies; backend remains a categorical parameter within each study.

Second, each complete mapping is encoded into a smaller latent search space:

| Deployment | Latent Parameter | Optimizer Type | Encoding |
|---|---|---|---|
| Both | `used_gpu_ratio` | Continuous float | Total GPUs divided by the branch GPU budget; range is the minimum and maximum ratio in the legal pool, default clamped from `1.0`. |
| Aggregated | `agg_num_gpus_per_engine_target` | Log-scale discrete | GPUs per worker, `tensor * pipeline * attention_data`; feasible values come from the legal pool and the default is the pool value nearest its geometric midpoint. |
| Aggregated | `agg_attention_mode` | Categorical | `tp` when attention data parallelism is `1`, otherwise `dp`. |
| Aggregated MoE | `agg_ffn_mode` | Categorical | `ep` when MoE expert parallelism is greater than `1`, otherwise `tp`. |
| Disaggregated | `prefill_gpu_share` | Continuous float | Prefill-pool GPUs divided by total candidate GPUs; range comes from the legal pool, default clamped from `0.5`. |
| Disaggregated | `prefill_num_gpus_per_engine_target` | Log-scale discrete | Prefill GPUs per worker. |
| Disaggregated | `decode_num_gpus_per_engine_target` | Log-scale discrete | Decode GPUs per worker. |
| Disaggregated | `prefill_attention_mode`, `decode_attention_mode` | Categorical | Per-role `tp` or `dp`. |
| Disaggregated MoE | `prefill_ffn_mode`, `decode_ffn_mode` | Categorical | Per-role `ep` or `tp`. |

The latent parameter names retain the existing Sweeper's `engine` wording; in this public schema,
`num_gpus_per_engine_target` means GPUs per worker.

Only `used_gpu_ratio` and, for disaggregated mode, `prefill_gpu_share` are continuous parallelism
parameters. GPUs per worker are discrete values sampled on a log scale; attention and FFN modes are
categorical. Replica count is not sampled directly: together, total GPU ratio and GPUs-per-worker
targets express the desired replica footprint. Constant latent parameters are omitted from the study
and injected at their defaults.

Third, every optimizer suggestion is snapped back to one complete mapping from the legal pool:

1. Remove mappings that do not support the suggested backend.
2. Count categorical mismatches for attention and FFN modes, and retain only mappings with the minimum
   mismatch count. An exact mode match wins whenever one exists.
3. Compute normalized squared distance over the numeric latent parameters. Ratios use linear values;
   each GPUs-per-worker target uses `log2`. Each dimension is normalized by its backend-compatible
   minimum-to-maximum span, and a constant dimension contributes zero:

   ```text
   distance = sum(((transform(actual) - transform(requested)) / span) ^ 2)
   ```

4. Select the mapping with minimum distance. Ties are deterministic: compare
   `(tensor, pipeline, attention_data, moe_tensor, moe_expert, replicas)` for aggregated mode, or the
   concatenated prefill tuple followed by the decode tuple for disaggregated mode.

The selected mapping supplies the concrete six YAML fields. Trial metadata records requested latent
features, actual snapped features, projection distance, whether a categorical mode was projected, and
the final complete parallel configuration.

## Role-specific context limits

`engine.workers.prefill.context_length` and
`engine.workers.decode.context_length` map to SDK search-space fields
`prefill_context_length` and `decode_context_length`. Both are fixed positive
integers. Each role's KV-feasibility check uses its override, then shared
`context_length`, then the resolved model maximum. Heterogeneous P/D workers
and AFD companions use their own role's limit.

Candidate serialization and generated worker payloads preserve explicit role
or shared limits. When neither is set, the payload leaves `max_model_len`
unset even though feasibility filtering used the model maximum. See
[worker context limits](../replay/engine/workers.md#context-limits) for public
configuration and synthetic workload validation.

<a id="pinned-parallel-configurations"></a>

## SDK: Pinned Parallel Configurations

Pinning `parallel_configs` requires exactly one deployment mode. An aggregated entry is one shape:

```yaml
search_space:
  deployment_mode: [agg]
  parallel_configs:
    - tp: 4
      attention_dp: 2
      replicas: 2
```

A disaggregated entry contains `prefill` and `decode` shapes. Every pinned shape must be legal,
KV-feasible, supported by at least one selected backend, and accepted by the configured Replay
runner. If every selected backend/topology pair is runner-incompatible, preflight raises
`aisimulate.sweeper.RunnerIncompatibleError` with the rejected mode and backend names.

<a id="sampler-algorithm-override"></a>

## SDK: Sampler Algorithm Override

The experimental `AISIMULATE_SWEEPER_VIZIER_ALGO` environment variable overrides the Vizier
algorithm. For example, set it to `RANDOM_SEARCH` to bypass the default GP-bandit designer.
`SPICA_VIZIER_ALGO` remains a deprecated fallback during migration; when both are set, the
AI Simulate variable takes precedence.

## `kv_load_ratio` (candidate-relative concurrency)

Sweeper resolves a KV-load trial after the backend, parallel shape, replicas, and batching
knobs have been selected. For every active role, it asks AI Configurator for the **per-rank** KV token
capacity using that candidate's `max_num_batched_tokens`, `max_num_seqs`, memory fraction,
parallel shape, and MTP setting. The scheduler-visible role capacity is:

```text
per_rank_usable_tokens = floor(per_rank_tokens / block_size) * block_size
role_capacity_tokens = per_rank_usable_tokens * attention_dp * replicas
```

Attention-DP ranks own independent sequence pools, so capacity is multiplied by
`attention_dp`; TP/EP ranks shard the same sequences and are not multipliers. For disagg,
both prefill and decode are checked for candidate-specific memory feasibility, but only
**decode** capacity drives load. For agg, **agg** capacity drives load.

The concrete closed-loop cap is:

```text
average_tokens_per_request = isl + floor(osl / 2)
capacity_concurrency = floor(role_capacity_tokens / average_tokens_per_request)
concurrency = max(1, floor(kv_load_ratio * capacity_concurrency))
```

`kv_load_ratio = 0` therefore maps to the minimum concurrency `1`; `1` means estimated
100% steady-state KV occupancy. It is an estimate, not a guarantee that replay sees no
temporary KV pressure or request retraction. Batching combinations that leave no KV budget
are reported to Vizier as infeasible before replay.

Every resulting candidate records `kv_load_ratio`, the derived `concurrency`,
`kv_load_concurrency_capacity` for traceability. Linear caches also record
`kv_load_capacity_tokens` and per-role `*_kv_capacity_tokens`. These token fields
are absent for grouped caches, which instead record `kv_load_capacity_bytes`,
`kv_load_request_cache_bytes`. Consumers must treat token-capacity fields as optional
and use the byte-based fields for grouped layouts.

## `num_request_ratio` (synthetic length scales with the load)

`resolved_request_count(concurrency_override=None)` computes the synthetic request count as

```text
num_requests = max(1, round(num_request_ratio * load))
```

where `load` is, in precedence order: the candidate-derived `concurrency_override` (KV-load
mode), else fixed `concurrency` (closed-loop), else `request_rate` (open-loop).

So the synthetic trace length **scales with the swept load automatically**: with
`num_request_ratio = 10`, concurrency `256` yields `2560` requests, concurrency `512`
yields `5120`. Result is floored at `1`; `num_request_ratio` itself is treated as `0.0`
when unset (`max(1, …)` keeps at least one request).

## Attention-FFN disaggregation

Public CLI uses `engine.mode: afd` and `engine.afd`; SDK uses deployment modes
`afd` and `afd+pd`. This is an analytical path, and runners must advertise the
backend/topology pair. It does not execute native A/F worker handoffs.

In the SDK, `afd_pinned_topologies` supplies complete shapes. Otherwise an
explicit `afd_batch_size_candidates` list bounds the finite domain; optional
`afd_tp_a_candidates`, `afd_f_moe_ep_size_candidates`, `afd_microbatch_candidates`,
`afd_pipeline_model_candidates` and `afd_max_candidates` constrain it further.
`afd+pd` uses `afd_phase: prefill` or `decode` and the opposite phase's ordinary
worker knobs; `afd_companion_parallel_configs` can pin companion shapes.

AFD requires concrete positive input/output lengths and synthetic request-rate
or absolute-concurrency traffic. Candidate-relative KV load is rejected because
the analytical path does not expose scheduler-visible KV capacity. The candidate
GPU total includes attention, FFN and any P/D companion. Read
[Replay AFD](../replay/engine/analytical.md#afd) for the full
shape contract, staging and analytical limits.

### Finite AFD topology enumeration

`AFDSearchConfig.pinned_topologies` preserves complete pinned shapes exactly.
Otherwise `enumerate_afd_topologies` enumerates A/F node counts, attention TP
and batch size, F-side EP, microbatches and pipeline model. One F replica spans
all F GPUs; F TP equals that pool size. EP must divide both F TP and the known
model expert count. The GPU budget is an upper bound: a remainder smaller than
`gpus_per_node` may be unused.

The finite domain retains canonical ordering. If it exceeds `max_candidates`,
enumeration fails with `candidate_limit` instead of silently truncating it.
For AFD+P/D, the full legal topology-by-companion product must also fit
`afd_max_candidates`. Candidate provenance records the phase, shape, A/F GPU
accounting, generated dimensions and filters; pinned shapes identify their
explicit source. Performance measurements and staged scheduling are documented
in [Replay's AFD contract](../replay/engine/analytical.md#afd).

## Analytical encoder search

EPD combines an analytical encoder pool with aggregated or P/D language Replay.
In public CLI, fixed `traffic.source.images` pairs with `engine.workers.encoder`;
encoder tensor parallelism, replicas and batch size accept concrete values or
`choices`. Encoder batch size is at most eight. The SDK uses
`search_space.encoder` and `workload.images` instead.

The workload must use fixed synthetic images and fixed concurrency. Encoder and
language GPUs both count toward the candidate budget. SLA bounds are aggregate
means; `strict_sla: true` is required when bounds are present, except `min_gpus`,
which always enforces them. Goodput objectives, a goodput rate floor,
`min_candidate_gpus`, adapters, variable loads, traces, sessions and per-request
capture are unsupported. EPD cannot combine with AFD. See
[Replay's analytical boundaries](../replay/engine/analytical.md) and the
[existing EPD example](../../examples/cli/epd-recommend.yaml).
`engine.workers.encoder.mode: native` lifts the fixed-concurrency,
aggregate-SLA, per-request and op-level restrictions; see
[native encoder pools](../replay/engine/analytical.md#native-encoder-pools).

<a id="removed-kvbm-fields"></a>

## Host Offload and Removed KVBM Fields

Sweeper rejects the old KVBM block-count, transfer-bandwidth, offload-batch-size, and cache-hit
search fields. Those legacy fields have no adapter migration.

The public `predict` and `recommend` commands support a separate native host-offload descriptor
at `engine.workers.<role>.kv_cache.host_offload`. It sets the G2 `scope`, `num_host_blocks` and
bandwidths as fixed values, not search dimensions. It requires vLLM and prefix caching; native
speculative decoding is not supported. For `recommend`, mode (`aggregated` or `disaggregated`)
and backend must be concrete; parallelism knobs, including `attention_data`, may be searched. When
both prefill and decode use `scope: cluster_shared`, both roles must set `tensor` and `pipeline` to
the same explicit integer (an omitted value is searched), their KV block geometry and pool fields
must match, and either both or neither must use default timing. This does not add disk offload or
restore the removed KVBM search fields.

See [Native vLLM host-offload prediction](../replay/engine/kv-cache.md#host-offload-g2)
for a complete YAML example and CLI command.
