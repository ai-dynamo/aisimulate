# `aisimulate-core` public API contract

The core API is delivered through the repository's two release artifacts at
the same version:

- the `aisimulate` Python wheel, whose estimator API is `aisimulate_core`;
- the `aisimulate-core` Rust crate, imported as `aisimulate_core`.

The single wheel owns the application, estimator SDK, model and system data,
and unified native PyO3 extension. It does not depend on another core
distribution or on Dynamo. The crate owns the compiled engine, forward-pass
model, Replay runtime, KV-cache request/response types, and the embedded
Rust-to-Python construction path. Legacy Python import namespaces are removed
in AISimulate 0.13.0; see the [Python migration guide](python-source-migration.md).

## Stable Python facade

New Python code should import from the small facade:

```python
from aisimulate_core.sdk import (
    EngineHandle,
    ModelConfig,
    RuntimeConfig,
    RustForwardPassPerfModel,
    compile_engine,
    estimate_kv_cache,
    estimate_num_gpu_blocks,
)
```

The explicit module paths remain supported:

```python
from aisimulate_core.sdk.engine import EngineHandle, compile_engine
from aisimulate_core.sdk.rust_engine_step import RustForwardPassPerfModel
from aisimulate_core.sdk.memory import estimate_kv_cache, estimate_num_gpu_blocks
```

`aisimulate_core.sdk.__all__` is the supported high-level surface. The
facade resolves lazily, so importing it does not load the model registry,
performance database, or native engine until a name is used.

The top-level `aisimulate_core` module exposes the lower-level native
extension contract:

- `AicEngine`
- `RustForwardPassPerfModel` (the raw PyO3 class, distinct from the ergonomic
  SDK wrapper)
- `engine_spec_bincode_from_json`
- `_build_smoke`

The wheel includes `py.typed` and a stub for that native extension. The SDK
Python modules carry their own annotations.

### Context-attention kernel queries

`AicEngine.evaluate_context_attention_kernels_json` accepts `ops_json` (a JSON
array of serialized `ContextAttention` operations), `batch_size`, and `s`
(sequence length in tokens), with optional `prefix=0`,
`imbalance_correction_scale=1.0`, and `visual_block_upper_triangle=False`.
It evaluates only the attention kernels, excluding fused QK normalization,
RoPE, and KV-write work. Invalid JSON and other operation families raise
`ValueError`.

With `visual_block_upper_triangle=True`, the query prices only the additional
strict upper-triangle attention pairs inside a bidirectional visual block,
using the underlying causal kernel's database policy and provenance. This
mode requires `prefix=0`; a nonzero prefix raises `ValueError`. A block with
zero or one token has zero additional latency and energy, with `source="sol"`.
The flag is a runtime query option and does not change the serialized op or
engine-spec formats.

Results are `list[tuple[str, float, float, str]]`, containing
`(name, latency_ms, energy_wms, source)` with repeated operation names folded
together. Energy is in watt-milliseconds and is zero when power data is
unavailable. An empty operation list returns an empty list. `aisimulate_core.AicEngine` exposes this
method; `EngineHandle` provides an annotated SDK wrapper with the same query
options.

## Engine context limits

`EngineConfig.max_model_len` is an optional positive prompt-plus-output token
limit for vLLM, TRT-LLM, and SGLang. Recipe adapters can map TRT-LLM
`max_seq_len` and SGLang `context_length` to this field.

- A prompt at or above the limit is rejected before prefill computation.
- A decode destination rejects such a prompt before queuing the handoff or
  reserving KV blocks, even when destination admission is deferred.
- Terminal rejection removes the request's outstanding Belady input demand
  before subsequent cache admission and eviction.
- Generation stops when prompt plus output reaches the limit, including
  speculative bursts and requests with explicit output token IDs. Reports
  retain the requested output length and count only tokens actually generated.
- TRT-LLM completion reservations and SGLang output reservations use the capped
  output budget. Physical KV capacity remains a separate constraint.
- The limit applies to aggregated workers and both roles in disaggregated replay.
  An unset limit preserves the existing backend behavior.

The prediction compiler and recommendation deployment builder pass explicit
`engine.context_length` values to every backend. Prediction with
`context_length: max` retains its existing behavior: vLLM resolves model
metadata, while TRT-LLM and SGLang leave the scheduler limit unset.

This is the simulator's normalized context-limit contract. It does not model
backend-version-specific frontend validation margins or automatic prompt
truncation. Successful replay alone does not establish silicon timing accuracy.
See the [SGLang GPU parity observations](https://github.com/ai-dynamo/aisimulate/pull/261#pullrequestreview-5250881045)
for stricter boundary behavior; they do not establish a fixed token offset
across versions or configurations.

## KV-cache capacity reservation

SGLang's native estimator treats `mem_fraction_static` as a static weights/KV
pool. Peak activation/workspace estimates remain visible in
`memory_breakdown.activations_bytes`, but are not deducted from that pool;
transient execution headroom is already outside the static fraction. Increasing
the prefill token budget alone therefore does not reduce SGLang KV capacity.
Resident runtime/communication estimates reduce the pre-load free-memory pool
before applying the fraction; weights are deducted afterward. The budget is
`(capacity - resident_overhead) * mem_fraction_static - weights`, less any
explicit additional graph reservation. For ordinary SGLang DeepSeek-V3/R1
(non-CP, non-PP, non-speculative, non-large-EP), the estimator also respects
checkpoint dense/MoE layer counts and TP-sharded embeddings independently of
the unchanged timing graph. Other model layouts retain their prior weight
accounting.
vLLM and TRT-LLM continue to deduct activation memory under their own budget
semantics. No measured server capacity is required by this calculation.

`estimate_kv_cache` and `estimate_num_gpu_blocks` accept
`cuda_graph_reserved_bytes=<rank-local bytes>`. The value must be a
non-negative integer no greater than `2**53` and defaults to zero. It is treated
as fixed non-KV memory before the backend-specific KV fraction is applied. For
SGLang, it is an additional reservation beyond graph/runtime headroom already
encoded by `mem_fraction_static`. Native estimates return it as
`memory_breakdown.cuda_graph_reserved_bytes`; naive fallback estimates apply it
but keep `memory_breakdown=None` because the other components are unavailable.

The Rust `KvCacheEstimateRequest` exposes the same field. Engine replay accepts
the value in its engine arguments and carries it through native AIC capacity
rematerialization, so the Python estimator and native scheduler use the same
rank-local KV capacity. The `aisimulate predict` YAML exposes it at
`engine.workers.<role>.kv_cache.capacity.cuda_graph_reserved_bytes`. This API
does not estimate the reservation; callers must supply a value from a source
they trust.

Serialized Rust requests and estimates that omit the field remain compatible
because deserialization defaults it to zero. Rust source that constructs
`KvCacheEstimateRequest` with a struct literal must add
`cuda_graph_reserved_bytes: 0`; exhaustive `MemoryBreakdown` literals and
patterns must include the new field. This source migration is part of the next
minor API update.

### FPM profile cache groups and byte budgets

FPM resources support `cache_layout: linear` with a positive
`kv_bytes_per_token`, or `cache_layout: grouped` with a nonempty `cache_groups`
list and no scalar token rate. Existing linear profiles keep their serialized
shape. Python exports `FpmCacheGroup` from `aisimulate.fpm_profile` and
`aisimulate_core.fpm_profile`; Rust exports it from `aisimulate_core::perfmodel`.

Memory can be pending, declared, or observed at runtime. The four legacy fields
`weights_bytes`, `activations_bytes`, `runtime_overhead_bytes`, and
`comm_overhead_bytes` are optional. A complete set retains the existing non-KV
budget calculation. A partial set is planning evidence only: missing values are
not zero, and cache sizing or replay fails with an instruction to finalize
runtime memory. Schema validation, collection planning, and direct timing queries
can use a pending profile. Omitted fields remain absent from serialized output.

Alternatively, `resources.runtime_memory` records the selected deployment's
initialized cache allocation:

```json
{
  "kv_cache_bytes": 1073741824,
  "gpu_memory_utilization": 0.9,
  "max_model_len": 32768,
  "provenance": "Verified collection worker initialization; see saved evidence."
}
```

The byte count is a positive integer no greater than `2**53`; utilization is
finite and in `(0, 1]`; context and provenance are required. Runtime memory
cannot coexist with any of the four legacy non-KV fields. Python exposes
`FpmRuntimeMemoryProfile` and resource properties `memory_source`
(`pending`, `declared`, or `runtime`) and `memory_ready`; `require_memory()`
rejects pending resources. Rust exposes `FpmRuntimeMemoryConfig` and
`FpmResourceConfig::require_memory()`. Rust callers using resource struct literals
must wrap legacy byte values in `Some(...)` and supply `runtime_memory: None`.

Each group has a unique `name`, `kind` (`attention` or `convolution`), positive
`num_layers`, `block_size_tokens`, and `page_size_bytes`, and an optional positive
`sliding_window`. An omitted or null window retains full history; convolution
groups require a window. `page_size_bytes` is the **rank-local aggregate for all
layers in the group**, including runtime padding. Do not multiply it by
`num_layers` again. Runtime block sizes and padding are deployment inputs; model
geometry alone does not establish them. See the
[grouped-profile review workflow](fpm-self-service.md#review-grouped-cache-resources).

Use `RustForwardPassPerfModel.estimate_cache_budget(config, budget)` with the
same canonical `ForwardPassPerfModelConfig` used for timing. Rust exposes
`ForwardPassPerfModelConfig::estimate_cache_budget(&request)` and
`ForwardPassPerfModel::estimate_cache_budget(&request)`, taking
`FpmCacheBudgetRequest` and returning `FpmCacheBudget`. The config method checks
the exact profile deployment and scheduler envelope without constructing a
timing model, loading timings, or building an operation graph. Grouped planning
requires the native extension; it does not require GPU access or measured data.

| Budget field | Meaning |
| --- | --- |
| `total_gpu_capacity_bytes` | Positive rank-local GPU capacity. |
| `memory_fraction_kind`, `memory_fraction_value` | `of_total` and the fraction of GPU memory available to the worker. |
| `max_num_tokens`, `max_batch_size` | Positive rank-local scheduler limits within a declared envelope; exact recorded values for runtime memory. |
| `context_length` | Optional positive request bound, at most the profile context and runtime `max_model_len` when present; defaults to the profile context. |
| `request_occupancy_tokens` | Optional positive logical request length within the selected context, for a separate steady decode footprint with one new token per forward pass. |
| `cuda_graph_reserved_bytes` | Separate nonnegative reservation for declared memory; must be zero for runtime memory. |
| `tolerance_fraction` | Optional fraction in `[0, 1)` deducted from the resulting cache byte budget. |

The result includes `total_kv_size_bytes`, `memory_breakdown`,
`resource_provenance`, `cache_layout`, `cache_groups`, and
`request_peak_cache_bytes`. The latter is a conservative single-request bound
including block alignment and the configured prefill chunk, not a reservation
for every scheduler slot. A smaller context changes this peak bound, not the
declared non-cache resource bounds. `tolerance_adjusted`, when present, provides
the reduced byte budget. For grouped caches, `kv_size_per_token_bytes`,
`total_kv_size_tokens`, and the adjusted token capacity are null (`None` in Python).
There is no equivalent scalar token capacity.

When `request_occupancy_tokens` is supplied, the result also includes
`request_occupancy_cache_bytes`. KV-relative synthetic loads use this native
footprint while retaining the actual scheduler settings and context for resource
admission. Observed linear pools retain their recorded capacity even when timing
coverage is smaller; uncovered queries report timing errors.

Runtime memory uses `kv_cache_bytes` directly and returns
`memory_breakdown: null` (`Option::None` in Rust). It does not invent activation
or other non-KV components. The requested memory fraction must exactly match the
recorded utilization; the device budget must contain the observed cache pool.
A different device capacity argument never rescales the pool or qualifies a new
deployment. Changed scheduler settings require new runtime evidence. Explicit
scalar cache-capacity overrides are rejected for runtime profiles in prediction,
recommendation and replay; use the recorded allocation. A tolerance margin may
reduce it through the canonical budget API.

Python `estimate_kv_cache(..., fpm_profile=..., context_length=...)` delegates
grouped and runtime-memory profiles to this native budget method.
`estimate_num_gpu_blocks` also accepts `context_length` for runtime validation;
it and the
legacy Rust scalar `KvCacheEstimate` transport reject grouped profiles; use the
groups and shared byte budget instead. Existing complete linear declarations
keep their budget calculation and serialized shape.

Grouped native execution supports cold aggregated vLLM, PP1/CP1, HBM-only cache,
non-speculative decoding and `prefix_caching: false`. Prefix reuse, host/G3
offload, disaggregation and fixed scalar cache capacity are rejected. Allocation
is atomic across all groups against one shared rank-local byte budget. A window
of `W` keeps the blocks covering the last `W - 1` computed tokens and all tokens
scheduled for the next forward. Expired completed pages are evicted before
subsequent forwards; temporary prefill pages remain charged through completion.
Full-attention groups retain full history. This physical retention never truncates
logical request progress or the context coordinates sent to FPM timing queries.

The Rust Replay observer's `ReplaySchedulerMetricsSnapshot` reports optional
`kv_cache_used_bytes` and `kv_cache_capacity_bytes` for grouped caches; its
`active_cache_usage` and `physical_cache_usage` use byte occupancy. `active_blocks`
counts resident group pages, while `total_blocks` is zero because heterogeneous
pages have no scalar block capacity. Use the byte fields for capacity comparisons.
These fields are currently exposed through the Rust observer API, not the Python
JSON replay runtime. Linear snapshots retain their existing block metrics.

## Choosing a forward-pass API

Use `RustForwardPassPerfModel.best_available(config)` from Python or
`ForwardPassPerfModel::best_available(config)` from Rust. The canonical
`ForwardPassPerfModelConfig` owns model, hardware, backend, topology, data
policies, a required immutable `worker_type`, and the complete nested
`estimator_config`. Worker roles are `prefill`, `decode`, and `aggregated`.

```python
from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel

config = ForwardPassPerfModelConfig(
    model="Qwen/Qwen3-32B",
    system="h200_sxm",
    backend="vllm",
    worker_type="aggregated",
    tp=2,
    estimation_mode="auto",
    fallback_policy="deny",
    estimator_config={
        "features": {"attention_kv_weight": 1.0},
        "fpm_regression": {
            "sampling": {"bins_per_axis": [4, 16], "max_observations": 128},
            "min_observations": 5,
        },
        "correction": {"enabled": True},
    },
)
model = RustForwardPassPerfModel.best_available(config)
print(model.diagnostics()["provenance"])
```

### External whole-forward FPM data

Set `estimator_config.fpm_interpolation.fpm_parquet_path` on the canonical configuration to use an external parquet and its required same-stem `.metadata.json` sidecar:

```python
config = ForwardPassPerfModelConfig(
    model="Qwen/Qwen3-0.6B",
    system="h200_sxm",
    backend="vllm",
    backend_version="0.25.1",
    worker_type="aggregated",
    estimation_mode="fpm_interpolation",
    estimator_config={
        "fpm_interpolation": {"fpm_parquet_path": "/data/reviewed-fpm.parquet"},
    },
)
model = RustForwardPassPerfModel.best_available(config)
```

The parquet identity must match the requested model, hardware, backend version, topology, and quantization. The systems YAML is still required, but a backend timing-data directory is unnecessary. Relative paths bind to the working directory when the model is constructed; resolved provenance stores the absolute path. The control applies when FPM interpolation is selected; other estimators retain it in provenance without opening the file. Saved legacy `timing.forward_model: fpm` and `timing.fpm_parquet_path` inputs migrate to the same canonical control, which is preserved in replay and per-role recommendation output.

`model.static_phase_latency(batch_size=1, input_tokens=512, output_tokens=4, prefill=False)` exposes the native engine's existing static integration before online correction. Prefill returns one prefill latency; decode returns total decode latency for the output sequence. This method requires a native estimator. AFD+PD uses it for an external-FPM regular companion, dividing total decode latency by `max(1, output_tokens - 1)` for TPOT. AFD attention and FFN workers retain their existing timing provider.

### Engine identity controls

The canonical configuration also carries quantization overrides and
`attention_backend`, `moe_backend`, `enable_eplb` (default `false`), and
`wideep_num_slots` (default absent). These controls reach model construction,
KV memory sizing, and replay provenance. EPLB/slots and nondefault MoE backend
selection require an MoE model. Collected FPM interpolation cannot represent
EPLB, slots, or MoE backend overrides; it rejects an explicit incompatible
request and is skipped during automatic selection for those identities.

Rust callers using exhaustive `ForwardPassPerfModelConfig` literals must add
`moe_backend: None`, `enable_eplb: false`, and `wideep_num_slots: None`.
`ForwardPassPerfModelConfig::new(...)` supplies these defaults. This extends
the canonical configuration introduced by #242.

Rust callers constructing `SyntheticTraceSpec` must also add
`cached_prefix_tokens: 0` to preserve existing prefix-sharing behavior. A positive
value creates shared input tokens; cache hits still depend on runtime state.
The value must align to the trace's `block_size` and must not exceed any sampled
input length. Unified replay uses one-token trace blocks for an exact prefix,
then applies the engine's cache block size when calculating reuse.

`nextn` remains compute-side identity. Expected accepted draft tokens are a
simulator workload assumption, supplied separately by the unified CLI as
`engine.nextn_accepted`; they do not tune the estimator.

### Selection and fallback

`estimation_mode` defaults to `auto`; `fallback_policy` defaults to `deny`.
Auto always searches `op_level`, `fpm_interpolation`, then `fpm_regression`,
including when fallback is denied. For an explicit mode, deny permits only
that estimator; allow tries the requested estimator first, then the remaining
estimators in the same global priority order. Each native mode tries the
ordered `systems_paths` before moving to another estimator. Omitted roots preserve
configured SDK discovery (or the systems-path environment override when the SDK
uses its packaged default); an explicit `default` entry selects the packaged root.
The resolved paths are shared by preflight, construction and capacity estimation.
Selection occurs
at construction; queries do not silently switch estimators on a data-domain error.

Invalid caller configuration does not trigger fallback. A constructed regression
may be unready: nonempty queries return `None` until their selected workload
store has a usable fit. Offline prediction/recommendation reject an untrained
regression instead of fabricating a latency. The current aggregated regression
retains its four workload stores; dedicated roles retain one each.

The returned provenance records the requested and selected modes, failed
selection attempts, effective backend version, data policy, selected root,
and complete estimator configuration. Its resolved config pins the selected
mode with deny so saved replay input repeats that selection.

Prompt-lookup verification uses the same constructor: set `speculation` to
`{"kind": "ngram", "params": {"num_speculative_tokens": 2}}` in Python/JSON, or
`ForwardPassSpeculationConfig::Ngram { num_speculative_tokens: 2 }` in Rust.
It supports vLLM op-level timing with 1–5 draft tokens and `nextn: 0`; auto can
select op-level but cannot fall back to an unsupported speculative estimator.
The cost configuration is retained in provenance and saved recommendations.
Acceptance rates and the scheduler seed stay in the CLI/Replay speculation
configuration; they do not change the model's target-verification graph.

### Estimator controls

`estimator_config` is passed intact through the Python facade, CLI, Sweeper,
and Replay, then parsed and validated in Rust. Unknown fields report their
nested paths. The supported namespaces are:

- `features`: `attention_kv_weight`, `prefill_attention_pair_weight`, and
  `ffn_token_weight`, each defaulting to 1.0. These currently affect regression
  only; positive finite values are required when regression is constructed.
- `fpm_regression`: independent `sampling`, `min_observations` (5), and `fit`.
  The fit kind is `standardized_nnls`, with a free intercept and nonnegative
  slopes. `singular_ridge_scale` defaults to `1e-9` and applies only when retrying
  a singular equation.
- `correction`: `enabled` (true), independent `sampling`, `min_observations`
  (5), `factor_bounds` (min 0.5, max 2.0), and the existing `max_num_tokens`
  (8192), `max_batch_size` (512), and `max_kv_tokens` (2000000) ranges.
- `fpm_interpolation.method`: `auto` (default), `sol`, or `direct`. Rust selects
  SOL for a registered architecture, or direct interpolation for an unknown
  architecture with a valid profile. Without a profile, auto retains SOL.
  Explicit SOL requires a registered analytical model; direct requires a profile.
- `fpm_interpolation.collect_coverage`: `false` (default). Opt in to bounded
  evidence from actual direct-FPM lookups, as described below. It requires
  explicit `estimation_mode: fpm_interpolation`, resolved `method: direct`, and
  `fallback_policy: deny`. It is omitted from normalized serialization when false.
- `op_level`: a reserved typed namespace; unknown fields are rejected.

The top-level `fpm_profile` contains the complete profile dictionary: pinned
model revision, architecture, context length, expert count, deployment precision
and topology, cache geometry, memory evidence, and provenance. A profile requires
an explicit literal `backend_version` that matches its selected deployment;
slot aliases and omitted versions are rejected. Profile/schema and precision
conflicts fail before estimator fallback. Omitted precision fields are filled
from the profile and preserved in the resolved canonical configuration.

For measured-only timing, pass `estimation_mode="fpm_interpolation"`,
`fallback_policy="deny"`, and
`estimator_config={"fpm_interpolation": {"method": "direct"}}` together with
`fpm_profile`. Direct interpolation requires `database_mode="SILICON"`, emits
whole-forward native operations without SOL operations, and never constructs
an analytical graph. Profile resource estimates and memory planning do not
require timing data or a native timing model.

The returned provenance pins both the selected estimation mode and interpolation
method, alongside the complete normalized profile. Reusing its `config` keeps
that selection across serialization and replay. Later timing coverage errors
never switch estimator or interpolation method. A registered model's graph
construction failure does not change SOL to direct; top-level fallback still
follows the configured estimator ordering and policy.

Sampling defaults to `bins_per_axis: [4, 4]` and `max_observations: 64` per
logical store. Rectangular grids are supported. Regression uses dynamic
`log1p` retention coordinates and fits standardized raw features. Correction
uses fixed raw workload coordinates; its one-dimensional prefill grid uses
the product of the two axis counts. Retention evicts the oldest sample from
the most populated cell when the store exceeds its budget.

Correction explicitly reports `feature_space: legacy_workload`. Its existing
prefill/decode/mixed stores and median-ratio calculation remain unchanged at
default settings. A shared role-based correction space requires separate
accuracy validation and is not accepted as a configuration value in this release.

Use `regression_store_diagnostics()` for per-store counts/readiness. Summary
readiness means at least one store is ready; another cold store can still
return `None`. `tune_with_fpms()` preserves the established FPM observation
contract. Native construction still uses Python model compilation; estimator
selection, regression, correction, and latency computation are owned by Rust.

### Direct-FPM query coverage

Enable coverage through the same canonical `best_available(config)` construction:

```yaml
estimation_mode: fpm_interpolation
fallback_policy: deny
estimator_config:
  fpm_interpolation:
    method: direct
    collect_coverage: true
```

The rest of the configuration must identify the exact deployment, including its
complete `fpm_profile`, literal backend version and data roots. CLI prediction
places these timing controls under `engine.workers.<role>.timing`. Coverage does
not change point selection, interpolation, correction or latency, and it does
not permit fallback or extrapolation after a missing query.

The returned model exposes `fpm_query_coverage()` as
`Result<Option<FpmQueryCoverage>, AicError>` in Rust and `dict[str, Any] | None`
in the Python SDK. The raw PyO3 method returns a JSON string, using `null` when
disabled. Evidence belongs to that model instance; cloning a Rust model starts
a fresh accumulator so candidate evaluations do not share counts. The public
Rust facade exports `FpmQueryCoverage`, `FpmQueryCoverageCounts`, `FpmQueryGap`
and `FpmQueryPurpose`.

A snapshot has cumulative `queries`, `prefill`, `decode` and
`mixed_decode_baseline` counts, each with `measured`, `interpolated` and
`unsupported`. Classification comes from the native lookup that supplied or
rejected the timing, including mixed-pass decode-floor lookups. Counts describe
`native_lookup_resolutions`; they are not request or replay-iteration counts,
and an external timing-cache hit adds no lookup. The snapshot retains up to 128
distinct gaps with phase, purpose, model/cell identity, query coordinates, reason
and occurrence count. `omitted_gap_queries` counts failed occurrences beyond
that storage limit; unsupported totals are not truncated. Snapshot reads do not
reset evidence, and lookup errors remain errors with partial evidence available.
The snapshot alone does not assess replay completion or prediction accuracy.

For existing static phase consumers, the same returned model provides these
uncorrected native timings, in milliseconds:

| Method | Coordinate contract |
| --- | --- |
| `predict_prefill_latency(batch_size, isl, prefix)` | Full input length and cached-prefix length per request; direct-FPM totals are `(batch_size, batch_size * (isl - prefix), batch_size * prefix)`. |
| `predict_decode_latency_total(batch_size, total_past_kv_tokens)` | Exact total past-KV across the decode batch, preserving the native FPM coordinate without a mean-context conversion. |
| `fpm_decode_kv_ceiling()` | Largest collected decode KV total, or `None`; reading this bound does not record a timing lookup or prove shape-specific coverage. |

These methods preserve the existing engine's phase behavior and record actual
direct lookup evidence when enabled. Empty work does not invent a measured
query. Whole-iteration `estimate_forward_pass_time_ms()` remains the API for
scheduled per-rank FPM telemetry, including mixed work and its existing online
correction behavior.

Ordinary replay uses this returned model for FPM timing and emits
`fpm_query_coverage` in its report when collection is enabled. The CLI also saves
`fpm-coverage.json`, including partial evidence when a native timing error stops
replay; the Python exception retains a `fpm_query_coverage` attribute for that
failure path. A passing replay coverage status requires completed requests,
nonempty resolved queries and no unsupported lookup. It is distinct from
operation/energy evidence, which whole-model FPM does not provide. See
[FPM replay validation](fpm-self-service.md#validate-fpm-query-coverage-with-agentx-replay)
for the stricter whole-corpus completion checks and saved onboarding artifacts.

### Migrating saved configuration

Use `ForwardPassPerfModelConfig.from_legacy_engine_config(old_config,
worker_type, old_options, allow_regression=False)` to convert a saved flat
EngineConfig and tuning options. It pins the old explicit native mode instead
of changing it to auto. Set `allow_regression=True` only for an old caller that
allowed direct regression fallback; the migration preserves that two-mode
order rather than adding interpolation. Legacy `forward_model: fpm` maps to
`fpm_interpolation`, and `fallback_policy: error` maps to deny. The deprecated
`regression` policy remains readable for these saved direct-fallback requests.
Legacy `extra.fpm_profile` and `extra.fpm_interpolation` migrate to the full
canonical profile and nested interpolation method; newly exported configuration
uses only the canonical fields.

Previously saved CLI timing with `forward_model` retains explicit selection
and deny. Newly authored requests without a selection use auto. `ForwardPassPerfOptions`
is retained as a legacy migration value type; new construction has one complete
config and no separate options argument. The raw PyO3 class also exposes
`normalize_config` and migration helpers for JSON-oriented consumers.

The [FPM regression design](../python/aisimulate/docs/fpm/aic-fpm-regression-design.md)
explains the retained workload routing and feature mathematics.

## Stable Rust facade

New embedded consumers should construct engines with
`aisimulate_core::perfmodel::AicEngineBuilder`. The
builder normalizes configuration into one private build request and enters
Python once to compile an engine specification. Calls on the returned
`AicEngine` are pure Rust and do not re-enter Python. Selected former AIC
crate-root types remain re-exported during the migration window, but the
`perfmodel` namespace is canonical for new code.

Native engine construction in standalone binaries requires the crate's `embed-python`
feature; applications hosted by an initialized Python interpreter do not need
auto-initialization. The matching `aisimulate` wheel must be importable for native
construction. Explicit `fpm_regression` construction works without the Python feature. See the
[crate README](../crates/core/README.md) for setup and usage examples.

The flat `build_aic_engine` adapter was removed from `main`; consumers must use
`AicEngineBuilder`.

The supported `aisimulate_core::perfmodel` Rust surface is grouped as follows:

- compiled engine: `AicEngineBuilder`, `AicEngine`, `AicError`;
- forward-pass estimation: `ForwardPassPerfModel`,
  `ForwardPassWorkerType`, `ForwardPassPerfModelConfig`, `EstimatorConfig`,
  diagnostics/readiness/source types, and the `ForwardPassMetrics` telemetry
  types;
- KV-cache estimation: `estimate_kv_cache`, `KvCacheEstimateRequest`,
  `KvCacheEstimateOptions`, `KvCacheMemoryFraction`, and estimate/result/error
  types;
- FPM profile cache resources: `FpmCacheGroup`, `FpmCacheKind`, `FpmCacheLayout`,
  `FpmResourceConfig`, `FpmRuntimeMemoryConfig`, `FpmCacheBudgetRequest`, `FpmCacheBudget`, and
  `FpmCacheBudgetAdjusted`, through the canonical model/config budget method;
- wire identity: `EngineConfig`, `ParallelMapping`, `QuantizationConfig`,
  `SpeculativeConfig`, `BackendKind`, `DatabaseMode`, and `DataType`;
- schema gates: `ENGINE_CONFIG_SCHEMA_VERSION`,
  `ENGINE_SPEC_SCHEMA_VERSION`, and `FPM_VERSION`.

Advanced consumers may use `perfmodel::engine::{Engine, RuntimeConfig, StaticMode,
StaticResult, PerOpValue}` and `engine::spec::{EngineSpec, OpSpec}` to load and
execute a previously compiled specification directly. `PerOpValue` is the
per-op result tuple `(name, latency_ms, energy_wms, source)` returned by the
`*_per_op` / `evaluate_*` methods (the thin op-list evaluation FFI); per-op
energy is 0.0 wherever the perf tables carry no power columns. That zero is a
missing-data sentinel, not evidence of a zero-power operation. See the
[modeled-power contract](power-model.md) for the latency-weighted coverage gate,
aggregation rules, and public output boundary. Typed per-op energy alone does
not make unified replay power available.

## Static phase diagnostics

The canonical `ForwardPassPerfModel` returned by `best_available` exposes
`static_phase_diagnostics(batch_size, context_length, prefix, prefill)` in Rust; the Python
`RustForwardPassPerfModel` wrapper accepts the same named arguments (with `prefix=0`).
It returns name-folded operation latency/energy, source tags, executed MoE communication
measurement substitutions, and optional SOL latency/compute/memory evidence. Decode means one
step at `context_length + 1`; prefill removes the cached prefix. Values precede learned online
correction. Whole-model estimators reject operation decomposition; missing SOL implementations
carry explicit reasons without changing the selected latency estimate.

Replay requests this evidence through `ReplayOutputRequirements(capture_performance_diagnostics=True)`.
`TimingOperationEvidence.details` is optional; providers without it must retain `None` (Rust
struct literals must initialize the new field). The constructor keeps it absent by default.
The CLI's time/source reports sum the observed phase work, including repeated cached timing
queries, and preserve distinct fallback records while folding repeated operation names.
Identical substitutions are deduplicated, so record counts are not execution counts.

## Replay timing evidence

The runtime-neutral `TimingModel` contract exposes optional accumulated
evidence through `evidence_summary()`. Op-level AIC providers return a
`TimingEvidenceSummary` split into prefill and decode phases. Each
`TimingPhaseEvidence` carries accumulated known energy in W-ms, total latency,
latency covered by nonzero energy data, merged provenance, and name-folded
`TimingOperationEvidence` records with the same fields. Missing operation
energy is represented by `None`, never by a synthesized zero. When coverage is
below one, the energy is partial: it includes only the portion with positive
operation-energy evidence. It is not a total-workload energy estimate.
Providers that assemble these public records directly should use
`TimingPhaseEvidence::try_from_operations` and `try_accumulate`; those paths
validate numeric fields and canonicalize covered latency to zero when energy is
missing. Nonempty operation lists must agree with phase totals; a relative
rounding tolerance applies only to this consistency check. The original infallible helpers remain available for already-valid
evidence.

Whole-model FPM timing and the built-in fixed and polynomial timing models are
latency-only and return `None` from `evidence_summary()`. Consumers must keep
that distinction when producing power metrics: absence of evidence is not a
zero-watt prediction. FPM decode timing continues to query the exact total
past-KV coordinate rather than the op-level mean-context coordinate.

## Agentic report source migration (0.13)

The replay report additions require the coordinated 0.13.0 wheel/crate version,
aligned with main's release preparation in PR #268. They must not be released
as a 0.12 patch. Downstream exhaustive Rust `ReplayReport` literals must supply
`agentic_phases: None` for a cold run (or its prepared phase evidence).
Exhaustive `PerRequestRecord` literals must supply `agentic_phase: None` for
cold replay, or `Some(AgenticReplayPhase::Profile)` for measured warmed requests.
Exhaustive destructuring must name these fields or use `..`.

The external-consumer compile fixture `rebuild_replay_report_literals` constructs
both public structs exhaustively against this boundary. JSON consumers retain
the existing cold shape: absent optional phase evidence is not serialized.
This source migration does not change the engine-config/spec or FPM wire schemas.

## Compatibility rules

- The `aisimulate` wheel and `aisimulate-core` crate versions must match for
  every release.
- A breaking `EngineConfig`, `EngineSpec`, or `ForwardPassMetrics` wire change
  must bump its corresponding schema constant. Consumers reject unsupported
  schema versions before using the payload.
- A supported facade name is not removed or given a new required parameter
  without a documented deprecation path. The package is pre-1.0, so an
  unavoidable incompatible API change also requires a minor-version bump.
- The canonical `best_available(config)` API replaces the old positional worker
  role/options signatures and separate estimator constructors. This is a source
  migration: use `ForwardPassPerfModelConfig::new(...)` in Rust or the SDK config
  class in Python, and use the explicit migration helper for saved EngineConfig
  values. Downstream Dynamo callers must migrate before this API's stable release;
  keep the crate and wheel versions aligned at the coordinated minor release.
- [The migration checklist](../.github/release-gates.json) and
  `scripts/check_release_migrations.py` apply before stable publication. Clear the
  pending entry in a reviewed change after downstream validation and merge.
  There is currently no standalone stable-publication workflow in this repository;
  that release process must invoke the checker for both its policy and target
  declarations (`--target-gates`). Missing or malformed declarations fail closed.
- The raw PyO3 class and ergonomic SDK wrapper intentionally share the name
  `RustForwardPassPerfModel`; callers should import from `aisimulate_core.sdk`
  unless they specifically need the JSON-oriented native binding.

The standalone AIC 0.12 compatibility distributions form a separate package
lineage: `aiconfigurator` pins `aiconfigurator-core==0.12.0`, and that core builds
its own native binding. They do not depend on the AISimulate wheel or crate.
Their retained binding declarations are not downstream callers of this API.
The [replacement installation](../README.md#upgrade-from-standalone-aiconfigurator)
removes both old distributions and installs AISimulate's complete migrated
application and core. Namespace compatibility does not preserve the removed
constructor signatures; direct SDK callers must perform the source migration
above. Release gates track consumers of the new artifacts, such as Dynamo.

## CI contract

Every change is checked from three consumer viewpoints:

1. Python source and isolated installed-wheel imports, including the public
   facade, native stub, bundled data, and upper/core ownership boundary;
2. Rust tests with embedding disabled and with all features enabled;
3. a separate workspace crate that depends on `aisimulate-core` and
   compiles only against its public exports.
