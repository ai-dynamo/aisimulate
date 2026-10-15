<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Whole-forward FPM

Whole-forward FPM consumes measured iteration latency from a Parquet file and
its same-stem `.metadata.json` sidecar. It does not need regression training.
Its exact cell identity includes the model, system, backend/version, parallelism,
and precision; model-specific execution identities must also match.

Use [FPM self-service](../fpm-self-service/README.md) to prepare, review, collect
or import a profile. The existing [implementation guide](../fpm-self-service/implementation.md)
owns artifact validation, publication, checkpoints, and recovery.

## Recorded speculative-round identity

A profile deployment may record the following optional identity. The parent
profile's `model_revision` and the draft revision must be lowercase 40-hex
immutable checkpoint commits. This schema describes a **complete target-plus-draft
round**, not an AR target-verification approximation:

```yaml
execution_profile:
  kind: speculative_round
  method: eagle3
  num_draft_tokens: 3
  draft_model: example/draft-model
  draft_revision: "2222222222222222222222222222222222222222"
  timing_scope: model_step_cuda
```

The initial identity contract permits vLLM pure TP with DP1/PP1/CP1 and 1–5
draft tokens. Acceptance length belongs to replay progress, not this timing
selector. Omitting `execution_profile` preserves existing AR profile JSON and
cell keys. Identity publication alone does not enable execution: canonical
construction rejects these profiles with either `nextn: 0` or nonzero `nextn`
until the corresponding consumer is supported.

Rust produces the exact table `execution_profile` string by prefixing
`speculative_round_v1:` to its canonical serialization of
`{"model_revision": ..., "execution_profile": ...}`. The descriptor's field
order is `kind`, `method`, `num_draft_tokens`, `draft_model`, `draft_revision`,
`timing_scope`. Use the profile's `execution_identity(model_revision)` helper;
do not hand-build a selector or substitute another checkpoint, draft, depth,
or timing boundary. The resulting four-column suffix is
`["", selector, "none", "text"]`; the existing Op layout and binary schema do
not change. Malformed, unknown-version, or noncanonical speculative selectors
are invalid data, not a fallback opportunity.

### Native-CUDA measured rows

Speculative-round rows require schema 7, backend `vllm`, sidecar
`measurement_policy: vllm_native_fpm_model_step_cuda_v1`, and sidecar
`timing_scope: model_step_cuda`. Every row carries that same measurement policy
and scope, plus:

| Field | Meaning |
|---|---|
| `measurement_reduction` | `single_sample` or `actual_upper_median` |
| `source_sample_count` | Integer 1 for a raw sample; at least 2 for an actual upper median |
| `latency_ms` | The selected real observation's complete native CUDA elapsed time |

An actual upper median selects a real sample and retains **that sample's own
workload coordinates**. It does not average latencies, use an unmeasured nominal
coordinate, or divide the recorded time by the sample count. Preserve raw
observations and selection provenance alongside the table. The policy does not
admit AR-labelled rows or scheduler/host wall-time rows; renaming a CPU clock is
not a valid conversion. Cached prefill and decode must retain the existing
`real_kv` provenance requirement. Missing or fabricated state cannot be healed
into these measured-only cells.

## Direct and SOL-assisted methods

- `method: direct` needs a verified resource profile and `database_mode: SILICON`.
  It does not build an analytical graph. Queries require exact measurements or
  supported measured brackets; there is no unmeasured extrapolation.
- `method: sol` needs a registered analytical architecture. It uses the
  supported native FPM SOL evaluator to transfer from measured anchors.
- `method: auto` chooses SOL for registered architectures, or direct for a
  profile with verified but unregistered architecture metadata. Without a
  profile it retains SOL.

Readiness is phase-specific: prefill needs real prefill rows, decode needs real
decode rows, and aggregated needs both. A ready table still can lack an individual
query's measured domain. Missing timing and pending [resource memory](../memory.md)
are independent: planning can validate a pending profile, but cache sizing and
Replay need finalized resource evidence.

### External whole-forward FPM data

Set `estimator_config.fpm_interpolation.fpm_parquet_path` on the canonical configuration to use an external parquet and its required same-stem `.metadata.json` sidecar:

```python
from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel

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
[FPM replay validation](../fpm-self-service/implementation.md#validate-fpm-query-coverage-with-agentx-replay)
for the stricter whole-corpus completion checks and saved onboarding artifacts.

### DCP self-benchmark profiles

`dcp` is an optional recorded decode-context-parallel dimension within TP.
It must be positive and divide `tp`; it does not multiply GPU or MoE group
counts. Missing DCP and explicit DCP1 remain distinct FPM identities. An
explicit DCP8 request cannot consume ordinary TP8 or unrecorded-DCP data.
For whole-forward timing, DCP greater than one requires a measured vLLM
`fpm_interpolation` profile with the matching identity. Regression and
non-vLLM whole-forward interpolation reject DCP greater than one. Op-level
DCP is a separate, supported path for model/backend combinations with native
sharded-attention and collective modeling; see [DCP scope](../configuration.md#decode-context-parallelism).
Within whole-forward FPM, SOL-dependent transfer paths can still reject DCP;
this is not a blanket rejection of op-level analytical DCP.

FPM v6 accepts optional `dcp` in the Parquet identity and sidecar selector.
The loader also accepts `per_row_single_sample_or_median_of_3` sidecars when
every row declares a consistent `measurement_policy`/`measurement_repeats`
pair: `dynamo_native_single_sample_v1`/1 or `kvwarm_median_of_3`/3. Values are
already aggregated by the producer; the loader preserves their latency.

Schema 7 also accepts two backend-bound median policies, each tied to its own
state protocol and timing boundary:

| Backend | `measurement_policy` | `state_protocol` | `timing_boundary` |
|---|---|---|---|
| vLLM | `vllm_native_fpm_prefix_seed_median_v1` | `glm53flash_prefix_cache_real_seed_v1` | `dynamo_vllm_instrumented_scheduler_fpm_wall_time` |
| SGLang | `sglang_native_fpm_radix_seed_median_v1` | `glm53flash_sglang_radix_real_seed_v1` | `sglang_native_fpm_rank0_device_timer_wall_time` |

These policies preserve the producer's already-reduced `latency_ms`; the
consumer neither averages it again nor divides it by the repeat count. The
sidecar and every row must declare integer counts of 5 warmups and 10
measurements, `global_warmup_iterations=0`, and the policy's protocol and
boundary. Missing or inconsistent median metadata and `fake_fallback` KV rows
are rejected before interpolation; the existing execution-identity and real-KV
requirements still apply. The legacy `dynamo_native_single_sample_v1` contract
is unchanged.

For a Kimi K3 text profile, an explicit configuration can be:

```python
from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel

config = ForwardPassPerfModelConfig(
    model="moonshotai/Kimi-K3", system="gb300", backend="vllm",
    backend_version="0.29.0", worker_type="aggregated",
    tp=8, pp=1, attention_dp=1, moe_tp_size=8, moe_ep_size=1, dcp=8,
    gemm_quant_mode="bfloat16", moe_quant_mode="w4a16_mxfp4",
    kvcache_quant_mode="fp8", attention_backend="FLASHINFER_MLA",
    estimation_mode="fpm_interpolation", fallback_policy="deny",
    systems_paths=("/absolute/profile/systems",),
    estimator_config={
        "fpm_interpolation": {
            "text_only": True,
            "unrecorded_quant_modes": ["fmha", "comm"],
        },
        "correction": {"enabled": False},
    },
)
model = RustForwardPassPerfModel.best_available(config)
```

Only use `unrecorded_quant_modes` for fields that the selected profile actually
leaves unrecorded. The runtime `FLASHINFER_MLA` label is retained for exact FPM
matching while the compiler uses its internal FlashInfer backend description.
The model architecture remains registered once; adding a parallel configuration
does not require another model class.

Place the reviewed primary pair at
`systems/data/gb300/vllm/0.29.0/fpm_forward_perf.{parquet,metadata.json}` and copy
the matching hardware YAML into the systems root. Keep source hashes and the
pinned dataset revision with the profile. Do not combine synthetic-attention
boundary points or nonuniform layouts with a balanced primary profile.

Replay YAML passes recorded DCP through
`engine.workers.<role>.parallelism.decode_context`. Per-worker `timing` accepts
the canonical quant-mode fields and `attention_backend`, alongside estimator
selection and controls. DCP FPM replay requires explicit fixed KV block capacity;
automatic DCP/hybrid capacity sizing is not implemented. Host offload or P/D
transfer also requires explicit KV bytes per token with DCP.
`decode_context` is currently supported by AISimulate's `--stack engine` only.
Dynamo Replay and Planner do not yet support this field; their configuration
propagation, cache identity, and dependency version need a downstream update.
These timing precision/backend overrides are prediction-only; recommendation
rejects them until its feasibility preflight supports the same identity.
Supplying a fixed pool does not add KDA checkpoint, eviction, or chunk-alignment
fidelity to the generic Replay cache/scheduler. This API change enables timing
consumption, not full hybrid-cache simulation or multimodal prediction from
text-only measurements.

The positional engine-spec wire format is version-gated. Recompile older
binary EngineSpecs with the matching wheel/crate; incompatible versions are
rejected before payload decoding. Configuration and profiles without DCP
remain accepted as unrecorded DCP.

Detailed per-query support and weights are described in the [Python API](../api/python.md#direct-fpm-query-evidence).
