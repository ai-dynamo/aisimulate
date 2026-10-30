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

## Hybrid FPM/OP deprecation and migration

The legacy `forward_model="fpm"` route that combines an autoregressive target
FPM with explicit operation-level draft operations is deprecated. Construction
emits a deprecation warning when it retains a draft-operation tail; its existing
timing and graph remain available during migration. Pure operation-level
speculative decoding remains supported. Ordinary autoregressive FPM and n-gram
FPM verification without a draft network are outside this deprecation.

Choose the replacement according to the evidence and caller you have:

| Route | Migration use | Current readiness |
| --- | --- | --- |
| **Option 1: non-SD FPM approximation** | Estimate a complete SD decode iteration from an ordinary STP FPM with query-only `draft_length`. | Available for pure-decode queries through the canonical Rust/Python API. It does not yet replace prefill/mixed, static or Replay calls. |
| **Option 2: measured SD FPM** | Query measurements of the complete target-plus-draft round at their recorded identity. | Available for the restricted measured-round profile contract below, with matching timing and observed memory evidence. |
| Transitional hybrid FPM + OP | Preserve existing applicable OP draft estimates while callers and replacement coverage migrate. | The legacy route remains available with a warning; do not remove its draft cost or add an Option 1 estimate to it. |

The transition preserves applicable OP draft estimates, with Option 1 as the
fallback destination when an ordinary STP FPM can support the request. Today that
fallback requires explicitly constructing an STP model and using the query API
below. `best_available` retains its existing estimator priority: `auto` prefers
OP timing, and explicit `speculation` currently admits only OP candidates. It
does not automatically convert an unsupported speculative configuration into
Option 1, or switch a selected estimator after a query failure. See the
[configuration contract](../configuration.md#speculative-decoding-configuration).

Removal has **no fixed date**. It depends on replacement readiness for affected
callers and workloads: compatible timing coverage, accepted accuracy evidence,
preserved draft memory and acceptance/progress semantics, and migrated caller
queries. Option 1's decode-only API and Option 2's restricted measured profiles
do not establish that readiness for every existing hybrid caller. Retain the
legacy route where those conditions have not been met. Before changing defaults
or removing it, document the supported native FPM replacements, validation
evidence and caller migration plan; coordinate that switch with their availability.

## Estimate speculative decode from an STP FPM

An ordinary single-token-prediction (STP) FPM can also estimate the complete
latency of one speculative decode iteration. Supply `draft_length` on the
existing query API; the model itself remains an STP estimator:

This is **Option 1**. Select an FPM explicitly so construction cannot choose an
OP estimator, and leave `speculation` unset and `nextn` at zero:

```python
from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel

stp_config = ForwardPassPerfModelConfig(
    model="Qwen/Qwen3-0.6B",
    system="h200_sxm",
    backend="vllm",
    worker_type="decode",
    estimation_mode="fpm_interpolation",
    fallback_policy="deny",
)
model = RustForwardPassPerfModel.best_available(stp_config)
metrics = {
    "scheduled_requests": {
        "num_decode_requests": 8,
        "sum_decode_kv_tokens": 32768,
    }
}
sd_iteration_ms = model.estimate_forward_pass_time_ms(metrics, draft_length=3)
detail = model.estimate_forward_pass_detailed(metrics, draft_length=3)
```

The selected model/system/backend must have ordinary STP data; configure the
existing [external FPM path](#external-whole-forward-fpm-data) or resource profile
when needed. The example supplies logical request counts, not the packed OP
verification counters. Do not retain a hybrid draft tail or add another draft
estimate: this query already estimates the whole SD iteration.

The fixed rule is:

$$\widehat T_{\mathrm{SD,iter}} = (0.08d + 1.25)\,F_{\mathrm{STP}}((d+1)B, K_{\mathrm{total}}).$$

- `d` is the positive integer draft length, with a chain verify width of `d+1`.
- `B` is each rank's logical decode request count; `K_total` is its total past
  KV token count, excluding the current input tokens. The STP reference uses
  `(d+1)B` virtual requests and preserves `K_total` exactly. With per-request
  feature lists, each original KV length is split evenly across `d+1` virtual
  requests, distributing integer remainders without dropping tokens; their
  STP extend lengths are one. This is an empirical mapping, not an equivalent
  attention graph.
- The result is **the complete SD decode iteration**, including draft and
  verification work. It is not verification-only or time per accepted token.
  Acceptance/progress and draft memory remain separate concerns. The STP
  reference is not an additional executed stage.

There is no model, GPU or SD-algorithm allowlist and no draft checkpoint is
required. Interpolation (direct or SOL-assisted), regression and learned FPM
estimators use the same rule, applying the multiplier once after the STP
prediction, its existing corrections and rank reduction. The example queries
STP at batch 32 and total KV 32768, then multiplies by 1.49. The widened
reference must still be supported by the selected estimator: missing data,
missing learned stores and regression readiness retain their normal behavior.

Omitting `draft_length` preserves the existing STP API and result schema.
Supplying it to the detailed query returns `latency_ms`, `draft_length`,
`scaling_factor` and the original `stp_reference` estimate/evidence. Recorded
STP observations are never relabelled as measured SD timings. This per-query
option does not modify the model, cache identity, saved configuration or
tuning state; `tune_with_fpms` continues to take STP observations. The Rust
counterparts are `estimate_speculative_forward_pass_time_ms(metrics, d)` and
`estimate_speculative_forward_pass_detailed(metrics, d)`.

V1 accepts pure decode iterations (including idle ranks). Prefill/mixed work,
operation-level estimators, hybrid draft graphs and already-measured SD-round
estimators are rejected. This option is a prediction API, not a new replay
executor or CLI deployment switch. Ordinary replay SD controls and measured
SD-round profiles retain their existing contracts.

### Calibration and limitations

This empirical approximation was compared with two models (native MTP and
EAGLE3) on H100, H200, B200 and GB200 at depths 1–4. Other models, depths and
production workloads are extrapolation; measured-reference errors exclude
actual FPM prediction error. Model/method and runtime/backend differences
confound comparisons. See the committed [calibration evidence](../../../benchmarks/evidence/sd-fpm-depth-v1/README.md)
for the setup, exclusions, error statistics, chart and provenance.

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
cell keys. Consumption requires explicit `estimation_mode: fpm_interpolation`,
`method: direct`, `database_mode: SILICON`, `fallback_policy: deny`, and `nextn`
equal to `num_draft_tokens`. Ordinary AR profiles cannot supply SD-round timing.

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
admit AR-labelled rows (use `vllm_native_ar_model_step_cuda_v1`) or
scheduler/host wall-time rows; renaming a CPU clock is not a valid conversion.
Cached prefill and decode must retain the existing `real_kv` provenance
requirement. Missing or fabricated state cannot be healed into these
measured-only cells.

Autoregressive native CUDA samples use schema 7 and the separate
`vllm_native_ar_model_step_cuda_v1` policy. They require the full text AR
execution identity, `timing_scope: model_step_cuda`, and the same sample fields
listed above. Decode and cached-prefill rows must carry
`kv_seed_regime: real_kv`. This policy records the complete model step,
including sampling. It cannot supply speculative-round timing. Keep these
observations separate from Dynamo scheduler wall-time measurements.

### Consuming a measured round

This is **Option 2**. Use the same canonical `ForwardPassPerfModel.best_available(config)` path as
ordinary FPM profiles. Both prefill and decode cells must carry the identical
execution binding. The decode lookup uses request count and actual total past
KV tokens, without multiplying the request count by `nextn + 1`. The measured
latency already includes target verification, sampling and draft work; no draft
operation tail is added. Prefill must likewise use the matching measured SD
prefill cell, not a relabelled AR prefill cell.

For a reviewed profile and table whose identity matches the values below:

```python
import json
from pathlib import Path

from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel

sd_config = ForwardPassPerfModelConfig(
    model="Qwen/Qwen3-8B",
    system="h200_sxm",
    backend="vllm",
    backend_version="0.24.0",
    worker_type="decode",
    fpm_profile=json.loads(Path("/data/measured-sd-profile.json").read_text()),
    nextn=3,  # Must match the profile's recorded draft depth.
    estimation_mode="fpm_interpolation",
    fallback_policy="deny",
    database_mode="SILICON",
    estimator_config={
        "fpm_interpolation": {
            "method": "direct",
            "fpm_parquet_path": "/data/measured-sd.parquet",
        },
    },
)
model = RustForwardPassPerfModel.best_available(sd_config)
sd_iteration_ms = model.estimate_forward_pass_time_ms(metrics)
```

Use the logical decode metrics from Option 1 above. Omit `draft_length` and
`speculation`: measured rounds already include their draft cost, and STP scaling
or an OP draft tail would count it again. The paths above are user-supplied
artifacts, including the table's required metadata sidecar; this example does
not assert packaged measured-SD coverage for that identity.

Resources must use a linear cache with observed `runtime_memory`. The recorded
usable KV pool already reflects draft and graph allocations; capacity sizing
does not subtract them again. Pending or analytically declared memory alone is
insufficient. Grouped caches, generic `speculation` configurations, recorded
DCP and non-pure-TP topologies remain unsupported on this route. Use `nextn`
with the exact recorded `execution_profile`; generic speculation cost graphs
must not be composed with measurements that already include the draft work.

Supported queries are pure-prefill and pure-decode `ForwardPassMetrics`, prefill
static/detailed queries, and replay's prefill and decode queries with actual
iteration totals. Raw mixed FPM, static multi-round generation, per-operation
and ad-hoc operation queries fail explicitly. Ordinary replay with a measured
round retains its existing separate P/D phase composition; this is an
approximation for mixed iterations, not a measured mixed-round model. Partial-width decode rounds still
use the configured depth's timing. Enable query coverage to identify holes
rather than extrapolating.

The SD identity does not expand the guided `onboard` collector or its replay
validation scope. Prepare and validate the measured table separately, then use
ordinary `predict` with the [replay SD controls](../../replay/engine/speculation.md#measured-sd-round-fpm).

## Direct and SOL-assisted methods

- `method: direct` needs a verified resource profile and `database_mode: SILICON`.
  It does not build an analytical graph. Queries require exact measurements or
  supported measured brackets; there is no unmeasured extrapolation.
- `method: sol` needs a registered analytical architecture. It uses the
  supported native FPM SOL evaluator to transfer from measured anchors.
- `method: auto` chooses SOL for registered architectures, or direct for a
  profile with verified but unregistered architecture metadata. Without a
  profile it retains SOL.

Direct prefill first uses an exact measurement or interpolates new-token counts at fixed KV, then brackets KV using curves that cover the requested token count. If this cannot bracket the query, it interpolates KV at each fixed measured token count and then uses the nearest covered token counts on either side of the query. The second order can cover irregular collection boundaries where different prompt lengths have different maximum KV coordinates. It preserves every result and supporting measurement from the first order when that order succeeds.

Both orders stay within measured bounds at the same batch size and selected cell. Neither uses SOL, synthetic `fake_fallback` rows, or extrapolation. The fallback is part of `method: direct` and needs no extra setting; decode interpolation is unchanged. Detailed evidence reports `cross_kv_interpolation` for the usual token-first order and `kv_first_interpolation` for the fallback, with weights flattened to original measurements. Additional query coverage does not establish interpolation accuracy or matching CUDA graph execution regimes.

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
| `predict_prefill_latency_total(batch_size, total_prefill_tokens, total_kv_read_tokens)` | Exact tokens computed and cached or earlier-chunk KV read across the prefill batch. Whole-forward FPM without operation-level draft tails queries these totals directly, so a ragged batch is never priced above its scheduled token count. Other graphs use the integer means of `predict_prefill_latency`; execution profiles reject totals that do not divide evenly across the batch. |
| `predict_decode_latency_total(batch_size, total_past_kv_tokens)` | Exact total past-KV across the decode batch, preserving the native FPM coordinate without a mean-context conversion. |
| `fpm_decode_kv_ceiling()` | Largest collected decode KV total, or `None`; reading this bound does not record a timing lookup or prove shape-specific coverage. |

These methods preserve the existing engine's phase behavior and record actual
direct lookup evidence when enabled. Empty work does not invent a measured
query. Whole-iteration `estimate_forward_pass_time_ms()` remains the API for
scheduled per-rank FPM telemetry, including mixed work and its existing online
correction behavior.

Ordinary replay uses this returned model for FPM timing and emits
`fpm_query_coverage` in its report when collection is enabled. A vLLM or
TensorRT-LLM replay pass that computes prefill alongside decode steps runs as
one forward. Replay charges the prefill at the pass's exact totals plus only
the latency the decode requests add, `max(0, decode - decode_pass_baseline)`
at their totals. A tuned model applies the online correction of that mixed
iteration to both parts, so the pass costs what
`estimate_forward_pass_time_ms()` returns for it. Measured SD-round profiles,
MTP drafts (`nextn > 0`) and operation-level draft tails define no such decode
share; replay prices their decode work as a separate decode pass. The CLI also
saves `fpm-coverage.json`, including partial evidence when a native timing
error stops replay; the Python exception retains a `fpm_query_coverage`
attribute for that failure path. A passing replay coverage status requires completed requests,
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

For independent full-grid repetitions, use the existing
`per_row_single_sample_or_full_grid_median_v1` sidecar policy. Every row must
declare `dynamo_native_single_sample_v1`/1 or
`independent_full_grid_median_of_3`/3. The latter records the producer's median
of three independent full-grid measurements; it is a different procedure from
`kvwarm_median_of_3`, and the two per-row sidecar policies cannot be exchanged.
Other repetition counts and unknown row policies are rejected.

Prediction needs only the aggregated Parquet table and its metadata sidecar.
The loader preserves the stored latencies and checks the pair's digest, row
count, identity, workload and duplicate coordinates, as well as the declared
row policy/count pairs. It does not open raw campaign files, recompute medians
or assess repeatability and serving accuracy. The publisher must validate
aggregation and quality evidence before publication; successful loading does
not qualify the dataset's accuracy.

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
