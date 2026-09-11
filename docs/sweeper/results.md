---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Sweeper Results
subtitle: A lossless candidate ledger with scalar and Pareto views
---

> [!WARNING]
> **Experimental.** Schema version `1.0` is the first machine-readable Sweeper result contract.
> Within a schema version, fields keep their meaning and units; incompatible changes require a new
> `schema_version` and an explicit converter.

`SweepResult` is the canonical result envelope for both exhaustive and optimizer-guided execution.
The complete candidate ledger is the source of truth. Scalar top-N and Pareto fronts are views that
refer to ledger rows by stable `candidate_id`; they do not discard the other evaluated candidates.
The unified `aisimulate recommend` command writes this envelope as `recommendation.json` beside
the selected `recommendations/*.yaml` prediction inputs.

```python
result = sweeper.run(config, top_n=5, candidate_retention="all")

result.to_json()                 # lossless interchange
result.to_csv()                  # documented analysis view
result.counts.failed             # machine-readable outcome counts
result.selected_candidates       # top-N or Pareto candidates
```

`Sweeper.run(config)` returns `SweepResult`, the single public execution result. Callers that need
only the scalar top-N or Pareto selection use `result.selected_candidates`; rejected candidates,
run provenance, and counts remain available on the same result envelope.

Strict aggregate SLA filtering happens before scalar ranking or Pareto dominance. Rejected candidates
remain in the ledger with status `infeasible` and reason category `sla_constraint`.

## Envelope

| Field | Meaning |
|---|---|
| `schema_version` | Result contract version. The only accepted value in this release is `1.0`. |
| `candidate_retention` | `all`, `feasible`, or `views`; counts always describe the complete run. |
| `counts` | Outcome and cache counts for the complete run. |
| `candidates` | Retained candidate records in evaluation order. |
| `views.top_n` | Best-first scalar candidate IDs, limited by `top_n`. |
| `views.pareto_front` | Non-dominated candidate IDs in frontier order. |
| `provenance` | Search strategy, run ID/time, implementation, input fingerprint, and validated input config. |

`views.top_n` and `views.pareto_front` are mutually exclusive. The unified CLI applies final adapter
canonicalization and concrete-config deduplication before writing the active view, so each selected
candidate ID maps one-to-one to one numbered prediction YAML. An empty result has empty views and an
empty candidate list, while its counts and run provenance remain present.

### Candidate record

Every materialized or capability-gated candidate attempt has one record when retention is `all`.

| Field | Meaning |
|---|---|
| `candidate_id` | Stable within the result, formatted `candidate-NNNNNN`. |
| `status` | `feasible`, `infeasible`, `unsupported`, `timed_out`, or `failed`. |
| `config` | Concrete backend, topology, engine knobs, load, and adapter configuration available at the terminal status. |
| `prediction_config` | Concrete public `aisimulate predict` configuration when produced by the unified CLI. |
| `used_gpus` | Provisioned GPU count, or `null` if materialization failed before it was known. Unit: GPUs. |
| `score` | Scalar score normalized so larger is better; for a Pareto row it is the first objective's raw value. |
| `metrics` | Normalized replay metrics in natural units for every attempt that completed replay, including attempts later classified infeasible. Empty only when no valid replay report exists. |
| `objectives` | Natural-unit Pareto objective values, otherwise `null`. |
| `reason_category` | Stable category for a non-feasible candidate. |
| `reason` | Human-readable detail; consumers branch on `reason_category`, not this text. |
| `provenance` | Model, hardware, backend/version, topology, workload, objective/SLA, data, power, and per-operation evidence. |

Metric names and units are explicit: throughput is `*_tok_s`, latency is `*_ms`, energy is `*_j`,
power is `*_w`, duration is `duration_ms`, and `gpu_hours` is GPU-hours. `score` is not assumed to
have a unit; use the named metric or `objectives` for display and comparisons.

### Counts

`evaluated` is the number of candidate attempts that reached materialization or replay and equals
`feasible + infeasible + timed_out + failed`. `unsupported` is separate because capability gating
rejects it before evaluation. `cache_hits` counts repeated optimizer suggestions served from the
run-local completed-result cache or coalesced with an identical suggestion in the same ask batch.
Each such suggestion consumes a trial budget slot and is counted explicitly as a cache hit, but does
not create a duplicate ledger row.

| Status | When used |
|---|---|
| `feasible` | Replay succeeded, required metrics were present, and budget gates passed. |
| `infeasible` | A modeled constraint such as GPU budget, KV capacity, or strict aggregate SLA rejected the candidate. |
| `unsupported` | The runner does not support the backend/topology pair. |
| `timed_out` | Replay exceeded `max_eval_seconds`. It remains infeasible to the optimizer, but is distinct in results. |
| `failed` | Materialization, runner execution, or the runner/result contract failed. |

Stable reason categories are `gpu_budget`, `kv_capacity`, `sla_constraint`, `backend_topology`, `runtime_timeout`,
`candidate_materialization`, `replay_runtime`, `runner_contract`, `invalid_metrics`, `no_samples`,
`parallel_projection`, `adapter_constraint`, and `unknown`.

## Provenance

Run provenance stores the complete validated `SmartSearchConfig` and a SHA-256 fingerprint of its
canonical JSON. It also distinguishes `optimizer_guided` from `exhaustive` execution.

Candidate provenance repeats the context required to interpret a row:

- model and hardware identifiers;
- backend and resolved performance-model version;
- concrete agg/disagg topology;
- the concrete materialized replay workload (including effective concurrency), objective, and SLA payloads;
- exact per-role performance-model identity from `BackendDeploymentSpec.performance_model_metadata`
  plus performance-data source records from runner metadata;
- power/energy measurements from report metrics or runner metadata;
- operation-level source and version records; and
- the complete validated runner metadata payload.

Runner authors may populate `ReplayReport.metadata.performance_data`, `.operations`, and `.power`.
Unknown JSON metadata is preserved under `runner_metadata`, so provenance is not lost when a newer
runner supplies evidence an older consumer does not yet promote into typed fields. Candidate
provenance is derived from the concrete `ReplaySpec`, not the run-level search domain or timing-model
heuristics. Pre-materialization rejections have no replay specification and therefore retain only the
concrete fields known at their rejection point; the full input domain remains in run provenance.

## JSON and CSV

`SweepResult.to_json()` is the lossless interchange format. It rejects non-finite numbers, sorts
mapping keys, includes empty-run provenance, and round-trips through `SweepResult.from_json()`.

`SweepResult.to_csv()` is a one-row-per-retained-candidate analysis view with these columns:

`schema_version`, `candidate_id`, `status`, `reason_category`, `reason`, `used_gpus`, `score`,
`config_json`, `prediction_config_json`, `metrics_json`, `objectives_json`, `provenance_json`,
`is_top_n`, and `is_pareto`.

Nested fields are canonical JSON cells rather than lossy dotted columns. CSV is not the interchange
format: an empty CSV has only its header and therefore cannot carry run provenance or counts.

## Legacy AIC mapping

The legacy AIC `ColumnsAgg`, `ColumnsDisagg`, `ColumnsAggEpd`, and `ColumnsAFD` DataFrame rows map to
candidate records as follows. Parenthesized role prefixes become explicit prefill/decode/encoder or
attention/FFN keys in `provenance.topology` and `config`; they are not retained as punctuation-based
field names.

| Legacy DataFrame/CLI field or artifact | Canonical field | Conversion |
|---|---|---|
| `best_config_topn.csv` row | `candidates[]` plus `views.top_n[]` | Convert the row, then place its ID in best-first view order. |
| `pareto.csv` row | `candidates[]` plus `views.pareto_front[]` | Convert the row, then place its ID in frontier order. |
| `model` | `provenance.model` | Exact string. |
| `system`, `(p)system`, `(d)system` | `provenance.hardware`, `provenance.topology` | A homogeneous system is candidate hardware; role-specific systems remain role topology fields. |
| `backend`, `(p)backend`, `(d)backend` | `provenance.backend`, `config` | Exact backend strings; heterogeneous role values remain explicit in `config`. |
| `version`, `(p)version`, `(d)version` | `provenance.backend_version`, `config` | Exact resolved versions; heterogeneous versions remain role-specific in `config`. |
| `isl`, `osl`, `prefix`, `concurrency`, `request_rate` | `provenance.workload` and concrete `config` load | Token counts are tokens; rate is requests/s. |
| `tp`, `pp`, `dp`, `moe_tp`, `moe_ep`, `cp`, role-prefixed variants | `provenance.topology` and `config` | Integers copy without reinterpretation; `dp` maps to `attention_dp`. |
| `workers`, `replicas`, `(p)workers`, `(d)workers`, `(e)workers` | `provenance.topology` and `config` | Counts of role replicas. |
| `bs`, `global_bs`, role-prefixed variants | `config` batching fields | Preserve concrete per-worker/global batching values. |
| `num_total_gpus` | `used_gpus` | Integer GPUs. |
| `ttft` | `metrics.mean_ttft_ms` | Milliseconds. |
| `tpot` | `metrics.mean_tpot_ms` | Milliseconds/token. |
| `request_latency` | `metrics.mean_e2e_latency_ms` | Milliseconds. |
| `encoder_latency`, `encoder_memory` | `metrics` and encoder fields in `provenance.topology` | Milliseconds and bytes/GiB as declared by the legacy source schema. |
| `tokens/s` | `metrics.output_throughput_tok_s` | Output tokens/s. |
| `tokens/s/gpu`, `tokens/s/gpu_cluster` | `objectives.throughput_per_gpu` or derived display field | Tokens/s/GPU; retain the source numerator and GPU normalization evidence. |
| `tokens/s/user` | `metrics.mean_output_token_throughput_per_user` | Tokens/s/user. |
| `seq/s`, `seq/s/gpu`, role worker rates | `metrics` | Sequences/s, with the legacy label preserved in migration metadata until a typed metric is added. |
| `balance_score`, `num_ctx_reqs`, `num_gen_reqs`, `num_tokens`, `ctx_tokens`, `gen_tokens` | `metrics` | Exact numeric values; request/token counts are counts. |
| `power_w` | `provenance.power.power_w` | Watts. |
| `gemm`, `kvcache`, `fmha`, `moe`, `comm`, `memory`, role variants | `metrics` | Legacy component estimates remain named metrics with original units recorded by the converter. |
| EPD `(a)workers` and `(e)workers`, `(e)tp`, `(e)pp`, `(e)bs`, `(e)parallel`, `(e)memory` | `config` and `provenance.topology` | Preserve the rate-matched aggregate and encoder cell as explicit roles. |
| AFD `phase`, `(a)nodes/tp/bs/micro_bs/workers`, `(f)nodes/tp/ep/workers` | `config` and `provenance.topology` | Preserve attention/FFN role topology and whether AFD applies to prefill, decode, or both. |
| AFD `t_a_layer`, `t_f_layer`, transfer/collective times, `t_step`, `balance_ratio`, `comm_hidden`, and prefill/decode variants | `metrics` | Preserve per-phase numeric values and source units; do not select one headline value for `phase=both`. |
| Legacy `is_oom` role flags | `status`, `reason_category`, `reason` | Map an OOM candidate to `infeasible`; preserve the role and memory detail in the reason/config. |
| `_per_ops_source` and saved `per_ops_source.json` | `provenance.operations` | One typed operation/source/version record per entry. |
| CLI warning/skip text | `status`, `reason_category`, `reason` | Convert known gates to stable categories; preserve original text as detail. |
| Missing/empty DataFrame | empty `SweepResult` | Preserve zero counts and run provenance instead of returning only `None`/empty output. |

Approved migration exceptions:

- Legacy DataFrames round many display columns to three decimal places. Canonical JSON retains the
  unrounded replay float; rounding is a presentation concern.
- `parallel` is a legacy display string. Canonical topology fields are authoritative; a converter
  may preserve the display string under runner metadata.
- Legacy component columns do not consistently declare units. A converter preserves their names and
  records the source schema instead of guessing units.
- PNG plots and generated deployment artifacts are not embedded. Plots derive from result views;
  deployable artifacts remain owned by the generator component and reference a selected candidate ID.

## Examples

Scalar result: two feasible rows may be retained while `views.top_n` contains only the best row.

```json
{
  "counts": {"evaluated": 2, "feasible": 2, "infeasible": 0, "unsupported": 0,
             "timed_out": 0, "failed": 0, "cache_hits": 0},
  "views": {"top_n": ["candidate-000002"], "pareto_front": []}
}
```

Pareto result: dominated feasible rows remain in the ledger but only non-dominated IDs are in the view.

```json
{"views": {"top_n": [], "pareto_front": ["candidate-000004", "candidate-000001"]}}
```

Empty result: no candidate was feasible, but failure evidence is still machine-readable.

```json
{
  "counts": {"evaluated": 3, "feasible": 0, "infeasible": 0, "unsupported": 1,
             "timed_out": 1, "failed": 2, "cache_hits": 0},
  "views": {"top_n": [], "pareto_front": []}
}
```

Mixed-mode result: agg and disagg rows share one ledger. Each candidate's
`provenance.topology.deployment_mode` identifies its branch; scalar ranking or Pareto dominance can
operate across branches without separate result schemas.

Failure row:

```json
{
  "candidate_id": "candidate-000003",
  "status": "failed",
  "reason_category": "runner_contract",
  "reason": "goodput objective requires goodput_output_throughput_tok_s"
}
```

## Replay specification

The result contract complements, rather than replaces, `ReplaySpec`. `ReplaySpec` version 1 contains
the concrete backend deployment, workload, goal, execution mode, concurrency, adapter configuration,
and runtime hooks. `RunnerCapabilities.require_compatible` checks it before execution, and
`canonical_json` creates deterministic strict JSON for the replay boundary.

Exact repeated suggestions reuse a result from the current `run` call. The cache does not persist
between calls, even when the same `Sweeper` instance is reused.

## Deployment Artifact Generation

A `Candidate` is a ranked simulation result, not a deployment manifest. The downstream
AIConfigurator generator owns artifact rendering. In the unified AISimulate application, pass the
selected candidate and its matching workload to
`aiconfigurator.generator.request.from_sweeper_candidate`, then render the resulting typed request
with `aiconfigurator.generator.api.generate_from_request`.

The bridge preserves evaluated engine limits and supported adapter configuration, and rejects
candidate data it cannot lower without loss. Pareto output has no implicit winner: callers must
select one point before requesting deployment artifacts.

AFD candidates are intentionally outside that native generator bridge because its renderers have
no A/F worker or routing contract. Pass a selected AFD recommendation to `aisimulate predict`
instead; the prediction writes deterministic `afd-replay-spec.json` and
`afd-qualification.json` analytical artifacts and marks native launch generation unsupported.
