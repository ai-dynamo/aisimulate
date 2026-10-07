---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Sweeper Results
subtitle: A lossless candidate ledger with scalar and Pareto views
---

> [!WARNING]
> **Experimental.** Schema version `1.1` adds explicit resource-limited candidate outcomes; `1.0` remains readable.
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
With `min_gpus`, measured goodput below the requested rate floor is `infeasible` with
`load_constraint`; missing rate evidence is `failed` with `runner_contract`. Candidate metrics
retain `request_throughput_rps`, `goodput_request_throughput_rps`, and `goodput_completed_requests`
when supplied by the runner. GPU-count selection happens before the top-N view is truncated.

## Envelope

| Field | Meaning |
|---|---|
| `schema_version` | Result contract version. New results emit `1.1`; reading `1.0` explicitly upgrades the envelope and defaults resource-limited counts to zero. |
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
| `status` | `feasible`, `infeasible`, `unsupported`, `timed_out`, `failed`, or `resource_limited`. |
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

`power_w` and `power_coverage` follow the
[modeled-power contract](../perf-model/power.md): active-forward-pass power per GPU,
energy-over-active-latency aggregation, and AIC's existing coverage rule.
Coverage is the share of modeled active time with operation-energy evidence.
For every candidate with a valid replay report, both keys must be present in
metrics and power provenance. Exactly 90% is sufficient to publish numeric
`power_w`; below 90%, `power_coverage` remains numeric while `power_w` is `null`.
A runner without typed operation-energy evidence must return `null` for both
values. Zero coverage is reserved for an energy-aware path with no covered
active latency. Null values must never be treated as zero watts or zero coverage.
Failed attempts without a valid replay report retain the empty `metrics`
envelope described above; they do not contain a power summary.

Valid replay reports preserve both nullable keys through runner normalization,
candidate metrics, power provenance, and `SweepResult.to_json()`. Publication
validation rejects numeric watts below the coverage gate. Null values never
become zero or enter objective arithmetic.

This contract applies to AISimulate `ReplayReport` and `SweepResult` outputs.
Raw DataFrames from the compatibility `aiconfigurator` sweep/picking APIs retain
their legacy schema and sentinels; the [AIC mapping](../aic-backward-compatibility/migration.md#legacy-aic-result-mapping) describes conversion targets,
not an automatic converter. They do not satisfy this power contract as-is. A
converter must establish coverage and preserve unavailable values before emitting
a conforming result; it cannot infer coverage from a legacy wattage column alone.

### Counts

`evaluated` is the number of candidate attempts that reached materialization or replay and equals
`feasible + infeasible + timed_out + failed`. `unsupported` is separate because capability gating
rejects it before evaluation. `resource_limited` is also separate: the host could not complete
an evaluation, so it is not a modeled constraint failure. `cache_hits` counts repeated optimizer suggestions served from the
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
| `resource_limited` | Host memory admission or bounded runtime recovery could not complete this candidate. |

Stable reason categories are `gpu_budget`, `kv_capacity`, `sla_constraint`, `load_constraint`, `backend_topology`, `runtime_timeout`,
`candidate_materialization`, `replay_runtime`, `runner_contract`, `invalid_metrics`, `no_samples`,
`parallel_projection`, `adapter_constraint`, `resource_limit`, and `unknown`.

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

## Replay and deployment boundaries

Exact repeated suggestions reuse a result within the current `run`; the cache
is not shared between runs. The [runner ABI](../adapters/runner-abi.md) owns
`ReplaySpec` and normalized report obligations. A ranked candidate is not a
serving manifest; use [deployment generation](deployment-generation.md) to
render supported candidates, and choose one point explicitly for a Pareto result.

## Host resource interruptions


`resource_limited` candidates have reason category `resource_limit` and no
simulated metrics or score. `counts.resource_limited` is separate from
`counts.evaluated`: a host admission refusal is not evidence about model
feasibility. A recommendation with resource-limited candidates covers only
completed evaluations. The CLI preserves completed results and exits with
status 3 to make this partial coverage visible. See [local execution resources](../reference/local-resources.md).

<a id="recommendation-directory"></a>

## CLI recommendation directory

```text
<output-dir>/
├── recommendation.json
├── recommendation.csv
├── resource-plan.json             # on refusal before the sweep starts
├── resource-runtime.json
├── execution-events.jsonl
└── recommendations/
    ├── 0001.yaml
    ├── 0002.yaml
    └── ...
```

- `recommendation.json` is the canonical lossless result (schema 1.1, with explicit upgrade of 1.0
  input). Its candidate ledger retains feasible, infeasible, unsupported, timed-out, failed, and
  `resource_limited` rows according to the declared retention policy;
  its counts describe the complete run. `views.top_n` or `views.pareto_front` lists the candidate IDs
  corresponding to numbered YAML files in order.
- Resource-limited rows have no simulated score or metrics. `counts.resource_limited` is separate
  from `counts.evaluated`; selected configurations cover completed evaluations. The CSV is a
  tabular view of the result. See [local resources](../reference/local-resources.md) for the diagnostic files.
- Each numbered YAML is a concrete prediction config. It excludes `optimization`, `optimizer`, and
  `preset`, contains no domains or `auto` values, and can be passed directly to
  `aisimulate predict`.

For scalar optimization, file numbering follows best-to-worst rank. For Pareto optimization, it
follows the deterministic display order of the complete nondominated front; that order does not
imply a scalar ranking.

If a completed search has no feasible candidate, the CLI still writes `recommendation.json` with empty views, zero
selected YAML files, complete counts and retained failure records. It exits with status `1` when
there are no resource-limited candidates. Any resource-limited candidate makes the exit status `3`,
even when fitting candidates and selected YAML files remain available. Other failed trials remain
in the ledger and permit status `0` when at least one selected configuration remains.
If the supervisor stops the entire execution, the event log may contain completed candidates
without a finalized `recommendation.json`; it is partial evidence, not a completed sweep.
