<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Understand your prediction

Start with the [CLI tutorial](quickstart.md#predict-one-deployment) and retain
per-request output when investigating a result:

```bash
aisimulate predict --stack engine --config prediction.yaml \
  --capture-per-request --output-dir prediction-output
```

`prediction.json` is the durable runner report. `--format json` changes the
summary on standard output; it does not replace the saved report. This guide
describes native engine replay. Analytical AFD/EPD and optional external runners
can expose different metrics; interpret them using their own report contracts.

## Read one report

This deliberately small, hypothetical excerpt illustrates the field names and
arithmetic. It is not measured performance or a complete serialized report:

```json
{
  "num_requests": 5,
  "completed_requests": 4,
  "total_output_tokens": 400,
  "duration_ms": 2000,
  "wall_time_ms": 150,
  "request_throughput_rps": 2,
  "output_throughput_tok_s": 200,
  "num_ttft_samples": 4,
  "num_tpot_samples": 4,
  "mean_ttft_ms": 100,
  "mean_tpot_ms": 10,
  "goodput_completed_requests": 3,
  "goodput_output_throughput_tok_s": 150,
  "gpu_hours": 0.0011111111111111111
}
```

| Field | Interpretation |
|---|---|
| `num_requests`, `completed_requests` | Requests recorded by this replay and requests that completed; compare them before interpreting latency. They need not equal all records in the input dataset. |
| `duration_ms` | Modeled reporting interval: 2 seconds here. It includes the elapsed interval represented by the report, rather than only active decode time. |
| `wall_time_ms` | Execution time of the simulator on your host: 150 ms here. It is not a serving latency. |
| `total_output_tokens`, `output_throughput_tok_s` | Output tokens from completed requests divided by modeled duration: `400 / 2 = 200` tokens/s. |
| `request_throughput_rps` | Completed requests divided by modeled duration: `4 / 2 = 2` requests/s. |
| `num_ttft_samples`, `num_tpot_samples` | Number of qualifying request samples for each latency distribution. A zero-valued latency with zero samples is not a fast deployment. |
| `goodput_output_throughput_tok_s` | Output tokens from completed, SLA-satisfying requests divided by the same duration; present when an SLA is configured. |
| `gpu_hours` | Provisioned GPU time integrated over the interval: every provisioned GPU over the simulated clock. Here it represents 2 GPUs for 2 seconds. |

Throughput per GPU uses average provisioned GPUs:
`avg_gpu = gpu_hours / (duration_ms / 3_600_000)`. Here, `200 / 2 = 100`
tokens/s/GPU. For a scaling runner, average provisioned GPUs can differ from
the peak allocation. See [optimization goals](../sweeper/optimization-goals.md).

An SGLang worker with `host_loop` ([SGLang VL host loop](../replay/engine/sglang-vl-host-loop.md))
splits TTFT into seven stage means: `mean_frontend_ms`, `mean_scheduler_inbox_wait_ms`,
`mean_receive_to_admit_ms`, `mean_prefill_elapsed_ms`, `mean_result_observation_delay_ms`,
`mean_handoff_to_first_token_ms` and `mean_detokenization_ms`. They sum to per-request
TTFT for requests that reached every milestone. `mean_detokenization_ms` is the time
from the engine emitting the first token to the client receiving it after
[`engine.detokenization`](../replay/engine/README.md#detokenization), and zero without
it. An encoder pool adds `encoder_latency_ms` (mean arrival to embedding
delivery, where the stage spans then start), `encoder_gpus` and `total_gpus`;
`gpu_hours` includes the pool and integrates every provisioned GPU over the
simulated clock, which under the host loop runs one overlap iteration past the
last observed token. With `--capture-per-request`, each record carries
the milestones `scheduler_received_ms`, `selected_ms` and `prefill_complete_ms`,
plus `frontend_ready_ms` when the worker prices frontend stages and, with a
native pool, `encoder_ready_ms`.

## Latency populations: ITL is not TPOT

| Metric | Native replay sample |
|---|---|
| TTFT | Arrival to first output token, including modeled queueing, for each completed request with an output token. |
| E2E latency | Arrival to last output token for each completed request with an output token. |
| TPOT | One average per completed request with at least two output tokens: `(last_token - first_token) / (output_tokens - 1)`. |
| ITL | Individual gaps between adjacent output tokens from completed, admitted requests. Longer outputs contribute more gaps. |
| Output token throughput per user | Individual `1000 / gap_ms` values for positive token gaps; its mean is not generally `1000 / mean_tpot_ms`. |

A token's time is the end of the engine pass that produced it, or, with
[`engine.detokenization`](../replay/engine/README.md#detokenization), the time the client
receives it.

For example, one request with a single 10 ms gap and another with three 30 ms
gaps give a mean TPOT of `(10 + 30) / 2 = 20` ms, but a mean ITL of
`(10 + 30 + 30 + 30) / 4 = 25` ms. Their percentiles also describe different
populations. A single-output-token request contributes TTFT but no TPOT sample.

The terminal table displays ITL. Aggregate SLA filtering uses mean TPOT for
the `itl_ms` bound. Use the saved report's explicit field names when comparing
with a benchmark; do not match columns solely by a similar label.

## Completed and incomplete requests

Native replay computes headline latency and throughput from completed requests.
Requests that are canceled, fail, or remain incomplete at a cutoff do not
contribute completed-request latency samples or output-token totals. A report
with many incomplete requests can therefore look deceptively fast if only its
latency columns are considered.

Inspect terminal records when `--capture-per-request` is enabled:

```bash
jq -s 'group_by(.terminal_status) | map({status: .[0].terminal_status, count: length})' \
  prediction-output/requests.jsonl
```

Per-request `itl_ms` is the request's average gap (TPOT), not its worst token
gap. Not every input request necessarily reaches a terminal record. Retain
the input, stop controls, report counts, and terminal records together.
Agentic virtual-time limits are soft scheduling cutoffs; see the
[trace contract](../replay/workloads.md#applied-compute-agentic-jsonl). A cutoff is not a
measurement warmup period or proof that the requested load was sustainable.

## Replay telemetry

Offline vLLM/SGLang engine replay supports optional native telemetry for both
aggregated and separate prefill/decode workers, including attention DP ranks.
Encoder pools are not supported by this capture path. It uses ordinary replay,
not the DP1-only detailed-artifact path.

```bash
aisimulate predict --stack engine --config examples/cli/agentic-profile.yaml \
  --capture-telemetry --telemetry-sample-interval-ms 1000 \
  --output-dir ./agentic-telemetry
```

The sampling interval is positive, finite, and measured in simulated
milliseconds. The default is 1,000 ms. Capture is disabled by default and does
not change scheduling, cache state, scaling clocks, or summary measurement.
It enables internal request capture without requiring `--capture-per-request`.

| Artifact | Contents |
| --- | --- |
| `telemetry-requests.jsonl` | Terminal requests from all phases: identity, lineage, lane, routing/admission history, lengths, timing, cache attribution, and status |
| `telemetry-samples.jsonl` | Settled scheduler/cache gauges and interval traffic counters, with worker/rank identity |
| `telemetry-lifecycle.jsonl` | Agentic request/play dispatch, terminal, quiescence, and skipped events |
| `prediction.json` → `telemetry` | Version, units, sample interval, clock and phase boundaries, artifact names, and record counts |

New telemetry uses one absolute virtual millisecond clock. Request/lifecycle
records distinguish `primer`, `warmup`, and `profile`. Scheduler samples use
`preparation` or `profile`: different lanes can prime and warm up concurrently.
Preparation intervals are closed at the profiling barrier. Existing
`requests.jsonl` remains profiling-only and barrier-relative, and existing
summary metrics still exclude preparation. Use the recorded profiling origin
to align the two clocks.

The manifest records whether replay completed, preparation aborted, the server
remained unsettled, or a simulation-time limit stopped it. As with existing
request output, requests still in flight at that limit have no terminal record;
their delivered tokens and executed prefill work remain in interval counters,
and pending work remains visible in scheduler gauges and lifecycle events.
Continuous-profile grace cancellation does produce terminal request records.

The exports provide the following inputs for warmup and profiling views:

| View | Native evidence and interpretation |
| --- | --- |
| Request/conversation/lane timeline | Absolute request timestamps, lineage, lane, and lifecycle events; retain canceled/failed requests |
| ISL/OSL, TTFT, E2E, interactivity | Request lengths and timing; `itl_ms` is per-request average token gap and is null with fewer than two delivered tokens |
| Queue depth | `prefill_scheduler_metrics` / `decode_scheduler_metrics`: `running_requests`, `waiting_requests`, worker/rank; plus `router_pending_*_requests` |
| KV use | Scheduler `active_blocks`, `inactive_blocks`, `total_blocks`, `active_cache_usage`, `physical_cache_usage`; grouped caches use `kv_cache_used_bytes` / `kv_cache_capacity_bytes` |
| Prefill/decode throughput | `traffic.committed_prefill_tokens` / `traffic.delivered_output_tokens`, divided by `traffic.duration_s` |
| Prefix lookup reuse | Role interval `native_prefix_cache_lookup`: `matched_tokens`, `queried_tokens`, `query_count`; actual retries count, even if admission later blocks |
| Prompt token sources | `traffic.first_admission`: `hbm_reused_tokens`, `host_reused_tokens`, `uncached_tokens`; distinct from executed work and readmission recomputation |
| Host KV use | `host_caches`: block/byte occupancy and capacity; shared pools appear once per ownership domain |
| Speculative acceptance | `traffic.avg_accept_length`, weighted by `accept_length_forward_count`; measured before output-limit truncation |

SGLang's pass-local cache-hit/total counters and native lookup counters describe
different populations. Router overlap and averages of per-request hit ratios
are not substitutes for either. Missing metrics remain unavailable, and a zero
query denominator is not a measured zero-percent hit rate. An absent host tier
is not evidence that an unmodeled host metric was measured as zero.
The exported pass-local cache counters are null when no such observation was
made in that interval. A measured cold miss has zero hits and a positive token
denominator. Grouped-cache `total_blocks` is null; use its byte capacity.

Delivered-token counters include tokens delivered before cancellation; headline
output throughput still uses the existing completed-request cohort. When
detokenization is configured, tokens enter the interval in which the client
receives them. Executed prefill counters include recomputation but exclude
unfinished passes at cutoff.
Both counters work with fixed timing and without FPM artifact capture.

Capture adds memory proportional to requests, lifecycle events, and samples
times live ranks, plus output tokens awaiting detokenization receipt. The CLI
retains host-memory supervision; low-level API callers
own their resource budgets. Capture adds no separate per-token trace files,
server logs, or power series. These are native AISim files, not AIPerf/infX export files, and
they do not establish timing accuracy or exact benchmark-input parity.

## Replay dashboard

Use `--export-html` to write an interactive, simulation-only `report.html` next
to the normal prediction output:

```bash
aisimulate predict --config examples/cli/agentic-profile.yaml \
  --export-html --output-dir ./agentic-dashboard
```

This flag enables telemetry capture internally, including request details;
neither `--capture-telemetry` nor `--capture-per-request` is required.
`--telemetry-sample-interval-ms` still controls sampling. The same offline
vLLM/SGLang engine paths listed above are supported. Rendering runs after replay
under the existing CLI resource supervisor and does not change native summary
metrics. Completed telemetry remains available even if HTML rendering fails.
Aborted preparation also produces a dashboard when native telemetry is available.

Open `report.html` directly in a browser. Data, JavaScript, and the Plotly
license are embedded; no server or network access is needed. The dashboard
has Dashboard and Request timeline tabs. It contains:

- Profiling and optional warmup views with request lengths, TTFT, E2E, and interactivity.
- Resident KV occupancy, host-cache pools, running/waiting queues, completed
  requests, preemptions, prefix reuse, native cache lookups, and prompt sources.
- Executed-prefill and delivered-output throughput, with a separate
  successful-completion-accounted output view, plus speculative acceptance.
- Conversation-level input footprint, completion status, lifecycle records,
  and a searchable conversation/worker timeline with request details.

The timeline uses thin interval lines and horizontal scrolling without zoom.
Profiling spans the native admission window; intervals crossing its end are
clipped in the view. Complete terminal records, including grace tails, remain
available in request details and the telemetry JSONL files. The Warmup stage
appears only when warmup was enabled and native preparation telemetry exists,
including empty or aborted preparation. It includes primer and warmup requests
and retains their phase labels.

Request percentile curves use the selected percentile on a trailing 50-request
window and on the cumulative cohort. Interactivity inverts the selected TPOT
percentile; fewer than two output tokens leaves it unavailable. Device-cache
and queue gauges and throughput rates use trailing 50-second smoothing. Rate
windows apportion counters by overlap, assuming uniform activity within each
native sample interval. Other counters retain their interval or cumulative
labels. Missing observations stay unavailable; grouped
caches retain byte capacity and shared host pools are shown once per ownership
domain. Conversation-level input footprint is the sum of the maximum active
input length per play and conversation, not a measurement of physical token
uniqueness. Footprint and in-flight request means include arrivals at the sample
time and exclude requests ending at that time. A single event sweep per phase
computes these series; redraws reuse them instead of scanning all requests per sample.
Native lookup reuse and scheduler pass reuse remain separate metrics.

This report presents simulated measurements only. It does not produce silicon
accuracy scores, server logs, or power curves. `--overwrite` replaces the HTML
along with known AISim outputs; omitting `--export-html` on the next run removes
any stale report. Unrelated files in the output directory remain untouched.

To render previously captured telemetry without another replay, use the Python
helper described in the [replay API](../replay/api/python.md#html-dashboard).
Large captures add rendering memory proportional to requests and samples;
keep the native JSONL files as the lossless record.

Request-derived charts use the recorded terminal cohort. When a simulation-time
limit leaves requests unfinished, native telemetry does not yet contain their
terminal details. The report shows the capture status and scheduler gauges;
missing terminal records do not imply that no work was admitted.

## Goodput and strict SLA answer different questions

With `evaluation.sla: {ttft_ms: 500, itl_ms: 50}`:

- **Per-request goodput** counts completed requests whose TTFT and average
  inter-token latency satisfy the bounds. Native replay skips the ITL bound
  for a request with fewer than two output tokens. Goodput throughput counts
  only the qualifying requests' output tokens.
- **`optimization.strict_sla: true`** additionally rejects a candidate when a
  configured aggregate mean exceeds its bound, or required completed-request
  counts, latency metrics, or sample counts are absent/invalid. It is not a
  p99 constraint and does not require every request to pass.

Bounds are inclusive. Unset bounds are unbounded; `e2e_ms` is a standalone
alternative to TTFT/ITL bounds. A goodput goal needs an SLA, while adding an SLA
to a throughput goal alone does not enable the aggregate-mean filter.

## Compare predictions with serving measurements

Keep model revision and quantization, GPU/topology, backend and version,
engine limits, timing/data identity, token lengths, arrivals/concurrency,
prefix reuse, stop criteria, and warmup policy aligned. The generated benchmark
helper warms up requests; a fresh replay's startup/cache history can differ.
Record the difference or align the measurement windows before attributing
error to the performance model.

Save the input YAML, concrete recommended YAML, AISimulate version/source SHA,
wheel identity when applicable, `prediction.json`, per-request records, and
the benchmark command/results. For recommendations also retain
`recommendation.json`: it records failed attempts and the candidate IDs behind
selected YAML files. Inspect them with:

```bash
jq '{counts, views, rejected: [.candidates[] | select(.status != "feasible") | {candidate_id, status, reason_category, reason}]}' \
  recommendation-output/recommendation.json
```

Check the [exit code](../reference/cli.md#errors-and-exit-codes) as well as the file:
a completed recommendation that selects no configuration and has zero resource-limited candidates writes its ledger and
exits `1`. Resource-limited candidates produce exit `3`, even when completed recommendations
remain available. A supervisor interruption can leave only checkpoint evidence; inspect the
[resource diagnostics](../reference/local-resources.md) before treating the search as complete.
Follow the [Dynamo deployment walkthrough](../sweeper/deployment-generation.md)
for artifact generation and benchmarking. The [accuracy overview](../../README.md#support-and-accuracy)
describes where measured validation exists; executing a prediction does not
establish its accuracy.
