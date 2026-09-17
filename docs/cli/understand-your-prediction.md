<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Understand your prediction

Start with the [CLI tutorial](user-guide.md#predict-one-deployment) and retain
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
| `gpu_hours` | Provisioned GPU time integrated over the interval. Here it represents 2 GPUs for 2 seconds. |

Throughput per GPU uses average provisioned GPUs:
`avg_gpu = gpu_hours / (duration_ms / 3_600_000)`. Here, `200 / 2 = 100`
tokens/s/GPU. For a scaling runner, average provisioned GPUs can differ from
the peak allocation. See [optimization goals](../sweeper/optimization-goals.md).

## Latency populations: ITL is not TPOT

| Metric | Native replay sample |
|---|---|
| TTFT | Arrival to first output token, including modeled queueing, for each completed request with an output token. |
| E2E latency | Arrival to last output token for each completed request with an output token. |
| TPOT | One average per completed request with at least two output tokens: `(last_token - first_token) / (output_tokens - 1)`. |
| ITL | Individual gaps between adjacent output tokens from completed, admitted requests. Longer outputs contribute more gaps. |
| Output token throughput per user | Individual `1000 / gap_ms` values for positive token gaps; its mean is not generally `1000 / mean_tpot_ms`. |

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
[trace contract](user-guide.md#applied-compute-agentic-jsonl). A cutoff is not a
measurement warmup period or proof that the requested load was sustainable.

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

Check the [exit code](user-guide.md#errors-and-exit-codes) as well as the file:
a completed recommendation with no feasible or resource-limited candidate writes its ledger and
exits `1`. Resource-limited candidates produce exit `3`, even when completed recommendations
remain available. A supervisor interruption can leave only checkpoint evidence; inspect the
[resource diagnostics](../local-resources.md) before treating the search as complete.
Follow the [Dynamo deployment walkthrough](../../python/aisimulate/docs/dynamo_deployment_guide.md)
for artifact generation and benchmarking. The [accuracy overview](../../README.md#support-and-accuracy)
describes where measured validation exists; executing a prediction does not
establish its accuracy.
