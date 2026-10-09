<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# CLI reference

Use `aisimulate predict` for one concrete deployment and `aisimulate recommend`
for a bounded search. Start with the [quickstart](../getting-started/quickstart.md)
for a complete input, or consult [configuration](configuration.md) for the YAML
shape. `aisimulate onboard` has its own [FPM self-service guide](../perf-model/fpm-self-service/implementation.md).
The separate [AIC compatibility CLI](../aic-backward-compatibility/cli.md) retains
its six established workflows.

<a id="commands"></a>

## Commands

| Command | Purpose | Example invocation | Result |
|---|---|---|---|
| `predict` | Evaluate one concrete deployment under a workload. | `aisimulate predict -c prediction.yaml --output-dir ./prediction-output` | A metrics summary and `prediction-output/prediction.json`. |
| `recommend` | Search deployment and load choices for an optimization goal. | `aisimulate recommend -c recommendation.yaml --output-dir ./recommendation-output` | Ranked configurations, `recommendation.json`, and concrete YAML files under `recommendations/`. |

`recommend` also accepts repeatable `--output NAME` options. Each selected output adapter requires
a same-named top-level configuration section, may observe live candidate and round notifications,
and writes additional artifacts after recommendation.

Unless labeled as captured output, metric values in example results are hypothetical and
illustrate the output format. Captured detail examples are simulation results, not hardware measurements.

<a id="common-options"></a>

## Common Options

| Option | Type | Default | Meaning |
|---|---|---:|---|
| `-c`, `--config PATH` | path | Required | Input YAML file. |
| `--stack NAME` | string | `engine` | Built-in or discovered execution stack. This selection is CLI-only and is never written into YAML. |
| `--set PATH=YAML_VALUE` | repeatable assignment | None | Set a supported configuration path after loading YAML. |
| `--output-dir PATH` | path | `./aisimulate-output` | Directory for durable results. |
| `--overwrite` | flag | `false` | Replace known AISimulate output files in an existing output directory. |
| `--format table\|json` | enum | `table` | Standard-output presentation. It does not change durable output files. |

`predict` also accepts:

| Option | Type | Default | Meaning |
|---|---|---:|---|
| `--capture-per-request` | flag | `false` | Write per-request prediction records to `requests.jsonl`. |
| `--detail` | comma-separated selectors | omitted | Add `summary`, `memory`, `time`, `energy`, `source`, or `all`. See [prediction details](#prediction-details). |
| `--online` | flag | `false` | Pace prediction against the real wall clock instead of virtual time. The selected stack must advertise online support. |

The CLI deliberately does not expose field-specific flags such as `--request-per-second` or
`--num-workers`. YAML is the authoritative semantic configuration surface.

<a id="override-semantics"></a>

### Override Semantics

`--set` uses a dot-separated path and parses its value as YAML:

```bash
aisimulate predict \
  --config prediction.yaml \
  --set traffic.load.concurrency=8 \
  --set engine.workers.aggregated.parallelism.replicas=2 \
  --output-dir ./prediction-overrides
```

The following rules apply:

- The path must be supported by the schema, but may be omitted from the input YAML. Unknown fields are rejected.
- Overrides are applied from left to right. The last assignment to a path wins.
- YAML scalar, sequence, and mapping syntax is accepted on the right-hand side.
- Sequence-index paths are not supported. Override the complete sequence instead.
- Normal schema and cross-field validation runs after all overrides are applied.
- Overrides affect the run. Recommended YAML files contain the selected concrete values;
  `predict` writes the runner report, not a separate resolved-input YAML.

<a id="choose-an-execution-stack"></a>

## Choose an execution stack

`--stack engine` is the default and uses the built-in offline runner. To use Dynamo-owned
routing and Planner behavior, install a [matched integration](../getting-started/installation.md#optional-dynamo-integration) and select it explicitly:

```bash
aisimulate predict --stack dynamo --config prediction.yaml --output-dir ./dynamo-prediction
aisimulate recommend --stack dynamo --config recommendation.yaml --output-dir ./dynamo-recommendation
```

A config containing `router` or `planner` requires the corresponding installed adapters; the
Dynamo integration provides them. Use the same stack when predicting a configuration saved by
that stack's recommendation run. See the complete [Dynamo prediction](../replay/dynamo.md#complete-dynamo-prediction-example)
and [recommendation](../replay/dynamo.md#dynamo-scalar-recommendation-example) examples.

`predict --online` requests wall-clock-paced execution from a stack that supports it.
The built-in `engine` stack supports offline execution only; `recommend` is always offline.
Online execution paces the simulation and does not launch a real serving endpoint.

If Dynamo is unavailable, the error includes the installed stack names. For example, when only
the built-in engine is installed:

```text
stack 'dynamo' is unavailable; installed stacks: engine. Install the distribution that provides the requested stack.
```

Install the integration in the same Python environment as `aisimulate`. Implementation details
for stack and adapter authors are in [adapter contracts](../adapters/README.md).

<a id="outputs"></a>

## Outputs

Read [Understand your prediction](../getting-started/understand-results.md) for an annotated
report, ITL/TPOT definitions, incomplete-request handling, and SLA interpretation.

Core output controls are CLI-only. A selected output adapter owns its same-named top-level input
section; that section configures artifact generation and is excluded from recommended prediction
YAML files. Output adapters receive the prepared output directory directly and must preserve
unrelated files.

Recommendation output uses the schema-versioned `SweepResult` contract documented in
[`docs/sweeper/results.md`](../sweeper/results.md). It preserves run metadata, a candidate-attempt
ledger, stable status and reason categories, counts, provenance, and candidate-ID selection views.
Replay metrics use unit-bearing names such as `*_tok_s`, `*_ms`, `*_w`, and `*_j`.

The fields `power_w` and `power_coverage` follow the
[modeled-power contract](../perf-model/power.md). That contract defines active-forward-pass per-GPU
scope, energy-over-active-latency aggregation, null semantics, and provenance requirements.
`power_coverage` is the share of modeled active time with operation-energy evidence; `power_w`
may be numeric at or above 90% coverage, so `0.90` passes while `0.899` does not. This formalizes
existing AIC semantics; it neither adds a new power calculation nor implies that every runner or
timing provider implements these fields. Normal prediction and recommendation
summaries always show both labels, with explicit unavailable values and reasons when needed.
Summary power is independent of `--detail`; the `energy` selector only adds a breakdown.
Both JSON keys are always present in conforming summaries: unavailable watts use `null`,
coverage stays numeric when computable, and an unsupported energy path uses `null` for both.
Consult the
[AIC migration guide](../aic-backward-compatibility/migration.md) for the current release boundary.

### Power and energy detail

Use `aisimulate predict --stack engine --config prediction.yaml --detail energy`
to show phase and operation evidence alongside the normal power summary.
`--detail all` includes energy. `--diagnostics power` remains a compatibility
alias for its original stdout envelope. `--diagnostics-top-n N` bounds table
rows per phase; `prediction.json` and JSON detail output retain all operations.
Each phase shows publication status, source kind, and the concrete source tag.
Missing or invalid display measurements render as `N/A`.

The native engine export supports this evidence path with op-level timing on
supported topologies. The external Dynamo Python adapter's diagnostics export
requires separate qualification: native Rust compatibility aliases do not establish
adapter parity. If a selected runner exports no typed evidence, energy details
state that reason. FPM, fixed, polynomial, AFD, and analytical EPD energy remain
unavailable; their summary fields are explicit nulls where unsupported.

<a id="prediction-directory"></a>

### Prediction Directory

```text
<output-dir>/
├── prediction.json
├── resource-plan.json             # on preflight refusal
├── resource-runtime.json
├── execution-events.jsonl         # when execution produced checkpoints
├── requests.jsonl                 # only with --capture-per-request
├── afd-replay-spec.json           # only for AFD
└── afd-qualification.json         # only for AFD
```

- `prediction.json` preserves the selected runner's existing full prediction report.
- `requests.jsonl` contains one record per request when explicitly enabled. SGLang workers with
  `host_loop` add `frontend_ready_ms`, `scheduler_received_ms`, `selected_ms` and
  `prefill_complete_ms` to each record; a native encoder pool adds `encoder_ready_ms`.
- `resource-plan.json` describes preflight refusal, with null for unavailable host, budget,
  or workload estimates. `resource-runtime.json` records the effective budget and supervision
  outcome. `execution-events.jsonl` retains complete checkpoints after interruption; see
  [local execution resources](local-resources.md) for their interpretation.
- `afd-replay-spec.json` is the exact, deterministic analytical replay contract for an AFD run,
  including topology, measurement provenance, workload, goal, and any P/D companion.
- `afd-qualification.json` validates and summarizes the A/F pools, routing order, backend version,
  measurement coverage, and GPU accounting. It explicitly records that native launch generation is
  unsupported; it is not a Kubernetes manifest or runnable shell artifact.

<a id="recommendation-directory"></a>

### Recommendation directory

See [recommendation results](../sweeper/results.md#cli-recommendation-directory)
for `recommendation.json`, CSV, selected YAML and partial-result semantics.

<a id="existing-output-directories"></a>

### Existing Output Directories

Without `--overwrite`, the CLI rejects an existing nonempty output directory. With `--overwrite`, it
replaces only the known output files listed below and preserves unrelated files.

Specifically, overwrite may replace `prediction.json`, `recommendation.json`, `recommendation.csv`,
`requests.jsonl`, `resource-plan.json`, `resource-runtime.json`, `execution-events.jsonl`,
`afd-replay-spec.json`, `afd-qualification.json`, and numbered `recommendations/NNNN.yaml` files.
Other files, including non-numbered files inside `recommendations/`, are preserved. Invalid
configuration loading, overrides, core-schema validation, unknown top-level sections, and stacks or
output adapters that are not installed leave existing artifacts intact. Errors raised while
importing an installed plugin or inside an adapter occur after the known outputs are removed.

<a id="standard-output"></a>

### Standard Output

`--format table` prints a concise human-readable summary. `--format json` prints the same summary as
one JSON value for shell automation. Durable artifact formats do not change with this option.

Prediction JSON without `--detail` on standard output is a summary object. SGLang workers with
`host_loop` add the mean time to first token split by stage: `mean_frontend_ms`,
`mean_scheduler_inbox_wait_ms`, `mean_receive_to_admit_ms`, `mean_prefill_elapsed_ms`,
`mean_result_observation_delay_ms` and `mean_handoff_to_first_token_ms`. The six spans sum to
per-request TTFT for requests that reached every milestone; with a native encoder pool they start
at the pool's delivery and `encoder_latency_ms` covers the wait. Encoder pools also report
`encoder_gpus` and `total_gpus`. Recommendation JSON is an array of selected
rows with `rank`, `score`, `objectives`, `used_gpus`, and `config_path`. Single-objective scores are
signed so higher is better; latency-minimizing targets report negative scores. Pareto rows carry
the raw objective values in `objectives`. Use `recommendation.json` for the complete candidate ledger.

<a id="prediction-details"></a>

### Prediction details

```bash
aisimulate predict -c prediction.yaml --detail summary,memory,time \
  --format json --output-dir ./prediction-details
```

`--detail` selects additional reports on `predict`. With no selector, stdout and durable
reports retain their existing shape. With a selector, JSON stdout contains `summary` and
`details`; the same versioned `details` object is added to `prediction.json` and follows the
[prediction-details schema](schemas/prediction-details.schema.json). Table output appends selected
sections and skipped-section reasons to the normal prediction summary.

- `summary`: existing serving metrics.
- `memory`: the existing initial per-rank memory capacity estimate, including components when
  available, with sizes in bytes and token counts in tokens. The only current `stage` value is
  `before_native_capacity_adjustments`; `estimated_num_gpu_blocks` is captured at this stage,
  before adjustments such as FPM profile-domain limits. It is neither a final runtime capacity
  nor observed memory usage. Explicit KV blocks, nested rank input,
  and unsupported providers/topologies may have no exported estimate.
  Analytical EPD retains available language-worker estimates and marks the encoder component
  breakdown unavailable; its memory section is partial when language estimates exist.
- `time`: existing TTFT, TTST, TPOT, inter-token, and end-to-end request latency statistics in
  milliseconds, plus trajectory latency statistics when exported by the runner. Replay duration
  and simulator wall time remain in the summary.
  On the native op-level engine path, `diagnostics` also contains accumulated prefill/decode
  and per-operation latency, speed-of-light (SOL) latency/compute/memory comparisons, and
  latency/SOL ratios. These are sums of scheduled rank-local forward-pass work across the replay,
  not request TTFT, critical-path duration, or whole-deployment GPU time. Synthetic speedup
  adjusts modeled latency; the SOL baseline remains unscaled. Missing SOL families have null
  comparisons and an explicit reason; phase SOL totals require complete operation coverage.
  Analytical EPD retains its approximation labels in the summary.
- `energy`: active forward-pass phase and operation energy evidence per GPU, with coverage,
  publication status, sources, and missing-evidence reasons. It preserves the normal summary
  power values. See [Power and energy detail](#power-and-energy-detail).
- `source`: per-phase operation source tags and executed MoE communication measurement
  substitutions (requested versus measured EP/node topology). An empty fallback list means no
  substitution was recorded; null means the provider did not export fallback metadata. This is
  operation evidence, not a full measurement-file lineage or estimator-selection audit.
- `all`: `summary,memory,time,energy,source`, with availability reported for each section.

`--detail-top-n N` (default 12; `--diagnostics-top-n` remains an alias) limits operation rows
in time, source, and energy tables only. JSON stdout and `prediction.json` retain every row.

Memory without evidence is omitted from `details.sections` and listed with a reason in
`details.skipped`. Time, source, and energy retain explicit unavailable evidence. A memory
section with only some estimated roles is `partial` and records
why other roles are unavailable. Energy retains an explicit unavailable status and reason when
the runner exports no typed evidence. Missing measurements are never invented as zero;
energy-aware runs with no covered operations report numeric zero coverage. Whole-model FPM,
fixed/polynomial timing, analytical EPD/AFD overlays, and adapters without the native export
report operation timing/source evidence unavailable. Serving time statistics remain available
where exported. See [diagnostic availability](../aic-backward-compatibility/migration.md#prediction-details-and-power).

Inspect a recommendation by running `predict --detail` on its saved YAML. Reporting options
are CLI-only; this change adds no YAML configuration fields.


<a id="errors-and-exit-codes"></a>

## Errors and Exit Codes

| Exit Code | Meaning |
|---:|---|
| `0` | Successful prediction or recommendation. |
| `1` | Execution failure, or a completed recommendation that selects no configuration and has zero resource-limited candidates. |
| `2` | CLI syntax, YAML parsing, schema, domain, override, or unsupported-combination error. |
| `3` | Resource refusal, including a partial recommendation containing resource-limited candidates. |
| `124` | Supervisor initialization or shutdown timeout. |
| `130` | Interrupted by the user. |

Configuration errors identify the input file and validation details. These shortened examples
illustrate the invalid field and cause; exact formatting can vary:

```text
recommendation.yaml: traffic.load.sessions_per_second.range.min:
must be greater than 0, got 0
```

Combination errors name conflicting values and explain the supported contract:

```text
recommendation.yaml: router.prefill_load_model.type:
'aic' is incompatible with router.policy='round_robin'; use policy='kv_router' or type='none'
```

Unsupported stack, backend, or policy combinations are reported as errors.

An unavailable timing model reports the rejected architecture, backend version,
or estimator selection reasons. Missing performance measurements during replay
name the missing data and suggest another parallelism or backend version.
An explicitly selected, untrained regression model still requires training
observations. For a new model, follow the
[FPM self-service guide](../perf-model/fpm-self-service/README.md).

<a id="troubleshooting"></a>

## Troubleshooting

| Symptom | What to check | Example fix |
|---|---|---|
| `aisimulate: command not found` | The environment containing AISimulate must be active. | From the tutorial directory, run `source .venv/bin/activate`, then `python -m pip show aisimulate`. |
| Installation reports an unsupported Python version | AISimulate requires Python 3.11–3.13. | Check `python3 --version` and create the environment with a supported interpreter. |
| Configuration or trace file cannot be found | Check the path and the directory where you ran the command. | Run from the directory containing `prediction.yaml`, or use absolute paths. |
| Output directory is not empty | Each run needs an empty directory or explicit overwrite. | Add `--output-dir ./another-prediction`, or `--overwrite` to replace known outputs. |
| Stack or config adapter is unavailable | Use the Python environment containing the selected integration. `router` and `planner` are Dynamo-owned sections. | Install a matched pair from the [installation guide](../getting-started/installation.md#optional-dynamo-integration) and use `--stack dynamo`. |
| `predict` rejects a domain or `optimization` | A search input was passed to a concrete prediction command. | Run `recommend` first, then predict `recommendations/0001.yaml`. |
| `--set` produces an unknown-field or load-validation error | Paths must be supported, and load fields must match the selected load type. | For the quick-start input, use `--set traffic.load.concurrency=8`. To change load type, replace the whole `traffic.load` mapping. |
| No selected configuration and zero resource-limited candidates, exit `1` | Inspect `recommendation.json` for candidate status, reason, and GPU/SLA constraints. | Check that the model fits within `max_candidate_gpus`, and that the workload can meet the SLA. |
| Resource refusal, exit `3` | Inspect resource diagnostics and any completed recommendation ledger. | Check the [local resource budget](local-resources.md); preserve completed results before choosing a smaller workload or another host. |
| A candidate fails to resolve performance data | Check the model, hardware, backend version, and timing mode. FPM needs a matching collected cell. | Use a covered combination from the [support reference](../../README.md#support-and-accuracy) or the [FPM workflow](../perf-model/methods/whole-forward.md). |
| `--online` is rejected | The selected stack must advertise online support. | Use offline execution with `--stack engine`, or an integration that supports online execution. |

For automation, check the exit code as well as standard output. `--format json` changes successful
summary output; validation and execution errors are reported on standard error. A completed recommendation
with no feasible result still saves its result ledger and does not provide a YAML to predict.
