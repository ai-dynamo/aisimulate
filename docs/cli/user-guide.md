<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AISimulate CLI User Guide

Predict serving behavior and search deployment configurations with `aisimulate`.

Use this guide for the unified CLI. For the six `aiconfigurator cli` commands still shipped
with AISimulate, see the [Legacy AIC CLI User Guide](legacy-aic-user-guide.md). The
[migration guide](migrate-from-aiconfigurator.md) explains which workflows have a unified replacement.

> [!WARNING]
> **Experimental.** Recommendation schemas and search behavior may change between releases
> without a standard deprecation period. Upgrading can require changes to your YAML or scripts.

**Contents**

- **Getting started**
  - [1. Start here](#start-here)
  - [2. Commands](#commands)
  - [3. Install](#install)
  - [4. Predict one deployment](#predict-one-deployment)
  - [5. Try your own workload](#try-your-own-workload)
  - [6. Recommend under a GPU budget](#recommend-under-a-gpu-budget)
- **Configuration and CLI reference**
  - [7. Common Options](#common-options)
  - [8. Choose an execution stack](#choose-an-execution-stack)
  - [9. Configuration Model](#configuration-model)
  - [10. Presets and Default Ranges](#presets-and-default-ranges)
  - [11. Traffic](#traffic)
  - [12. Engine](#engine)
  - [13. Router (Dynamo Adapter)](#router-dynamo-adapter)
  - [14. Planner (Dynamo Adapter)](#planner-dynamo-adapter)
  - [15. Evaluation](#evaluation)
  - [16. Recommendation Domains](#recommendation-domains)
  - [17. Optimization Goal](#optimization-goal)
  - [18. Optimizer Controls](#optimizer-controls)
- **Examples**
  - [19. Complete Dynamo Prediction Example](#complete-dynamo-prediction-example)
  - [20. Dynamo Scalar Recommendation Example](#dynamo-scalar-recommendation-example)
  - [21. Pareto Recommendation Example](#pareto-recommendation-example)
- **Outputs and troubleshooting**
  - [22. Outputs](#outputs)
  - [23. Errors and Exit Codes](#errors-and-exit-codes)
  - [24. Troubleshooting](#troubleshooting)
  - [25. Related documentation](#related-documentation)

<a id="start-here"></a>

## 1. Start here

1. [Install AISimulate](#install) and [predict one deployment](#predict-one-deployment) to see the results.
2. [Try your own workload](#try-your-own-workload) by changing token lengths or load.
3. [Recommend under a GPU budget](#recommend-under-a-gpu-budget), then [predict the selected configuration](#predict-a-recommended-configuration).

For a specific task, jump to [Dynamo integration](#choose-an-execution-stack),
[AgentX and other trace formats](#trace-format-compatibility), or [troubleshooting](#troubleshooting).
Use the [configuration reference](#configuration-model) when you need individual fields.

<a id="commands"></a>

## 2. Commands

| Command | Purpose | Example invocation | Result |
|---|---|---|---|
| `predict` | Evaluate one concrete deployment under a workload. | `aisimulate predict -c prediction.yaml --output-dir ./prediction-output` | A metrics summary and `prediction-output/prediction.json`. |
| `recommend` | Search deployment and load choices for an optimization goal. | `aisimulate recommend -c recommendation.yaml --output-dir ./recommendation-output` | Ranked configurations, `recommendation.json`, and concrete YAML files under `recommendations/`. |

Unless labeled as captured output, metric values in example results are hypothetical and
illustrate the output format. Captured detail examples are simulation results, not hardware measurements.

<a id="install"></a>

## 3. Install

Check the [installation guide](../installation.md) for the selected wheel's
platform requirements and publication status. Use its source-install workflow
for features documented on `main` that are not yet in a published wheel.

Use **Python 3.11–3.13**. The commands below use Bash or Zsh. Check that `python3` selects a
supported version; substitute a versioned command such as `python3.13` if needed.

When upgrading an existing AIConfigurator environment, follow the
[package migration instructions](../../README.md#upgrade-from-standalone-aiconfigurator) before installing.

Create a working directory and virtual environment, then install the built-in engine:

```bash
python3 --version
mkdir -p aisimulate-tutorial
cd aisimulate-tutorial
python3 -m venv .venv
source .venv/bin/activate
python -m pip install aisimulate
aisimulate --help
aisimulate predict --help
aisimulate recommend --help
```

The help output should list `predict` and `recommend`. Save the YAML files below in this working
directory and run the commands from there. Reactivate `.venv` when opening a new terminal.

The built-in engine predicts behavior offline without launching a GPU serving deployment.
`engine.hardware: h200_sxm` names the hardware being modeled; you do not need an H200 to run the example.

<a id="predict-one-deployment"></a>

## 4. Predict one deployment

Save this as `prediction.yaml`. It evaluates one Qwen3-32B-FP8 worker on an H200 with modeled
vLLM serving behavior, four concurrent requests, and twelve requests in total. The vLLM examples
explicitly select H200 performance-data version `0.24.0`:

```yaml
traffic:
  source: {type: synthetic, input_tokens: 1024, output_tokens: 128}
  load: {type: concurrency, concurrency: 4}
  stop: {requests: 12}

engine:
  mode: aggregated
  model: Qwen/Qwen3-32B-FP8
  hardware: h200_sxm
  backend: vllm
  backend_version: "0.24.0"
  workers:
    aggregated:
      parallelism: {replicas: 1, tensor: 1}
```

`input_tokens` and `output_tokens` set the prompt and response lengths. `replicas` counts worker
copies; `tensor` counts GPUs used to split each worker's model. With the other parallelism settings
at their defaults, the GPU count is `replicas × tensor`: this example models one GPU.

Run it and retain the per-request records:

```bash
aisimulate predict \
  --config prediction.yaml \
  --output-dir ./prediction-output \
  --capture-per-request
```

To rerun a command, choose a new `--output-dir` or add `--overwrite` to replace its known output files.

Example output (illustrative values):

This compact view shows `avg` and `p99`; expand the terminal output below for all statistics.

| Metric | avg | p99 |
|---|---:|---:|
| Time to first token (ms) | 120.00 | 149.00 |
| Time to second token (ms) | 130.00 | 159.00 |
| Request latency (ms) | 1,390.00 | 1,419.00 |
| Inter-token latency (ms) | 10.00 | 10.00 |
| Output per user (tokens/s) | 100.00 | 100.00 |
| Total output (tokens/s) | 320.00 | N/A |
| Request throughput (requests/s) | 2.50 | N/A |
| Request count | 12 | N/A |

Wall time: **1,250 ms**. Saved report: `prediction-output/prediction.json`.

<details>
<summary>Full terminal output (all statistics)</summary>

```text
NVIDIA AIPerf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━┓
┃ Metric                                             ┃      avg ┃      min ┃      max ┃      p99 ┃      p90 ┃      p75 ┃   std ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━┩
┃ Time to First Token (ms)                           ┃   120.00 ┃    90.00 ┃   150.00 ┃   149.00 ┃   144.00 ┃   135.00 ┃ 18.00 ┃
┃ Time to Second Token (ms)                          ┃   130.00 ┃   100.00 ┃   160.00 ┃   159.00 ┃   154.00 ┃   145.00 ┃ 18.00 ┃
┃ Request Latency (ms)                               ┃ 1,390.00 ┃ 1,360.00 ┃ 1,420.00 ┃ 1,419.00 ┃ 1,414.00 ┃ 1,405.00 ┃ 18.00 ┃
┃ Inter Token Latency (ms)                           ┃    10.00 ┃    10.00 ┃    10.00 ┃    10.00 ┃    10.00 ┃    10.00 ┃  0.00 ┃
┃ Output Token Throughput Per User (tokens/sec/user) ┃   100.00 ┃   100.00 ┃   100.00 ┃   100.00 ┃   100.00 ┃   100.00 ┃  0.00 ┃
┃ Output Token Throughput (tokens/sec)               ┃   320.00 ┃      N/A ┃      N/A ┃      N/A ┃      N/A ┃      N/A ┃   N/A ┃
┃ Request Throughput (requests/sec)                  ┃     2.50 ┃      N/A ┃      N/A ┃      N/A ┃      N/A ┃      N/A ┃   N/A ┃
┃ Request Count (requests)                           ┃    12.00 ┃      N/A ┃      N/A ┃      N/A ┃      N/A ┃      N/A ┃   N/A ┃
└━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┴━━━━━━━━━━┴━━━━━━━━━━┴━━━━━━━━━━┴━━━━━━━━━━┴━━━━━━━━━━┴━━━━━━━━━━┴━━━━━━━┘
Wall Time (ms): 1,250.00
Saved full report to: prediction-output/prediction.json
```

</details>

The `avg` column shows average latency or throughput. `p99` is the 99th percentile: about 99% of
the corresponding measurements fall at or below that value. Time to First Token measures how long a request waits
for its first output token. Inter Token Latency measures gaps between output tokens, and Request
Latency measures the complete request. Output Token Throughput counts generated output tokens
per second across the deployment. A short workload is useful for checking the
workflow; choose a representative workload for performance comparisons. `Wall Time` measures the
simulator's execution time on your machine; latency and throughput describe the modeled deployment.

The command writes:

```text
prediction-output/
├── prediction.json
└── requests.jsonl
```

`prediction.json` contains the full runner report, including its `summary`. Add `--format json`
to print the summary as one compact JSON object instead of the metrics table and saved-path line.
`--capture-per-request` is optional and is unavailable for analytical EPD.

`predict` accepts concrete values only. Search domains, `optimization`, and `optimizer` belong
in a recommendation input.

<a id="try-your-own-workload"></a>

## 5. Try your own workload

Reuse `prediction.yaml` and change supported fields with `--set`. The file itself is unchanged.

<a id="change-token-lengths-and-concurrency"></a>

### 5.1 Change token lengths and concurrency

This run uses 4,096 input tokens, 512 output tokens, and up to 16 requests in flight.
It submits 160 requests in total:

```bash
aisimulate predict \
  --config prediction.yaml \
  --set traffic.source.input_tokens=4096 \
  --set traffic.source.output_tokens=512 \
  --set traffic.load.concurrency=16 \
  --set traffic.stop.requests=160 \
  --output-dir ./prediction-c16
```

The result is the same metrics table and a report at `prediction-c16/prediction.json`.
Concurrency is a limit on active requests; `traffic.stop.requests` is the total workload size.

<a id="use-an-incoming-request-rate"></a>

### 5.2 Use an incoming request rate

To model eight arrivals per second, replace the entire load mapping so its fields match the new type:

```bash
aisimulate predict \
  --config prediction.yaml \
  --set 'traffic.load={type: constant_rate, requests_per_second: 8}' \
  --set traffic.stop.requests=200 \
  --output-dir ./prediction-8rps
```

The report is saved to `prediction-8rps/prediction.json`. A request rate controls arrivals, so requests
may queue if the deployment cannot keep up. Use `concurrency` to cap in-flight requests, or
`constant_rate` / `poisson` to study an offered arrival rate. A request-rate input does not guarantee
that the deployment meets that rate within a latency target.

To evaluate a different model or system, update `engine.model`, `engine.hardware`, and
`engine.backend` in the YAML. Choose a combination covered by the
[support reference](../../README.md#support-and-accuracy). Keep the workload fixed when comparing
deployments; changing both the deployment and the workload makes the scores harder to compare.

<a id="recommend-under-a-gpu-budget"></a>

## 6. Recommend under a GPU budget

Save this as `recommendation.yaml`. It searches one-GPU and four-GPU aggregated configurations,
at concurrency four or eight, to maximize throughput per GPU within a four-GPU budget:

```yaml
traffic:
  source: {type: synthetic, input_tokens: 1024, output_tokens: 128}
  load: {type: concurrency, concurrency: {choices: [4, 8]}}
  stop: {requests: 32}

engine:
  mode: aggregated
  model: Qwen/Qwen3-32B-FP8
  hardware: h200_sxm
  backend: vllm
  backend_version: "0.24.0"
  workers:
    aggregated:
      parallelism:
        preset:
          - {replicas: 1, tensor: 1, pipeline: 1, attention_data: 1, moe_tensor: 1, moe_expert: 1}
          - {replicas: 2, tensor: 2, pipeline: 1, attention_data: 1, moe_tensor: 1, moe_expert: 1}
      scheduler: {max_batched_tokens: 8192, max_sequences: 256}
      kv_cache:
        block_size: 64
        capacity: {type: default, memory_fraction: 0.9}

optimization:
  target: throughput_per_gpu
  constraints: {max_candidate_gpus: 4}

optimizer:
  algorithm: random
  max_trials: 4
  parallelism: 1
  seed: 42
```

The two parallelism presets represent `1 replica × 1 GPU` and `2 replicas × 2 GPUs`.
The GPU budget limits each candidate deployment. `optimizer.max_trials` limits search attempts,
and `optimizer.parallelism` controls concurrent simulation trials on your machine.

This example searches concurrency as well as deployment shape. To compare deployments at one
fixed load, replace `{choices: [4, 8]}` with a concrete concurrency such as `8`.

Run the bounded search:

```bash
aisimulate recommend \
  --config recommendation.yaml \
  --output-dir ./recommendation-output
```

Example output (illustrative values, four selected configurations):

```text
AISimulate recommendations
1: score=400 used_gpus=4 config=recommendation-output/recommendations/0001.yaml
2: score=360 used_gpus=1 config=recommendation-output/recommendations/0002.yaml
3: score=320 used_gpus=4 config=recommendation-output/recommendations/0003.yaml
4: score=280 used_gpus=1 config=recommendation-output/recommendations/0004.yaml
Saved full result to: recommendation-output/recommendation.json
```

The first row identifies a selected configuration and its score. For `throughput_per_gpu`,
`score=400` means output tokens per second per GPU; four GPUs would correspond to 1,600 output
tokens per second in aggregate. Higher scores rank first. The selected count can be smaller than
the trial budget because candidates may be infeasible, fail, or resolve to the same configuration.

For this example result, the command writes:

```text
recommendation-output/
├── recommendation.json
└── recommendations/
    ├── 0001.yaml
    ├── 0002.yaml
    ├── 0003.yaml
    └── 0004.yaml
```

`recommendation.json` contains the complete result ledger, including candidate metrics, statuses,
and selection views. The numbered YAML files contain the selected concrete prediction inputs in
rank order. Add `--format json` to print the selected rows as a JSON array instead of the ranked
text and saved-path line; the output files stay the same.

A four-trial search is a small starting example. Increase `optimizer.max_trials` to explore more
candidates; the trial budget does not guarantee an exhaustive search or a globally optimal result.
Use `parallelism: {preset: default}` to let AISimulate generate the parallelism search space.

<a id="choose-a-goal-and-add-latency-limits"></a>

### 6.1 Choose a goal and add latency limits

Use `throughput` to maximize total output within the GPU budget, or `throughput_per_gpu` to favor
efficiency. Use a `goodput` target when only output meeting your service-level agreement (SLA)
should count toward the score. The [optimization reference](#optimization-goal) lists all targets.

For latency-constrained selection, add this `evaluation` section and replace `optimization` with:

```yaml
evaluation:
  sla: {ttft_ms: 500, itl_ms: 50}
optimization:
  target: goodput_per_gpu
  strict_sla: true
  constraints: {max_candidate_gpus: 4}
```

`goodput_per_gpu` rewards SLA-compliant throughput per GPU. `strict_sla: true` additionally
filters candidates by the configured aggregate mean latency bounds. This is an efficiency search;
see the [minimum-GPU migration example](migrate-from-aiconfigurator.md#keep-minimum-gpu-sizing-on-the-compatibility-cli)
for the legacy sizing workflow.

<a id="predict-a-recommended-configuration"></a>

### 6.2 Predict a recommended configuration

If the search found a feasible candidate, pass a saved YAML directly to `predict`:

```bash
aisimulate predict \
  --config ./recommendation-output/recommendations/0001.yaml \
  --output-dir ./best-prediction
```

The saved YAML contains concrete values with no search domains, `preset`, `optimization`, or
`optimizer`. The command prints the prediction metrics table shown earlier and writes
`best-prediction/prediction.json`. It is a prediction input; deployment
manifests and launch scripts are covered in the [migration guide](migrate-from-aiconfigurator.md).

<a id="common-options"></a>

## 7. Common Options

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
| `--detail` | comma-separated selectors | omitted | Add `summary`, `memory`, `time`, or `all`. See [prediction details](#prediction-details). |
| `--online` | flag | `false` | Pace prediction against the real wall clock instead of virtual time. The selected stack must advertise online support. |

The CLI deliberately does not expose field-specific flags such as `--request-per-second` or
`--num-workers`. YAML is the authoritative semantic configuration surface.

<a id="override-semantics"></a>

### 7.1 Override Semantics

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

## 8. Choose an execution stack

`--stack engine` is the default and uses the built-in offline runner. To use Dynamo-owned
routing and Planner behavior, install the optional integration and select it explicitly:

```bash
python3 -m pip install aisimulate ai-dynamo
aisimulate predict --stack dynamo --config prediction.yaml --output-dir ./dynamo-prediction
aisimulate recommend --stack dynamo --config recommendation.yaml --output-dir ./dynamo-recommendation
```

A config containing `router` or `planner` requires the corresponding installed adapters; the
Dynamo integration provides them. Use the same stack when predicting a configuration saved by
that stack's recommendation run. See the complete [Dynamo prediction](#complete-dynamo-prediction-example)
and [recommendation](#dynamo-scalar-recommendation-example) examples below.

`predict --online` requests wall-clock-paced execution from a stack that supports it.
The built-in `engine` stack supports offline execution only; `recommend` is always offline.
Online execution paces the simulation and does not launch a real serving endpoint.

If Dynamo is unavailable, the error includes the installed stack names. For example, when only
the built-in engine is installed:

```text
stack 'dynamo' is unavailable; installed stacks: engine. Install the distribution that provides the requested stack.
```

Install the integration in the same Python environment as `aisimulate`. Implementation details
for stack and adapter authors are in [Sweeper architecture](../sweeper/architecture.md#unified-cli-integration).

<a id="configuration-model"></a>

## 9. Configuration Model

The sections below are a reference. Jump to [traffic](#traffic), [engine](#engine),
[search domains](#recommendation-domains), [presets](#presets-and-default-ranges),
[optimization goals](#optimization-goal), [search controls](#optimizer-controls), or [outputs](#outputs).

These fragments show the available top-level sections; they are not complete runnable inputs.
Use the prediction and recommendation examples above for complete configurations.

Both commands use one strict YAML model:

```yaml
traffic: {}
engine: {}
router: {}
planner: {}
evaluation: {}
```

`recommend` extends that model with:

```yaml
optimization: {}
optimizer: {}
```

The command determines the document type. There is no top-level `kind` or stack field.

| Section | `predict` | `recommend` | Purpose |
|---|---|---|---|
| `traffic` | Optional | Optional | Request source, load shape, and stopping condition. Uses the default synthetic request traffic when omitted. |
| `engine` | Required | Required | Model, hardware, backend, topology, and worker roles. |
| `router` | Optional adapter | Optional adapter | Dynamo routing policy; round robin when omitted. Requires the integration when configured. |
| `planner` | Optional adapter | Optional adapter | Dynamo runtime scaling; disabled when omitted. Requires the integration when configured. |
| `evaluation` | Optional | Optional | Service-level objective (SLA) thresholds used for reporting and goals. |
| `optimization` | Rejected | Required | Recommendation objective and candidate GPU constraints. |
| `optimizer` | Rejected | Optional | Public search controls. |

Unknown fields are rejected everywhere. Every semantic configuration knob is an explicit, typed YAML
field. The selected stack, backend, policy, timing model, or capacity model determines which
conditional fields are legal; there is no generic configuration passthrough mapping.

<a id="presets-and-default-ranges"></a>

## 10. Presets and Default Ranges

The reference tables below use five columns:

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
  repeated prose below the table.

A preset is a list of complete mappings. Each mapping must specify every knob belonging to the
smallest preset class shown in the table, including a `null` value for a conditionally inactive knob.
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

### 10.1 Parallelism Preset Behavior

`engine.workers.<role>.parallelism` uses the same `preset: default` spelling as every other
preset-capable object:

```yaml
parallelism:
  preset: default
```

The default preset selects complete, feasible parallelism mappings within the GPU budget.
It accounts for the model, backend, memory capacity, and supported worker shapes. Its current
worker-size ladder is `1, 2, 4, 8, 16` GPUs with pipeline parallelism fixed at `1`; replica counts
share the candidate budget. See the [projection algorithm](../sweeper/architecture.md#parallelism-search-projection)
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

<a id="traffic"></a>

## 11. Traffic

Traffic is always expressed as three concepts:

```yaml
traffic:
  source: {}
  load: {}
  stop: {}
```

`source` defines requests or sessions, `load` defines when they begin, and `stop` defines when the run
ends. The load and stop unit follows the source type.

Omitting `traffic` is equivalent to this concrete default:

```yaml
traffic:
  source:
    type: synthetic
    input_tokens: 1024
    output_tokens: 128
  load:
    type: concurrency
    concurrency: 10
  stop:
    requests: 100
```

The default source contains independent requests with fixed input sequence length (ISL) and output
sequence length (OSL); it does not sample token-length distributions or create sessions. A supplied
`traffic` mapping does not merge recursively with this example. Once `traffic` is present, its normal
source, load, and stop validation applies.

The default run uses `100` requests at concurrency `10`, or 10 times the active concurrency, following
the current SA convention.

<a id="traffic-fields"></a>

### 11.1 Traffic Fields

| Knob | Default | Default Range | Preset | Rules |
|---|---:|---|---|---|
| `traffic.source.type` | `synthetic` | `x` | `-` | `synthetic`, `synthetic-session`, or `trace`. |
| `traffic.source.input_tokens` | `1024` | `x` | `-` | Positive; `synthetic` only. |
| `traffic.source.output_tokens` | `128` | `x` | `-` | Positive; `synthetic` only. |
| `traffic.source.images` | Unset | `x` | `-` | Fixed positive `height`, `width`, `count` (default 1); synthetic analytical EPD only; requires `engine.workers.encoder`. |
| `traffic.source.new_input_tokens_per_turn` | `1024` | `x` | `-` | Positive; `synthetic-session` only. |
| `traffic.source.output_tokens_per_turn` | `128` | `x` | `-` | Positive; `synthetic-session` only. |
| `traffic.source.session.turns` | `4` | `x` | `-` | At least `2`. |
| `traffic.source.session.shared_prefix_ratio` | `0` | `x` | `-` | From `0` through `1`. |
| `traffic.source.session.prefix_groups` | `0` | `x` | `-` | Nonnegative; positive when prefix ratio is positive. |
| `traffic.source.session.inter_turn_delay_ms` | `0` | `x` | `-` | Nonnegative. |
| `traffic.source.paths` | Required for trace | `x` | `-` | One path except `dynamo`, which permits multiple. |
| `traffic.source.format` | `mooncake` | `x` | `-` | See [Trace Format Compatibility](#trace-format-compatibility). |
| `traffic.source.block_size` | `512`; embedded for `dynamo` and `weka` | `x` | `-` | Positive. For embedded formats, an explicit value is an equality assertion. |
| `traffic.source.nested_timestamp_basis` | `auto` | `x` | `-` | `auto`, `absolute`, or `relative`; Weka only. |
| `traffic.load.type` | `concurrency` | `x` | `-` | Synthetic: `concurrency`, `poisson`, `constant_rate`, or `kv_capacity_fraction`; trace: `trace_timestamps` or `concurrency`. |
| `traffic.load.concurrency` | `10` | `-` | `-` | Positive integer; explicit domains are allowed in `recommend`. |
| `traffic.load.requests_per_second` | `null` | `-` | `-` | Positive; synthetic request open-loop load only. |
| `traffic.load.sessions_per_second` | `null` | `-` | `-` | Positive; synthetic session open-loop load only. |
| `traffic.load.seed` | `42` | `x` | `-` | Nonnegative; `poisson` only. |
| `traffic.load.fraction` | `null` | `-` | `-` | Positive finite number; `kv_capacity_fraction` only and may exceed `1`. |
| `traffic.load.speedup` | `1` | `-` | `-` | Positive; trace timestamp load only. |
| `traffic.load.agentic_lanes` | `null` | `x` | `-` | Positive integer; `weka`, `agentic_mooncake`, or agentic `dynamo` timestamp replay only. |
| `traffic.stop.requests` | `100` for default traffic | `x` | `-` | Positive integer; 10× default concurrency; synthetic request source only. |
| `traffic.stop.requests_per_load_unit` | `null` | `x` | `-` | Positive; synthetic request source only. |
| `traffic.stop.sessions` | `null` | `x` | `-` | Positive integer; synthetic session source only. |
| `traffic.stop.sessions_per_load_unit` | `null` | `x` | `-` | Positive; synthetic session source only. |
| `traffic.stop.max_virtual_time_seconds` | `null` | `x` | `-` | Positive; supported trace formats only. |

<a id="synthetic-request-source"></a>

### 11.2 Synthetic Request Source

```yaml
traffic:
  source:
    type: synthetic
    input_tokens: 1024
    output_tokens: 128
  load:
    type: poisson
    requests_per_second: 8
    seed: 42
  stop:
    requests: 100
```

`synthetic` generates independent single requests. It does not accept a `session` mapping.

<a id="synthetic-session-source"></a>

### 11.3 Synthetic Session Source

```yaml
traffic:
  source:
    type: synthetic-session
    new_input_tokens_per_turn: 1024
    output_tokens_per_turn: 128
    session:
      turns: 4
      shared_prefix_ratio: 0.5
      prefix_groups: 16
      inter_turn_delay_ms: 1000
  load:
    type: constant_rate
    sessions_per_second: 8
  stop:
    sessions: 100
```

`synthetic-session` generates multi-turn sessions. It requires a `session` mapping, and requests
inside one session execute in turn order.

<a id="synthetic-load-and-stop"></a>

### 11.4 Synthetic Load and Stop

Both synthetic source types support closed-loop concurrency, Poisson arrivals, constant-rate
arrivals, and recommendation-only KV-capacity-relative load as listed in the Traffic table.

The source-specific open-loop rate field is:

- `requests_per_second` for `source.type: synthetic`.
- `sessions_per_second` for `source.type: synthetic-session`.

The rate is a positive finite number. `concurrency` is a positive integer and counts independent
requests for `synthetic` or active sessions for `synthetic-session`. At most one turn of a session is
active at a time. `seed` is a nonnegative integer and defaults to `42`. `fraction` is greater than `0`
and finite; it has no upper bound. `fraction: 1` targets a concurrency whose estimated KV working set
equals the candidate's usable KV capacity. Values greater than `1` intentionally oversubscribe that
capacity. They remain valid because excess work can queue; they do not mean that the engine has more
physical KV memory.

The stop fields follow the source unit. The fixed-count field is a positive integer. The
load-relative field is
a positive number and resolves to `max(1, round(count_per_load_unit * load_unit))`. The load unit is
concurrency for `concurrency` and resolved `kv_capacity_fraction` traffic, requests per second for a
`synthetic` open-loop source, or sessions per second for a `synthetic-session` open-loop source.

For `synthetic-session`, a session with four turns contributes four requests but only one unit to the
load and stopping condition.

`sessions_per_second` controls the arrival rate of new sessions. For a multi-turn session, it schedules
the first turn; later turns follow that session's completion and `inter_turn_delay_ms` rules and do not
count as new load arrivals. `requests_per_second` schedules independent single requests.

In a recommendation input, source type, token fields, session shape, and stopping condition stay
concrete. Only `traffic.load` rows whose Default Range is not `x` can be search domains. A
`kv_capacity_fraction` recommendation is materialized as a concrete `concurrency` load in each
recommended prediction YAML.

<a id="trace-source"></a>

### 11.5 Trace Source

```yaml
traffic:
  source:
    type: trace
    paths:
      - traces/requests.jsonl
    format: mooncake
    block_size: 512
  load:
    type: trace_timestamps
    speedup: 1.0
  stop:
    max_virtual_time_seconds: 300
```

Omitting `traffic.stop` for a trace runs to end of trace. `max_virtual_time_seconds` is trace-only and
cannot be used for synthetic traffic. Trace source, format, and token/session content stay concrete in
a recommendation input; only a numeric trace-load field can be a domain.

`speedup: N` divides authored timing by `N`; for example, `2` replays the timing twice as fast. For
Mooncake session traces it scales both first-turn arrival timestamps and inter-turn delays. For
agentic traces it scales root-node timestamps and the combined dependency delay and tool wait.
`speedup` is deliberately rejected with `load.type: concurrency`: concurrency replaces authored
first-arrival pacing, and inter-turn or dependency delays remain unscaled.

<a id="trace-format-compatibility"></a>

### 11.6 Trace Format Compatibility

| Format | JSONL Unit | Allowed Load | `speedup` | `max_virtual_time_seconds` | Other Constraints |
|---|---|---|---|---|---|
| `mooncake` | One request or session turn with a full prompt | `trace_timestamps`, `concurrency` | Timestamp load only | Supported | None specific to the format. |
| `mooncake-delta` | One session turn; follow-up input is only the new input delta | `trace_timestamps`, `concurrency` | Timestamp load only | Supported | Aggregated deployment only; `planner.policy` must be `disabled`. |
| `agentic_mooncake` | One request node in a dependency graph | `trace_timestamps` | Supported | Not supported; omit it | Aggregated deployment only; `planner.policy` must be `disabled`. |
| `weka` | A raw kv-cache-tester or published AgentX JSON/JSONL corpus; directories are traversed recursively and JSONL files may contain multiple plays | `trace_timestamps` | Supported | Not supported; omit it | Aggregated deployment only; source block size is embedded and the result is functionally qualified. |
| `applied_compute_agentic` | One complete session, expanded into `num_turns + 1` requests | `concurrency` | Not supported; omit it | Supported | Source rows have no first-turn timestamps. |
| `dynamo` standard trace | Native request-trace records, possibly across multiple files | `trace_timestamps`, `concurrency` | Timestamp load only | Supported | The embedded trace block size is authoritative. |
| `dynamo` agentic trace | Native agentic request-trace records, possibly across multiple files | `trace_timestamps` | Supported | Not supported; omit it | Aggregated deployment only; `planner.policy` must be `disabled`. |

The `dynamo` loader detects whether its records are standard or agentic and applies the corresponding
row above. If `traffic.source.block_size` is supplied for `dynamo` or `weka`, it must match the
embedded block size. For the other formats, `block_size` is the trace hash-block size used to
reconstruct prompts.

Weka is the public AgentX source format and AISimulate is its prediction entry point. AISimulate
deterministically lowers Weka into Agentic Mooncake v2, the versioned producer-neutral interchange
format, and then validates that lower IR as a `ValidatedAgenticGraph`, the runtime representation.
Dynamo is an optional integration and is not required to parse, convert, or predict a Weka corpus.
Two producer timestamp conventions exist: raw kv-cache-tester nested request timestamps are relative
to their subagent marker, while SemiAnalysis-published AgentX timestamps are root-trace absolute.
`nested_timestamp_basis` may select either convention explicitly. When omitted (or set to `auto`),
AISimulate scans every nested request in every JSON/JSONL row before lowering. If any child timestamp
is earlier than its subagent marker by more than the join epsilon, the complete corpus is interpreted
as relative; otherwise it is interpreted as absolute. This is one corpus-wide heuristic, never a
per-request rewrite. It cannot prove that a corpus is homogeneous: a malformed absolute request can
select relative for the entire corpus, while relative offsets that are all at or above their markers
can select absolute. Producers with ambiguous data should set the basis explicitly. Both conventions
lower uniformly to root-absolute canonical timestamps. The selected basis and whether it was inferred
heuristically or configured are logged; the resolved value is reported as
`weka_nested_timestamp_basis` and included in source identity.
The neutral importer accepts mixed source models and preserves each request's model label in graph
provenance and identity. Execution currently supports one target: before the graph enters
the model-neutral `WorkloadDriver`, AISimulate projects every request onto the one model configured by
`engine.model`. The report records the sorted source-model set, target model, and
`project_to_configured_target` policy under `agentic_model_projection`; per-node heterogeneous timing
models are not supported yet.
The lowering records a zero-based `source_play_ordinal` on every v2 row so materialized graphs retain
deterministic directory and JSONL order; missing ordinals remain valid for older v2 inputs, but an
ordered graph must provide one unique contiguous ordinal for every play.
An explicit `agentic_lanes: N` assigns plays round-robin to N client lanes. The next play starts when
the current play's client work ends: all authored requests complete on success, or all dispatched
requests become terminal after a failure skips undispatched work. Background requests remain part
of their play even without a parent join. P/D source holds and other server cleanup may outlive this
boundary; they still constrain engine admission and final drain, but do not delay client submission.
Omitting the field preserves authored timestamp behavior; corpus wrapping and fixed-duration lane
orchestration are outside the version 1 contract.

<a id="mooncake-and-mooncake-delta-jsonl"></a>

### 11.7 Mooncake and Mooncake Delta JSONL

Both formats use the same row schema. Rows with the same `session_id` are turns in file order.

| Field | Required | Semantics |
|---|---|---|
| `request_id` | No | Request identity. |
| `session_id` | No | Groups rows into a session; an omitted value creates a one-row session. |
| `input_length` or `input_tokens` | No | Input token count; defaults to the capacity represented by `hash_ids`. |
| `output_length` or `output_tokens` | Yes | Output token count. |
| `output_token_ids` | No | Exact output tokens; its length must equal the output token count. |
| `hash_ids` | Yes | Prompt hash blocks at `traffic.source.block_size`. |
| `timestamp` or `created_time` | Conditional | Virtual timestamp in milliseconds. Required for the first row of every session under `trace_timestamps`. |
| `delay` or `delay_ms` | No | Inter-turn delay in milliseconds. It must be omitted or zero on a session's first row; later rows use it instead of a timestamp difference. |
| `priority`, `strict_priority`, `policy_class` | No | Scheduling metadata. |

With `format: mooncake`, every row describes that turn's complete prompt. With
`format: mooncake-delta`, the first row describes the initial prompt, while each later row's input and
`hash_ids` describe only new input for that turn. Replay builds the next complete prompt by appending
the prior generated output and the new delta. Using `mooncake-delta` on a full-prompt trace would
double-count prior context.

<a id="agentic-mooncake-jsonl"></a>

### 11.8 Agentic Mooncake JSONL

`agentic_mooncake` includes all Mooncake request fields above, but `request_id` is required, nonempty,
and unique. Each row is an independently schedulable request node rather than a turn inferred only
from session order. It adds these fields:

| Field | Default | Semantics |
|---|---:|---|
| `wait_for` | `[]` | Request IDs that must all complete first. Unknown IDs, self-dependencies, and cycles are rejected. |
| `delay` / `delay_ms` | `0` | Delay after the last dependency completes. |
| `tool_wait_ms` | `0` | Additional tool wait after dependencies; scheduling delay is `delay + tool_wait_ms`. |
| `timestamp` / `created_time` | `0` for roots | Ready time for a node with an empty `wait_for`; dependent-node timestamps do not control release. |
| `request_kind`, `branches`, `prefix_reset`, `tool_events` | Empty | Producer metadata accepted by the format. Scheduling is controlled by `wait_for`, and cache identity is carried by `hash_ids`; these metadata fields do not independently change replay behavior. |

A dependent node becomes ready after the latest request in `wait_for` completes, plus its `delay` and
`tool_wait_ms`. `speedup` therefore scales both authored root arrivals and those post-dependency waits.

<a id="applied-compute-agentic-jsonl"></a>

### 11.9 Applied Compute Agentic JSONL

Each row is one session with `num_turns`, `input_prompt_length`, arrays
`assistant_response_length`, `tool_call_output_length`, and `tool_call_latency`, plus
`final_assistant_response_length`. Each array length must equal `num_turns`; tool latency is in
seconds. The row expands to `num_turns` assistant/tool turns plus one final assistant request. The
input grows cumulatively by each assistant response and tool output. Because the format has no
first-session arrival timestamps, it requires `load.type: concurrency` and rejects `speedup`.

> [!NOTE]
> `max_virtual_time_seconds` limits the total simulated virtual time of one prediction or recommendation
> candidate. It is not a separate processing-time limit for each path in `traffic.source.paths`, and it
> is not a real wall-clock timeout. The limit is a soft scheduling cutoff: events at the cutoff are
> processed, replay stops before the first event after the cutoff, and requests still in flight can be
> reported as incomplete. The reported duration can extend slightly past the cutoff while an already
> running engine pass finishes.

<a id="engine"></a>

## 12. Engine

`engine` owns topology, model and runtime identity, worker pools, scheduling, KV cache, timing, and
prefill-to-decode transfer behavior.

```yaml
engine:
  mode: aggregated
  model: Qwen/Qwen3-32B-FP8
  hardware: h200_sxm
  backend: vllm
  backend_version: "0.24.0"
  context_length: max
  workers:
    aggregated:
      parallelism:
        replicas: 2
        tensor: 1
        pipeline: 1
        attention_data: 1
        moe_tensor: 1
        moe_expert: 1
      scheduler:
        max_batched_tokens: 8192
        max_sequences: 256
        prefill_schedule_interval: 1
      kv_cache:
        block_size: 64
        prefix_caching: true
        bytes_per_token: auto
        capacity:
          type: default
          memory_fraction: 0.9
          cuda_graph_reserved_bytes: 0
      timing:
        type: default
        forward_model: op_level
      startup_seconds: 0
```

<a id="engine-fields"></a>

### 12.1 Engine Fields

`<role>` is `aggregated`, `prefill`, or `decode` as selected by `engine.mode`. AFD uses
`engine.afd` for its A/F pool and an optional opposite-phase regular worker.

| Knob | Default | Default Range | Preset | Rules |
|---|---:|---|---|---|
| `engine.mode` | `aggregated` | `{choices: [aggregated, disaggregated]}` | `-` | `aggregated`, `disaggregated`, or explicit `afd`. AFD cannot be mixed into a recommendation mode domain. |
| `engine.model` | Required | `x` | `-` | Nonempty and fixed during recommendation. |
| `engine.hardware` | Required | `auto` | `-` | Fallback hardware identifier; `recommend` also accepts `auto` resolved from `optimization.hardware`. P/D workers may override it. |
| `engine.backend` | `vllm` | `{choices: [vllm, sglang]}` | `-` | `vllm`, `sglang`, or `trtllm`; explicit choices may include supported alternatives. |
| `engine.backend_version` | `null` | `x` | `-` | Fixed when set. |
| `engine.context_length` | `"max"` | `x` | `-` | `"max"` derives the effective maximum from the resolved Hugging Face model config; a concrete value must be positive. |
| `engine.workers` | Mode-dependent | `x` | `-` | Aggregated role; prefill plus decode roles; or the optional opposite-phase companion for AFD+P/D. Aggregated and disaggregated modes also support an optional analytical `encoder` pool. |
| `engine.workers.prefill.hardware`, `.decode.hardware` | Inherit `engine.hardware` | `x` | `-` | Concrete nonempty SKU; no `auto` or search domain. Disaggregated roles only; aggregated workers and AFD companions reject hardware overrides. Saved recommendations retain the overrides. |
| `engine.workers.encoder.tensor`, `.replicas`, `.batch_size` | `1` | Scalar or finite `choices` | `encoder` | Positive; batch size at most 8. Not a language-worker parallelism preset. |
| `engine.workers.encoder.hardware`, `.backend_version` | Inherit/resolve | `x` | `-` | Encoder hardware and performance data; backend follows language backend. Saved prediction YAML pins resolved values. |
| `engine.workers.encoder.latency_correction`, `.rate_degradation` | `1.0`, `0.9` | `x` | `-` | Finite positive factors; degradation at most 1. See [EPD CLI semantics](../sweeper/epd.md#unified-cli). |
| `engine.workers.<role>.parallelism.preset` | `default` in `recommend` | `auto` | `-` | Generated default space, complete mapping list, `false`, or `{}`. |
| `engine.workers.<role>.parallelism.replicas` | `1` | Feasible positive values within GPU budget | `parallelism` | Positive. |
| `engine.workers.<role>.parallelism.tensor` | `1` | Feasible registry values | `parallelism` | Positive and model/backend compatible. |
| `engine.workers.<role>.parallelism.pipeline` | `1` | Feasible registry values | `parallelism` | Positive and model/backend compatible. |
| `engine.workers.<role>.parallelism.attention_data` | `1` | Feasible registry values | `parallelism` | Positive and model/backend compatible. |
| `engine.workers.<role>.parallelism.moe_tensor` | `1` | Feasible registry values | `parallelism` | Positive and model/backend compatible. |
| `engine.workers.<role>.parallelism.moe_expert` | `1` | Feasible registry values | `parallelism` | Positive and model/backend compatible. |
| `engine.workers.<role>.scheduler.max_batched_tokens` | Aggregated/prefill/decode: `8192` | Prefill/aggregated: `{choices: [8192, 16384, 32768]}`; decode: `-` | `-` | Positive. |
| `engine.workers.<role>.scheduler.max_sequences` | Aggregated `256`; prefill `1`; decode `256` | Prefill: `{choices: [1, 2, 4, 8, 16, 32, 64, 128, 256]}`; aggregated/decode: `{choices: [256, 512, 1024]}` | `-` | Positive. |
| `engine.workers.<role>.scheduler.prefill_schedule_interval` | `1` | `x` | `-` | `predict` only. Positive. Values above one throttle prefill admission only for vLLM attention-DP groups. |
| `engine.workers.<role>.kv_cache.block_size` | vLLM `64`; SGLang `1`; TensorRT-LLM `32` | `-` | `-` | Positive and backend-supported. Defaults are backend-specific, not version-specific. |
| `engine.workers.<role>.kv_cache.prefix_caching` | `true` | `x` | `-` | Backend-supported. |
| `engine.workers.<role>.kv_cache.bytes_per_token` | `auto` | `x` | `-` | Positive when concrete. `auto` resolves once per worker role from the model and that role's TP/PP/MoE shape. |
| `engine.workers.<role>.kv_cache.capacity.type` | `default` | `x` | `-` | `default` or `fixed`. |
| `engine.workers.<role>.kv_cache.capacity.memory_fraction` | vLLM/TensorRT-LLM `0.9`; SGLang `0.88` | `-` | `-` | `(0, 1]`; `default` capacity only. |
| `engine.workers.<role>.kv_cache.capacity.blocks` | `null` | `x` | `-` | Positive and required for `fixed` capacity. |
| `engine.workers.<role>.kv_cache.capacity.cuda_graph_reserved_bytes` | `0` | `-` | `-` | `predict` only. Integer from `0` through `2**53`; `default` capacity only. |
| `engine.workers.<role>.kv_cache.host_offload.num_host_blocks` | Required when `host_offload` is present | `x` | `-` | Positive; fixed descriptor, aggregated vLLM only. |
| `engine.workers.<role>.kv_cache.host_offload.d2h_bandwidth_gbps` | `32.0` | `x` | `-` | Finite and nonnegative. |
| `engine.workers.<role>.kv_cache.host_offload.h2d_bandwidth_gbps` | `32.0` | `x` | `-` | Finite and nonnegative. |
| `engine.workers.<role>.timing.type` | `default` | `x` | `-` | `default`, `fixed`, or `polynomial`. |
| `engine.workers.<role>.timing.prefill_ms` | `null` | `x` | `-` | Nonnegative and required for `fixed` timing. |
| `engine.workers.<role>.timing.decode_ms` | `null` | `x` | `-` | Nonnegative and required for `fixed` timing. |
| `engine.workers.<role>.timing.forward_model` | `op_level` | `x` | `-` | `op_level` or `fpm`; `default` timing only. `fpm` replays whole-forward (FPM) latency measured for the role's exact model, hardware, backend version, parallel shape and quantization, and fails closed when no such cell exists. |
| `engine.workers.<role>.startup_seconds` | `0` | `x` | `-` | Nonnegative. |
| `engine.kv_transfer.bytes_per_token` | `auto` | `x` | `-` | Positive when concrete. Independent from worker KV-cache geometry; `auto` resolves from the prefill/source role's TP/PP/MoE shape. |
| `engine.kv_transfer.bandwidth_gb_per_second` | `null` | `x` | `-` | Positive when set; `null` disables transfer delay. |
| `engine.kv_transfer.timing_mode` | `destination_missing` | `x` | `-` | `full_prompt` or `destination_missing`; disaggregated mode only. |
| `engine.afd.phase` | Required for AFD | `x` | `-` | `both` for pure AFD; `prefill` or `decode` when `combined_with_pd: true`. |
| `engine.afd.combined_with_pd` | Required for AFD | `x` | `-` | Selects pure `afd` or internal `afd+pd`; it is never inferred from workers. |
| `engine.afd.a_batch_size` | Required for AFD | User-supplied finite domain | `-` | Positive, memory-qualified A-worker batch size. Prediction requires one value. |
| `engine.afd.n_a_nodes`, `n_f_nodes`, `tp_a` | Required for AFD prediction | Enumerated within the GPU budget | `-` | Positive concrete topology fields. Recommendation may optionally constrain `tp_a`. |
| `engine.afd.f_moe_ep_size` | `1` for prediction; model-derived domain for recommendation | Optional choices | `-` | Positive; recommendation also accepts `n_f_nodes` or `ffn_tp`. Dense models require `1`. |
| `engine.afd.num_microbatches` | `3` for prediction | `{choices: [2, 3, 4]}` | `-` | Positive. |
| `engine.afd.pipeline_model` | `optimistic` for prediction | `{choices: [optimistic, conservative]}` | `-` | `optimistic`, `conservative`, or `serial`. |
| `engine.afd.comm_overhead_factor` | `1.0` | `x` | `-` | Positive factor applied once by the AFD evaluator. |
| `engine.afd.boundary_on_attn` | `true` | `x` | `-` | Fixed A/F boundary convention. |

`engine.hardware: auto` is valid only in `recommend` and requires the single hardware identifier under
`optimization.hardware`. Every recommended prediction YAML replaces `auto` with that concrete
identifier. Language workers inherit that fallback unless a P/D role overrides it;
an optional analytical encoder pool can also specify its own hardware.

For heterogeneous P/D, set `engine.workers.prefill.hardware` and/or
`engine.workers.decode.hardware`. An omitted role inherits `engine.hardware`. Both roles
share the model, backend and backend version; when an override is present, an omitted
version must resolve identically on both effective SKUs. Pin a common supported version
if their latest versions differ. Prediction uses each role's hardware for timing and KV
capacity. Recommendation checks each role against its own hardware within the shared GPU
budget and saves the overrides in prediction YAML. See the
[complete YAML and CLI example](migrate-from-aiconfigurator.md#48-migrate-heterogeneous-pd-hardware).

An aggregated configuration uses `workers.aggregated`. A disaggregated configuration uses
`workers.prefill` and `workers.decode`:

```yaml
engine:
  mode: disaggregated
  model: Qwen/Qwen3-32B-FP8
  hardware: h200_sxm
  backend: vllm
  backend_version: "0.24.0"
  context_length: max
  kv_transfer:
    bandwidth_gb_per_second: 400
    timing_mode: destination_missing
  workers:
    prefill:
      parallelism: {replicas: 2, tensor: 2, pipeline: 1, attention_data: 1, moe_tensor: 1, moe_expert: 1}
      scheduler: {max_batched_tokens: 8192, max_sequences: 64}
      kv_cache:
        block_size: 64
        prefix_caching: true
        bytes_per_token: auto
        capacity: {type: default, memory_fraction: 0.9}
      timing: {type: default}
      startup_seconds: 0
    decode:
      parallelism: {replicas: 4, tensor: 1, pipeline: 1, attention_data: 1, moe_tensor: 1, moe_expert: 1}
      scheduler: {max_batched_tokens: 8192, max_sequences: 256}
      kv_cache:
        block_size: 64
        prefix_caching: true
        bytes_per_token: auto
        capacity: {type: default, memory_fraction: 0.9}
      timing: {type: default}
      startup_seconds: 0
```

Language-worker roles share the top-level model, backend, backend version, and context length.
Per-role overrides for those settings are rejected. P/D workers may override the hardware fallback.
The optional analytical encoder pool has its own supported hardware and backend-version fields.
If `engine.mode` is a
recommendation domain containing both modes, `workers` declares all three roles. Each concrete
candidate retains only the role or roles active for its selected mode.

An AFD prediction is concrete: pure AFD requires `phase: both` and explicit `n_a_nodes`,
`n_f_nodes`, `tp_a`, and `a_batch_size`. A single-phase topology sets
`combined_with_pd: true` and supplies only its opposite regular worker: a decode-side AFD pool uses
`workers.prefill`, while a prefill-side AFD pool uses `workers.decode`. Recommendation enumerates
the node split under `optimization.constraints.max_candidate_gpus`; it still requires an explicit,
memory-qualified `a_batch_size` domain. AFD supports fixed-length synthetic request traffic and an
absolute load only. The implementation is an analytical foreground-engine execution path, not a
claim that the selected topology was physically served.

`kv_cache.capacity.type: default` and `timing.type: default` replace the previous public name `aic`.
They select the stack's default capacity estimator and timing provider. The initial default registry
preserves current replay and Sweeper behavior, including the backend and role defaults in the table.
Backend-version-specific defaults are not selected automatically by this registry.

`timing.forward_model` selects the forward-pass model behind the default timing provider. `op_level`
composes per-operator measurements; `fpm` replays whole-forward measurements from a collected FPM
cell and requires an exact match on model, hardware, backend version, parallel shape and
quantization. A candidate without a matching cell fails at replay and is recorded as a failed
candidate (reason category `replay_runtime`) rather than silently falling back to `op_level`. In
`fpm` mode with `capacity.type: default`, the KV capacity is also capped to the cell's collected
decode-KV ceiling. The bundled FPM cells are collected at backend versions outside the queryable
version slots; until FPM cells are slot-queryable, set the transitional escape hatch
`AIC_ALLOW_UNLISTED_VERSIONS=1` to use them.

`kv_cache.capacity.type: fixed` requires `blocks`, so users can directly provide cache size. It rejects
`memory_fraction` and nonzero `cuda_graph_reserved_bytes`. Conversely, `type: default` rejects `blocks`
and derives block count from model, hardware, parallelism, block size, backend, memory fraction, and
the caller-provided CUDA graph reservation.

The physical GPU count of a worker role is:

```text
parallelism.replicas * parallelism.tensor * parallelism.pipeline * parallelism.attention_data
```

`moe_tensor` and `moe_expert` describe partitioning within that physical shape and do not multiply
the GPU count again. Aggregated candidate GPU count is the aggregated worker count. Disaggregated
candidate GPU count is the sum of the prefill and decode worker counts.

`full_prompt` charges transfer for the complete prompt KV footprint. `destination_missing` charges
only the prompt KV not already present at the selected decode worker. `kv_transfer` is rejected for
aggregated mode. All `kv_transfer` fields are concrete-only; their Default Range is `x`, and
`recommend` rejects domains on them. Transfer bytes per token describe the PD link payload and may
differ from each worker role's physical `kv_cache.bytes_per_token`.

<a id="native-vllm-host-offload-prediction"></a>

### 12.2 Native vLLM host-offload prediction

The initial public host-offload surface is deliberately fail-closed: it supports one aggregated
vLLM worker role with prefix caching enabled, attention DP equal to one, and no native speculative
decoding. The descriptor is fixed in both `predict` and `recommend`; host capacity and bandwidths
are not search dimensions. `bytes_per_token` belongs to `kv_cache`, not `host_offload`, and is
resolved for the worker role before lowering to the native rank.

```yaml
# host-offload-prediction.yaml
engine:
  mode: aggregated
  model: Qwen/Qwen3-32B-FP8
  hardware: h200_sxm
  backend: vllm
  backend_version: "0.24.0"
  context_length: 4096
  workers:
    aggregated:
      parallelism: {replicas: 1, tensor: 1, pipeline: 1, attention_data: 1, moe_tensor: 1, moe_expert: 1}
      scheduler: {max_batched_tokens: 8192, max_sequences: 16}
      kv_cache:
        block_size: 16
        prefix_caching: true
        bytes_per_token: auto
        capacity: {type: fixed, blocks: 2499}
        host_offload:
          num_host_blocks: 4096
          d2h_bandwidth_gbps: 32.0
          h2d_bandwidth_gbps: 32.0
      timing: {type: default}

traffic:
  source: {type: synthetic, input_tokens: 1024, output_tokens: 128}
  load: {type: concurrency, concurrency: 4}
  stop: {requests: 16}
```

Run it with:

```bash
aisimulate predict --stack engine --config host-offload-prediction.yaml
```

<a id="optional-g3-offload"></a>

#### 12.2.1 Optional G3 offload

G3 is an optional extension to native vLLM host offload, not a standalone cache
mode. It requires `host_offload` to be enabled. Add this mapping inside the
existing `engine.workers.aggregated.kv_cache`, as a sibling of `host_offload`,
and use the same `predict --stack engine` command above:

```yaml
g3_offload:
  scope: cluster_shared
  num_g3_blocks: 8192
```

Only `scope` and `num_g3_blocks` are required. `num_g3_blocks` is a positive integer,
not a nested `capacity` object. Optional controls and defaults are:

- `latency_to_first_byte_ms`: 0.1 ms per transfer.
- `read_bandwidth_gbps` and `write_bandwidth_gbps`: 10 GB/s each per worker.
- `shared_read_bandwidth_gbps` and `shared_write_bandwidth_gbps`: 80 GB/s each
  across the deployment, applied only in `cluster_shared` scope.

These are modeling defaults, not measured or GPU-calibrated values.
Latency is in milliseconds; bandwidth is in decimal
GB/s. Latency and bandwidth must be finite and non-negative. Zero bandwidth
means unlimited, not disabled. Block bytes are `block_size * bytes_per_token`;
`bytes_per_token: auto` uses the existing model/parallelism estimate. G3 stores
complete prefix-block identities, not real tensors or files. In native
ReplaySpec JSON, `g3_offload` and `native_host_offload` are sibling rank fields.

The ownership and bandwidth unit is a replica worker, not a physical host or
an individual TP rank. Set the initial worker count
with `engine.workers.aggregated.parallelism.replicas`:

- `worker_local`: each worker gets `num_g3_blocks` of independent capacity,
  like G2's `num_host_blocks`. Workers cannot reuse each other's stored blocks.
- `cluster_shared`: workers share one pool of `num_g3_blocks`. Duplicate prefix
  blocks occupy capacity once, regardless of how many workers use them.

Independent runs never share cached blocks. With N fixed workers, equal total
capacity means local `num_g3_blocks: C` versus shared `num_g3_blocks: N*C`.
Keep workload, G1/G2 settings, latency, and bandwidth identical for that comparison.
To isolate cache sharing from backend contention, make both shared bandwidth
caps non-binding (for example, set them to zero for unlimited bandwidth).
During scaling, local total capacity changes with worker count; shared capacity
does not. New workers start with cold G1/G2 and local G3, but can read existing
shared G3 blocks. Scale-in drains that worker's accepted I/O and releases its
pins before removing its local pool. Shared blocks survive their writer's exit.
Worker IDs are not reused. Replay requires at least one initial worker; its
existing scaling lifecycle permits scaling to zero and later adding new workers.

Completed G2 stores asynchronously write through to G3. Reads restore a
contiguous prefix through G3 → G2 → G1: G3 completion alone does not make GPU
blocks ready. The adapter reserves G2 destinations one block at a time; a later
miss or capacity failure keeps earlier accepted promotions. New promotions in
one lookup form one read job, and pending promotions defer H2D until a retry.
G3 writes retain the leading new blocks that fit while preserving blocks from
the same write cohort already in G3. A cohort larger than the available capacity is
partially stored; if no block fits, that optional insertion is skipped.
Resident, unpinned G3 blocks use deterministic LRU eviction.

Pending G2 destinations cannot be evicted. Completed promotions become ordinary
evictable G2 entries; a lookup hit alone does not pin them. H2D takes its own
source pins. Request termination detaches from accepted promotions, which may
still finish into G2 without activating G1 for the terminated request.

Transfers use the replay virtual clock and begin first-byte latency when
accepted. Reads start at the current lookup time. First-byte waiters consume
no bandwidth. Read and write budgets are independent; each moving job gets
an equal share of its worker's bandwidth. In `cluster_shared` scope, this is
also capped by its equal share of shared backend bandwidth. `worker_local`
ignores shared bandwidth limits entirely: for example, 16 workers at 10 GB/s
can reach 160 GB/s combined, while the default shared backend caps that at
80 GB/s. Unused shares are not redistributed. This is a fluid bandwidth
model, with no thread-pool or job-concurrency limit; finite backend execution
concurrency can therefore make real transfers slower.

For zero-duration I/O, a repeated request/key at the same timestamp falls back
to cache-miss handling while the recoverable prefix cannot fit in G2. Capacity
relief or time advancement permits retry. This simulator guard adds no pins or
invented latency and does not model native CPU retry overhead.

The prediction summary includes `g3_offload` only when enabled, alongside the
existing TTFT, TPOT, and throughput metrics:

For a reused runtime, G3 counters accumulate across reports and its cache remains warm.

- `lookup_probes`, `lookup_hits`, `lookup_pending` count block probes, including
  retries. Hit ratio is hits / probes; pending probes are not hits.
- `read` and `write` report submitted, completed, and canceled jobs, completed
  bytes, and summed transfer milliseconds including first-byte latency.
  Canceled jobs add no completed bytes.
- `evictions`, `resident_blocks`, `pending_blocks`, and
  `cross_worker_read_blocks` describe tier state and reuse. Cross-worker reuse
  counts completed reads of blocks first written by another worker, not lookup
  hits. `--capture-per-request` retains the existing `requests.jsonl` output.

G3 supports aggregated vLLM with fixed or dynamically scaled workers, prefix caching enabled,
attention DP equal to one, and no native speculative decoding. It does not
support `recommend`, disaggregated mode, or hardware integration.
Replay owns the deployment-wide tier; direct scheduler construction cannot
provide it. Omit `g3_offload` to keep existing G1/G2 behavior.

These controls do not establish filesystem or real-GPU performance parity.
The existing G2 full-external-hit boundary remains: Replay may recompute one
full block where the reference vLLM external-receive path recomputes one token.
G3 byte counters do not resolve that difference.

<a id="router-dynamo-adapter"></a>

## 13. Router (Dynamo Adapter)

Router is not part of the AISimulate core schema. The `dynamo.router` config adapter owns this
section's concrete model, defaults, recommendation domains, validation, and runtime lowering. The
section is accepted only when the selected stack provides that adapter; omitting it keeps an
engine-only configuration engine-only.

| Knob | Default | Default Range | Preset | Rules |
|---|---:|---|---|---|
| `router.policy` | `round_robin` | `{choices: [round_robin, kv_router]}` | `-` | `round_robin` or `kv_router`. |
| `router.prefill_load_model.type` | `none` | `{choices: [none, aic]}` | `-` | `aic` is the current legacy Router identifier and is KV-router-only. |
| `router.overlap_score_credit` | `1.0` | `{choices: [0.0, 0.5, 1.0]}` | `-` | Finite, nonnegative, and KV-router-only. |
| `router.prefill_load_scale` | `1.0` | `{choices: [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]}` | `-` | Finite, nonnegative, and KV-router-only. |
| `router.temperature` | `0.0` | `{choices: [0.0, 0.2, 0.5, 1.0]}` | `-` | Finite, nonnegative, and KV-router-only. |

`round_robin` requires `prefill_load_model.type: none` and has no KV-router-only knobs. The
`kv_router` policy may use either load model. Production-only Router fields remain outside the version
1 contract. Replacing the legacy `aic` load-model name with an implementation-neutral public name is
deferred until the Router exposes that name.

<a id="planner-dynamo-adapter"></a>

## 14. Planner (Dynamo Adapter)

Planner is not part of the AISimulate core schema. The `dynamo.planner` config adapter owns this
section's concrete model, presets, recommendation domains, validation, and runtime lowering. The
section is accepted only when the selected stack provides that adapter.

```yaml
planner:
  policy: disabled
```

| Knob | Default | Default Range | Preset | Rules |
|---|---:|---|---|---|
| `planner.scaling_policy.preset` | `default` in `recommend` | `{choices: [disabled, throughput_180_5, throughput_600_5, load_180_5, load_180_10, hybrid_180_5, hybrid_600_5]}` | `-` | Throughput and hybrid presets require `planner.target: sla` plus TTFT/ITL thresholds. |
| `planner.fpm_sampling.preset` | `default` in `recommend` | `{choices: [small, default, large, fine]}` | `-` | Built-in preset choices, complete mapping list, `false`, or `{}`. |
| `planner.load_sensitivity.preset` | `default` in `recommend` | `{choices: [aggressive, default, conservative]}` | `-` | Built-in preset choices, complete mapping list, `false`, or `{}`. |
| `planner.load_predictor.preset` | `default` in `recommend` | `{choices: [constant_last, arima_raw, arima_log1p, prophet_w20_raw, prophet_w20_log1p, prophet_w50_raw, prophet_w50_log1p, kalman_default_raw, kalman_default_log1p, kalman_reactive_raw, kalman_reactive_log1p]}` | `-` | Interval-level predictor pre-sweep candidates; complete mapping list, `false`, or `{}`. |
| `planner.policy` | `disabled` | `{choices: [disabled, enabled]}` | `-` | `disabled` or `enabled`. |
| `planner.target` | `throughput` | `x` | `-` | Derived from `optimization.target` in `recommend`. |
| `planner.enable_throughput_scaling` | `true` | `{choices: [false, true]}` | `scaling_policy` | `true` requires `planner.target: sla` plus TTFT/ITL thresholds. |
| `planner.enable_load_scaling` | `false` | `{choices: [false, true]}` | `scaling_policy` | Planner policy only. |
| `planner.throughput_adjustment_interval_seconds` | `180` | `{choices: [180, 600]}` | `scaling_policy` | Positive; throughput scaling only. |
| `planner.load_adjustment_interval_seconds` | `5` | `{choices: [5, 10]}` | `scaling_policy` | Positive and shorter than throughput interval when used. |
| `planner.max_num_fpm_samples` | `64` | `{choices: [32, 64, 128]}` | `fpm_sampling` | Positive. |
| `planner.fpm_sample_bucket_size` | `16` | `{choices: [4, 16, 64]}` | `fpm_sampling` | Positive perfect square. |
| `planner.load_scaling_down_sensitivity` | `80` | `{choices: [70, 80, 90]}` | `load_sensitivity` | From `0` through `100`; load scaling only. |
| `planner.load_min_observations` | `5` | `{choices: [3, 5, 8]}` | `load_sensitivity` | Positive; load scaling only. |
| `planner.load_predictor` | `arima` | `{choices: [constant, arima, prophet, kalman]}` | `load_predictor` | Throughput scaling only. |
| `planner.load_predictor_log1p` | `false` | `{choices: [false, true]}` | `load_predictor` | Throughput scaling only. |
| `planner.prophet_window_size` | `50` | `{choices: [20, 50]}` | `load_predictor` | Positive; Prophet only. |
| `planner.kalman_q_level` | `1.0` | `{choices: [1.0, 10.0]}` | `load_predictor` | Positive; Kalman only. |
| `planner.kalman_q_trend` | `0.1` | `{choices: [0.1, 1.0]}` | `load_predictor` | Positive; Kalman only. |
| `planner.kalman_r` | `10.0` | `{choices: [5.0, 10.0]}` | `load_predictor` | Positive; Kalman only. |
| `planner.kalman_min_points` | `5` | `{choices: [3, 5]}` | `load_predictor` | Positive; Kalman only. |
| `planner.max_num_gpus` | `8` | `x` | `-` | Positive Planner runtime scaling ceiling; maps to Dynamo Planner `max_gpu_budget`. |
| `planner.min_workers` | `1` | `-` | `-` | Nonnegative. |
| `planner.prefill_min_workers` | `null` | `-` | `-` | Positive when set. |
| `planner.decode_min_workers` | `null` | `-` | `-` | Positive when set. |

Planner has four independent preset sub-items rather than one whole-Planner preset. Each named or
custom mapping covers every knob in exactly one sub-item. The nested `*.preset` selectors disappear
after materialization; expanded knobs are written directly under `planner` in concrete prediction
YAML.

When `planner.load_predictor.preset` is off, the predictor-name knob is written as
`planner.load_predictor.type` in the recommendation input because `planner.load_predictor` is the
sub-item mapping. It materializes back to the concrete scalar `planner.load_predictor` field in a
recommended prediction YAML.

`scaling_policy`, `fpm_sampling`, and `load_sensitivity` are composed as independent main-search
dimensions. `load_predictor` is different: its candidates run in a pre-sweep for every selected
throughput-adjustment interval, and the winning predictor mapping is materialized into the candidate.

`planner.policy`, `planner.target`, `planner.max_num_gpus`, and the three runtime minimum-worker knobs
are not covered by a preset. `predict` may set a concrete target and otherwise uses `throughput`. In
`recommend`, target is derived: throughput targets and Pareto map to `throughput`, `ttft` and
`e2e_latency` map to `latency`, and goodput targets map to `sla`.

Throughput-based Planner scaling is legal only for the `sla` target with concrete
`evaluation.sla.ttft_ms` and `evaluation.sla.itl_ms`. For `throughput`, `latency`, or `load` Planner
targets, the adapter rejects any scaling-policy preset or custom mapping that enables throughput
scaling before search begins.

Planner runtime limits and recommendation candidate GPU constraints are separate:

- `planner.max_num_gpus`, `min_workers`, `prefill_min_workers`, and `decode_min_workers` constrain
  runtime scaling during one predicted candidate run.
- `optimization.constraints` constrains which static candidate deployments the recommender evaluates.

When `planner.policy: disabled` or the `disabled` scaling-policy preset is selected, no Planner
runtime hook is materialized and conditionally inactive fields are omitted from concrete output.

<a id="evaluation"></a>

## 15. Evaluation

```yaml
evaluation:
  sla:
    ttft_ms: 500
    itl_ms: 50
```

| Knob | Default | Default Range | Preset | Rules |
|---|---:|---|---|---|
| `evaluation.sla.ttft_ms` | `null` | `x` | `-` | Positive and independently optional; an unset field is unbounded. |
| `evaluation.sla.itl_ms` | `null` | `x` | `-` | Positive and independently optional; an unset field is unbounded. |
| `evaluation.sla.e2e_ms` | `null` | `x` | `-` | Positive; mutually exclusive with TTFT plus ITL. |

`goodput` and `goodput_per_gpu` optimization require at least one SLA bound. Planner throughput
scaling specifically uses the `ttft_ms` plus `itl_ms` form when the recommendation target is
SLA-based.

<a id="recommendation-domains"></a>

## 16. Recommendation Domains

A field in a recommendation input is either a concrete value or one explicit domain object.

<a id="choices"></a>

### 16.1 Choices

```yaml
engine:
  backend:
    choices: [vllm, sglang]
```

`choices` must be nonempty and contain unique values valid for the field. A bare YAML sequence never
means a search domain.

<a id="numeric-range"></a>

### 16.2 Numeric Range

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

### 16.3 Domain Validation

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

<a id="optimization-goal"></a>

## 17. Optimization Goal

`optimization` exists only in a recommendation input:

```yaml
optimization:
  target: goodput_per_gpu
  constraints:
    min_candidate_gpus: null
    max_candidate_gpus: 32
```

| Knob | Default | Default Range | Preset | Rules |
|---|---:|---|---|---|
| `optimization.target` | `throughput` | `x` | `-` | Maximize `throughput`, `throughput_per_gpu`, `throughput_per_user`, `goodput`, or `goodput_per_gpu`; minimize `ttft` or `e2e_latency`; or compute `pareto`. |
| `optimization.hardware` | `null` | `x` | `-` | One nonempty hardware identifier; required for `engine.hardware: auto`. |
| `optimization.strict_sla` | `false` | `x` | `-` | When true, reject candidates whose aggregate mean metrics exceed any configured SLA bound before ranking or Pareto analysis. |
| `optimization.constraints.min_candidate_gpus` | `null` | `x` | `-` | Positive when set and no greater than the maximum. |
| `optimization.constraints.max_candidate_gpus` | `32` | `x` | `-` | Positive. |

`pareto` is always the fixed `throughput_per_gpu` and `throughput_per_user` frontier. Goodput targets
require at least one `evaluation.sla` bound. Strict SLA requires at least one bound and controls only
the additional aggregate-mean filter. `optimization.hardware` never accepts a list or inventory
mapping; it supplies the fallback hardware identifier, which P/D workers may override.

<a id="optimizer-controls"></a>

## 18. Optimizer Controls

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

<a id="complete-dynamo-prediction-example"></a>

## 19. Complete Dynamo Prediction Example

Save this as `dynamo-prediction.yaml`:

```yaml
traffic:
  source:
    type: synthetic
    input_tokens: 1024
    output_tokens: 128
  load:
    type: poisson
    requests_per_second: 8
    seed: 42
  stop:
    requests: 100

engine:
  mode: aggregated
  model: Qwen/Qwen3-32B-FP8
  hardware: h200_sxm
  backend: vllm
  backend_version: "0.24.0"
  context_length: max
  workers:
    aggregated:
      parallelism:
        replicas: 2
        tensor: 1
        pipeline: 1
        attention_data: 1
        moe_tensor: 1
        moe_expert: 1
      scheduler:
        max_batched_tokens: 8192
        max_sequences: 256
      kv_cache:
        block_size: 64
        prefix_caching: true
        capacity:
          type: default
          memory_fraction: 0.9
      timing:
        type: default
      startup_seconds: 0

router:
  policy: round_robin
  prefill_load_model: {type: none}

planner:
  policy: disabled

evaluation:
  sla:
    ttft_ms: 500
    itl_ms: 50
```

Run it with the Dynamo integration installed:

```bash
aisimulate predict --stack dynamo --config dynamo-prediction.yaml --output-dir ./dynamo-full-prediction
```

The result is a metrics summary and `dynamo-full-prediction/prediction.json`.

<a id="dynamo-scalar-recommendation-example"></a>

## 20. Dynamo Scalar Recommendation Example

Save this as `dynamo-recommendation.yaml`:

```yaml
traffic:
  source:
    type: synthetic-session
    new_input_tokens_per_turn: 1024
    output_tokens_per_turn: 128
    session: {turns: 4, shared_prefix_ratio: 0, prefix_groups: 0, inter_turn_delay_ms: 1000}
  load:
    type: poisson
    sessions_per_second: {range: {min: 4, max: 32, step: 4, scale: linear}}
    seed: 42
  stop:
    sessions_per_load_unit: 10

engine:
  mode: {choices: [aggregated, disaggregated]}
  model: Qwen/Qwen3-32B-FP8
  hardware: auto
  backend: {choices: [vllm, sglang]}
  backend_version: null
  context_length: max
  workers:
    aggregated:
      parallelism: {preset: default}
      scheduler: {max_batched_tokens: {choices: [8192, 16384]}, max_sequences: {choices: [256, 512]}}
    prefill:
      parallelism: {preset: default}
      scheduler: {max_batched_tokens: 8192, max_sequences: 64}
    decode:
      parallelism: {preset: default}
      scheduler: {max_batched_tokens: 8192, max_sequences: 256}

router:
  policy: {choices: [round_robin, kv_router]}
  prefill_load_model: {type: none}

evaluation:
  sla: {ttft_ms: 500, itl_ms: 50}

optimization:
  target: goodput_per_gpu
  hardware: h200_sxm
  constraints:
    min_candidate_gpus: 1
    max_candidate_gpus: 32

optimizer:
  algorithm: bayesian
  max_trials: 320
  parallelism: 16
  candidate_timeout_seconds: 600
  seed: 42
```

Run the search:

```bash
aisimulate recommend --stack dynamo --config dynamo-recommendation.yaml --output-dir ./dynamo-full-recommendation
```

The result contains `recommendation.json` and any selected prediction YAML files under
`dynamo-full-recommendation/recommendations/`.

Conditional validation applies after a domain is materialized. For example, a round-robin candidate
must resolve the load model to `none`; a recommendation must not rely on an invalid combination being
silently ignored.

<a id="pareto-recommendation-example"></a>

## 21. Pareto Recommendation Example

The following replaces the `optimization` mapping from the scalar example. The single hardware value
is still required because that example uses `engine.hardware: auto`:

```yaml
optimization:
  target: pareto
  hardware: h200_sxm
  constraints:
    min_candidate_gpus: 1
    max_candidate_gpus: 32
```

A scalar recommendation ranks every feasible candidate and saves all distinct concrete prediction
configs. A Pareto recommendation saves the distinct concrete configs on the nondominated front.
The durable result keeps the complete candidate ledger; its selected candidate-ID view is rebuilt
after adapter canonicalization and deduplication so it maps one-to-one to the numbered YAML files.

<a id="outputs"></a>

## 22. Outputs

Read [Understand your prediction](understand-your-prediction.md) for an annotated
report, ITL/TPOT definitions, incomplete-request handling, and SLA interpretation.

Output controls are CLI-only. They never appear in an input or recommended YAML file.

Recommendation output uses the schema-versioned `SweepResult` contract documented in
[`docs/sweeper/results.md`](../sweeper/results.md). It preserves run metadata, a candidate-attempt
ledger, stable status and reason categories, counts, provenance, and candidate-ID selection views.
Replay metrics use unit-bearing names such as `*_tok_s`, `*_ms`, `*_w`, and `*_j`.

The optional names `power_w` and `power_coverage` are reserved by the
[modeled-power contract](../power-model.md). That contract defines active-forward-pass per-GPU
scope, energy-over-active-latency aggregation, omission semantics, and provenance requirements.
`power_coverage` is the share of modeled active time with operation-energy evidence; `power_w`
may be published at or above 90% coverage, so `0.90` passes while `0.899` does not. This formalizes
existing AIC semantics; it neither adds a new power calculation nor implies that every runner or
timing provider implements these fields. Once implemented, normal prediction and recommendation
summaries will always show both labels, with explicit unavailable values and reasons when needed.
Summary power is independent of `--detail`; the planned `energy` selector only adds a breakdown.
JSON fields remain optional under the publication gate. Consult the
[AIC migration guide](migrate-from-aiconfigurator.md) for the current release boundary.

<a id="prediction-directory"></a>

### 22.1 Prediction Directory

```text
<output-dir>/
├── prediction.json
├── requests.jsonl                 # only with --capture-per-request
├── afd-replay-spec.json           # only for AFD
└── afd-qualification.json         # only for AFD
```

- `prediction.json` preserves the selected runner's existing full prediction report.
- `requests.jsonl` contains one record per request when explicitly enabled.
- `afd-replay-spec.json` is the exact, deterministic analytical replay contract for an AFD run,
  including topology, measurement provenance, workload, goal, and any P/D companion.
- `afd-qualification.json` validates and summarizes the A/F pools, routing order, backend version,
  measurement coverage, and GPU accounting. It explicitly records that native launch generation is
  unsupported; it is not a Kubernetes manifest or runnable shell artifact.

<a id="recommendation-directory"></a>

### 22.2 Recommendation Directory

```text
<output-dir>/
├── recommendation.json
└── recommendations/
    ├── 0001.yaml
    ├── 0002.yaml
    └── ...
```

- `recommendation.json` is the canonical lossless result. Its candidate ledger retains feasible,
  infeasible, unsupported, timed-out, and failed rows according to the declared retention policy;
  its counts describe the complete run. `views.top_n` or `views.pareto_front` lists the candidate IDs
  corresponding to numbered YAML files in order.
- Each numbered YAML is a concrete prediction config. It excludes `optimization`, `optimizer`, and
  `preset`, contains no domains or `auto` values, and can be passed directly to
  `aisimulate predict`.

For scalar optimization, file numbering follows best-to-worst rank. For Pareto optimization, it
follows the deterministic display order of the complete nondominated front; that order does not
imply a scalar ranking.

If no feasible candidate exists, the CLI still writes `recommendation.json` with empty views, zero
selected YAML files, complete counts and retained failure records, then exits with status `1`. If
some trials fail but at least one selected config remains, those failures stay in the ledger and the
recommendation succeeds with status `0`.

<a id="existing-output-directories"></a>

### 22.3 Existing Output Directories

Without `--overwrite`, the CLI rejects an existing nonempty output directory. With `--overwrite`, it
replaces only the known output files listed below and preserves unrelated files.

Specifically, overwrite may replace `prediction.json`, `recommendation.json`, `requests.jsonl`,
`afd-replay-spec.json`, `afd-qualification.json`, and numbered `recommendations/NNNN.yaml` files.
Other files, including non-numbered files inside `recommendations/`, are preserved.

<a id="standard-output"></a>

### 22.4 Standard Output

`--format table` prints a concise human-readable summary. `--format json` prints the same summary as
one JSON value for shell automation. Durable artifact formats do not change with this option.

Prediction JSON without `--detail` on standard output is a summary object. Recommendation JSON is an array of selected
rows with `rank`, `score`, `objectives`, `used_gpus`, and `config_path`. Single-objective scores are
signed so higher is better; latency-minimizing targets report negative scores. Pareto rows carry
the raw objective values in `objectives`. Use `recommendation.json` for the complete candidate ledger.

<a id="prediction-details"></a>

### 22.5 Prediction details

```bash
aisimulate predict -c prediction.yaml --detail summary,memory,time \
  --format json --output-dir ./prediction-details
```

`--detail` selects additional reports on `predict`. With no selector, stdout and durable
reports retain their existing shape. With a selector, JSON stdout contains `summary` and
`details`; the same versioned `details` object is added to `prediction.json` and follows the
[prediction-details schema](prediction-details.schema.json). Table output appends selected
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
  This does not provide per-phase/operation timings or SOL. Analytical EPD retains its
  approximation labels in the summary.
- `all`: the three supported sections above, when evidence exists.

Sections without evidence are omitted from `details.sections` and listed with reasons in
`details.skipped`. A memory section with only some estimated roles is `partial` and records
why other roles are unavailable. Missing values are never filled with zero. `energy` and
`source` are unsupported selectors; `all` does not include them. Per-operation diagnostics,
SOL, and power remain [migration gaps](migrate-from-aiconfigurator.md#detailed-diagnostics).

Inspect a recommendation by running `predict --detail` on its saved YAML. Reporting options
are CLI-only; this change adds no YAML configuration fields.

<a id="captured-detail-output"></a>

#### 22.5.1 Captured detail output

Use `prediction.yaml` from [Predict one deployment](#predict-one-deployment): Qwen3-32B-FP8,
one H200, vLLM performance-data version `0.24.0`, 1,024 input tokens, 128 output tokens,
concurrency four, and twelve requests. These outputs were captured from the built-in engine
on 2026-09-15 with AISimulate 0.12.0 and this detail implementation. They are simulation
results; values may change with the implementation or performance data.

```bash
aisimulate predict -c prediction.yaml --detail all \
  --output-dir ./prediction-details-table
```

Selected results, with latency and throughput rounded for readability:

| Section | Field | Captured value |
| --- | --- | ---: |
| `summary` | Completed requests | 12 |
| `summary` | Output throughput (tokens/s) | 164.96 |
| `time` | Mean TTFT (ms) | 262.58 |
| `time` | Mean inter-token latency (ms) | 22.37 |
| `time` | Mean request latency (ms) | 3103.77 |
| `memory` | Weights per rank (bytes) | 34,317,271,040 |
| `memory` | KV capacity estimate per rank (bytes) | 96,706,075,033 |
| `memory` | Estimated GPU blocks per rank | 11,528 |

<details>
<summary>Terminal detail output (selected lines)</summary>

The normal metrics table appears first. This excerpt keeps the complete memory section and
selected summary/time lines; `...` marks omitted lines. Memory is explicitly labeled as the
estimate before native adjustments.

```text
Detail: summary
  scope: serving_workload
  status: available
  completed_requests: 12
  output_throughput_tok_s: 164.96061998541668
  ...

Detail: memory
  scope: capacity_estimate_per_rank
  status: available
  aggregated: available
    stage: before_native_capacity_adjustments
    total_gpu_capacity_bytes: 151397597184
    total_kv_size_bytes: 96706075033
    kv_size_per_token_bytes: 131072
    total_kv_size_tokens: 737808
    source: native
    scheduler_block_size_tokens: 64
    estimated_num_gpu_blocks: 11528
    weights_bytes: 34317271040
    activations_bytes: 1476395008
    runtime_overhead_bytes: 3758096384
    comm_overhead_bytes: 0
    cuda_graph_reserved_bytes: 0

Detail: time
  scope: serving_workload
  status: available
  mean_e2e_latency_ms: 3103.7710699999993
  mean_itl_ms: 22.37155098425194
  mean_ttft_ms: 262.58409499999993
  ...
```

</details>

For machine-readable output, select just memory and use a separate output directory:

```bash
aisimulate predict -c prediction.yaml --detail memory --format json \
  --output-dir ./prediction-details-json
```

JSON stdout has `summary` and `details`. The same `details` object is saved in
`prediction-details-json/prediction.json`.

<details>
<summary>Complete captured JSON details object (summary omitted)</summary>

This is the value of `details`, pretty-printed without changing its values:

```json
{
  "schema_version": "1.0",
  "sections": {
    "memory": {
      "roles": {
        "aggregated": {
          "estimated_num_gpu_blocks": 11528,
          "kv_size_per_token_bytes": 131072,
          "memory_breakdown": {
            "activations_bytes": 1476395008,
            "comm_overhead_bytes": 0,
            "cuda_graph_reserved_bytes": 0,
            "runtime_overhead_bytes": 3758096384,
            "weights_bytes": 34317271040
          },
          "scheduler_block_size_tokens": 64,
          "scope": "capacity_estimate_per_rank",
          "source": "native",
          "stage": "before_native_capacity_adjustments",
          "status": "available",
          "tolerance_adjusted": null,
          "total_gpu_capacity_bytes": 151397597184,
          "total_kv_size_bytes": 96706075033,
          "total_kv_size_tokens": 737808
        }
      },
      "scope": "capacity_estimate_per_rank",
      "status": "available"
    }
  },
  "skipped": {}
}
```

</details>

**When memory evidence is unavailable:** explicitly configure KV blocks and request memory:

```bash
aisimulate predict -c prediction.yaml --detail memory \
  --set 'engine.workers.aggregated.kv_cache.capacity={type: fixed, blocks: 512}' \
  --output-dir ./prediction-details-fixed
```

The normal prediction summary still prints, followed by this captured line:

```text
Skipped memory: aggregated: explicit KV blocks, nested rank input, or a non-AIC capacity provider; no memory component estimate was used by the Python materializer
```

For this run, `details.sections` is empty and `details.skipped.memory` contains the reason
above. `--detail all` would still include summary and time while skipping memory. Power and
energy are absent from all these examples.

<a id="errors-and-exit-codes"></a>

## 23. Errors and Exit Codes

| Exit Code | Meaning |
|---:|---|
| `0` | Successful prediction or recommendation. |
| `1` | Execution failure or a completed recommendation with no feasible candidate. |
| `2` | CLI syntax, YAML parsing, schema, domain, override, or unsupported-combination error. |
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

<a id="troubleshooting"></a>

## 24. Troubleshooting

| Symptom | What to check | Example fix |
|---|---|---|
| `aisimulate: command not found` | The environment containing AISimulate must be active. | From the tutorial directory, run `source .venv/bin/activate`, then `python -m pip show aisimulate`. |
| Installation reports an unsupported Python version | AISimulate requires Python 3.11–3.13. | Check `python3 --version` and create the environment with a supported interpreter. |
| Configuration or trace file cannot be found | Check the path and the directory where you ran the command. | Run from the directory containing `prediction.yaml`, or use absolute paths. |
| Output directory is not empty | Each run needs an empty directory or explicit overwrite. | Add `--output-dir ./another-prediction`, or `--overwrite` to replace known outputs. |
| Stack or config adapter is unavailable | Use the Python environment containing the selected integration. `router` and `planner` are Dynamo-owned sections. | Install `aisimulate ai-dynamo` in that environment and use `--stack dynamo`. |
| `predict` rejects a domain or `optimization` | A search input was passed to a concrete prediction command. | Run `recommend` first, then predict `recommendations/0001.yaml`. |
| `--set` produces an unknown-field or load-validation error | Paths must be supported, and load fields must match the selected load type. | For the quick-start input, use `--set traffic.load.concurrency=8`. To change load type, replace the whole `traffic.load` mapping. |
| No feasible candidate, exit `1` | Inspect `recommendation.json` for candidate status, reason, and GPU/SLA constraints. | Check that the model fits within `max_candidate_gpus`, and that the workload can meet the SLA. |
| A candidate fails to resolve performance data | Check the model, hardware, backend version, and timing mode. FPM needs a matching collected cell. | Use a covered combination from the [support reference](../../README.md#support-and-accuracy) or the [FPM workflow](../../python/aisimulate/docs/fpm/README.md). |
| `--online` is rejected | The selected stack must advertise online support. | Use offline execution with `--stack engine`, or an integration that supports online execution. |

For automation, check the exit code as well as standard output. `--format json` changes successful
summary output; validation and execution errors are reported on standard error. A recommendation
with no feasible result still saves its result ledger and does not provide a YAML to predict.

<a id="related-documentation"></a>

## 25. Related documentation

- [AIC migration guide](migrate-from-aiconfigurator.md)
- [Legacy AIC CLI User Guide](legacy-aic-user-guide.md)
- [Sweeper architecture](../sweeper/architecture.md)
- [Sweeper result schema](../sweeper/results.md)
