<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# CLI reference

Use `aisimulate predict` for one concrete deployment and `aisimulate recommend`
for a bounded search. Start with the [quickstart](../getting-started/quickstart.md)
for a complete input, or consult [configuration](configuration.md) for the YAML
shape. `aisimulate onboard` has its own [FPM self-service guide](../perf-model/fpm-self-service/implementation.md).
`aisimulate perf-model` checks and converts perf-model inputs; see [perf-model tools](#perf-model).
The separate [AIC compatibility CLI](../aic-backward-compatibility/cli.md) retains
its six established workflows.

<a id="commands"></a>

## Commands

| Command | Purpose | Example invocation | Result |
|---|---|---|---|
| `predict` | Evaluate one concrete deployment under a workload. | `aisimulate predict -c prediction.yaml --output-dir ./prediction-output` | A metrics summary and `prediction-output/prediction.json`. |
| `recommend` | Search deployment and load choices for an optimization goal. | `aisimulate recommend -c recommendation.yaml --output-dir ./recommendation-output` | Ranked configurations, `recommendation.json`, and concrete YAML files under `recommendations/`. |
| `perf-model` | Check a perf-model input, convert old estimator fields to it, or fetch FPM data. | `aisimulate perf-model check ./perf-model.yaml` | One line per worker type with `ok` or `failed` and the error. See [perf-model tools](#perf-model). |

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

### Eager execution and CUDA graphs

These flags control the vLLM worker used for FPM collection. `--enforce-eager`
is a worker flag, not an `aisimulate predict` or `onboard init` option.

| Setting | CUDA graphs | `torch.compile` |
|---|---|---|
| Worker `--enforce-eager` | Disabled; eager execution. | Disabled. |
| Collector `--fpm-enforce-eager` | Disabled; emits worker `--enforce-eager`. | Disabled. |
| Onboarding `--cudagraph-mode NONE` / worker `cudagraph_mode=NONE` | Disabled. | Unchanged; may remain enabled. |

See [vLLM's compilation guide](https://docs.vllm.ai/en/stable/design/debug_vllm_compile/)
for the worker flag semantics. DSV4.1 onboarding automatically selects the
required eager setting from model metadata; users need no extra flag. The
[FPM self-service guide](../perf-model/fpm-self-service/implementation.md#serving-roles-and-cuda-graph-settings)
explains the current qualification limit and conflicting graph settings.

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
- `engine.ais_perf_model.<key>` changes one perf-model config, or creates it. A list of
  configs can only be replaced as a whole. See
  [Unified perf-model input](../perf-model/configuration.md#unified-perf-model-input).

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
state that reason. Native AFD exports operation-energy evidence; numeric power
depends on operation-energy coverage. FPM, fixed, polynomial, and analytical EPD
energy remain unavailable; their summary fields are explicit nulls where unsupported.

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
  `host_loop` add `scheduler_received_ms`, `selected_ms` and `prefill_complete_ms` to each
  record; a worker with `frontend` or `host_profile` also adds `frontend_ready_ms`, and a native
  encoder pool adds `encoder_ready_ms`.
- `resource-plan.json` describes preflight refusal, with null for unavailable host, budget,
  or workload estimates. `resource-runtime.json` records the effective budget and supervision
  outcome. `execution-events.jsonl` retains complete checkpoints after interruption; see
  [local execution resources](local-resources.md) for their interpretation.
- `afd-replay-spec.json` is the exact, deterministic native replay contract for an AFD run,
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

For `engine.ais_perf_model`, an entry key that is not a perf-model field is a validation error, so
existing artifacts stay. A key written twice inside `engine.ais_perf_model` in the configuration
file is an error when the file is read, with exit code `2`, so existing artifacts stay too. In
`predict`, an
[identity key](../perf-model/configuration.md#identity-keys) whose value differs from the engine
section is found later, when the run builds its workers. With `--overwrite`, the known outputs are
already removed by then.

<a id="standard-output"></a>

### Standard Output

`--format table` prints a concise human-readable summary. `--format json` prints the same summary as
one JSON value for shell automation. Durable artifact formats do not change with this option.

Prediction JSON without `--detail` on standard output is a summary object. SGLang workers with
`host_loop` add the mean time to first token split by stage: `mean_frontend_ms`,
`mean_scheduler_inbox_wait_ms`, `mean_receive_to_admit_ms`, `mean_prefill_elapsed_ms`,
`mean_result_observation_delay_ms`, `mean_handoff_to_first_token_ms` and `mean_detokenization_ms` (the
engine emitting the first token to the client receiving it with `engine.detokenization`, otherwise
zero). The seven spans sum to per-request TTFT for requests that reached every milestone; with a native encoder pool they start
at the pool's delivery and `encoder_latency_ms` covers the wait. Encoder pools also report
`encoder_gpus` and `total_gpus`. Recommendation JSON is an array of selected
rows with `rank`, `score`, `objectives`, `used_gpus`, and `config_path`. Single-objective scores are
signed so higher is better; latency-minimizing targets report negative scores. Pareto rows carry
the raw objective values in `objectives`. Use `recommendation.json` for the complete candidate ledger.

Normal prediction output includes timing provenance without `--detail`. The
`prediction_provenance` summary maps each worker role to its effective `estimator`,
`timing_source`, and `data_version`. Operation-level AIC timing reports the resolved
performance-database backend version, including resolution of version aliases.
FPM estimators report the resolved profile revision as `data_version` and preserve
the profile model/revision in `data_identity`. Without a profile revision, they
report a null version with a reason. Fixed/polynomial timing also has no data
version; polynomial timing reports source `built_in`, which covers both the default
and explicit polynomial selection. Its unavailable reason names built-in timing;
fixed timing uses `configured` and a matching reason. Custom runners that omit provenance report an
explicit `unavailable_reason`. Malformed runner provenance fails before report
output with a diagnostic naming the invalid field. These are the executed estimator's values, not the
requested `auto` mode. The same fields appear in saved `prediction.json`.

<a id="prediction-details"></a>

### Prediction details

```bash
aisimulate predict -c prediction.yaml --detail summary,memory,time \
  --format json --output-dir ./prediction-details
```

Ordinary prediction output includes effective KV precision and its source for roles
with native or profile capacity estimates. JSON summaries expose this as
`kv_cache_precision.<role>.{dtype,source}`; see
[KV precision](../perf-model/configuration.md#kv-precision-and-token-capacity).

`--detail` selects additional reports on `predict`. With a selector, JSON stdout contains `summary` and
`details`; the same versioned `details` object is added to `prediction.json` and follows the
[prediction-details schema](schemas/prediction-details.schema.json). Table output appends selected
sections and skipped-section reasons to the normal prediction summary.

- `summary`: existing serving metrics.
- `memory`: the existing initial per-rank memory capacity estimate, including components when
  available, with sizes in bytes and token counts in tokens. The only current `stage` value is
  `before_native_capacity_adjustments`. Native and profile estimates include the resolved
  `kvcache_quant_mode`: compare it with the deployment before interpreting token capacity.
  An omitted engine override can infer FP8; see [KV precision](../perf-model/configuration.md#kv-precision-and-token-capacity).
  The field is omitted when the estimator cannot establish the quantization mode.
  When exported, `memory_breakdown.reserved_bytes` reports fixed rank-local non-KV
  reservations, such as the VL embedding cache, separately from `cuda_graph_reserved_bytes`.
  Older reports may omit `reserved_bytes`.
  `estimated_num_gpu_blocks` is captured at this stage,
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
fixed/polynomial timing, analytical EPD overlays, and adapters without the native export
report operation timing/source evidence unavailable. Serving time statistics remain available
where exported. See [diagnostic availability](../aic-backward-compatibility/migration.md#prediction-details-and-power).

Inspect a recommendation by running `predict --detail` on its saved YAML. Reporting options
are CLI-only; this change adds no YAML configuration fields.


<a id="perf-model"></a>

## Perf-model tools

`aisimulate perf-model` works on the
[unified perf-model input](../perf-model/configuration.md#unified-perf-model-input).

```bash
aisimulate perf-model check ./perf-model.yaml
aisimulate perf-model migrate prediction.yaml
aisimulate perf-model fetch --list
```

| Action | Purpose |
|---|---|
| `check` | Build every entry of a perf-model input and report the ones that fail. |
| `migrate` | Convert the deprecated estimator fields of a configuration file to `ais_perf_model`. |
| `fetch` | List published FPM profiles, or download one to the local cache. |

These actions run no simulation and do not take the common options above. Each
action accepts `--format table|json`; the default is `table`.

| Exit Code | Meaning |
|---:|---|
| `0` | Success. `check`: every entry builds. `migrate`: the file converts, or has nothing to convert. `fetch`: the profile is in the cache, or the profiles are listed. |
| `1` | The action ran and reports a failure. `check`: at least one entry fails. `migrate`: the file cannot be converted. `fetch`: the profile cannot be fetched. |
| `2` | Wrong usage, or an input that cannot be read or parsed. `migrate`: also an output file that cannot be written. `fetch`: also an unknown manifest or profile. |
| `130` | Interrupted by the user. |

### `perf-model check`

```bash
aisimulate perf-model check ./perf-model.yaml
aisimulate perf-model check ./perf-model.yaml --worker-type decode --format json
aisimulate perf-model check \
  '{"model": "Qwen/Qwen3-32B", "system": "h200_sxm", "backend": "vllm"}' \
  --worker-type aggregated
```

The argument is a file path or inline JSON. `check` builds the perf model of
every entry, the same way a consumer does. A consumer builds only the entry
for its own worker type, so run `check` in CI to cover all of them.

`check` reads the value on its own. A `predict`, `recommend` or search
configuration that holds the value adds rules of its own: the identity keys
must agree with the engine, and an entry must not set `afd_routing_location`
or `afd_pool_role`. A value that `check` reports as `ok` can still be rejected
by such a configuration; see
[Identity keys](../perf-model/configuration.md#identity-keys).

| Option | Meaning |
|---|---|
| `--worker-type T` | Repeatable. Build only these worker types. Required when the input is one config without `worker_type`; without it the command exits `2` and names this option. |
| `--fill KEY=VALUE` | Repeatable. A value that a component would observe at run time, such as `tp=4`. The value is parsed as YAML. It is added when the entry leaves the key out, and must be equal when the entry writes it. |
| `--workspace DIR` | Folder in which dataset entries are prepared. Default: a temporary directory. |

`check` takes the value of the key, not a whole configuration file. Given a
`predict`, `recommend` or search configuration, it exits `2`, with or without
`--worker-type`:

```text
VALUE is a whole config: it has a top-level engine section. check takes the value of engine.ais_perf_model, in a file of its own or as inline JSON
```

For a search configuration the message names `search_space` and
`search_space.ais_perf_model`.

For each entry the report shows the worker type, the `estimation_mode` and
the result:

```text
worker type  estimation_mode  result
prefill      op_level         ok
decode       fpm_regression   ok, not ready: pure_decode (insufficient_data)
aggregated   (not set)        failed: ValueError: <error text>
```

- `estimation_mode` is shown as written: `auto` stays `auto`, and an entry
  that leaves it out shows `(not set)`.
- `ok, not ready` means the perf model builds but cannot estimate yet, for
  example a regression without observations. The line lists the stores that
  still need data. It counts as `ok` for the exit code.
- `--format json` prints `{"ok": ..., "entries": [...]}`. Each entry has
  `worker_type`, `estimation_mode`, `ok`, `error`, the resolved `config` and
  the `readiness` report.

### `perf-model migrate`

```bash
aisimulate perf-model migrate prediction.yaml
aisimulate perf-model migrate prediction.yaml --output prediction.new.yaml
aisimulate perf-model migrate prediction.yaml --output prediction.new.yaml --overwrite
```

`migrate` converts the [deprecated estimator fields](../perf-model/configuration.md#old-estimator-fields)
of a `predict`, `recommend` or Sweeper search configuration to
`engine.ais_perf_model` or `search_space.ais_perf_model`. It converts
AISimulate configuration files only.

| Option | Meaning |
|---|---|
| `--kind predict\|recommend\|search` | The kind of file. Detected when omitted: a top-level `search_space` means `search`; an `engine` section next to `optimization` or `optimizer` means `recommend`; any other file with an `engine` section means `predict`. When the kind cannot be detected, the command exits `2` and asks for this option. |
| `--output FILE` | Write the converted config to `FILE`: as JSON when `FILE` ends with `.json`, as YAML otherwise. `FILE` may be the configuration file itself; another file that exists is replaced only with `--overwrite`. A write that fails leaves a file that exists as it was. Comments and formatting of the original file are not kept. |
| `--overwrite` | Let `--output` replace a file that exists and is not the configuration file itself. Without `--output` it has no effect. |

Without `--output`, `migrate` prints the plan and writes nothing: the new
`ais_perf_model` block, the fields it removes, and notes.

- **One config or a list.** When every old field is at the engine level, the
  result is one config without `worker_type`. Otherwise it is a list with one
  entry per active worker type.
- **Older spellings.** `systems_path`, `forward_model` and `fpm_parquet_path`
  are converted too. `forward_model` and `fpm_parquet_path` are per-worker
  fields, so they give a list.
- **Defaults stay.** A field that only repeats its default, such as
  `fallback_policy: deny`, is not an old field for `migrate`. It stays in the
  file and does not conflict with the new key.
- **A `null` role policy.** In a search configuration,
  `role_estimator_controls.<role>.transfer_policy: null` is an old field. It
  gives that role every transfer kind. `migrate` removes it, and the entry for
  that role leaves `transfer_policy` out.
- **Relative paths.** Old fields are relative to the working directory. The new
  key is relative to the configuration file. `migrate` rewrites each relative
  path to start at the folder of `FILE`, or of the configuration file without
  `--output`. It prints a note such as
  `path 'fpm-data' (relative to the working directory) rewritten as '../fpm-data' (relative to the folder of the converted config, /srv/run/configs)`.
  Run it from the directory where you run the configuration; the plan names
  the working directory it assumed. Absolute paths, `~` paths and the
  `systems_paths` item `default` stay as written.
- **Symbolic links.** Through a symbolic link, the file that the link names
  is replaced and the link stays. When the link, every link on the way to the
  file and that file are in one folder, relative paths are rewritten as
  above. When one of them is in another folder, the converted config has a
  name in each of these folders, and a relative path would start at a
  different folder for each name. `migrate` then writes each relative path as
  an absolute path, and the note says so:
  `path 'fpm-data' (relative to the working directory) rewritten as '/srv/run/fpm-data' (an absolute path: the converted config has names in /srv/run and /srv/run/configs, and a relative path would start at a different folder for each name)`.
  The same holds without `--output` when the configuration file is such a
  link. `migrate` sees the links on the way from `FILE` to the file, and
  those from the configuration file when it names the same file. A link
  elsewhere that names the converted file still starts relative paths at its
  own folder. To get absolute paths, name that link as the configuration file
  or as `FILE` when you convert the file; a file that is already converted
  has nothing left to convert.
- **Existing files.** When `FILE` exists and is not the configuration file,
  `migrate` needs `--overwrite`. Without it nothing is written and the exit
  code is `2`:
  `output file prediction.new.yaml exists; pass --overwrite to replace it`.
- **One step.** `migrate` writes a new file next to `FILE` and moves it into
  place, so it needs permission to create a file in that folder. Without that
  permission it exits `2` with `could not write <FILE>: <reason>`, and the
  file stays as it was. Other hard links to `FILE` keep the old text.
- **JSON.** A `FILE` that ends with `.json`, in any letter case, is written as
  JSON. A `.json` configuration converted in place stays JSON. AISimulate
  reads configuration files as YAML. When the JSON text would not be read
  back as the same config, `migrate` writes nothing and exits `2`:
  `cannot write small.json as JSON: engine.ais_perf_model.estimator_config.ridge is 1e-05 and would be read back from the file as '1e-05'; name a file that does not end with .json to write YAML`.
- **All or nothing.** When one field cannot be converted, nothing is
  converted. `migrate` prints a `cannot convert: <field>: <reason>` line for
  each such field and then `Nothing was converted.`, `--output` writes no
  file, and the exit code is `1`. A file that already has `ais_perf_model`
  next to old fields is such a case; old fields are not merged into the key.
  A value that the new key cannot hold, such as a YAML date, is another:
  `cannot convert: engine.ais_perf_model cannot hold a value of the old fields: estimator_config.op_level.as_of has a value of type date; use a string, a number, true or false, a list or a mapping`.
  A third is `engine.estimator_config.state_snapshot` in a config that runs
  several worker types and has no old field under `engine.workers`: the key
  takes a state snapshot for one worker type, and `migrate` does not choose
  the worker type. Write `engine.ais_perf_model` as a list with one entry per
  `worker_type`, as the line says. When a worker has an old field of its own,
  `migrate` writes the list, and each entry names the state snapshot that
  worker used.
- **Circular values.** A YAML alias that makes a value contain itself is a
  usage error with exit code `2`:
  `engine.estimator_config.again contains itself: the value is the mapping or list it is written in, as a YAML alias can say. A config cannot be circular; write the value out`.
- **Not covered.** A configuration with AFD, an encoder pool, or `fixed` or
  `polynomial` timing keeps the old fields. `migrate` reports
  `cannot convert: <reason>` and exits `1`, also when the file has no old
  field.
- **Nothing to convert.** `migrate` says so and exits `0`.
- **JSON report.** `--format json` prints `config`, `kind`,
  `working_directory`, `status` (`converted`, `nothing_to_convert` or
  `cannot_convert`), the new `ais_perf_model` value, the `removed` fields,
  `notes`, `problems` and the `output` file.

Do not migrate a replay configuration written by `aisimulate onboard`.
`onboard` compares that file with the workload it reviewed, and rejects the
converted file:

```text
replay configuration differs from the reviewed direct-FPM workload
```

Reading the unchanged replay configuration back prints the deprecation
warning without the `migrate` command; see
[Limits in this release](../perf-model/configuration.md#limits-in-this-release).

### `perf-model fetch`

```bash
aisimulate perf-model fetch --list
aisimulate perf-model fetch --list kimi_k3_fpm
aisimulate perf-model fetch kimi_k3_fpm gb300-vllm-mxfp4-tp8-dcp8-primary-v0.29.0 --allow-unqualified
```

`fetch MANIFEST PROFILE` downloads one [published FPM profile](../perf-model/fpm-self-service/README.md#reuse-a-published-profile)
and verifies it. It prints the local folder and the `systems_paths` line to
put in an entry. The folder is printed as an absolute path with symbolic
links resolved, also when `--cache-dir` or `$XDG_CACHE_HOME` is relative, so
the line names it from an entry file in any folder. Run it ahead of time for
a component that must not download data. `MANIFEST` is a manifest file or the
ID of a packaged manifest.
`fetch --list [MANIFEST]` lists profiles instead and takes no `PROFILE`.

| Option | Meaning |
|---|---|
| `--list` | List the profiles of `MANIFEST` and their admission without downloading data. Without `MANIFEST`, list the profiles of every packaged manifest. |
| `--cache-dir DIR` | Cache folder. Default: `aisimulate/fpm` under `$XDG_CACHE_HOME`, or under `~/.cache` when that variable is unset. |
| `--local-files-only` | Never download; use only what the cache holds. |
| `--allow-unqualified` | Accept a profile that is not admitted for serving, such as development data. The profile stays unadmitted. |

Without `--allow-unqualified`, a profile that is not admitted for serving is
not fetched. `fetch` exits `1`, and the error ends with a line that names the
option:

```text
could not fetch profile 'gb300-vllm-mxfp4-tp8-dcp8-primary-v0.29.0': FPM profile 'gb300-vllm-mxfp4-tp8-dcp8-primary-v0.29.0' is not admitted for serving; No serving qualification or archived collection system YAML is published for this table; the bundled system configuration is a reference for reproduction.
Pass --allow-unqualified to fetch the profile anyway.
```

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

`perf-model` uses only `0`, `1`, `2` and `130`, with the meanings listed under
[perf-model tools](#perf-model).

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

When no requests complete, `predict` saves its report but exits `1`. The table
marks the prediction invalid; JSON includes `prediction_status: invalid`,
`stop_reason: no_completed_requests`, and an actionable `diagnostic`. Serving
latency and throughput values, including processing-rate aliases, are `null`
(unavailable), not valid zero samples. The invalid banner also appears in analytical EPD tables.
The stop reason identifies the observed outcome, not a proven scheduler cause.
Check KV capacity against request lengths, context limits, and traffic stop limits.

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
| `host cost table ... does not exist yet` (or a missing-row message) | A native SGLang VL prediction reads the frontend stage from a host cost table row measured on the serving host; the message prints the `collect` command that adds the row. | Run the printed `python -m aisimulate.vl.collect ...` with `--sglang-python` pointing at the serving venv, then rerun `predict`; see [SGLang VL host loop](../replay/engine/sglang-vl-host-loop.md). |
| `--online` is rejected | The selected stack must advertise online support. | Use offline execution with `--stack engine`, or an integration that supports online execution. |

For automation, check the exit code as well as standard output. `--format json` changes successful
summary output; validation and execution errors are reported on standard error. A completed recommendation
with no feasible result still saves its result ledger and does not provide a YAML to predict.
