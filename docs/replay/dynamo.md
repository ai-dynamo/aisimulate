<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

<a id="router-dynamo-adapter"></a>
<a id="13-router-dynamo-adapter"></a>
<a id="14-planner-dynamo-adapter"></a>

# Dynamo integration

Select `--stack dynamo` to use Dynamo-owned Router/Planner behavior. Install the
integration in the same environment as AISimulate and select the same stack when
replaying a saved recommendation. The YAML does not embed stack selection.
The engine stack remains sufficient for reading native Dynamo trace files.

`predict --online` requests wall-clock-paced simulation only when the selected
stack advertises it; it does not launch a serving endpoint. `recommend` is
always offline. AgentX, G2 and telemetry support require qualification in the
installed Dynamo integration, independently of Engine-stack acceptance.

## Planner installation check

This example runs the public AISimulate CLI with Planner enabled. It uses
synthetic traffic, local model metadata, fixed timing, and fixed KV capacity;
no model weights, GPU, or Kubernetes cluster are needed. Its timings are
synthetic and do not measure model performance or qualify scaling accuracy.

### Install matching dependencies

Use Linux with Python 3.12, matching the release validation environment. From
the AISimulate repository root, place the compatible AISimulate, `ai-dynamo`,
and `ai-dynamo-runtime` wheel artifacts in `./wheels/` (one of each):

```bash
python3 -m venv .venv-planner
source .venv-planner/bin/activate
python3 -m pip install \
  ./wheels/aisimulate-*.whl \
  ./wheels/ai_dynamo-*.whl \
  ./wheels/ai_dynamo_runtime-*.whl
```

For an RC, select the exact release artifacts rather than relying on a
package version shared by multiple builds. RC wheels may not be available
from the public package index.

The basic [With Dynamo installation](../getting-started/installation.md#optional-dynamo-integration)
does not install all Planner dependencies. Install the complete Planner
requirements from the **same Dynamo tag or commit as those wheels**:

```bash
# Dynamo 1.5.0 RC9 example; replace with the revision of your installed build.
DYNAMO_REF=ffd7c1a90eb403c0d43911690c5c9b8457acd826
python3 -m pip install "grpcio-tools<=1.76.0" -r \
  "https://raw.githubusercontent.com/ai-dynamo/dynamo/${DYNAMO_REF}/container/deps/requirements.planner.txt"
python3 -m pip check
```

The `grpcio-tools` cap matches RC9's
[common requirements](https://github.com/ai-dynamo/dynamo/blob/ffd7c1a90eb403c0d43911690c5c9b8457acd826/container/deps/requirements.common.txt).
AISimulate's `google-vizier` dependency can install a newer `grpcio-tools`
that requires a different protobuf version. Include the cap in the same pip
invocation as the Planner requirements so the resolver can select compatible
versions together, including when repairing an existing environment. When
selecting another Dynamo revision, check its common requirements and update
this cap together with `DYNAMO_REF`.

Use the full requirements file, which includes `scikit-learn` and other
Planner dependencies. The supported prebuilt alternative is the matching
`dynamo-planner` image, which already includes these prerequisites. Use the
image from the same Dynamo release you intend to validate.

### Run a Planner-enabled prediction

From the AISimulate repository root, using the environment above:

```bash
cd examples/cli/dynamo-planner
python3 -m aisimulate predict \
  --stack dynamo \
  --config prediction.yaml \
  --output-dir ./planner-output \
  --capture-per-request \
  --format json
```

Keep this working directory: `prediction.yaml` resolves its `./model` path
relative to it. The included `model/config.json` is synthetic metadata based
on this repository's unified CLI test fixture, with no model weights.

A successful run exits zero, completes all 12 requests, and writes
`planner-output/prediction.json` plus `planner-output/requests.jsonl`.
The configuration explicitly enables load-based Planner scaling, with a
five-second adjustment interval and a two-GPU simulated budget. The
`--stack dynamo` option is required to resolve its top-level `planner` section.
For another run, choose a new output directory or pass `--overwrite` to
replace the known output files.

If the command fails while loading `dynamo.planner` with
`ModuleNotFoundError: No module named 'sklearn'`, the active Python environment
is missing Planner prerequisites. Install the matching requirements there,
run `python3 -m pip check`, and repeat this prediction. A successful dependency
check, basic prediction without Planner, or `predict --help` alone does not
verify that Planner loads.

## Router configuration

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

## Planner configuration

Planner is not part of the AISimulate core schema. The `dynamo.planner` config adapter owns this
section's presets, recommendation domains, and runtime lowering. Concrete enabled settings use
Dynamo's production `PlannerConfig` for defaults, validation, and target normalization. The section
is accepted only when the selected stack provides that adapter.

> [!IMPORTANT]
> The defaults, normalization, and interval rules below require a Dynamo build containing
> [Dynamo #15678](https://github.com/ai-dynamo/dynamo/pull/15678). The older `ai-dynamo==1.6.0.dev20260930` build predates that change and uses a separate simulation configuration
> model. Upgrade Dynamo to a build containing the change before using these rules.

```yaml
planner:
  policy: disabled
```

| Knob | Default | Default Range | Preset | Rules |
|---|---:|---|---|---|
| `planner.scaling_policy.preset` | `default` in `recommend` | `{choices: [disabled, throughput_180_5, throughput_600_5, load_180_5, load_180_10, hybrid_180_5, hybrid_600_5]}` | `-` | Default lists are filtered for the derived target and SLA. Explicit lists must be compatible. |
| `planner.fpm_sampling.preset` | `default` in `recommend` | `{choices: [small, default, large, fine]}` | `-` | Built-in preset choices, complete mapping list, `false`, or `{}`. |
| `planner.load_sensitivity.preset` | `default` in `recommend` | `{choices: [aggressive, default, conservative]}` | `-` | Built-in preset choices, complete mapping list, `false`, or `{}`. |
| `planner.load_predictor.preset` | `default` in `recommend` | `{choices: [constant_last, arima_raw, arima_log1p, prophet_w20_raw, prophet_w20_log1p, prophet_w50_raw, prophet_w50_log1p, kalman_default_raw, kalman_default_log1p, kalman_reactive_raw, kalman_reactive_log1p]}` | `-` | Interval-level predictor pre-sweep candidates; complete mapping list, `false`, or `{}`. |
| `planner.policy` | `disabled` | `{choices: [disabled, enabled]}` | `-` | `disabled` or `enabled`. |
| `planner.target` | `throughput` | `x` | `-` | Derived from `optimization.target` in `recommend`. |
| `planner.enable_throughput_scaling` | `false` for the default target; `true` for `sla` | `{choices: [false, true]}` | `scaling_policy` | Non-SLA targets normalize this to `false`; effective throughput scaling requires TTFT/ITL thresholds. |
| `planner.enable_load_scaling` | `true` for the default target; `false` for `sla` | `{choices: [false, true]}` | `scaling_policy` | Non-SLA targets normalize this to `true`. |
| `planner.throughput_adjustment_interval_seconds` | `180` | `{choices: [180, 600]}` | `scaling_policy` | Omitted uses the production default; `null` is accepted only when this scaling mode is disabled. |
| `planner.load_adjustment_interval_seconds` | `5` | `{choices: [5, 10]}` | `scaling_policy` | Also controls FPM updates. Must be shorter than the throughput interval only when both modes are enabled. Inactive `null` uses the production default. |
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
| `planner.max_num_gpus` | `8` | `x` | `-` | Planner runtime ceiling; maps to `max_gpu_budget`. Positive in `recommend`; concrete `predict` also accepts `-1` for unlimited. |
| `planner.min_num_gpus` | `-1` | `x` | `-` | Concrete `predict` only; maps to `min_gpu_budget` (`-1` disables the floor). Recommendations export `optimization.constraints.min_candidate_gpus` here when set. |
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
dimensions. FPM sampling is included only when a retained policy enables throughput scaling; load
sensitivity is included only when a retained policy enables load scaling. Independent knob domains
(`preset: false`) use the same compatible subset, and invalid combinations are skipped as infeasible.
Explicit preset and predictor candidate lists remain within the user-selected subset.
`load_predictor` is different: its candidates run in a pre-sweep for every selected throughput-adjustment
interval, and the winning predictor mapping is materialized into the candidate.

`planner.policy`, `planner.target`, the runtime GPU limits, and the three minimum-worker knobs
are not covered by a preset. `predict` may set a concrete target and otherwise uses `throughput`. In
`recommend`, target is derived: throughput targets map to `throughput`, `ttft` and `e2e_latency`
map to `latency`, and goodput targets map to `sla`. Pareto uses `sla` when it includes a goodput
objective and otherwise uses `throughput`. Thus `planner: {policy: enabled}` is a valid minimal
prediction configuration: the production Planner normalizes the default target to load-only scaling.

Throughput-based Planner scaling is legal only for the `sla` target with concrete
`evaluation.sla.ttft_ms` and `evaluation.sla.itl_ms`. For `throughput`, `latency`, or `load` Planner
targets, the default recommendation search removes throughput and hybrid presets before searching.
An explicitly selected incompatible preset or custom mapping is rejected before search begins.
Concrete prediction flags are normalized by the production Planner for the selected target.

Custom scaling-policy mappings may use `null` for an inactive mode's adjustment interval. An
omitted concrete interval also uses the production default; an explicitly null active interval is
rejected. Enabled configurations export both intervals as numbers: the load interval still drives
FPM updates during throughput-only scaling. A fully disabled custom policy uses null for both
intervals. These rules avoid passing null into the Planner runtime.

Planner runtime limits and recommendation candidate GPU constraints are separate:

- `planner.max_num_gpus`, `min_workers`, `prefill_min_workers`, and `decode_min_workers` constrain
  runtime scaling during one predicted candidate run.
- `optimization.constraints` constrains which static candidate deployments the recommender evaluates.
  When set, `min_candidate_gpus` also becomes the Planner runtime floor and is preserved in the
  exported prediction YAML as `planner.min_num_gpus`.

When `planner.policy: disabled` or the `disabled` scaling-policy preset is selected, no Planner
runtime hook is materialized and only `policy: disabled` is emitted. Enabled output contains the
normalized Planner settings, including defaults for inactive knobs, so the same concrete configuration
can be replayed without changing its effective Planner settings.




## Complete Dynamo prediction example

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

## Dynamo scalar recommendation example

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




The versioned integration contracts are described under [configuration ABI](../adapters/configuration-abi.md) and [native composition](../adapters/native-composition.md).
