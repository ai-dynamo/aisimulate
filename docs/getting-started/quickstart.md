<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Predict and recommend a deployment

AISimulate runs on the host CPU; the selected GPU is the hardware being modeled.
Follow [installation](installation.md) first. The examples below match the
[executable repository README](../../README.md), which CI runs against source
and the selected release profiles. Source documentation can require a newer
version than a published wheel.

## Predict one deployment

Save this as `prediction.yaml`:

```yaml
engine:
  mode: aggregated
  model: Qwen/Qwen3-32B-FP8
  hardware: h200_sxm
  backend: vllm
  workers:
    aggregated: {}
```

```bash
aisimulate predict \
  --stack engine \
  --config prediction.yaml \
  --output-dir ./aisimulate-prediction
```

Read `aisimulate-prediction/prediction.json` for the complete report. Add
`--capture-per-request` for `requests.jsonl`, or `--format json` for a JSON
summary on standard output. The simulator's wall time and the modeled serving
latencies are different quantities; see [understand results](understand-results.md).

## Choose a workload

The minimal prediction uses default synthetic traffic. Pin it explicitly for
comparisons by adding the following top-level section:

```yaml
traffic:
  source: {type: synthetic, input_tokens: 1024, output_tokens: 128}
  load: {type: concurrency, concurrency: 10}
  stop: {requests: 100}
```

Concurrency limits in-flight requests; `stop.requests` is the total workload.
To model arrivals at eight requests per second, replace the whole load object:

```bash
aisimulate predict --config prediction.yaml \
  --set 'traffic.load={type: constant_rate, requests_per_second: 8}' \
  --set traffic.stop.requests=100 --output-dir ./prediction-8rps
```

See [workload inputs](../replay/workloads.md) for trace replay, synthetic
sessions, arrival distributions, and stopping conditions.

## Recommend a deployment

Save this as `recommendation.yaml`:

```yaml
traffic:
  source: {type: synthetic, input_tokens: 1024, output_tokens: 128}
  load: {type: concurrency, concurrency: 10}
  stop: {requests: 100}

engine:
  mode: aggregated
  model: Qwen/Qwen3-32B-FP8
  hardware: auto
  backend: {choices: [vllm, sglang]}
  workers:
    aggregated:
      parallelism: {preset: default}

optimization:
  target: throughput_per_gpu
  hardware: h200_sxm
  constraints:
    max_candidate_gpus: 8

optimizer:
  max_trials: 8
```

```bash
aisimulate recommend \
  --stack engine \
  --config recommendation.yaml \
  --output-dir ./aisimulate-recommendation
```

This is an eight-trial search within an eight-GPU candidate budget. It evaluates
both vLLM and SGLang at the same fixed load. A trial limit is not an exhaustive
search or a guarantee of the optimal configuration.

## Predict the selected configuration

If the result contains a selected candidate:

```bash
aisimulate predict \
  --config ./aisimulate-recommendation/recommendations/0001.yaml \
  --output-dir ./aisimulate-best-prediction
```

Each recommendation YAML is a concrete prediction input. It does not deploy
GPU workers. Read the [Sweeper quickstart](../sweeper/quickstart.md) to inspect
the ledger, choose an objective, and distinguish partial from complete results.
Use [deployment generation](../sweeper/deployment-generation.md) to render and
validate a selected candidate on real hardware.

For another run, choose a fresh output directory or explicitly use
`--overwrite`. Invalid core configuration does not remove earlier results;
see the [CLI output policy](../reference/cli.md#existing-output-directories).
The optional Dynamo stack requires a matching installation and should be used
consistently for recommendation and the selected prediction; see
[Dynamo integration](../replay/dynamo.md).
