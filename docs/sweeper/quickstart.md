<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Run a recommendation search

[Install AISimulate](../getting-started/installation.md) and activate the
matching environment. This example uses the built-in offline engine on your
CPU. The H200 is the target being modeled.

<a id="recommend-under-a-gpu-budget"></a>

## Recommend under a GPU budget

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
├── recommendation.csv
├── resource-runtime.json
├── execution-events.jsonl
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

### Choose a goal and add latency limits

Use `throughput` to maximize total output within the GPU budget, or `throughput_per_gpu` to favor
efficiency. Use a `goodput` target when only output meeting your service-level agreement (SLA)
should count toward the score. The [optimization reference](optimization-goals.md#optimization-goal) lists all targets.

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
use `target: min_gpus` to select the smallest qualifying configuration found instead. See the
[minimum-GPU migration mapping](../aic-backward-compatibility/migration.md#traffic-parallelism-and-minimum-gpus) for load
constraints and the bundled AIC sizing alternative.

<a id="predict-a-recommended-configuration"></a>

### Predict a recommended configuration

If the search found a feasible candidate, pass a saved YAML directly to `predict`:

```bash
aisimulate predict \
  --config ./recommendation-output/recommendations/0001.yaml \
  --output-dir ./best-prediction
```

The saved YAML contains concrete values with no search domains, `preset`, `optimization`, or
`optimizer`. The command prints prediction metrics and writes
`best-prediction/prediction.json`. It is a prediction input; deployment
manifests and launch scripts are covered in the [migration guide](../aic-backward-compatibility/migration.md).

## Next steps

Use [search space](search-space.md) for presets, domains and budgets;
[optimization goals](optimization-goals.md) for scoring and SLA boundaries;
and [results](results.md) for candidate statuses and partial outcomes.
A selected YAML is a simulation input. [Deployment generation](deployment-generation.md)
turns supported candidates into artifacts for a separate hardware validation.
