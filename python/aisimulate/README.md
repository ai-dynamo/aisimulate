<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AISimulate

AISimulate predicts LLM serving behavior and searches for deployment
configurations offline, without bringing up a GPU serving cluster. The Python
package includes the simulation engine, performance estimator and profiles,
Replay, Sweeper, and deployment generation.

[Website](https://ai-dynamo.org/aisimulate/) ·
[Repository](https://github.com/ai-dynamo/aisimulate) ·
[CLI guide](https://github.com/ai-dynamo/aisimulate/blob/main/docs/cli/user-guide.md)

## Install

Use Python 3.11–3.13:

```bash
python3 -m pip install aisimulate
aisimulate --help
```

The built-in engine runs without Dynamo. See the
[installation guide](https://github.com/ai-dynamo/aisimulate/blob/main/docs/installation.md)
for platform requirements, source builds, nightly artifacts, and compatible
Dynamo installations. Documentation on `main` may describe features newer than
the latest published wheel. The optional Dynamo Planner integration requires
additional dependencies described in that guide.

## Predict one deployment

Save the following as `prediction.yaml`:

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
aisimulate predict --config prediction.yaml --output-dir ./prediction
```

The CLI prints a serving summary and writes the full report to
`prediction/prediction.json`. Prediction uses the built-in `engine` stack by
default; `--stack dynamo` selects a separately installed compatible Dynamo runner.

## Search for a deployment

Save the following as `recommendation.yaml`:

```yaml
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
```

```bash
aisimulate recommend --config recommendation.yaml --output-dir ./recommendation
```

Inspect `recommendation/recommendation.json` for search results and
`recommendation/recommendations/` for concrete prediction configurations.
The [CLI guide](https://github.com/ai-dynamo/aisimulate/blob/main/docs/cli/user-guide.md)
covers workload inputs, latency constraints, detailed output, and deployment
artifact generation.

## Documentation and coverage

- [Replay](https://github.com/ai-dynamo/aisimulate/blob/main/crates/core/src/replay/README.md) and
  [Sweeper](https://github.com/ai-dynamo/aisimulate/blob/main/docs/sweeper/overview.md)
  — embed replay and configuration search through Python APIs.
- [Core API](https://github.com/ai-dynamo/aisimulate/blob/main/docs/core-api.md)
  — engine, performance-model, and memory contracts.
- [FPE Support Matrix](https://ai-dynamo.org/aisimulate/fpe-support-matrix/)
  — forward-pass estimator coverage.
- [E2E Accuracy Overview](https://ai-dynamo.org/aisimulate/e2e-accuracy/) and
  [FPM Accuracy Overview](https://ai-dynamo.org/aisimulate/fpm-accuracy/)
  — published accuracy evidence.
- [Development guide](https://github.com/ai-dynamo/aisimulate/blob/main/DEVELOPMENT.md)
  — source setup, tests, and contribution workflow.

Estimator coverage is separate from serving-accuracy and deployment validation.
Use the published evidence for your workload and verify deployment choices with
real benchmarks.

## Legacy CLI compatibility

The wheel also provides the legacy `aiconfigurator` command for existing workflows.
Use `aisimulate` for new prediction and recommendation workflows. The
[migration guide](https://github.com/ai-dynamo/aisimulate/blob/main/docs/cli/migrate-from-aiconfigurator.md)
explains replacements and remaining differences; the
[legacy CLI guide](https://github.com/ai-dynamo/aisimulate/blob/main/docs/cli/legacy-aic-user-guide.md)
documents retained commands.

When upgrading from standalone AIConfigurator, remove the old distributions first
so that AISimulate owns the installed files:

```bash
python3 -m pip uninstall -y aiconfigurator aiconfigurator-core
python3 -m pip install --upgrade aisimulate
```
