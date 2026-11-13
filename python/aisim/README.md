<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AISim

AISim predicts LLM serving behavior and searches for deployment
configurations offline, without bringing up a GPU serving cluster. The Python
package includes the simulation engine, performance estimator and profiles,
Replay, Sweeper, and deployment generation through the compatibility CLI and
generator SDK.

[Website](https://ai-dynamo.org/aisim/) ·
[Repository](https://github.com/ai-dynamo/aisim) ·
[CLI guide](https://github.com/ai-dynamo/aisim/blob/main/docs/reference/cli.md)

## Install

Use Python 3.11–3.13. This branch builds the renamed `aisim` distribution;
these commands require its publication on your selected index. Until then, use
a historical `aisim` release in a separate environment or build from source:

```bash
python3 -m pip install aisim
aisim --help
```

The `aisim` wheel provides both `aisim` and `aisimulate`; both commands
use the same entry point. Python imports remain `aisim` and `aisim_core`.

The built-in engine runs without Dynamo. See the
[installation guide](https://github.com/ai-dynamo/aisim/blob/main/docs/getting-started/installation.md)
for platform requirements, source builds, nightly artifacts, and compatible
Dynamo installations. Documentation on `main` may describe features newer than
the latest published wheel. The optional Dynamo Planner integration requires
additional dependencies described in that guide.

## Migrate from the `aisim` distribution

The distributions share their Python files and CLI, so uninstall `aisim`
**before** installing or upgrading to `aisim`:

```bash
python3 -m pip uninstall -y aisim
python3 -m pip install --upgrade aisim
python3 -m pip check
```

If both names are already installed, use a fresh environment or uninstall
`aisimulate` and force-reinstall your selected `aisim` wheel to repair overlapping
files. Keep a Dynamo environment on the old distribution until its Dynamo
release requires `aisim`; otherwise dependency resolution can reinstall
`aisim`. Import names remain `aisim` and `aisim_core`.

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
aisim predict --config prediction.yaml --output-dir ./prediction
```

The CLI prints a serving summary and writes the full report to
`prediction/prediction.json`. Prediction uses the built-in `engine` stack by
default; `--stack dynamo` selects a separately installed compatible Dynamo runner.

## Search for a deployment

Save the following as `recommendation.yaml`:

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
aisim recommend --config recommendation.yaml --output-dir ./recommendation
```

This quickstart evaluates suggestions for the explicit 1,024-input/128-output-token,
10-concurrent-request workload above, with an eight-trial budget.

Inspect `recommendation/recommendation.json` for search results and
`recommendation/recommendations/` for concrete prediction YAML files to replay.
The [CLI guide](https://github.com/ai-dynamo/aisim/blob/main/docs/reference/cli.md)
covers workload inputs, latency constraints, and detailed output. To create
deployment manifests and launch scripts, use the bundled compatibility CLI or
generator SDK; see the
[deployment generation guide](https://github.com/ai-dynamo/aisim/blob/main/docs/sweeper/deployment-generation.md).

## Documentation and coverage

- [Replay](https://github.com/ai-dynamo/aisim/blob/main/docs/replay/README.md) and
  [Sweeper](https://github.com/ai-dynamo/aisim/blob/main/docs/sweeper/README.md)
  — embed replay and configuration search through Python APIs.
- [Core API](https://github.com/ai-dynamo/aisim/blob/main/docs/perf-model/api/python.md)
  — engine, performance-model, and memory contracts.
- [FPE Support Matrix](https://ai-dynamo.org/aisim/fpe-support-matrix/)
  — forward-pass estimator coverage.
- [Feature Support Matrix](https://ai-dynamo.org/aisim/feature-support-matrix/)
  — serving techniques replayed per engine.
- [E2E Accuracy Overview](https://ai-dynamo.org/aisim/e2e-accuracy/) and
  [FPM Accuracy Overview](https://ai-dynamo.org/aisim/fpm-accuracy/)
  — published accuracy evidence.
- [Development guide](https://github.com/ai-dynamo/aisim/blob/main/DEVELOPMENT.md)
  — source setup and tests.
- [Contributing](https://github.com/ai-dynamo/aisim/blob/main/CONTRIBUTING.md) and
  [Code of Conduct](https://github.com/ai-dynamo/aisim/blob/main/CODE_OF_CONDUCT.md)
  — contribution requirements and community guidelines.

Estimator coverage is separate from serving-accuracy and deployment validation.
Use the published evidence for your workload and verify deployment choices with
real benchmarks.

## Legacy CLI compatibility

The wheel provides the legacy `aiconfigurator` command through AISim 0.13.0.
Removal is targeted for AISim 0.14.0, after every remaining workflow has a
verified replacement in the unified CLI. Use `aisim` for new prediction and
recommendation workflows. The
[migration guide](https://github.com/ai-dynamo/aisim/blob/main/docs/aic-backward-compatibility/migration.md)
explains replacements and remaining differences; the
[legacy CLI guide](https://github.com/ai-dynamo/aisim/blob/main/docs/aic-backward-compatibility/cli.md)
documents retained commands.

The canonical Python imports are `aisim` and `aisim_core`. AISim
0.13.0 removes the `aiconfigurator` and `aiconfigurator_core` import namespaces;
see the [Python source migration guide](https://github.com/ai-dynamo/aisim/blob/main/docs/aic-backward-compatibility/migration.md#python-imports-and-resources)
for replacement imports. The legacy executable remains available as described above.

When upgrading from standalone AIConfigurator, remove the old distributions first
so that AISim owns the installed files:

```bash
python3 -m pip uninstall -y aiconfigurator aiconfigurator-core aisim
python3 -m pip install --upgrade aisim
```
