<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AISimulate

AISimulate predicts LLM serving behavior and searches for strong deployment
configurations offline, without bringing up a GPU serving cluster.

AISimulate is the successor to the
[AIConfigurator (AIC)](https://github.com/ai-dynamo/aiconfigurator)
repository. It brings the complete AIC application and estimator into one
standalone home with Dynamo-independent Replay and Sweeper capabilities.

The performance-modeling methodology is described in
[AIConfigurator: Lightning-Fast Configuration Optimization for Multi-Framework
LLM Serving](https://arxiv.org/abs/2601.06288).

## Install

### Engine-only

Install AISimulate by itself to use the built-in simulation engine without a
Dynamo dependency:

```bash
python3 -m pip install aisimulate
aisimulate --help
```

### With Dynamo

Install AISimulate with Dynamo to enable the `dynamo` runner plus Dynamo-owned
Router and Planner configuration adapters:

```bash
python3 -m pip install aisimulate ai-dynamo
aisimulate predict --help
```

AISimulate remains the CLI owner in both profiles. Select the integration at
runtime with `--stack dynamo`; installing Dynamo does not add another
simulation command.

### Upgrade from standalone AIConfigurator

Remove the former standalone distributions first so that only AISimulate owns
the compatibility imports and command:

```bash
python3 -m pip uninstall -y aiconfigurator aiconfigurator-core
python3 -m pip install --upgrade aisimulate
```

## Predict one deployment

`predict` evaluates one pinned deployment configuration. Save this example as
`prediction.yaml`:

```yaml
engine:
  mode: aggregated
  model: Qwen/Qwen3-32B-FP8
  hardware: h200_sxm
  backend: vllm
  workers:
    aggregated: {}
```

### Engine-only prediction

```bash
aisimulate predict \
  --stack engine \
  --config prediction.yaml \
  --output-dir ./aisimulate-prediction
```

### Dynamo-integrated prediction

```bash
aisimulate predict \
  --stack dynamo \
  --config prediction.yaml \
  --output-dir ./aisimulate-dynamo-prediction
```

The CLI prints a concise summary and writes the selected runner's complete
report to `<output-dir>/prediction.json`. Add `--capture-per-request` to also
write `requests.jsonl`, or use `--format json` for machine-readable standard
output.

## Recommend a deployment

`recommend` searches the prediction schema plus search domains and an
optimization goal. Save this example as `recommendation.yaml`:

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

### Engine-only recommendation

```bash
aisimulate recommend \
  --stack engine \
  --config recommendation.yaml \
  --output-dir ./aisimulate-recommendation
```

### Dynamo-integrated recommendation

```bash
aisimulate recommend \
  --stack dynamo \
  --config recommendation.yaml \
  --output-dir ./aisimulate-dynamo-recommendation
```

Each file under `<output-dir>/recommendations/` is a fully materialized,
concrete configuration. It contains no search domains and can be passed
directly back to `predict`:

```bash
aisimulate predict \
  --config ./aisimulate-recommendation/recommendations/0001.yaml \
  --output-dir ./aisimulate-best-prediction
```

Both commands support `--set PATH=YAML_VALUE`, `--output-dir`, `--overwrite`,
and `--format table|json`. See the [CLI reference](docs/cli/design.md) for the
complete schema, traffic models, search domains, presets, outputs, and error
contract.

## AIConfigurator compatibility CLI

The `aisimulate` wheel preserves the established `aiconfigurator` command for
workflows that have not yet moved to the unified CLI. AISimulate 0.12.0 keeps
this compatibility surface, while new prediction and search integrations
should start with `aisimulate predict` and `aisimulate recommend`.

```bash
# Check whether a model/system combination is supported.
aiconfigurator cli support \
  --model-path Qwen/Qwen3-32B-FP8 \
  --system h200_sxm

# Generate a starting deployment configuration.
aiconfigurator cli generate \
  --model-path Qwen/Qwen3-32B-FP8 \
  --total-gpus 8 \
  --system h200_sxm
```

The compatibility CLI preserves six workflows:

| Mode | Purpose |
|---|---|
| `default` | Compare aggregated and disaggregated candidates and select a strong starting point |
| `estimate` | Estimate one explicitly configured deployment |
| `recommend` | Find the minimum GPU count and configuration for a load target and SLA |
| `exp` | Run custom experiments from YAML |
| `generate` | Generate deployment artifacts without a parameter sweep |
| `support` | Check model and system coverage |

Read the [AIC CLI and Python API guide](python/aisimulate/README.md) for the
complete compatibility surface. The
[AIC migration guide](docs/cli/migrate-from-aiconfigurator.md)
explains which AIC workflows map to `predict` or `recommend` and which ones
must continue using the compatibility command for now.

## AIConfigurator repository transition

The standalone [AIConfigurator repository](https://github.com/ai-dynamo/aiconfigurator)
will publish its final 0.12.0 `aiconfigurator` and `aiconfigurator-core`
artifacts and then be archived. AISimulate is the canonical home for ongoing
development, releases, issues, and pull requests; open all new issues and pull
requests in this repository.

The `aiconfigurator` compatibility command remains available from the
`aisimulate` wheel in 0.12.0. It is targeted for removal in AISimulate 0.13.0,
after every remaining AIC workflow has a verified replacement in the unified
`aisimulate` CLI. Until then, use the compatibility command for the workflows
identified in the migration guide.

## Experimental status and validation boundary

> [!WARNING]
> Replay and Sweeper are experimental surfaces intended for evaluation and
> feedback, not production capacity planning. Their APIs, schemas, search
> behavior, and output may change without a standard deprecation period.

AISimulate narrows a deployment search and identifies candidates; it does not
replace validation on the target hardware. Benchmark shortlisted
configurations on a real deployment before making production capacity or SLA
decisions.

## SDKs

Use the focused SDK documentation instead of treating CLI internals as public
APIs:

- [Estimator/FPE Python and Rust SDK](docs/core-api.md)
- [Replay SDK and artifact contract](crates/core/src/replay/README.md)
- [Sweeper SDK](docs/sweeper/overview.md)
- [AIConfigurator compatibility Python API](python/aisimulate/README.md#python-api)

## Support and accuracy

Support coverage and accuracy are separate evidence. A supported cell means a
specific path can execute with the required data; it does not establish that
the resulting end-to-end prediction is accurate.

### FPE support matrix — in development

The new strict-native Forward Pass Engine (FPE) matrix measures estimator
coverage across a curated roster of current models, GPU systems, backends, and
backend versions. It probes native prefill, decode-start, decode-end, and mixed
forward-pass calls without fallback. It does not certify the CLI, Replay,
Sweeper, serving orchestration, or prediction accuracy.

The FPE matrix is currently under review in
[AISimulate PR #41](https://github.com/ai-dynamo/aisimulate/pull/41). Treat it
as an in-development coverage surface until that work merges and publishes the
interactive matrix.

### AIC CLI support matrix

The compatibility support matrix covers AIC command-based aggregated and
disaggregated workflows by model, system, backend, and backend version:

- [Interactive legacy AIC support matrix](https://ai-dynamo.github.io/aiconfigurator/support-matrix/)
- [AIC support-matrix data](python/aisimulate/src/aiconfigurator_core/systems/support_matrix/)
- [Curated model roster](python/aisimulate/docs/support-matrix/model-roster.md)

Check one exact cell from the installed package with:

```bash
aiconfigurator cli support \
  --model-path Qwen/Qwen3-32B-FP8 \
  --system h200_sxm \
  --backend vllm \
  --backend-version 0.14.0
```

### Accuracy matrix — under construction

The accuracy matrix is not yet a published support contract. The current work
tracks AISimulate predictions against curated measured-silicon anchors and
keeps forward-pass accuracy distinct from end-to-end serving accuracy. See the
[prediction regression and accuracy design](python/aisimulate/docs/design/prediction_regression_gate_design.md)
and the current [silicon anchor set](python/aisimulate/tools/accuracy_tracking/silicon_refs.csv).

## Release artifacts and compatibility

Starting with 0.12.0, this repository owns the complete AIConfigurator product
surface—not only its native core. The application, CLI, generator, SDK,
Collector, tests, documentation, and development tooling live under
`python/aisimulate/`.

This repository produces exactly two release artifacts:

1. `aisimulate` Python wheel — the application, both console commands,
   estimator SDK, model/performance data, Replay, Sweeper, and unified native
   extension;
2. `aisimulate-core` Rust crate — the native estimator and simulation core for
   Rust consumers.

It does **not** publish an `aiconfigurator` or `aiconfigurator-core` wheel, a
Python `aisimulate-core` distribution, or an `aiconfigurator-core` crate. The
`aisimulate` wheel preserves the `aiconfigurator`, `aiconfigurator_core`, and
`aisimulate_core` Python import namespaces during the compatibility window.

The AISimulate wheel does not declare Dynamo as an installation dependency.
Dynamo-owned Router, Planner, runtime, transport, and live-Mocker integrations
consume AISimulate through optional adapters.

## Develop from source

```bash
git clone https://github.com/ai-dynamo/aisimulate.git
cd aisimulate

uv venv .venv
source .venv/bin/activate
uv pip install -e ./python/aisimulate
```

Current performance profiles are checked-in Parquet files, so normal builds
and usage do not require Git LFS. Install Git LFS and run `git lfs pull` only
when working with retained legacy `*.txt` performance assets or their
compatibility tests.

## Accuracy evidence

The public-ready [E2E Accuracy Overview](python/aisimulate/docs/e2e-accuracy/)
reports matched client-observed TTFT and TPOT accuracy against measured silicon
operating points. It keeps accuracy, evidence coverage, and curve-shape error
separate and includes a machine-readable aggregate with exact snapshot digests.

The checked-in snapshot excludes multi-node configurations and applies only to
the exact model, hardware, framework, topology, workload, and concurrency cells
that were measured. It is not a universal support or deployment-certification
claim. Forward-pass accuracy and strict-native estimator coverage remain
separate evidence lanes.

Run the repository validation suites with:

```bash
cargo test --workspace
python -m pytest -c pytest.ini tests
python -m pytest -c python/aisimulate/pytest.ini python/aisimulate/tests -m "unit or build"
```

See [DEVELOPMENT.md](python/aisimulate/DEVELOPMENT.md) for environment and test details and
[CONTRIBUTING.md](CONTRIBUTING.md) before sending a change.
