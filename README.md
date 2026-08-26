<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AISimulate

AISimulate predicts LLM serving behavior and searches for strong deployment
configurations offline, without bringing up a GPU serving cluster.

One installation exposes two console entry points covering three primary
workflows:

| Goal | Command |
|---|---|
| Predict the behavior of one concrete deployment | `aisimulate predict` |
| Search a deployment space and recommend concrete configurations | `aisimulate recommend` |
| Use the established AIConfigurator estimator, generator, and support workflows | `aiconfigurator cli ...` |

The `aiconfigurator` executable is a compatibility surface shipped by the
`aisimulate` wheel. It is not a separate package to install. AISimulate also
provides Python SDKs for embedding the estimator, Replay, and Sweeper, plus the
`aisimulate-core` crate for Rust consumers.

> [!WARNING]
> Replay and Sweeper are experimental surfaces intended for evaluation and
> feedback, not production capacity planning. Their APIs, schemas, search
> behavior, and output may change without a standard deprecation period.

## Install

```bash
python3 -m pip install aisimulate

aisimulate --help
aiconfigurator --help
```

When upgrading from the former standalone AIConfigurator distributions, remove
them first so that only AISimulate provides the compatibility imports and
command:

```bash
python3 -m pip uninstall -y aiconfigurator aiconfigurator-core
python3 -m pip install --upgrade aisimulate
```

## Predict one deployment

`predict` accepts a pinned deployment configuration. For example, save this as
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

Run the prediction with the built-in `engine` stack:

```bash
aisimulate predict \
  --config prediction.yaml \
  --output-dir ./aisimulate-prediction
```

The CLI prints a concise summary and writes the complete report to
`aisimulate-prediction/prediction.json`. Add `--capture-per-request` to also
write `requests.jsonl`, or use `--format json` for machine-readable standard
output.

## Recommend a deployment

`recommend` accepts the prediction schema plus search domains and an
optimization goal. For example, save this as `recommendation.yaml`:

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

Run the search:

```bash
aisimulate recommend \
  --config recommendation.yaml \
  --output-dir ./aisimulate-recommendation
```

Each file under `aisimulate-recommendation/recommendations/` is a fully
materialized, concrete configuration. It contains no search domains and can be
passed directly back to `predict`:

```bash
aisimulate predict \
  --config ./aisimulate-recommendation/recommendations/0001.yaml \
  --output-dir ./aisimulate-best-prediction
```

Both commands support `--set PATH=YAML_VALUE` overrides, an explicit
`--output-dir`, `--overwrite`, and `--format table|json`. See the
[CLI reference](docs/cli-design.md) for the complete schema, traffic models,
search domains, presets, outputs, and error contract.

## Use the AIConfigurator compatibility CLI

Existing AIC workflows remain available under the established
`aiconfigurator` command:

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

For AIC command options, the Python API, supported systems, configuration
generation, and data collection, read the
[AIConfigurator compatibility guide](python/aisimulate/README.md).

## Execution stacks and Python APIs

The built-in `engine` stack is the default for `aisimulate predict` and
`aisimulate recommend`. Independently installed packages can register optional
runner and configuration adapters without adding another simulation CLI. For
example, the `ai-dynamo` wheel registers the `dynamo` stack:

```bash
python3 -m pip install aisimulate ai-dynamo
aisimulate predict --stack dynamo --config prediction.yaml
```

`aisimulate` is the only Replay/Sweeper CLI. The former module entry points are
replaced by `aisimulate predict` and `aisimulate recommend`; the Replay and
Sweeper Python APIs remain available to embedded callers.

For example, a caller can inject an explicit runner into Sweeper:

```python
from aisimulate.sweeper import SmartSearchConfig, Sweeper
from dynamo.replay.simulation import DynamoReplayRunnerFactory

config = SmartSearchConfig.from_yaml("smart_sweep.yaml")
candidates = Sweeper(
    runner_factory=DynamoReplayRunnerFactory(),
).run(config)
```

The standalone module validates the backend-neutral core schema but has no
implicit replay runtime. Adapter-owned search spaces are validated when the
selected adapters are resolved by `Sweeper.run`. KVBM sweep fields have been
removed and have no adapter migration.

Read the canonical [Sweeper documentation](docs/sweeper/overview.md) for its
configuration, search-space, and replay behavior. Backend-neutral and Dynamo
integration examples live under [`examples/sweeper`](examples/sweeper/README.md).
Dynamo owns its [Sweeper integration
guide](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/developer-guide/knowledge-base/modular-components/ai-simulate-experimental/sweeper-experimental/dynamo-integration.md).

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
`aisimulate_core` Python import namespaces for compatibility.

The AISimulate wheel does not declare Dynamo as an installation dependency.
Dynamo-owned Router, Planner, runtime, transport, and live-Mocker integrations
consume AISimulate through optional adapters.

## Repository layout

```text
crates/
  core/                 sole product crate: AIC perf model, Mocker, Replay, and PyO3 runtime
  tests/public-api/     external-consumer compile contract
python/
  aisimulate/           application, AIC compatibility source/data, Replay, Sweeper, and runtime
docs/
  cli-design.md         prediction and recommendation CLI contract
  artifact-contract.md  two-artifact release boundary
  aic-sync.md           deterministic AIC source synchronization workflow
  core-api.md           public core API and compatibility contract
  migration.md          AIC and Dynamo source/history mapping
scripts/
  build_release_artifacts.py
```

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

Run the repository validation suites with:

```bash
cargo test --workspace
python -m pytest -c pytest.ini tests
python -m pytest -c python/aisimulate/pytest.ini python/aisimulate/tests -m "unit or build"
```

For focused development contracts, see the
[core API](docs/core-api.md), [artifact contract](docs/artifact-contract.md),
and [AIC synchronization guide](docs/aic-sync.md).

## Source provenance

The repository retains both imported histories: the path-filtered AIC core
ancestry and the path-filtered Dynamo ancestry for the former `aisimulate/`
directory. The complete AIC upper application was initially imported from
AIConfigurator `main` commit `13b5cf2697876692b0a52098266c81162add11fc`
and is synchronized through commit
`095f58a51c4ca8e61b66ec108d86f223f8d559ce`. See
[`docs/migration.md`](docs/migration.md) for provenance and
[`docs/aic-sync.md`](docs/aic-sync.md) for the stable mirror mapping.
