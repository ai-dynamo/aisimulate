<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AISimulate

AISimulate is the standalone home for GPU-free inference simulation and
deployment configuration. The AISimulate wheel does not declare Dynamo as an
installation dependency. Dynamo-owned Router, Planner, runtime, transport,
and live-Mocker integrations consume AISimulate through optional adapters.

Starting with 0.12.0, this repository owns the complete AIConfigurator product
surface—not only its native core. The full AIC application, CLI, generator,
SDK, Collector, tests, documentation, and development tooling are preserved
under `python/aisimulate/`. The estimator SDK ships in that same wheel, while
the native estimator and Replay runtime remain independently consumable from
Rust through `aisimulate-core`.

> [!WARNING]
> Replay and Sweeper are experimental surfaces intended for evaluation and
> feedback, not production capacity planning. Their APIs, schemas, search
> behavior, and output may change without a standard deprecation period.

## Release artifacts

This repository produces exactly two release artifacts:

1. `aisimulate` Python wheel — the application, CLI, estimator SDK,
   model/performance data, Replay, Sweeper, and unified native extension;
2. `aisimulate-core` Rust crate — the native estimator and simulation core for
   Rust consumers.

It does **not** publish an `aiconfigurator` wheel or an `aiconfigurator-core`
wheel/crate, nor a Python `aisimulate-core` distribution. The legacy Python
import namespaces remain in the `aisimulate` wheel, which also preserves the supported
`aiconfigurator` console command. The `aisimulate` distribution does not
rename that compatibility command; it additionally installs the public
`aisimulate predict`/`aisimulate recommend` application.

## CLIs

Installing the `aisimulate` wheel provides the unified simulation CLI and preserves the established
AIC command name:

```bash
uv pip install aisimulate
aisimulate predict --config prediction.yaml
aisimulate recommend --config recommendation.yaml
aiconfigurator cli generate --model-path Qwen/Qwen3-32B-FP8 --total-gpus 8 --system h200_sxm
```

`aisimulate` is the only Replay/Sweeper CLI. The built-in `engine` stack is the default;
`--stack dynamo` selects the optional runner and Router/Planner configuration adapters registered by
an independently installed `ai-dynamo` wheel. The existing Replay and Sweeper Python APIs remain
available to embedded callers.

For example, a minimal prediction input is:

```yaml
engine:
  model: Qwen/Qwen3-32B-FP8
  hardware: h200_sxm
  backend: vllm
  workers:
    aggregated: {}
```

See [`docs/cli/design.md`](docs/cli/design.md) for the complete schema and search-domain contract.

Install AISimulate by itself for engine-only development:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e ./python/aisimulate
```

For Dynamo feature development, install `ai-dynamo` separately. Its optional
Router and Planner adapters consume the released `aisimulate` artifact:

```bash
uv pip install aisimulate ai-dynamo
```

The `ai-dynamo` wheel registers the `dynamo` runner factory plus `dynamo.planner` and
`dynamo.router` configuration adapters and Sweeper providers. Its runner composes the materialized
runtime hooks with the shared AISimulate Replayer.

Run a sweep from Python with an explicit runner:

```python
from aisimulate.sweeper import SmartSearchConfig, Sweeper
from dynamo.replay.simulation import DynamoReplayRunnerFactory

config = SmartSearchConfig.from_yaml("smart_sweep.yaml")
candidates = Sweeper(
    runner_factory=DynamoReplayRunnerFactory(),
).run(config)
```

The standalone module validates the backend-neutral core schema but intentionally has no implicit
replay runtime. Adapter-owned search spaces are validated when the selected adapters are resolved
by `Sweeper.run`.
KVBM sweep fields have been removed and have no adapter migration.

Read the canonical [Sweeper documentation](docs/sweeper/overview.md) for its configuration,
search-space, and replay behavior. Backend-neutral and Dynamo integration examples live under
[`examples/sweeper`](examples/sweeper/README.md). Dynamo owns its [Sweeper integration
guide](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/developer-guide/knowledge-base/modular-components/ai-simulate-experimental/sweeper-experimental/dynamo-integration.md),
including the Dynamo development environment and adapter contracts.

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

## Repository layout

```text
crates/
  core/                 sole product crate: AIC perf model, Mocker, Replay, and PyO3 runtime
  tests/public-api/     external-consumer compile contract
python/
  aisimulate/           application, AIC core mirror/data, Replay, Sweeper, and native runtime
docs/
  cli/
    design.md           public CLI schema and output contract
    migrate-from-aiconfigurator.md
                        AIConfigurator-to-AISimulate CLI translation
  artifact-contract.md  two-artifact release boundary
  aic-sync.md           deterministic AIC source synchronization workflow
  core-api.md           public core API and compatibility contract
  migration.md          AIC and Dynamo source/history mapping
scripts/
  build_release_artifacts.py
```

## Development

```bash
cargo test --workspace
python -m pytest -c pytest.ini tests
python -m pytest -c python/aisimulate/pytest.ini python/aisimulate/tests -m "unit or build"
```

Read the canonical [Sweeper documentation](docs/sweeper/overview.md) for its
configuration, search-space, and replay behavior. For focused AIC development
instructions, see [`python/aisimulate/README.md`](python/aisimulate/README.md)
and [`docs/core-api.md`](docs/core-api.md).

## Source provenance

The branch retains both imported histories: the path-filtered AIC core ancestry
and the path-filtered Dynamo ancestry for the former `aisimulate/` directory.
The complete AIC upper application was initially imported from AIConfigurator
`main` commit `13b5cf2697876692b0a52098266c81162add11fc` and is synchronized
through commit `095f58a51c4ca8e61b66ec108d86f223f8d559ce`. See
[`docs/migration.md`](docs/migration.md) for provenance and
[`docs/aic-sync.md`](docs/aic-sync.md) for the stable mirror mapping.
