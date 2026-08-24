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
install a second top-level application command named `aisimulate`.

## CLI continuity

Installing the `aisimulate` wheel preserves the established command name:

```bash
uv pip install aisimulate
aiconfigurator cli generate --model-path Qwen/Qwen3-32B-FP8 --total-gpus 8 --system h200_sxm
```

AISimulate becomes the package and source owner without renaming the CLI that
users already invoke. Any future `predict`/`recommend` actions must satisfy the
tracked AIC parity and product gates and evolve the retained `aiconfigurator`
command rather than introducing a second top-level executable.

For an engine-only replay, use `python -m aisimulate.replay`. Dynamo Router,
Planner, or online adapters remain available through `python -m dynamo.replay`
when `ai-dynamo` is installed separately. Both commands share the engine,
topology, traffic, replay-mode, SLA, and output arguments; Dynamo adds its
adapter options. For configuration search, call
`Sweeper(runner_factory=...).run(config)` or start from an example under
[`examples/sweeper`](examples/sweeper/README.md).

For example, run one engine-only synthetic replay with fixed timing:

```bash
python -m aisimulate.replay \
  --extra-engine-args '{"engine_type":"vllm","num_gpu_blocks":1024,"block_size":16,"timing_model":{"type":"fixed","prefill_ms":10,"decode_ms":2}}' \
  --input-tokens 1024 \
  --output-tokens 128 \
  --request-count 16 \
  --replay-concurrency 4
```

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

The `ai-dynamo` wheel registers the `dynamo.planner` and `dynamo.router` Sweeper provider entry
points. Its Dynamo runner composes the materialized runtime hooks with the shared AI Simulate
Replayer.

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

## Repository layout

```text
crates/
  core/                 sole product crate: AIC perf model, Mocker, Replay, and PyO3 runtime
  tests/public-api/     external-consumer compile contract
python/
  aisimulate/           application, AIC core mirror/data, Replay, Sweeper, and native runtime
docs/
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
