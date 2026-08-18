<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AISimulate

AISimulate is the standalone home for GPU-free inference simulation and
deployment configuration. Neither AISimulate wheel declares Dynamo as an
installation dependency. Dynamo-owned Router, Planner, runtime, transport,
and live-Mocker integrations consume AISimulate through optional adapters.

Starting with 0.12.0, this repository owns the complete AIConfigurator product
surface—not only its native core. The full AIC application, CLI, generator,
SDK, Collector, tests, documentation, and development tooling are preserved
under `python/aisimulate/`. The native estimator remains independently
consumable from Python and Rust.

> [!WARNING]
> Replay and Sweeper are experimental surfaces intended for evaluation and
> feedback, not production capacity planning. Their APIs, schemas, search
> behavior, and output may change without a standard deprecation period.

## Release artifacts

This repository produces exactly three release artifacts:

1. `aisimulate` Python wheel — the complete application, CLI, Replay, and Sweeper;
2. `aisimulate-core` Python wheel — the estimator SDK, native extension,
   model metadata, and performance data;
3. `aisimulate-core` Rust crate — the native estimator and simulation core for
   Rust consumers.

It does **not** publish an `aiconfigurator` wheel or an `aiconfigurator-core`
wheel/crate. The legacy Python import namespaces and the `aiconfigurator`
console command remain compatibility surfaces inside the two AISimulate
wheels for the 0.12 transition.

## CLI transition

The `aisimulate` wheel installs both compatibility commands:

```bash
aisimulate cli generate --model-path Qwen/Qwen3-32B-FP8 --total-gpus 8 --system h200_sxm
aiconfigurator cli generate --model-path Qwen/Qwen3-32B-FP8 --total-gpus 8 --system h200_sxm
```

Both currently execute the complete, proven AIC CLI. The newer
`predict`/`recommend` design must satisfy the tracked AIC-to-AISimulate parity
matrix, product requirements, and approved exceptions before it replaces this
delegation.

For an engine-only replay, use `python -m aisimulate.replay`. Dynamo Router,
Planner, or online adapters remain available through `python -m dynamo.replay`
when `ai-dynamo` is installed separately. For configuration search, call
`Sweeper(runner_factory=...).run(config)` or start from an example under
[`examples/sweeper`](examples/sweeper/README.md).

## Repository layout

```text
crates/
  aisimulate-core/      migrated AIC estimator and native PyO3 extension
  core/                 generalized Mocker engine and deterministic Replayer
  python/               Python binding for the Replay runtime
  tests/public-api/     external-consumer compile contract
python/
  aisimulate/           complete AIC application and compatibility CLI
  aisimulate-core/      Python estimator SDK, metadata, and performance data
src/aisimulate/         standalone Replay and Sweeper Python source
docs/
  artifact-contract.md  three-artifact release boundary
  core-api.md           public core API and compatibility contract
  migration.md          AIC and Dynamo source/history mapping
scripts/
  build_release_artifacts.py
```

## Development

```bash
cargo test --workspace
python -m pytest
```

Read the canonical [Sweeper documentation](docs/sweeper/overview.md) for its
configuration, search-space, and replay behavior. For focused AIC development
instructions, see [`python/aisimulate/README.md`](python/aisimulate/README.md)
and [`docs/core-api.md`](docs/core-api.md).

## Source provenance

The branch retains both imported histories: the path-filtered AIC core ancestry
and the path-filtered Dynamo ancestry for the former `aisimulate/` directory.
The complete AIC upper application is a snapshot from AIConfigurator `main`
commit `13b5cf2697876692b0a52098266c81162add11fc`. See
[`docs/migration.md`](docs/migration.md) for the complete mapping.
