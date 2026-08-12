# AISimulate

AISimulate is the standalone, engine-neutral home for GPU-free inference
simulation and configuration search. Dynamo integrates through Dynamo-owned
adapters; this repository must not import or depend on Dynamo runtime types or
services.

This initial migration establishes the shared estimator core from
AIConfigurator. The public CLI, Sweeper, Replayer, generalized Mocker, and
Dynamo provider work will land in their corresponding follow-up issues.
The layout and dependency boundary follow the
[AISimulate Standalone Repo Roadmap (Overview)](https://docs.google.com/document/d/1teYyqyf64K9h0mhiQEJbJLqZAHTjyZD98szHrEY0J_8/edit).

## Repository layout

```text
crates/
  aisimulate-core/      Rust estimator and native PyO3 extension
  tests/public-api/     external-consumer compile contract
python/
  aisimulate-core/      Python SDK, model metadata, and performance data
docs/
  core-api.md           public API and compatibility contract
  migration.md          AIC-to-AISimulate migration and history notes
```

The `aisimulate-core` wheel exports the new `aisimulate_core` facade and keeps
the existing `aiconfigurator_core` namespace as a compatibility surface for
AIC 0.12.0. The `aisimulate-core` crate is imported as `aisimulate_core`.

## Build and test

```bash
cargo test --workspace
python -m venv .venv
.venv/bin/python -m pip install -U pip maturin pytest
cd python/aisimulate-core
../../.venv/bin/python -m maturin build --interpreter ../../.venv/bin/python \
  --release --out ../../dist
```

For an editable Python install:

```bash
cd python/aisimulate-core
../../.venv/bin/python -m maturin develop --release
../../.venv/bin/python -m pytest tests
```

## History

The core was imported with filtered AIC commit ancestry, then reorganized with
Git renames. Use `git log --follow -- <path>` to traverse a file back through
the former `aic-core/`, `rust/aiconfigurator-core/`, and
`src/aiconfigurator/` locations. See [docs/migration.md](docs/migration.md).
