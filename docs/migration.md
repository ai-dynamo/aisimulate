# AIC core migration

This repository is the canonical source for the engine-neutral estimator core
starting with AISimulate 0.12.0. The migration is tracked by
[AIC-1702](https://linear.app/nvidia/issue/AIC-1702/repo-migrate-aiconfigurator-core-to-the-aisimulate-repository).

## API mapping

| AIC 0.11 surface | AISimulate 0.12 surface | Compatibility |
| --- | --- | --- |
| Python distribution `aiconfigurator-core` | `aisimulate-core` | Install name changes |
| `aiconfigurator_core` | `aisimulate_core` | Old import remains available in 0.12.0 |
| `aiconfigurator_core.sdk` | `aisimulate_core.sdk` | Old facade and explicit submodules remain available in 0.12.0 |
| Rust package/import `aiconfigurator-core` / `aiconfigurator_core` | `aisimulate-core` / `aisimulate_core` | Cargo consumers may temporarily alias the new package under the old dependency key |

Temporary Cargo alias:

```toml
[dependencies]
aiconfigurator-core = { package = "aisimulate-core", version = "0.12" }
```

## Ownership boundary

AISimulate owns the estimator, model and performance data, neutral scheduling
and replay contracts, and engine-native simulation. It must not depend on
Dynamo. Dynamo retains Router, Planner, topology, KVBM, runtime, transport, and
live-Mocker integrations behind Dynamo-owned adapters.

## Preserved Git history

The migration branch merges a path-filtered AIC history. It keeps commits that
touched the core's former locations and prunes unrelated application,
collector, generator, and web code. The final reorganization uses Git renames.

For example:

```bash
git log --follow -- crates/aisimulate-core/src/lib.rs
git log --follow -- python/aisimulate-core/src/aiconfigurator_core/sdk/engine.py
```

GitHub's per-file History view normally follows the current path. When a rename
boundary is not detected in the web UI, the commands above remain authoritative
and `git blame` still traverses the imported ancestry.
