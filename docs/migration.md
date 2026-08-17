# Full AIConfigurator migration

This repository is the canonical source for the complete AIConfigurator
product and its engine-neutral estimator core starting with AISimulate 0.12.0.
The migration is tracked by
[AIC-1702](https://linear.app/nvidia/issue/AIC-1702/repo-migrate-aiconfigurator-core-to-the-aisimulate-repository).

## API mapping

| AIC 0.11 surface | AISimulate 0.12 surface | Compatibility |
| --- | --- | --- |
| Python distribution `aiconfigurator` | `aisimulate` | The `aiconfigurator` command and import namespace ship inside `aisimulate` during the compatibility window |
| CLI `aiconfigurator ...` | `aisimulate ...` | Both commands delegate to the complete AIC CLI until the new CLI passes the parity gate |
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
and replay contracts, engine-native simulation, and the migrated AIC
application. Neither AISimulate wheel declares Dynamo as an installation
dependency. The imported generator still uses Dynamo's deployment config
modifiers through function-local imports when a caller explicitly requests
Dynamo manifests; moving that integration behind a Dynamo-owned adapter is a
separate boundary cleanup and is not a fourth release artifact.

## Source provenance

The migration branch keeps the earlier path-filtered core history. The
complete upper application is imported as a reviewable snapshot from
AIConfigurator source commit
`13b5cf2697876692b0a52098266c81162add11fc`. The final tree moves that upper
application beneath `python/aisimulate/` and updates the existing core layout
without copying a second buildable core manifest.

The imported upper tree includes the CLI, generator, SDK compatibility layer,
Collector, tests, docs, Docker/development assets, and the original inactive
workflow definitions. Only the repository-root `.github/workflows/` directory
is active in AISimulate.

The source commit is the future synchronization boundary. For example:

```bash
git log --follow -- crates/aisimulate-core/src/lib.rs
git log --follow -- python/aisimulate-core/src/aiconfigurator_core/sdk/engine.py
git diff 13b5cf2697876692b0a52098266c81162add11fc:src/aiconfigurator/main.py HEAD:python/aisimulate/src/aiconfigurator/main.py
```

## CLI cutover gate

Copying all AIC code removes repository-placement risk; it does not by itself
prove that the newer `predict`/`recommend` CLI is a behavioral replacement.
Until AIC-1480/AIC-1472/AIC-1476 have passing evidence or approved exceptions,
the `aisimulate` executable delegates to the full AIC command implementation.
The legacy `aiconfigurator` executable is an alias in the same wheel, not a
fourth release artifact.

## Artifact boundary

The only publishable packages are `aisimulate`, `aisimulate-core` (wheel), and
`aisimulate-core` (crate). See [artifact-contract.md](artifact-contract.md).
