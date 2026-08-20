# AISimulate artifact contract

AISimulate 0.12.0 has one product version and exactly three release artifacts:

| Artifact | Build manifest | Public purpose |
| --- | --- | --- |
| `aisimulate` wheel | `python/aisimulate/pyproject.toml` | Complete application, CLI, generator, compatibility SDK, Replay, Sweeper, native runtime, and package data |
| `aisimulate-core` wheel | `python/aisimulate-core/pyproject.toml` | Native Python estimator, model metadata, and performance data |
| `aisimulate-core` crate | `crates/aisimulate-core/Cargo.toml` | Native Rust estimator |

The internal Replay engine and Python binding crates and the workspace test
crate are `publish = false`. Imported AIConfigurator source
does not retain another buildable `aiconfigurator` or `aiconfigurator-core`
manifest. The preserved command and compatibility namespaces live inside the
approved wheels and therefore do not add artifacts.

`scripts/build_release_artifacts.py` validates the manifest set before it
builds and validates the output directory afterward. A release build fails if
an additional wheel, source distribution, or crate appears.

All three artifacts use version `0.12.0`. The application wheel depends on the
exact matching core wheel. The core wheel and crate also remain version-locked
because they share wire-schema constants and native behavior.
