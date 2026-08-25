# AIC source compatibility view

This directory is not a Python project, Rust crate, or release artifact. It
contains only narrow, non-recursive symlinks that preserve repository-relative
paths used by imported AIConfigurator tests and tooling during the deprecation
window.

- `src/` points at the source tree packaged by the single `aisimulate` wheel.
- `rust/aiconfigurator-core/{src,tests,parity_tests,docs}` point at the
  performance-model mirrors owned by the single `aisimulate-core` crate.

Do not add a `pyproject.toml` or `Cargo.toml` here. New code should use the
canonical paths recorded in the repository-root `scripts/aic_sync.toml`.
