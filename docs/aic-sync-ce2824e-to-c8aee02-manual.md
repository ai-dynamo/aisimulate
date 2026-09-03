# AIConfigurator manual synchronization report

- From: `ce2824e8abd9bef71c3b162f651e63704a0eb4c1`
- To: `c8aee02f0887547a334c3d6cd192c42757e4b40e`
- Source PR: [ai-dynamo/aiconfigurator#1602](https://github.com/ai-dynamo/aiconfigurator/pull/1602)

This final transfer carries the worker-type-bound FPM regression redesign from
the frozen AIConfigurator repository. The feature-only source delta is
`7bdd746c..c8aee02f`; the preceding `ce2824e8..7bdd746c` commit changes only
standalone AIConfigurator package versions in the release paths accounted for
below.

## `pyproject.toml`, `aic-core/pyproject.toml`, and `uv.lock`

Reason: AISimulate has one wheel manifest and a combined dependency set.

The source range bumps the standalone AIConfigurator distributions from 0.11.0
to 0.12.0. AISimulate already owns a unified 0.12.0 release boundary, so no
Python manifest or lockfile change is required.

## `aic-core/rust/aiconfigurator-core/Cargo.toml`

Reason: AISimulate owns the combined `aisimulate-core` crate.

The source change is only the matching 0.12.0 package-version bump. The
combined crate is already versioned 0.12.0, so its manifest and workspace lock
remain unchanged.

## Other standalone release files

`README.md`, `aic-core/uv.lock`,
`aic-core/rust/aiconfigurator-core/Cargo.lock`, and
`aic-core/rust/aiconfigurator-core/README.md` also change only their standalone
AIConfigurator version references. AISimulate's root documentation and unified
locks already describe the 0.12.0 artifacts, so none of these source edits is
ported.

## Public API documentation and external-crate tests

`aic-core/API.md` and `aic-core/rust/tests/public-api/src/lib.rs` are outside
the automatic mirror table. Their feature-relevant changes are adapted into
`docs/core-api.md` and `crates/tests/public-api/src/lib.rs` respectively.

## AISimulate integration adaptations

- The source crate-root FPM re-export is applied to both
  `crates/core/src/perfmodel/mod.rs` and the compatibility exports in
  `crates/core/src/lib.rs`.
- FPM fixtures continue to use `perfmodel::engine`, because `crate::engine` is
  AISimulate's Replay scheduler namespace, and retain the unified wheel's
  system-data path.
- Python-backed native constructors remain gated by the target crate's
  `python` feature. The regression-only constructor remains available without
  that feature.
- No standalone package version, active workflow, schema constant, or
  generated CODEOWNERS file changes in this transfer.
