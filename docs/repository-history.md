# AISimulate repository history

This repository combines two independently preserved source streams: the full
AIConfigurator product and the former Dynamo `aisimulate/` package.

## Full AIConfigurator migration

AISimulate is the canonical source for the complete AIConfigurator product and
its engine-neutral estimator core starting with AISimulate 0.12.0. The
migration is tracked by
[AIC-1702](https://linear.app/nvidia/issue/AIC-1702/repo-migrate-aiconfigurator-core-to-the-aisimulate-repository).

### API mapping

| AIC 0.11 surface | AISimulate 0.12 surface | Compatibility |
| --- | --- | --- |
| Python distribution `aiconfigurator` | `aisimulate` | The `aiconfigurator` command and import namespace ship inside `aisimulate` during the compatibility window |
| CLI `aiconfigurator ...` | `aisimulate predict` / `aisimulate recommend` for new simulation workflows; `aiconfigurator ...` for compatibility-only workflows | See the [AIC migration guide](cli/migrate-from-aiconfigurator.md); this is not a flag-compatible rename |
| Python distribution `aiconfigurator-core` | included in `aisimulate` | No separate core distribution is installed |
| `aiconfigurator_core` | `aiconfigurator_core` from the `aisimulate` wheel | Existing import remains available in 0.12.0 |
| `aiconfigurator_core.sdk` | `aisimulate_core.sdk` facade in the same wheel | Both import paths remain available in 0.12.0 |
| Rust package/import `aiconfigurator-core` / `aiconfigurator_core` | `aisimulate-core` / `aisimulate_core` | Cargo consumers may temporarily alias the new package under the old dependency key |

Temporary Cargo alias:

```toml
[dependencies]
aiconfigurator-core = { package = "aisimulate-core", version = "0.12", features = ["python"] }
```

The former AIC crate-root types and builders remain available through this
alias. Its `engine` module is the one unavoidable name collision: the combined
crate keeps Replay's scheduler at `aisimulate_core::engine`, so the former AIC
compiled-engine module is available at `aisimulate_core::perfmodel::engine`.
Consumers that do not embed Python can omit the `python` feature and use the
pure-Rust performance-model types; the feature is required for
`AicEngineBuilder` and `AicEngine`.

### Ownership boundary

AISimulate owns the estimator, model and performance data, neutral scheduling
and replay contracts, engine-native simulation, and the migrated AIC
application. The AISimulate wheel does not declare Dynamo as an installation
dependency. The imported generator retains function-local use of Dynamo's
deployment config modifiers only when a caller explicitly requests Dynamo
manifests; moving that integration behind a Dynamo-owned adapter is a separate
boundary cleanup and is not another release artifact.

### Source provenance

The migration branch keeps the earlier path-filtered AIC core history. The
complete upper application was initially imported as a snapshot from
AIConfigurator source commit `13b5cf2697876692b0a52098266c81162add11fc`.
The current synchronization boundary is source commit
`095f58a51c4ca8e61b66ec108d86f223f8d559ce`. It includes the initial boundary
at `ff2be1fd434fd516474e42b77f94cd5a5f841b9b` plus the 18 first-parent commits
in the frozen `ff2be1f..095f58a` range. The final tree moves the upper
application and Python core beneath `python/aisimulate/`, and moves the AIC
Rust core beneath `crates/core/src/perfmodel/`, without copying a second
buildable manifest.

The imported upper tree includes the CLI, generator, SDK compatibility layer,
Collector, tests, docs, Docker/development assets, and the original inactive
workflow definitions. Only the repository-root `.github/workflows/` directory
is active in AISimulate.

The source commit is the future synchronization boundary. With an
AIConfigurator clone at the sibling `../aiconfigurator` path and its origin
fetched, for example:

```bash
git log --follow -- crates/core/src/perfmodel/mod.rs
git log --follow -- python/aisimulate/src/aiconfigurator_core/sdk/engine.py
git -C ../aiconfigurator fetch origin
diff -u <(git -C ../aiconfigurator show 095f58a51c4ca8e61b66ec108d86f223f8d559ce:src/aiconfigurator/main.py) <(git show HEAD:python/aisimulate/src/aiconfigurator/main.py)
```

### Bulk synchronization ledger

[AIC-1788](https://linear.app/nvidia/issue/AIC-1788/repo-bulk-sync-post-ff2be1f-aiconfigurator-changes-into-aisimulate-through-095f58a) tracks the single AISimulate bulk synchronization from `ff2be1f` through `095f58a`. The review units below preserve the source range's first-parent order.

| AIC PR | Source commit | Review unit | AISimulate disposition |
| --- | --- | --- | --- |
| [#1565](https://github.com/ai-dynamo/aiconfigurator/pull/1565) | [`61613b9`](https://github.com/ai-dynamo/aiconfigurator/commit/61613b9) | AIC-1802 | Preserved the macOS wheel action byte-for-byte under inactive `python/aisimulate/.github/`; active workflow applicability remains AIC-1706. |
| [#1564](https://github.com/ai-dynamo/aiconfigurator/pull/1564) | [`a9e012a`](https://github.com/ai-dynamo/aiconfigurator/commit/a9e012a) | AIC-1790 | Mapped the DSA data correction and fail-open coverage; final Parquet and metadata blobs remain byte-identical. |
| [#1568](https://github.com/ai-dynamo/aiconfigurator/pull/1568) | [`c7bc4cf`](https://github.com/ai-dynamo/aiconfigurator/commit/c7bc4cf) | AIC-1802 | Preserved the platform-wheel workflow byte-for-byte under inactive `python/aisimulate/.github/`; no active root workflow changed. |
| [#1545](https://github.com/ai-dynamo/aiconfigurator/pull/1545) | [`1d76ac0`](https://github.com/ai-dynamo/aiconfigurator/commit/1d76ac0) | AIC-1791 | Mapped the Dynamo recipe adapter to the AIS application/core layout. |
| [#1571](https://github.com/ai-dynamo/aiconfigurator/pull/1571) | [`298cd36`](https://github.com/ai-dynamo/aiconfigurator/commit/298cd36) | AIC-1789 | Mapped the database fixture cache-invalidation guard and tests. |
| [#1541](https://github.com/ai-dynamo/aiconfigurator/pull/1541) | [`a81829d`](https://github.com/ai-dynamo/aiconfigurator/commit/a81829d) | AIC-1792 | Mapped the Nemotron-3.5-Lightning NVFP4 Collector declarations and data. |
| [#1548](https://github.com/ai-dynamo/aiconfigurator/pull/1548) | [`2dd1fb4`](https://github.com/ai-dynamo/aiconfigurator/commit/2dd1fb4) | AIC-1792 | Mapped the DeepSeek-V4 NVFP4 checkpoint declarations and model metadata. |
| [#1540](https://github.com/ai-dynamo/aiconfigurator/pull/1540) | [`724f763`](https://github.com/ai-dynamo/aiconfigurator/commit/724f763) | AIC-1790 | Mapped DSA all-full fail-open behavior when skip-indexer rows are absent. |
| [#1544](https://github.com/ai-dynamo/aiconfigurator/pull/1544) | [`71f49c7`](https://github.com/ai-dynamo/aiconfigurator/commit/71f49c7) | AIC-1794 | Mapped MTP decode-share scaling and reversion guards. |
| [#1445](https://github.com/ai-dynamo/aiconfigurator/pull/1445) | [`40c1e74`](https://github.com/ai-dynamo/aiconfigurator/commit/40c1e74) | AIC-1793 | Mapped the perf-data reuse manifest/tool rename; regenerated the manifest from the final AIS data tree. |
| [#1402](https://github.com/ai-dynamo/aiconfigurator/pull/1402) | [`87caf68`](https://github.com/ai-dynamo/aiconfigurator/commit/87caf68) | AIC-1795 | Mapped removal of the legacy `build_aic_engine` adapter into the AIS core API and implementation. |
| [#1569](https://github.com/ai-dynamo/aiconfigurator/pull/1569) | [`163f662`](https://github.com/ai-dynamo/aiconfigurator/commit/163f662) | AIC-1802 | Preserved the inactive build-test workflow byte-for-byte and mapped the sanity selector, trigger paths, and notebook; active CI wiring remains AIC-1706. |
| [#1513](https://github.com/ai-dynamo/aiconfigurator/pull/1513) | [`90f7fc0`](https://github.com/ai-dynamo/aiconfigurator/commit/90f7fc0) | AIC-1798 | Mapped current NVIDIA NVFP4 variants, including B60, and imported all final support-matrix CSVs byte-for-byte. |
| [#1473](https://github.com/ai-dynamo/aiconfigurator/pull/1473) | [`0dc8a2b`](https://github.com/ai-dynamo/aiconfigurator/commit/0dc8a2b) | AIC-1796 | Mapped the shared Generator/Collector FPM contract. |
| [#1575](https://github.com/ai-dynamo/aiconfigurator/pull/1575) | [`b28ab8f`](https://github.com/ai-dynamo/aiconfigurator/commit/b28ab8f) | AIC-1800 | Mapped DeepSeek-V4 NVFP4 Hopper redirects. |
| [#1507](https://github.com/ai-dynamo/aiconfigurator/pull/1507) | [`dbb322e`](https://github.com/ai-dynamo/aiconfigurator/commit/dbb322e) | AIC-1801 | Mapped MiniMax-M3 MSA collectors, SDK tables, model metadata, and multi-platform data; the bounded evidence-waiver correction is recorded below. |
| [#1486](https://github.com/ai-dynamo/aiconfigurator/pull/1486) | [`899034f`](https://github.com/ai-dynamo/aiconfigurator/commit/899034f) | AIC-1797 | Mapped TRT-LLM DeepSeek-V4 mHC and CSA/HCA collectors and source data; the unrelated-case filtering correction is recorded below. |
| [#1475](https://github.com/ai-dynamo/aiconfigurator/pull/1475) | [`095f58a`](https://github.com/ai-dynamo/aiconfigurator/commit/095f58a) | AIC-1799 | Mapped the FPM forward-pass collection workflow. |

The closure audit accounts for all 406 source paths and all six detected renames. Its exact-data envelope contains 96 changed Parquet blobs totaling 26,373,223 bytes.

### Intentional AISimulate adaptations

- Source paths map to the combined AIS layout: `aic-core/rust/aiconfigurator-core/src/` to `crates/core/src/perfmodel/`, `aic-core/src/aiconfigurator_core/` to `python/aisimulate/src/aiconfigurator_core/`, and the upper application, Collector, tests, docs, and tools beneath `python/aisimulate/`. Migration-introduced crate, distribution, package-data, and tool references follow those destinations; pre-existing source-provenance comments may retain historical AIC paths.
- The source `.gitattributes` snapshot remains byte-identical under `python/aisimulate/`. Root `.gitattributes` carries equivalent mapped rules for the active AIS data and generated model-config paths and is owned by AISimulate Infra plus maintainers.
- The source engine-step golden is byte-identical. Source model configs, collection metadata, reuse declarations, Parquet files, `collector_ref` values, framework image digests, and other pinned SHAs are preserved unless a row is explicitly named here.
- `perf_data_reuse_manifest.yaml` is intentionally regenerated from the final AIS data tree because the source snapshot predates the data added by #1507 and #1486. Its generator defaults and rendered instructions use the unified `python/aisimulate/src/aiconfigurator_core` path.
- Source workflows and actions are provenance-only snapshots under `python/aisimulate/.github/`. The repository-root workflows remain unchanged by this synchronization; [AIC-1706](https://linear.app/nvidia/issue/AIC-1706/repo-establish-aisimulate-ci-release-and-ownership-contract) owns active CI and release applicability.
- AISimulate publishes two artifacts: one `aisimulate` wheel and one `aisimulate-core` crate. The application manifest includes the migrated FPM workflow, its model-plan YAML, in-pod runtime assets, compatibility SDKs, and performance data in the `aisimulate` wheel so `python -m collector.fpm_forward` remains usable after installation; this package-data mapping does not add an artifact or console script.
- Rust formatting, AIS API documentation, and Python lint adaptations are limited to crate/module identity and existing AIS checks. Commit `f6e2f7a` explicitly defers Qwen W4A16 Collector cases that the retained runtime cannot execute instead of silently relabeling them.
- Commit `7f07f3b` removes unrelated-case filtering from the DSV4 Collector and bounds the MSA evidence waiver to its approved scope. These are post-port policy corrections, not untracked source drift.
The stable path mapping and last synchronized AIC commit are recorded in
[`scripts/aic_sync.toml`](../scripts/aic_sync.toml). Follow the binary-safe,
path-rewritten workflow in [aic-sync.md](aic-sync.md) for later AIC code, data,
and test commits. Packaging, CI, and repository-policy changes are adapted
manually because AISimulate owns the combined release boundary.

### CLI transition and compatibility gate

The `aisimulate` wheel installs both the unified `aisimulate` application and
the established `aiconfigurator` compatibility command. `predict` and
`recommend` are the preferred entry points for new simulation and search
integrations, but copying AIC code and adding the new commands does not prove
behavioral parity for every legacy workflow. AIC-1480/AIC-1472/AIC-1476 track
the remaining parity and product evidence.

Keep using `aiconfigurator` for the no-direct-replacement workflows in the
[AIC migration guide](cli/migrate-from-aiconfigurator.md) until an explicit
replacement and migration path land. The compatibility command is planned for
deprecation, but its removal schedule is a separate release decision and must
not be inferred from the presence of the new CLI.

For features already implemented by the standalone Sweeper, see
[Migrate from AIConfigurator](cli/migrate-from-aiconfigurator.md) for explicit legacy
command-to-configuration examples and current execution boundaries.

## Dynamo-to-AISimulate package migration

The standalone Replay and Sweeper package migration is tracked by
[AIC-1703](https://linear.app/nvidia/issue/AIC-1703/repo-migrate-the-aisimulate-package-to-the-aisimulate-repository).
It includes the generalized Mocker engine and deterministic Replayer from
[Dynamo PR #12525](https://github.com/ai-dynamo/dynamo/pull/12525). The package
remains Dynamo-independent; Dynamo-owned Router, Planner, runtime, transport,
and live-Mocker integrations consume it through optional adapter contracts.

The migration merged a path-filtered copy of the Dynamo history for the former
`aisimulate/` directory. Original commits, authorship, and blame context are
retained. For example:

```bash
git log --follow -- crates/core/src/engine/generalized/engine.rs
git log --follow -- python/aisimulate/src/aisimulate/sweeper/search.py
```

The two migration branches initially carried separate Python and Rust
manifests. The post-merge reconciliation keeps the preserved source histories
but builds one 0.12 wheel from `python/aisimulate/`: its mixed Maturin layout
packages the unified native extension together with the complete AIC
application/core and Replay/Sweeper Python sources. One published
`aisimulate-core` crate in `crates/core/` combines the estimator and Replay
runtime behind feature-gated Python bindings.

The imported history requires a merge commit. Squashing would retain the files
but discard that Dynamo ancestry from this repository's `main` history.

## Artifact boundary

The only publishable packages are the `aisimulate` wheel and the
`aisimulate-core` crate. See [artifact-contract.md](artifact-contract.md).
