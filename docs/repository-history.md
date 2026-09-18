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
| `aisimulate_core` | `aisimulate_core` from the `aisimulate` wheel | Existing import remains available in 0.12.0 |
| `aisimulate_core.sdk` | `aisimulate_core.sdk` facade in the same wheel | Both import paths remain available in 0.12.0 |
| Rust package/import `aiconfigurator-core` / `aisimulate_core` | `aisimulate-core` / `aisimulate_core` | Cargo consumers may temporarily alias the new package under the old dependency key |

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
`c8aee02f0887547a334c3d6cd192c42757e4b40e`. It includes the initial boundary
at `ff2be1fd434fd516474e42b77f94cd5a5f841b9b`, the 41 first-parent commits in
the frozen `ff2be1f..ce2824e8` range, and the final worker-type-bound FPM
regression transfer from AIConfigurator PR #1602. The final tree moves the
upper application and Python core beneath `python/aisimulate/`, and moves the
AIC Rust core beneath `crates/core/src/perfmodel/`, without copying a second
buildable manifest.

The imported upper tree includes the CLI, generator, SDK compatibility layer,
Collector, tests, docs, Docker/development assets, and the original inactive
workflow definitions. Only the repository-root `.github/workflows/` directory
is active in AISimulate.

The recorded source commit is the final synchronization boundary. With an
AIConfigurator clone at the sibling `../aiconfigurator` path and its origin
fetched, for example:

```bash
git log --follow -- crates/core/src/perfmodel/mod.rs
git log --follow -- python/aisimulate/src/aisimulate_core/sdk/engine.py
git -C ../aiconfigurator fetch origin
diff -u <(git -C ../aiconfigurator show c8aee02f0887547a334c3d6cd192c42757e4b40e:src/aisimulate/legacy_cli/entrypoint.py) <(git show HEAD:python/aisimulate/src/aisimulate/legacy_cli/entrypoint.py)
```

### Selective speculative-decoding migration

[AIConfigurator PR #1563](https://github.com/ai-dynamo/aiconfigurator/pull/1563) is selectively adapted at source head `6290c161a354da5250c391bd43372b2e9c6f4a51` for pluggable speculation schemes and verify-on-FPM. This feature transfer does not advance the contiguous synchronization boundary above. The [migration ledger](aic-pr1563-migration.md) records the source paths, adaptations, and modeling limits.

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

#### September 2026 tail synchronization

The next synchronization advances the boundary from `095f58a` through
`ce2824e8`. It preserves the following 23 first-parent review units in source
order:

| AIC PR | Source commit | AISimulate disposition |
| --- | --- | --- |
| [#1578](https://github.com/ai-dynamo/aiconfigurator/pull/1578) | [`2ab190d`](https://github.com/ai-dynamo/aiconfigurator/commit/2ab190d) | Mapped cross-node EP resolution through DeepEP data. |
| [#1551](https://github.com/ai-dynamo/aiconfigurator/pull/1551) | [`2dc2406`](https://github.com/ai-dynamo/aiconfigurator/commit/2dc2406) | Removed the completed Rust-core migration ladder documents. |
| [#1591](https://github.com/ai-dynamo/aiconfigurator/pull/1591) | [`805a14e`](https://github.com/ai-dynamo/aiconfigurator/commit/805a14e) | Confirmed the combined wheel already constrains `plotext<6`. |
| [#1557](https://github.com/ai-dynamo/aiconfigurator/pull/1557) | [`0e3aaa5`](https://github.com/ai-dynamo/aiconfigurator/commit/0e3aaa5) | Mapped GLM-5.3 BF16, FP8, and NVFP4 support. |
| [#1478](https://github.com/ai-dynamo/aiconfigurator/pull/1478) | [`231c1e4`](https://github.com/ai-dynamo/aiconfigurator/commit/231c1e4) | Mapped two-column intra-batch prefill imbalance pricing. |
| [#1441](https://github.com/ai-dynamo/aiconfigurator/pull/1441) | [`5261c8c`](https://github.com/ai-dynamo/aiconfigurator/commit/5261c8c) | Mapped the vLLM XPU Collector 0.26.0 upgrade. |
| [#1506](https://github.com/ai-dynamo/aiconfigurator/pull/1506) | [`dafcd61`](https://github.com/ai-dynamo/aiconfigurator/commit/dafcd61) | Mapped Kimi-K3 DSPARK recommendation defaults. |
| [#1590](https://github.com/ai-dynamo/aiconfigurator/pull/1590) | [`2ed278a`](https://github.com/ai-dynamo/aiconfigurator/commit/2ed278a) | Mapped missing-power normalization. |
| [#1598](https://github.com/ai-dynamo/aiconfigurator/pull/1598) | [`ea551a0`](https://github.com/ai-dynamo/aiconfigurator/commit/ea551a0) | Mapped FPM fake-fallback latency extrapolation. |
| [#1554](https://github.com/ai-dynamo/aiconfigurator/pull/1554) | [`17e8169`](https://github.com/ai-dynamo/aiconfigurator/commit/17e8169) | Mapped GB300 multi-node custom all-reduce collection. |
| [#1581](https://github.com/ai-dynamo/aiconfigurator/pull/1581) | [`d2e6290`](https://github.com/ai-dynamo/aiconfigurator/commit/d2e6290) | Mapped three queryable version slots and the old-version data prune. |
| [#1542](https://github.com/ai-dynamo/aiconfigurator/pull/1542) | [`4ca39f9`](https://github.com/ai-dynamo/aiconfigurator/commit/4ca39f9) | Mapped vLLM and TensorRT-LLM MoE all-to-all collectors. |
| [#1574](https://github.com/ai-dynamo/aiconfigurator/pull/1574) | [`e6a151c`](https://github.com/ai-dynamo/aiconfigurator/commit/e6a151c) | Mapped recovered Lightning and DeepSeek-V4 NVFP4 recipes. |
| [#1559](https://github.com/ai-dynamo/aiconfigurator/pull/1559) | [`93d6974`](https://github.com/ai-dynamo/aiconfigurator/commit/93d6974) | Mapped vLLM KDA serving metadata and GB300 0.27.0 data. |
| [#1576](https://github.com/ai-dynamo/aiconfigurator/pull/1576) | [`af2bc05`](https://github.com/ai-dynamo/aiconfigurator/commit/af2bc05) | Mapped AISimulate migration warnings into the compatibility package. |
| [#1482](https://github.com/ai-dynamo/aiconfigurator/pull/1482) | [`2b76936`](https://github.com/ai-dynamo/aiconfigurator/commit/2b76936) | Mapped WideEP MLA kernel-source resolution. |
| [#1558](https://github.com/ai-dynamo/aiconfigurator/pull/1558) | [`f7fa7b5`](https://github.com/ai-dynamo/aiconfigurator/commit/f7fa7b5) | Replaced borrowed GB300 measurements with B300 silicon data. |
| [#1562](https://github.com/ai-dynamo/aiconfigurator/pull/1562) | [`32e3c41`](https://github.com/ai-dynamo/aiconfigurator/commit/32e3c41) | Mapped MLA 0.27.0 and exact Kimi-K3 module rows. |
| [#1471](https://github.com/ai-dynamo/aiconfigurator/pull/1471) | [`5995bb0`](https://github.com/ai-dynamo/aiconfigurator/commit/5995bb0) | Mapped implicit framework communication-data reuse. |
| [#1533](https://github.com/ai-dynamo/aiconfigurator/pull/1533) | [`b31d899`](https://github.com/ai-dynamo/aiconfigurator/commit/b31d899) | Mapped Blackwell GEMM and GDN serving parity. |
| [#1594](https://github.com/ai-dynamo/aiconfigurator/pull/1594) | [`340bba0`](https://github.com/ai-dynamo/aiconfigurator/commit/340bba0) | Mapped explicit DSPARK acceptance. |
| [#1600](https://github.com/ai-dynamo/aiconfigurator/pull/1600) | [`b2a80e9`](https://github.com/ai-dynamo/aiconfigurator/commit/b2a80e9) | Removed a duplicate WideEP version override. |
| [#1519](https://github.com/ai-dynamo/aiconfigurator/pull/1519) | [`ce2824e`](https://github.com/ai-dynamo/aiconfigurator/commit/ce2824e) | Mapped attention lane selection and Qwen3.5-397B NVFP4 MoE cases. |

This tail changes 1,445 upstream mirror paths, including 554 binary blobs. The
manual-path decisions and the bounded final-tree adaptations are recorded in
[`aic-sync-095f58a-to-ce2824e-manual.md`](aic-sync-095f58a-to-ce2824e-manual.md).

### Intentional AISimulate adaptations

- Source paths map to the combined AIS layout: `aic-core/rust/aiconfigurator-core/src/` to `crates/core/src/perfmodel/`, `src/aisimulate_core/` to `python/aisimulate/src/aisimulate_core/`, and the upper application, Collector, tests, docs, and tools beneath `python/aisimulate/`. Migration-introduced crate, distribution, package-data, and tool references follow those destinations; pre-existing source-provenance comments may retain historical AIC paths.
- The source `.gitattributes` snapshot remains byte-identical under `python/aisimulate/`. Root `.gitattributes` carries equivalent mapped rules for the active AIS data and generated model-config paths and is owned by AISimulate Infra plus maintainers.
- The source engine-step golden is byte-identical. Source model configs, collection metadata, reuse declarations, Parquet files, `collector_ref` values, framework image digests, and other pinned SHAs are preserved unless a row is explicitly named here.
- `perf_data_reuse_manifest.yaml` is intentionally regenerated from the final AIS data tree because the source snapshot predates the data added by #1507 and #1486. Its generator defaults and rendered instructions use the unified `python/aisimulate/src/aisimulate_core` path.
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
replacement and migration path land. The standalone AIConfigurator repository
will publish its final 0.12.0 `aiconfigurator` and `aiconfigurator-core`
artifacts and then be archived; ongoing development, releases, issues, and pull
requests move to AISimulate.

The compatibility command remains in the AISimulate 0.12.0 wheel and is
targeted for removal in AISimulate 0.13.0. Removal is gated on verified unified
CLI replacements for every remaining workflow in the migration guide.

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
