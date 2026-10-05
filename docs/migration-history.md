# AISimulate migration history

This document records historical transfers, not current API guidance.
For current migration instructions, see [MIGRATION.md](MIGRATION.md).

This repository combines two independently preserved source streams: the full
AIConfigurator product and the former Dynamo `aisimulate/` package.

## Full AIConfigurator migration

AISimulate is the canonical source for the complete AIConfigurator product and
its engine-neutral estimator core starting with AISimulate 0.12.0. The
migration is tracked by
[AIC-1702](https://linear.app/nvidia/issue/AIC-1702/repo-migrate-aiconfigurator-core-to-the-aisimulate-repository).

For current package, command, and API mappings, see [MIGRATION.md](MIGRATION.md).

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
diff -u <(git -C ../aiconfigurator show c8aee02f0887547a334c3d6cd192c42757e4b40e:src/aiconfigurator/main.py) <(git show HEAD:python/aisimulate/src/aisimulate/legacy_cli/entrypoint.py)
```

### Selective speculative-decoding migration

[AIConfigurator PR #1563](https://github.com/ai-dynamo/aiconfigurator/pull/1563) is selectively adapted at source head `6290c161a354da5250c391bd43372b2e9c6f4a51` for pluggable speculation schemes and verify-on-FPM. This feature transfer does not advance the contiguous synchronization boundary above. The [migration ledger](migration-history.md#pr-1563-selective-transfer) records the source paths, adaptations, and modeling limits.

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
[the manual synchronization report](migration-history.md#synchronization-095f58a-to-ce2824e).

### Intentional AISimulate adaptations

- Source paths map to the combined AIS layout: `aic-core/rust/aiconfigurator-core/src/` to `crates/core/src/perfmodel/`, `aic-core/src/aiconfigurator_core/` to `python/aisimulate/src/aisimulate_core/`, and the upper application, Collector, tests, docs, and tools beneath `python/aisimulate/`. Migration-introduced crate, distribution, package-data, and tool references follow those destinations; pre-existing source-provenance comments may retain historical AIC paths.
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
[AIC migration guide](MIGRATION.md) until an explicit
replacement and migration path land. The standalone AIConfigurator repository
will publish its final 0.12.0 `aiconfigurator` and `aiconfigurator-core`
artifacts and then be archived; ongoing development, releases, issues, and pull
requests move to AISimulate.

The compatibility command remains in the AISimulate 0.13.0 wheel and is
targeted for removal in AISimulate 0.14.0. Removal is gated on verified unified
CLI replacements for every remaining workflow in the migration guide.

For current command and API replacements, see
[Migrate from AIConfigurator](MIGRATION.md) for explicit legacy
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

## Synchronization ff2be1 to 095f58a

- From: `ff2be1fd434fd516474e42b77f94cd5a5f841b9b`
- To: `095f58a51c4ca8e61b66ec108d86f223f8d559ce`

### `.github`

Reason: Only AISimulate repository-root workflows are active.

AISimulate disposition:

- The imported workflow/action snapshots remain inactive under
  `python/aisimulate/.github/`; applicable platform-wheel behavior is adapted
  to the unified `aisimulate` wheel.
- Active repository-root CI remains AISimulate-owned and validates the
  two-artifact release contract.
- The daily manifest rename is reflected by the canonical
  `perf_data_reuse_manifest.yaml`, its generator, and its path-trigger tests.

Changed upstream entries:

- `M	.github/actions/build-platform-wheel/action.yml`
- `M	.github/workflows/build-platform-wheels.yml`
- `M	.github/workflows/build-test.yml`
- `R076	.github/workflows/op-kernel-source-manifest-daily-run.yml	.github/workflows/perf-data-reuse-manifest-daily-run.yml`

## Synchronization 095f58a to ce2824e

- From: `095f58a51c4ca8e61b66ec108d86f223f8d559ce`
- To: `ce2824e8abd9bef71c3b162f651e63704a0eb4c1`
- AISimulate base: `0f0f4b33d61283d7c05a95289ac9da58c57e92c2`

### `pyproject.toml`

Reason: AISimulate has one wheel manifest and a combined dependency set.

Upstream changed `plotext` to `<6`. The combined AISimulate manifest already
carried that bound, so no further manifest edit was required.

### `uv.lock`

Reason: regenerate from the combined AISimulate manifest.

The upstream lockfile change was the matching `plotext` constraint. The
combined AISimulate lock already resolved the compatible dependency set and
remained authoritative.

### `aic-core/rust/aiconfigurator-core/Cargo.toml`

Reason: adapt dependency and feature changes into `crates/core/Cargo.toml`.

The upstream core added `log = "0.4"`. The dependency was added to the
combined AISimulate crate and the repository `Cargo.lock` was regenerated.

### Migration-only adaptations

- Extended rename and copy headers are rewritten into the configured mirror
  paths by `scripts/render_aic_sync_patch.py`; a regression test covers the
  renderer behavior.
- AISimulate-specific packaging, active workflows, and generated root
  `CODEOWNERS` remain owned by this repository rather than copied from AIC.
- The performance-data reuse manifest and report were regenerated from the
  final AISimulate data tree. That correctly keeps the pruned vLLM
  `chunk_gated_delta_rule` lane absent; the imported AIC test is adjusted to
  assert the final data state instead of reintroducing the stale donor lane.
- The vLLM 0.20.1 generator golden allowlist includes the migrated
  `--gpu-memory-utilization` flag, whose rendering is separately covered by
  sweeper request tests.

## Synchronization ce2824e to c8aee02

- From: `ce2824e8abd9bef71c3b162f651e63704a0eb4c1`
- To: `c8aee02f0887547a334c3d6cd192c42757e4b40e`
- Source PR: [ai-dynamo/aiconfigurator#1602](https://github.com/ai-dynamo/aiconfigurator/pull/1602)

This final transfer carries the worker-type-bound FPM regression redesign from
the frozen AIConfigurator repository. The feature-only source delta is
`7bdd746c..c8aee02f`; the preceding `ce2824e8..7bdd746c` commit changes only
standalone AIConfigurator package versions in the release paths accounted for
below.

### `pyproject.toml`, `aic-core/pyproject.toml`, and `uv.lock`

Reason: AISimulate has one wheel manifest and a combined dependency set.

The source range bumps the standalone AIConfigurator distributions from 0.11.0
to 0.12.0. AISimulate already owns a unified 0.12.0 release boundary, so no
Python manifest or lockfile change is required.

### `aic-core/rust/aiconfigurator-core/Cargo.toml`

Reason: AISimulate owns the combined `aisimulate-core` crate.

The source change is only the matching 0.12.0 package-version bump. The
combined crate is already versioned 0.12.0, so its manifest and workspace lock
remain unchanged.

### Other standalone release files

`README.md`, `aic-core/uv.lock`,
`aic-core/rust/aiconfigurator-core/Cargo.lock`, and
`aic-core/rust/aiconfigurator-core/README.md` also change only their standalone
AIConfigurator version references. AISimulate's root documentation and unified
locks already describe the 0.12.0 artifacts, so none of these source edits is
ported.

### Public API documentation and external-crate tests

`aic-core/API.md` and `aic-core/rust/tests/public-api/src/lib.rs` are outside
the automatic mirror table. Their feature-relevant changes are adapted into
`docs/core-api.md` and `crates/tests/public-api/src/lib.rs` respectively.

### AISimulate integration adaptations

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

## PR 1563 selective transfer

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->


This selectively ports the four commits of [AIConfigurator PR #1563](https://github.com/ai-dynamo/aiconfigurator/pull/1563), from `b28ab8fff6fdc85c7c0d62da4506682e40164308` through `6290c161a354da5250c391bd43372b2e9c6f4a51`, onto AISimulate base `7dbd110f6459f012d284fdb3080214b6ba204e6e`. The source PR was unmerged when this migration was prepared. This is an individual feature transfer; `scripts/aic_sync.toml` retains the contiguous main-branch synchronization boundary `c8aee02f0887547a334c3d6cd192c42757e4b40e`.

### Behavior

The migrated SDK builds n-gram, EAGLE-3, DFlash, DSpark, standalone-draft, and MTP speculation schemes. It accounts for draft compute and memory, physical target verification width, and caller-supplied accepted-token progress. Whole-forward FPM verification keeps draft operations explicit after the target FPM operation. The compatibility estimate CLI and Task speculative block expose the source workflow; unified predict/recommend integration is outside this transfer.

Usage and modeling limits are documented in the [speculation package](../python/aisimulate/src/aisimulate_core/sdk/speculation/README.md).

### AISimulate adaptations

- Map Rust and Python into the existing combined distribution. Keep one wheel and one crate, with identity-preserving compatibility imports.
- Advance the current operation schema from 17 to 18 rather than copying the source's intermediate schema numbers. Retain current positional APIs, attention-lane selection, worker-role FPM lookup, per-curve baseline handling, and structured performance results.
- Keep speculation configuration keyword-only on ModelConfig to preserve existing positional construction.
- Keep the new Task field keyword-only as well. Materialized models snapshot speculation inputs so later caller mutation cannot reuse a cache entry for a different draft graph; resolving legacy MTP does not turn it into a persistent explicit scheme.
- Map draft query tokens and batch size before operation lookup through the typed native TokenScale wrapper. This preserves nonlinear collective and compute costs for integer and fractional draft widths. Standalone drafts retain their checkpoint layer count independently of target-only layer overrides.
- Include draft prefill and generation work in op-level mixed/aggregate estimates, with phase-specific execution and complete per-operation metadata. Draft prefill follows the existing full-prefill amortization convention; draft generation uses its own width for every decode round.
- Resolve standalone MoE draft parallelism through the existing configuration rules and repeat its native generation graph for each draft step. This supports composite operations without mutating the independent draft's weights or prefill graph; the compiled draft graph grows linearly with the draft count.
- Validate malformed draft geometry, token counts, target-layer IDs, empty injection taps, TP divisibility, checkpoint layer bounds, and unsupported operation combinations before estimates can silently use a different shape. Explicit disabled schemes reject legacy MTP conflicts, and MTP blocks preserve checkpoint-derived `nextn="auto"` when depth is omitted.
- Carry dense draft sliding windows through native attention in both phases. Reject unrepresentable `DraftOpSpec.query_overrides` before mutating either phase; this currently excludes DeepSeek-V4 DSpark. N-gram requires `trigger_rate=1.0` until mixed-round costs are modeled.
- Preserve separate FPM draft rows, energy, sources, and executed fallback metadata within the existing prefill/decode component totals. Native scalar setters reject operation families that cannot carry the field.
- Expose estimate flags only where the API consumes them; preserve legacy MTP equivalence and reject unconsumed scheme configurations in unsupported task modes.
- Preserve existing engine goldens. Added synthetic verification cases and source-parity checks establish software behavior; no new GPU accuracy campaign is claimed, and the rejected fitted calibration is not introduced.

### Source paths

All source material below is NVIDIA-authored and licensed under Apache-2.0. Source copyright and license headers are retained. Imported files are adapted for AISimulate; source links name the exact source revision, and the root/packaged THIRD_PARTY_NOTICES records the transfer.

| Original source at the pinned revision | AISimulate path |
| --- | --- |
| [`aic-core/rust/aiconfigurator-core/parity_tests/test_engine_step_parity.py`](https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/aic-core/rust/aiconfigurator-core/parity_tests/test_engine_step_parity.py) | `crates/core/parity_tests/perfmodel/test_engine_step_parity.py` |
| [`aic-core/rust/aiconfigurator-core/src/config.rs`](https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/aic-core/rust/aiconfigurator-core/src/config.rs) | `crates/core/src/perfmodel/config.rs` |
| [`aic-core/rust/aiconfigurator-core/src/engine/runtime.rs`](https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/aic-core/rust/aiconfigurator-core/src/engine/runtime.rs) | `crates/core/src/perfmodel/engine/runtime.rs` |
| [`aic-core/rust/aiconfigurator-core/src/engine/spec.rs`](https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/aic-core/rust/aiconfigurator-core/src/engine/spec.rs) | `crates/core/src/perfmodel/engine/spec.rs` |
| [`aic-core/rust/aiconfigurator-core/src/fpm/tests.rs`](https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/aic-core/rust/aiconfigurator-core/src/fpm/tests.rs) | `crates/core/src/perfmodel/fpm/tests.rs` |
| [`aic-core/rust/aiconfigurator-core/src/operators/attention.rs`](https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/aic-core/rust/aiconfigurator-core/src/operators/attention.rs) | `crates/core/src/perfmodel/operators/attention.rs` |
| [`aic-core/rust/aiconfigurator-core/src/operators/fpm_forward.rs`](https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/aic-core/rust/aiconfigurator-core/src/operators/fpm_forward.rs) | `crates/core/src/perfmodel/operators/fpm_forward.rs` |
| [`aic-core/rust/aiconfigurator-core/src/operators/fpm_sol.rs`](https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/aic-core/rust/aiconfigurator-core/src/operators/fpm_sol.rs) | `crates/core/src/perfmodel/operators/fpm_sol.rs` |
| [`aic-core/rust/aiconfigurator-core/src/operators/op.rs`](https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/aic-core/rust/aiconfigurator-core/src/operators/op.rs) | `crates/core/src/perfmodel/operators/op.rs` |
| [`aic-core/rust/aiconfigurator-core/src/perf_database/fpm_forward.rs`](https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/aic-core/rust/aiconfigurator-core/src/perf_database/fpm_forward.rs) | `crates/core/src/perfmodel/perf_database/fpm_forward.rs` |
| [`aic-core/rust/aiconfigurator-core/src/py.rs`](https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/aic-core/rust/aiconfigurator-core/src/py.rs) | `crates/core/src/perfmodel/py.rs` |
| [`aic-core/rust/aiconfigurator-core/src/py_ops.rs`](https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/aic-core/rust/aiconfigurator-core/src/py_ops.rs) | `crates/core/src/perfmodel/py_ops.rs` |
| [`aic-core/rust/tests/public-api/src/lib.rs`](https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/aic-core/rust/tests/public-api/src/lib.rs) | `crates/tests/public-api/src/lib.rs` |
| [`aic-core/src/aiconfigurator_core/sdk/backends/base_backend.py`](https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/aic-core/src/aiconfigurator_core/sdk/backends/base_backend.py) | `python/aisimulate/src/aisimulate_core/sdk/backends/base_backend.py` |
| [`aic-core/src/aiconfigurator_core/sdk/config.py`](https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/aic-core/src/aiconfigurator_core/sdk/config.py) | `python/aisimulate/src/aisimulate_core/sdk/config.py` |
| [`aic-core/src/aiconfigurator_core/sdk/config_builders.py`](https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/aic-core/src/aiconfigurator_core/sdk/config_builders.py) | `python/aisimulate/src/aisimulate_core/sdk/config_builders.py` |
| [`aic-core/src/aiconfigurator_core/sdk/engine.py`](https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/aic-core/src/aiconfigurator_core/sdk/engine.py) | `python/aisimulate/src/aisimulate_core/sdk/engine.py` |
| [`aic-core/src/aiconfigurator_core/sdk/models/__init__.py`](https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/aic-core/src/aiconfigurator_core/sdk/models/__init__.py) | `python/aisimulate/src/aisimulate_core/sdk/models/__init__.py` |
| [`aic-core/src/aiconfigurator_core/sdk/models/base.py`](https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/aic-core/src/aiconfigurator_core/sdk/models/base.py) | `python/aisimulate/src/aisimulate_core/sdk/models/base.py` |
| [`aic-core/src/aiconfigurator_core/sdk/operations/fpm_forward.py`](https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/aic-core/src/aiconfigurator_core/sdk/operations/fpm_forward.py) | `python/aisimulate/src/aisimulate_core/sdk/operations/fpm_forward.py` |
| [`aic-core/src/aiconfigurator_core/sdk/rust_engine_step.py`](https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/aic-core/src/aiconfigurator_core/sdk/rust_engine_step.py) | `python/aisimulate/src/aisimulate_core/sdk/rust_engine_step.py` |
| [`aic-core/src/aiconfigurator_core/sdk/speculation/__init__.py`](https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/aic-core/src/aiconfigurator_core/sdk/speculation/__init__.py) | `python/aisimulate/src/aisimulate_core/sdk/speculation/__init__.py` |
| [`aic-core/src/aiconfigurator_core/sdk/speculation/base.py`](https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/aic-core/src/aiconfigurator_core/sdk/speculation/base.py) | `python/aisimulate/src/aisimulate_core/sdk/speculation/base.py` |
| [`aic-core/src/aiconfigurator_core/sdk/speculation/dense_draft.py`](https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/aic-core/src/aiconfigurator_core/sdk/speculation/dense_draft.py) | `python/aisimulate/src/aisimulate_core/sdk/speculation/dense_draft.py` |
| [`aic-core/src/aiconfigurator_core/sdk/speculation/dflash.py`](https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/aic-core/src/aiconfigurator_core/sdk/speculation/dflash.py) | `python/aisimulate/src/aisimulate_core/sdk/speculation/dflash.py` |
| [`aic-core/src/aiconfigurator_core/sdk/speculation/draft_model.py`](https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/aic-core/src/aiconfigurator_core/sdk/speculation/draft_model.py) | `python/aisimulate/src/aisimulate_core/sdk/speculation/draft_model.py` |
| [`aic-core/src/aiconfigurator_core/sdk/speculation/dspark.py`](https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/aic-core/src/aiconfigurator_core/sdk/speculation/dspark.py) | `python/aisimulate/src/aisimulate_core/sdk/speculation/dspark.py` |
| [`aic-core/src/aiconfigurator_core/sdk/speculation/eagle.py`](https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/aic-core/src/aiconfigurator_core/sdk/speculation/eagle.py) | `python/aisimulate/src/aisimulate_core/sdk/speculation/eagle.py` |
| [`aic-core/src/aiconfigurator_core/sdk/speculation/materialize.py`](https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/aic-core/src/aiconfigurator_core/sdk/speculation/materialize.py) | `python/aisimulate/src/aisimulate_core/sdk/speculation/materialize.py` |
| [`aic-core/src/aiconfigurator_core/sdk/speculation/mtp.py`](https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/aic-core/src/aiconfigurator_core/sdk/speculation/mtp.py) | `python/aisimulate/src/aisimulate_core/sdk/speculation/mtp.py` |
| [`aic-core/src/aiconfigurator_core/sdk/speculation/ngram.py`](https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/aic-core/src/aiconfigurator_core/sdk/speculation/ngram.py) | `python/aisimulate/src/aisimulate_core/sdk/speculation/ngram.py` |
| [`src/aiconfigurator/cli/api.py`](https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/src/aiconfigurator/cli/api.py) | `python/aisimulate/src/aisimulate/legacy_cli/api.py` |
| [`src/aiconfigurator/cli/main.py`](https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/src/aiconfigurator/cli/main.py) | `python/aisimulate/src/aisimulate/legacy_cli/main.py` |
| [`src/aiconfigurator/sdk/speculation/__init__.py`](https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/src/aiconfigurator/sdk/speculation/__init__.py) | `python/aisimulate/src/aisimulate/sdk/speculation/__init__.py` |
| [`src/aiconfigurator/sdk/speculation/base.py`](https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/src/aiconfigurator/sdk/speculation/base.py) | `python/aisimulate/src/aisimulate/sdk/speculation/base.py` |
| [`src/aiconfigurator/sdk/speculation/dense_draft.py`](https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/src/aiconfigurator/sdk/speculation/dense_draft.py) | `python/aisimulate/src/aisimulate/sdk/speculation/dense_draft.py` |
| [`src/aiconfigurator/sdk/speculation/dflash.py`](https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/src/aiconfigurator/sdk/speculation/dflash.py) | `python/aisimulate/src/aisimulate/sdk/speculation/dflash.py` |
| [`src/aiconfigurator/sdk/speculation/draft_model.py`](https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/src/aiconfigurator/sdk/speculation/draft_model.py) | `python/aisimulate/src/aisimulate/sdk/speculation/draft_model.py` |
| [`src/aiconfigurator/sdk/speculation/dspark.py`](https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/src/aiconfigurator/sdk/speculation/dspark.py) | `python/aisimulate/src/aisimulate/sdk/speculation/dspark.py` |
| [`src/aiconfigurator/sdk/speculation/eagle.py`](https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/src/aiconfigurator/sdk/speculation/eagle.py) | `python/aisimulate/src/aisimulate/sdk/speculation/eagle.py` |
| [`src/aiconfigurator/sdk/speculation/materialize.py`](https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/src/aiconfigurator/sdk/speculation/materialize.py) | `python/aisimulate/src/aisimulate/sdk/speculation/materialize.py` |
| [`src/aiconfigurator/sdk/speculation/mtp.py`](https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/src/aiconfigurator/sdk/speculation/mtp.py) | `python/aisimulate/src/aisimulate/sdk/speculation/mtp.py` |
| [`src/aiconfigurator/sdk/speculation/ngram.py`](https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/src/aiconfigurator/sdk/speculation/ngram.py) | `python/aisimulate/src/aisimulate/sdk/speculation/ngram.py` |
| [`src/aiconfigurator/sdk/speculative.py`](https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/src/aiconfigurator/sdk/speculative.py) | `python/aisimulate/src/aisimulate/sdk/speculative.py` |
| [`src/aiconfigurator/sdk/task_v2.py`](https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/src/aiconfigurator/sdk/task_v2.py) | `python/aisimulate/src/aisimulate/sdk/task_v2.py` |
| [`tests/cross_package/test_import_contract.py`](https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/tests/cross_package/test_import_contract.py) | `python/aisimulate/tests/cross_package/test_import_contract.py` |
| [`tests/unit/cli/test_estimate_speculative.py`](https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/tests/unit/cli/test_estimate_speculative.py) | `python/aisimulate/tests/unit/cli/test_estimate_speculative.py` |
| [`tests/unit/sdk/speculation/__init__.py`](https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/tests/unit/sdk/speculation/__init__.py) | `python/aisimulate/tests/unit/sdk/speculation/__init__.py` |
| [`tests/unit/sdk/speculation/test_base.py`](https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/tests/unit/sdk/speculation/test_base.py) | `python/aisimulate/tests/unit/sdk/speculation/test_base.py` |
| [`tests/unit/sdk/speculation/test_consumer_equivalence.py`](https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/tests/unit/sdk/speculation/test_consumer_equivalence.py) | `python/aisimulate/tests/unit/sdk/speculation/test_consumer_equivalence.py` |
| [`tests/unit/sdk/speculation/test_dense_draft_schemes.py`](https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/tests/unit/sdk/speculation/test_dense_draft_schemes.py) | `python/aisimulate/tests/unit/sdk/speculation/test_dense_draft_schemes.py` |
| [`tests/unit/sdk/speculation/test_draft_model_scheme.py`](https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/tests/unit/sdk/speculation/test_draft_model_scheme.py) | `python/aisimulate/tests/unit/sdk/speculation/test_draft_model_scheme.py` |
| [`tests/unit/sdk/speculation/test_dspark_scheme.py`](https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/tests/unit/sdk/speculation/test_dspark_scheme.py) | `python/aisimulate/tests/unit/sdk/speculation/test_dspark_scheme.py` |
| [`tests/unit/sdk/speculation/test_mtp_scheme.py`](https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/tests/unit/sdk/speculation/test_mtp_scheme.py) | `python/aisimulate/tests/unit/sdk/speculation/test_mtp_scheme.py` |
| [`tests/unit/sdk/task_v2/test_speculative_block.py`](https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/tests/unit/sdk/task_v2/test_speculative_block.py) | `python/aisimulate/tests/unit/sdk/task_v2/test_speculative_block.py` |
| [`tests/unit/sdk/test_fpm_forward.py`](https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/tests/unit/sdk/test_fpm_forward.py) | `python/aisimulate/tests/unit/sdk/test_fpm_forward.py` |
