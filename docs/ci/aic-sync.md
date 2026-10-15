<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Synchronizing AIConfigurator

AISimulate records an immutable AIConfigurator synchronization boundary and
explicit mappings from the pinned upstream layout into the
two canonical Python packages:

| Upstream AIConfigurator path | Stable AISimulate mirror |
| --- | --- |
| `src/aiconfigurator/sdk/` | `python/aisimulate/src/aisimulate/sdk/` |
| `src/aiconfigurator/generator/` | `python/aisimulate/src/aisimulate/generator/` |
| `src/aiconfigurator/cli/` | `python/aisimulate/src/aisimulate/legacy_cli/` |
| `aic-core/src/aiconfigurator_core/` | `python/aisimulate/src/aisimulate_core/` |
| `tests/` | `python/aisimulate/tests/` |
| `collector/`, `tools/` | matching folders under `python/aisimulate/` |
| `docs/` | manually adapted into the component documentation under `docs/` |
| `aic-core/rust/aiconfigurator-core/src/` | `crates/core/src/perfmodel/` |
| `aic-core/rust/aiconfigurator-core/tests/` | `crates/core/tests/perfmodel/` |
| `aic-core/rust/aiconfigurator-core/parity_tests/` | `crates/core/parity_tests/perfmodel/` |

The patch renderer maps directory paths, not Python imports. Review application
changes in the manual report: root `main.py` and `deprecation.py` move to
`aisimulate/legacy_cli/entrypoint.py` and `aisimulate/legacy_cli/deprecation.py`,
and removed resource aliases must not be restored. Update imported code to use
`aisimulate` and `aisimulate_core` while preserving immutable upstream citations.
Rust composition, Replay, and PyO3 integration remain outside the estimator
source mirror.

## Source provenance

The machine-readable path mapping and current upstream boundary are in
[scripts/aic_sync/aic_sync.toml](../../scripts/aic_sync/aic_sync.toml).
The full application import originated at upstream commit
`13b5cf2697876692b0a52098266c81162add11fc`; the recorded final synchronization
boundary is `c8aee02f0887547a334c3d6cd192c42757e4b40e`. Earlier path-filtered
history and authorship remain in Git. The Replay/engine source also retains
the history imported from Dynamo's former `aisimulate/` tree.

Preserve immutable source revisions, license notices, collector/runtime
identities, performance-data hashes and deliberate adaptations. The canonical
[third-party notices](../../THIRD_PARTY_NOTICES.md) describe imported material;
the [synchronization ledger](../../scripts/aic_sync/aic_sync.toml) records
mirrored subtrees and paths requiring manual adaptation. Use those current
records when applying another upstream change; completed PR transfer logs are
not a second synchronization procedure.

Documentation is no longer a mirrored subtree: upstream prose must be adapted
to the appropriate component rather than recreating `python/aisimulate/docs/`.

## Apply an upstream change

To generate a binary-safe patch from the recorded AIC boundary to a newer AIC
commit:

```bash
python scripts/aic_sync/render_aic_sync_patch.py \
  --source ../aiconfigurator \
  --to-ref <new-aic-sha> \
  --manual-report /tmp/aic-manual-changes.md \
  --output /tmp/aic-sync.patch
git apply --check /tmp/aic-sync.patch
git apply /tmp/aic-sync.patch
```

Then review and test the result, update `upstream.last_synced` to the exact AIC
commit, and commit the source changes and ledger update together. The renderer
uses Git binary patches, so performance data is synchronized along with code.

If any configured `[[manual]]` path changed, the renderer fails unless
`--manual-report` is provided. Review and adapt every entry in that report
before advancing `upstream.last_synced`; generating the mirror patch alone is
not evidence that manifests, workflows, or ownership policy were synchronized.

Packaging and repository policy are intentionally manual. Adapt upstream
changes to `pyproject.toml`, Cargo manifests and lockfiles, release workflows,
or CODEOWNERS into the combined AISimulate equivalents; never copy an upstream
publishable manifest or active workflow into the repository. The ledger lists
these manual paths and the reason for each exception.
