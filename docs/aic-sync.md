# Synchronizing AIConfigurator

AIConfigurator remains active during its deprecation window. AISimulate
therefore retains explicit mappings from the pinned upstream layout into the
two canonical Python packages:

| Upstream AIConfigurator path | Stable AISimulate mirror |
| --- | --- |
| `src/aiconfigurator/sdk/` | `python/aisimulate/src/aisimulate/sdk/` |
| `src/aiconfigurator/generator/` | `python/aisimulate/src/aisimulate/generator/` |
| `src/aiconfigurator/cli/` | `python/aisimulate/src/aisimulate/legacy_cli/` |
| `aic-core/src/aiconfigurator_core/` | `python/aisimulate/src/aisimulate_core/` |
| `tests/` | `python/aisimulate/tests/` |
| `collector/`, `tools/`, `docs/` | matching folders under `python/aisimulate/` |
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

The machine-readable mapping and synchronization ledger live in
`scripts/aic_sync.toml`. Manual-path reviews for each synchronization range are
recorded in
[`aic-sync-ff2be1-to-095f58a-manual.md`](aic-sync-ff2be1-to-095f58a-manual.md)
and
[`aic-sync-095f58a-to-ce2824e-manual.md`](aic-sync-095f58a-to-ce2824e-manual.md).
The final feature transfer from the frozen AIConfigurator repository is
recorded in
[`aic-sync-ce2824e-to-c8aee02-manual.md`](aic-sync-ce2824e-to-c8aee02-manual.md).
To generate a binary-safe patch from the recorded AIC boundary to a newer AIC
commit:

```bash
python scripts/render_aic_sync_patch.py \
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
