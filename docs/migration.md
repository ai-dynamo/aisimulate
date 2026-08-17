# Dynamo-to-AISimulate package migration

This repository is the canonical source for the `aisimulate` package starting
with the standalone repository migration tracked by
[AIC-1703](https://linear.app/nvidia/issue/AIC-1703/repo-migrate-the-aisimulate-package-to-the-aisimulate-repository).

The imported snapshot includes the generalized Mocker engine and deterministic
Replayer from [Dynamo PR #12525](https://github.com/ai-dynamo/dynamo/pull/12525).
The package remains Dynamo-independent. Dynamo-owned Router, Planner, runtime,
transport, and live-Mocker integrations consume the released package through
optional adapter contracts.

## Preserved Git history

The migration branch merges a path-filtered copy of the Dynamo history for the
former `aisimulate/` directory. Original commits, authorship, and blame context
are retained. For example:

```bash
git log --follow -- crates/core/src/engine/generalized/engine.rs
git log --follow -- src/aisimulate/sweeper/search.py
```

Merge the migration pull request with a merge commit. Squashing it would keep
the final files but discard the imported Dynamo ancestry from this repository's
`main` history.
