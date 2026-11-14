# Workflow: write a new op collector

Entry: kernels a probe record leaves unclassified (onboard_model) or an
owner-declared new op family that no collector measures yet. Exit: a collector module whose
execution path is proven aligned with serving, producing sane rows.

| # | Step | Who | Instrument / artifact |
|---|------|-----|------------------------|
| 1 | Serving truth first | script | a `probe_driver.py` campaign per system in scope for models exercising this op: which module classes run, which kernel families, under which parallel topologies (the server route covers tp>1) |
| 2 | Define the measured unit | owner | op granularity + table schema; recorded before code |
| 3 | Implement | AI | build through the framework's own builder/selector wherever possible; every hand-constructed metadata field or pinned backend carries a serving citation (file:line at the pinned version) |
| 4 | Path alignment gate | script | a `components/gates.yaml` row per phase of the op (`op`, a `select` naming one planned case, `repo`, `kv`; the gate name starts with the family), then `gates.py plan` -> the collection pipeline -> `gates.py grade` on each SM in scope: the new collector's kernels vs the serving records from step 1 — MUST be green before any perf row is trusted; a collector that happily invokes the wrong backend is worse than one that crashes. An op no gate can prove gets its reason under `ungated` in `gates.yaml` instead (a unit test fails on a registry op with neither) |
| 5 | Parallel coverage | script | if the op's kernels fork by topology (measured, not assumed — see the tp/ep invariance evidence in benchmarks/evidence/collector/op-probe-2026-10/), the case plan carries the forking dimension |
| 6 | Collect + run diff | script | a first slice (`collect.py --limit` / `--case-filter`), then the full plan through the collection pipeline; `run_diff.py failures <result dirs>` after the first window and at the end: a key bucket that fails as a whole is fixed or left out before the table is trusted |
| 7 | Facts | AI | anything learned that constrains future work goes where the collector reads it: framework walls -> `collector/skips.py` entries, dispatch -> the op plan's `dispatch`, a path measured differently from serving -> a deviation note at the code, citing the run |

The one law: the collector may execute a case or raise — it never silently
skips, and it never substitutes a backend to keep collecting.

`components/gates.py plan --sm <sm> --family <family>` schedules the family's gates and `gates.py grade --sm <sm> --op <op> --results <job dirs>` exits 0 when they are aligned.
