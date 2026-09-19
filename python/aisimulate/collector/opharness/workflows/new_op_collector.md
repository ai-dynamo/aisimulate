# Workflow: write a new op collector

Entry: a decomposition residue (onboard_model step 6) or an owner-declared new
op family that no collector measures yet. Exit: a collector module whose
execution path is proven aligned with serving, producing sane rows.

| # | Step | Who | Instrument / artifact |
|---|------|-----|------------------------|
| 1 | Serving truth first | script | the G1 identity records for models exercising this op: which module classes run, which kernel families, under which parallel topologies (the multi-rank injector covers tp/ep/a2a routes that change kernels) |
| 2 | Define the measured unit | owner | op granularity + table schema; recorded before code |
| 3 | Implement | AI | build through the framework's own builder/selector wherever possible; every hand-constructed metadata field or pinned backend carries a serving citation (file:line at the pinned version) |
| 4 | Path alignment gate | script | `components/path_diff.py`: the new collector's kernel set vs the serving records from step 1 — MUST be green before any perf row is trusted; a collector that happily invokes the wrong backend is worse than one that crashes |
| 5 | Parallel coverage | script | if the op's kernels fork by topology (measured, not assumed — see the tp/ep invariance findings), the case plan carries the forking dimension |
| 6 | Smoke -> full sanity | script | deterministic boundary cases first, then full plan; row-level sanity |
| 7 | Facts | AI | anything learned that constrains future work (framework walls, dispatch quirks) -> `results/findings.yaml` with evidence + pinned versions |

The one law: the collector may execute a case or raise — it never silently
skips, and it never substitutes a backend to keep collecting.
