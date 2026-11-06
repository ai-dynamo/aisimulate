# Workflow: upgrade an op (framework version or collector change)

Entry: one op family, one framework, a new pin. Exit: the family's tables are
recollected where its kernels changed (reuse declared for the rest), and the
collector's cells are shown to run what serving runs on the new pin.

Collection itself follows `docs/perf-model/collector/upgrade.md` and the
collector overview's "Every pin bump" row; this table adds the serving side.
`workflow_check.py upgrade_op` checks the steps marked Check.

Steps 4-8 run one op at a time across the frameworks (`--op`): an op's gates on every
backend are aligned or triaged before the next op starts.

| # | Step | Who | How |
|---|------|-----|-----|
| 1 | Bump the pin | owner | `collector/framework_manifest.yaml`: the default, or a family pin for the ops in scope. Check: `pin_is` |
| 2 | Plan diff | script | `components/case_inventory.py --fw --version --sm --base <rev>` on any CPU. A case or op leaving the plan without a `collector/skips.py` entry is a regression before any GPU hour. Check: `case_set_no_regression` |
| 3 | Source check | AI | diff the framework between the two pins for the family's selector, kernels and metadata on every SM in scope; cite file and symbol for anything the collector replicates |
| 4 | Serving identity | script | per system in scope: `probe_driver.py --system <system> --campaign <name> --gates --op <op> --emit-shards --url-base <share>` (the serving records the op's gates read, every backend, at each gate's release) -> `gates.py plan --serving <campaign dir>` (step 6) -> the collection pipeline, one job per (backend, release) -> `--import-raws <job dir> --records --matrix` per job; the campaign's matrix diffed against the previous pin's campaign for the same models is the identity report |
| 5 | Smoke | script | `collect.py --smoke` / `--limit` / `--case-filter` for the family's ops through the collection pipeline, on one system per SM generation |
| 6 | Path alignment | script | per SM: `components/gates.py plan --sm <sm> --op <op> --serving <campaign dir> --out <dir>` (the same jobs as step 4) -> `gates.py grade --sm <sm> --op <op> --results <job dirs>`: each gate's planned case, traced by `probe.trace_case` in its release's image, vs the serving record of the same SM and release. Check: `gates_aligned` (`--param op=<op>`) |
| 7 | Triage | AI | new fails vs the previous run of the same (backend, version). A model the backend cannot serve or a block that cannot run -> a `collector/skips.py` entry with its reason; a collector bug -> a fix; a path the table measures differently -> a deviation note at that code |
| 8 | Re-check skips and notes | AI | rule 7: `components/case_inventory.py --fw --skips [--ops]` prints the record skeleton; every in-scope entry is kept by a source check or a run, or deleted -> `results/<sm>/rechecks/<fw>-<ver>.yaml`. Check: `skips_rechecked` |
| 9 | Collect | script | recollect the families whose kernels changed; declare reuse (`reuse.yaml` with `approved_by`) for the rest |

## Across SMs

An upgrade validated on one SM is validated only for the fields and code
paths that SM reads (the 0.29 MLA block-table width passed every SM90 smoke and
crashed on B300).

1. Prefer values asked from the framework (`get_supported_kernel_block_sizes`,
   spec fields) over literals; a platform-dependent literal is a review finding.
2. When the framework branches by SM, mirror its branch with a citation.
3. Every other SM with shipped data gets a smoke and its own path verdicts, or
   an `unverified_sms` mark in the registry; never leave a sibling SM silently
   green.
4. A new SM adds taxonomy rules scoped to it with `sms:`.
