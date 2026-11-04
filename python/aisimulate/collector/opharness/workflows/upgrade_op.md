# Workflow: upgrade an op (framework version or collector change)

Entry: one op family, one framework, a new pin. Exit: the family's tables are
recollected where its kernels changed (reuse declared for the rest), and the
collector's cells are shown to run what serving runs on the new pin.

Collection itself follows `docs/perf-model/collector/upgrade.md` and the
collector overview's "Every pin bump" row; this table adds the serving side.
`upgrade_op.yaml` lists the steps `workflow_check.py` checks.

| # | Step | Who | How |
|---|------|-----|-----|
| 1 | Bump the pin | owner | `collector/framework_manifest.yaml`: the default, or a family pin for the ops in scope. Check: `pin_is` |
| 2 | Plan diff | script | `components/case_inventory.py --fw --version --sm --base <rev>` on any CPU. A case or op leaving the plan without a `collector/skips.py` entry is a regression before any GPU hour. Check: `case_set_no_regression` |
| 3 | Source check | AI | diff the framework between the two pins for the family's selector, kernels and metadata on every SM in scope; cite file and symbol for anything the collector replicates |
| 4 | Serving identity | script | `probe_driver.py --plan --only <representative models>` -> `--emit-queues` -> `--records` -> `--matrix`: a new `results/<sm>/<fw>-<ver>.yaml`; its diff against the old one is the identity report |
| 5 | Smoke | script | `collect.py --smoke` / `--limit` / `--case-filter` for the family's ops through the collection pipeline, on one system per SM generation |
| 6 | Path alignment | script | `components/gates.py plan` -> the collection pipeline -> `gates.py grade`: each gate's planned case traced by `probe.trace_case` vs the serving records. Check: `gates_aligned` |
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
