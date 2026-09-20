# Workflow: upgrade an op (framework version / collector change)

Entry: one op family, one framework, a new pinned version (targets.yaml pin
bump). Exit: the family's perf data is re-collected where needed and its
identity facts are re-verified on the new pin.

Workflows are thin orchestration for a human + AI pair; every step that must
give the same answer twice is a component invocation, never free-hand.

| # | Step | Who | Instrument / artifact |
|---|------|-----|------------------------|
| 1 | Bump the pin | owner | `targets.yaml` backends.<fw> (single version per backend; result files are named per version, so old matrices stay) |
| 2 | Rebuild env | script | `components/build_images.sh` |
| 3 | Scope the re-probe | script | `probe_driver.py --plan --only <representative models of this family>` — the reverse mapping (family -> representative models) comes from the previous matrix |
| 4 | Re-run identity | script | `probe_driver.py --emit-queues` + queues -> `--records` -> `--matrix` (a NEW `results/<sm>/<fw>-<ver>.yaml`; diff vs the old file IS the upgrade report) |
| 5 | Path alignment | script | `components/path_diff.py` for the family's collector op vs the fresh serving records |
| 6 | Expired customizations | AI | any `cli_extra_args` whose fact cites the old version: re-run the bare default with `--only`; drop args that became unnecessary, record in findings |
| 7 | Failure triage | AI | new fails vs old matrix; root-cause per the failure taxonomy; terminal facts -> `results/findings.yaml` (pinned to the new version) |

AI never: re-derives comparison baselines, renames labels outside
`kernel_taxonomy.yaml`, or concludes "unsupported" from reading code — every
"the framework does X" claim needs a component-produced record.

Progress is derived, not self-reported: `components/workflow_check.py upgrade_op --param ...` evaluates the sibling `upgrade_op.yaml` manifest against artifacts and names the first actionable step.

Steps 8-9 (recollect sanity, e2e spot check) were removed 2026-09-20 (owner decision): they had been permanent stubs. Row-level sanity is collect.py's executor + classified-failure machinery; an e2e-alignment component gets built the day a workflow actually needs it.
