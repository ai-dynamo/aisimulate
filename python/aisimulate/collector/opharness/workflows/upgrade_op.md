# Workflow: upgrade an op (framework version / collector change)

Entry: one op family, one framework, a new pinned version. Exit: the family's
perf data is re-collected where its kernels changed (reuse declared for the
rest), and its identity facts are re-verified on the new pin.

The collector's own vocabulary (`.claude/rules/collector/overview.md`) names
what this workflow moves: the **Pin** (`collector/framework_manifest.yaml`),
the **Plan** (`collector/op_plans.py` + `collector/skips.py`), the **Record**
(`collection_meta.yaml` or `reuse.yaml` next to the parquet), **Notes**
(deviation comments at the collector code) and the **Rules**. The harness adds
what the collector cannot see on its own: what serving deploys (identity
probes) and whether the collector's path matches it (path_diff gates).

Workflows are thin orchestration for a human + AI pair; every step that must
give the same answer twice is a component invocation, never free-hand.

| # | Step | Who | Instrument / artifact |
|---|------|-----|------------------------|
| 1 | Bump the pin | owner | The collector pin is `collector/framework_manifest.yaml` (the default, or a family pin for the ops in scope); `targets.yaml` backends.<fw> names the version the probes deploy (one per backend; result files are named per version, so old matrices stay in the local results/ and the previous campaign's evidence bundle) |
| 2 | Rebuild env | script | `components/build_images.sh` |
| 3 | Scope the re-probe | script | `probe_driver.py --plan --only <representative models of this family>` — the reverse mapping (family -> representative models) comes from the previous matrix |
| 4 | Re-run identity | script | `probe_driver.py --emit-queues` + queues -> `--records` -> `--matrix` (a NEW `results/<sm>/<fw>-<ver>.yaml`; diff vs the old file IS the upgrade report) |
| 4b | Lane guards vs evidence | script | `components/lane_evidence.py --fw --version --sm`: every collector version/SM lane guard cell must agree with the fresh identity records. The runner a lane declares is the op plan's dispatch on that SM; `components/lane_evidence.yaml` only says which checkpoints are the evidence and which serving identity confirms a runner. A guard is edited ONLY with this report in hand, never from source reading |
| 4c | Plan diff | script | `components/case_inventory.py --fw --version --sm --base <rev>` on any CPU: the collector's case plan (`collect.py --plan-only --plan-table`) diffed against the base revision -> `results/<sm>/cases/<fw>-<ver>.yaml`. An op leaving the plan, or cases and categorical values leaving it without a `collector/skips.py` entry accounting for them, is a regression BEFORE any GPU hour. There is no waiver file: a deliberate loss is a skips.py entry with its reason |
| 5 | Path alignment | script | `components/path_diff.py` for the family's collector op vs the fresh serving records; a capture names its cell as a planned case (`op_smoke.py --select` / `planned_row`) |
| 5b | Executor smoke | script | `components/executor_smoke.py --run --limit 1` in the framework image: every planned op once through collect.py's real path (`--model-cases-full` plan, checkpoint, `--resume`, finalize) -> `results/<sm>/executor_smoke/<fw>-<ver>.yaml`. `--shards <pipeline plan>` (CPU) checks a shard plan against the case plan before any GPU hour |
| 6 | Expired customizations | AI | any `cli_extra_args` whose fact cites the old version: re-run the bare default with `--only`; drop args that became unnecessary |
| 7 | Failure triage | AI | new fails vs old matrix; root-cause per the failure taxonomy. A conclusion goes where the collector reads it: a model the backend cannot serve, or a block that cannot run -> a `collector/skips.py` entry with its reason; a collector bug -> a fix; a path the table measures differently from serving -> a deviation note at that code; the evidence behind any of them -> `benchmarks/evidence/collector/<campaign>/` (README + raw). There is no findings file |
| 8 | Re-check skips and notes | AI | rule 7: every `collector/skips.py` entry in scope (`components/case_inventory.py --fw --skips [--ops]` prints the record skeleton) is re-checked by a source check or a targeted run and kept or deleted -> `results/<sm>/rechecks/<fw>-<ver>.yaml`; deviation notes in scope are re-checked the same way and reported in the PR |

Collection itself follows the collector's upgrade playbook
(`docs/perf-model/collector/upgrade.md`): recollect only the families whose
kernels changed, declare reuse (`reuse.yaml` with `approved_by`) for the rest,
and compare the failure set with the previous run of the same (backend,
version).

## Evidence flow (added 2026-10-06, from the b200_sxm sglang 0.5.21 shard run)

Seven of the eight failed shards of GitLab job 469988017 were knowledge the
harness already held and had not moved: the identity records contradicted a
lane guard edited on another box from source reading (int4_wo, 3,078 rows
refused); a conclusion root-caused Kimi-K3's 96-head trtllm-gen decode gap and
the collector still swept num_heads=96 (362 "unexpected"); op_smoke proved
every collector could time a case while collect.py's plan/resume/finalize
layers — which the pipeline actually runs — were never exercised (3 ops not
in the case plan, 3 skip_indexer shards + 1 all-failed shard dead in
finalization). The steps above make each movement a checked artifact:

- identity records -> lane guards: `lane_evidence.py` (the plan's dispatch vs serving, graded cells);
- case YAML + skips.py -> the collected set: `case_inventory.py` (a CPU plan diff against the base revision);
- a framework gap -> the plan: a `collector/skips.py` entry, which every later plan reads (`skips_rechecked`);
- collector -> pipeline: `executor_smoke.py` runs the same command line the pipeline runs, and
  `--shards` validates the pipeline's plan against the collector's plan on a CPU.

Probe geometry is part of the evidence: a tp1-only identity probe cannot see
per-rank head counts (`topology_policy.tp_required` with `sms:` scopes a
checkpoint to a tp on one SM — Kimi-K3 tp2 on sm100).

## SM compatibility (read this on EVERY upgrade)

An upgrade validated on one SM is validated ONLY for the fields and code
paths that SM consumes. Origin: the 0.29 mla lane passed every SM90 smoke
while handing the indexer builder a block table narrower than its declared
`block_table_width` — only the family-100 decode path reads the
width-sized expanded buffer, so the contract violation stayed latent on
H20 and crashed on B300 (2026-09-20).

1. Adapted code prefers values ASKED from the framework
   (`get_supported_kernel_block_sizes`, `get_block_table_width`, spec
   fields) over literals; a platform-dependent literal (32-vs-64 class) is
   a review finding even when the current SM passes.
2. When the framework itself branches by SM, mirror ITS branch
   (`is_device_capability_family(...)` + serving citation) — never invent
   your own condition.
3. On completing an upgrade, every OTHER SM with shipped data either gets a
   re-smoke + fresh per-SM path_diff verdict (minutes per op) or an
   `unverified_sms` mark in the registry — the gate then shows blocked, not
   done. Never leave a sibling SM silently green.
4. Verdict-plane artifacts are per-SM by construction
   (`kernel_taxonomy_<sm>.yaml`, `results/pathdiff/<sm>/`,
   `results/retests/<sm>/`); a new SM starts with its own vocabulary file,
   never by editing another SM's.
5. Arch-neutral fixes discovered on one SM propagate to the other SMs via
   an explicit cross-SM work order (state the fix, the serving citation,
   and the requested re-smoke — the B300->H20 block-table exchange is the
   template).

AI never: re-derives comparison baselines, renames labels outside
`kernel_taxonomy_<sm>.yaml`, or concludes "unsupported" from reading code — every
"the framework does X" claim needs a component-produced record.

Progress is derived, not self-reported: `components/workflow_check.py upgrade_op --param ...` evaluates the sibling `upgrade_op.yaml` manifest against artifacts and names the first actionable step.

Steps 8-9 (recollect sanity, e2e spot check) were removed 2026-09-20 (owner decision): they had been permanent stubs. Row-level sanity is collect.py's executor + classified-failure machinery; an e2e-alignment component gets built the day a workflow actually needs it.

## SM of a run

The SM is `targets.yaml` `platform.sm` (see README "Per-SM runs"); `AIS_SM` only
overrides it. `workflow_check upgrade_op --param sm=<sm>` selects which SM's
results/ are judged; gates a framework cannot serve on an SM are declared
`FLOOR_SM=<sm>` in `captures/verdicts_*.sh` instead of being dropped.
