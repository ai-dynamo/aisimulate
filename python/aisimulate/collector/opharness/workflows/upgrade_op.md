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
| 4b | Lane guards vs evidence | script | `components/lane_evidence.py --fw --version --sm`: every collector version/SM lane guard cell must agree with the fresh identity records (`components/lane_evidence.yaml` says which checkpoints are the evidence). A guard is edited ONLY with this report in hand, never from source reading |
| 4c | Case-set regression | script | `AIS_SM=<sm> components/case_inventory.py --fw --version --sm` inside the new image (CPU): every registry op's case set -> `results/<sm>/cases/<fw>-<ver>.yaml`; diffed against the previous version's inventory. An op leaving the plan, a getter dropping a model, a lane or dtype value disappearing is a regression BEFORE any GPU hour; deliberate losses are signed in `<fw>-<ver>.waivers.yaml` |
| 5 | Path alignment | script | `components/path_diff.py` for the family's collector op vs the fresh serving records |
| 5b | Executor smoke | script | `components/executor_smoke.py --run --limit 1` in the framework image: every planned op once through collect.py's real path (`--model-cases-full` plan, checkpoint, `--resume`, finalize) -> `results/<sm>/executor_smoke/<fw>-<ver>.yaml`. `--shards <pipeline plan>` (CPU) checks a shard plan against the case plan before any GPU hour |
| 6 | Expired customizations | AI | any `cli_extra_args` whose fact cites the old version: re-run the bare default with `--only`; drop args that became unnecessary, record in findings |
| 7 | Failure triage | AI | new fails vs old matrix; root-cause per the failure taxonomy; terminal facts -> `results/findings.yaml` (pinned to the new version) |
| 8 | Propagate findings | AI | every finding pinned to this (fw, version, sm) carries `collector_impact:` — which op, what changed (`guard` / `cases` / `collector_fix` / `pipeline` / `none`) and the ref. A model-level framework gap (Kimi-K3 96 heads) becomes a collector grid guard in the SAME campaign, not two days later in a 362-failure shard |

## Evidence flow (added 2026-10-06, from the b200_sxm sglang 0.5.21 shard run)

Seven of the eight failed shards of GitLab job 469988017 were knowledge the
harness already held and had not moved: the identity records contradicted a
lane guard edited on another box from source reading (int4_wo, 3,078 rows
refused); a finding root-caused Kimi-K3's 96-head trtllm-gen decode gap and
the collector still swept num_heads=96 (362 "unexpected"); op_smoke proved
every collector could time a case while collect.py's plan/resume/finalize
layers — which the pipeline actually runs — were never exercised (3 ops not
in the case plan, 3 skip_indexer shards + 1 all-failed shard dead in
finalization). The three steps above (4b, 5b, 8) make each movement a
checked artifact:

- identity records -> lane guards: `lane_evidence.py` (declarative rules, graded cells);
- case files + getters -> the collected set: `case_inventory.py` (derived, no GPU; version-to-version diff);
- findings -> collector grid: `collector_impact` on every pinned finding (`findings_propagated`);
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
