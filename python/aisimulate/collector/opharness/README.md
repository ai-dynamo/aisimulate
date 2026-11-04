# opharness: what serving runs, and whether the collector measures it

The collector (`.claude/rules/collector/overview.md`) already holds the pin,
the plan, the records, the notes and the rules. The harness adds the two
things the collector cannot see by itself: which kernels serving deploys for a
generator-rendered config, and whether a collector cell runs the same path.
It exists to make an op upgrade short and repeatable; anything a V4 tool
already answers (`collect.py --plan-only`, `--smoke`, `--case-filter`,
`tools/perf_database/parquet_diff.py`) is used directly, not wrapped.

## What is here

| Path | What it does |
|---|---|
| `targets.yaml` | the probe roster's overlay: per-model generate args (each with its fact), dummy overrides, owner exclusions, topologies, the box's platform. Versions and images come from `collector/framework_manifest.yaml` (`probe_driver.load_targets`) |
| `components/probe_driver.py` | golden `cli generate` render -> probe queues -> `archive/records.jsonl` -> per-model matrix `results/<sm>/<fw>-<ver>.yaml` |
| `components/probes/` | in-container probes: in-process for tp1, the real server (`probe_*_server.py`) for tp>1 |
| `components/dummies.py`, `fetch_inputs.py` | depth-cut, width-true dummy checkpoints from fetched configs |
| `components/kernel_taxonomy.yaml` | kernel name -> backend label and role, one file for every SM (`sms:` scopes a rule) |
| `components/gates.yaml` + `gates.py` | the path gates: each names one planned collector case; `gates.py plan` writes the `probe.trace_case` shards the collection pipeline runs, `gates.py grade` grades their kernels against the serving record |
| `components/path_diff.py` | a collector case's kernels vs the serving record, through the same taxonomy |
| `components/case_inventory.py` | the CPU plan diff against a base revision; `--skips` lists the entries a pin bump re-checks |
| `components/workflow_check.py` | the four checks the workflows use |
| `workflows/` | `upgrade_op`, `onboard_model`, `new_op_collector`: a step table (.md) and its checked steps (.yaml) |

## Driving a workflow

```
python3 components/workflow_check.py upgrade_op --param fw=vllm --param version=0.33.0
```

prints each checked step as done or todo with the reason; do the first todo and
re-run. The checks are `pin_is` (the manifest), `case_set_no_regression` (the
plan diff), `gates_aligned` (path_diff verdicts, scoped by `repo=` or
`family=`) and `skips_rechecked` (rule 7). Steps that need judgment (triage, a
deviation note) are prose in the .md and land where the collector reads them:
a `collector/skips.py` entry, the op plan's `dispatch`, a note at the code, a
fix. There is no findings file. Commit evidence only when a test or script
reads it; otherwise cite the run (pipeline id, system, version) at the code.

## Running

- **Workspace.** `AIS_PROBE_WORKSPACE` holds data only: `configs/`,
  `dummy_models/`, `archive/` (plan, raw probes with `.fp` fingerprint
  sidecars, records), `facts/`, `jitcache/`. A raw counts as evidence only when
  its fingerprint (engine invocation, dummy config, probe code, image, kv)
  matches the plan; the matrix marks the rest `stale evidence`.
- **SM.** `targets.yaml` `platform.sm` names the box; `AIS_SM` overrides it to
  grade another box's evidence. SM103 uses the SM100 rules: every framework
  selects kernels by capability family (`major == 10`, `in (100, 103)`), so
  branch on the family, never on `sm == 100`.
- **Gates** run on the collection pipeline: `gates.py plan --fw <fw> --sm <sm> --out <dir>`
  writes a shard plan of `probe.trace_case` shards (collect.py runs each planned case once
  under the profiler, its measurement child processes included), and `gates.py grade` reads
  the unpacked job result. A gate exists on an SM where its case is planned; `no_serving`
  marks an SM whose probe box cannot serve the model, which gets a `platform-floor` verdict.
- **Serving probes** (`probe_driver.py`, `probes/`) still run on a box with the
  framework images.

## Rules

1. A component answers one question the same way twice; judgment and
   trade-offs stay with the agent and the owner.
2. A serving claim needs a record from a probe or a capture; a source citation
   (framework file and symbol) explains it, as the collector rules require.
3. One home per fact: versions and images in the manifest, cases in the plan,
   skips in `collector/skips.py`, kernel labels in `kernel_taxonomy.yaml`.
4. Results stay in the workspace; golden commands stamp the generator commit,
   and result files carry their framework version.
