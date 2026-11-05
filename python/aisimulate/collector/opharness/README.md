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
| `targets.yaml` | the probe roster's overlay: per-model generate args (each with its fact), dummy overrides, owner exclusions, topologies. Versions and images come from `collector/framework_manifest.yaml` (`probe_driver.load_targets`) |
| `components/probe_driver.py` | a campaign for one system: golden `cli generate` render -> `probe.serving` shards -> raws -> `records.jsonl` -> per-model matrix |
| `components/probes/` | in-container probes: in-process for tp1, the real server (`probe_*_server.py`) for tp>1 |
| `components/dummies.py`, `fetch_inputs.py` | depth-cut, width-true dummy checkpoints from fetched configs |
| `components/kernel_taxonomy.yaml` | kernel name -> backend label and role, one file for every SM (`sms:` scopes a rule) |
| `components/gates.yaml` + `gates.py` | the path gates: each names one planned collector case; `gates.py plan` writes the `probe.trace_case` shards the collection pipeline runs, `gates.py grade` grades their kernels against the serving record |
| `components/path_diff.py` | a collector case's kernels vs the serving record, through the same taxonomy |
| `components/case_inventory.py` | the CPU plan diff against a base revision; `--skips` lists the entries a pin bump re-checks |
| `components/workflow_check.py` | the four checks the workflows use |
| `workflows/` | `upgrade_op`, `onboard_model`, `new_op_collector`: the step tables; `workflow_check.py` holds the checked steps |

## Driving a workflow

```
python3 components/workflow_check.py upgrade_op --param fw=vllm --param version=0.33.0 --param sm=sm100
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

Everything runs from one control machine; GPU work goes through the collection
pipeline (the `aisim-campaign` skill). Nothing generated is committed.

- **Workspace.** `AIS_PROBE_WORKSPACE` (default `collector/opharness/workspace/`,
  git-ignored) holds `configs/`, `dummy_models/`, `campaigns/<name>/` and `results/`.
- **Serving probes.** A campaign is one system: `probe_driver.py --system <system>
  --backends <fw> --only <repos> --emit-shards --url-base <share dir>` renders each run's
  golden config for that system, uploads its inputs (dummy model dir and rendered config)
  and writes `probe.serving` shard plans per tp under `campaigns/<name>/shards/`
  (`collector/probes/serving.py` runs `probes/probe_<fw>.py` in the job). Then
  `--import-raws <job result dir>` (only shards that ran this emit), `--records` (kernels
  labeled with the rules of the SM each raw ran on) and `--matrix`. Two campaigns'
  matrices for the same models (old pin, new pin) diff into the identity report.
- **Gates.** `gates.py plan --fw <fw> --sm <sm> --out <dir>` writes the `probe.trace_case`
  shards for one SM (collect.py runs each planned case once under the profiler, its
  measurement child processes included); `gates.py grade --fw <fw> --sm <sm> --results
  <job result dir>` grades them against the serving records of the same SM and writes the
  verdicts to `results/pathdiff/<sm>/<fw>-<version>/`. A gate exists on an SM where its case
  is planned; `no_serving` marks an SM where the model cannot be served, which gets a
  `platform-floor` verdict.
- **SM.** A campaign's SM is its system's (`src/aisimulate_core/systems/<system>.yaml`);
  a record's is the one its raw reports. Records keep every kernel the taxonomy labels, so
  after a taxonomy edit re-run `--records` before grading. SM103 uses the SM100 rules where every framework
  selects kernels by capability family (`major == 10`): branch on the family, never on
  `sm == 100`.

## Rules

1. A component answers one question the same way twice; judgment and
   trade-offs stay with the agent and the owner.
2. A serving claim needs a record from a probe or a capture; a source citation
   (framework file and symbol) explains it, as the collector rules require.
3. One home per fact: versions and images in the manifest, cases in the plan,
   skips in `collector/skips.py`, kernel labels in `kernel_taxonomy.yaml`.
4. Results stay in the workspace on the control machine; a plan records the
   generator commit, and result files carry their system, SM and framework version.
