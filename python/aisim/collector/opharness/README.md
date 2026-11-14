# opharness: what serving runs, and whether the collector measures it

The collector (`.claude/rules/collector/overview.md`) already holds the pin,
the plan, the records, the notes and the rules. The harness adds what the
collector cannot see by itself, with three checks and nothing else:

| Check | Question | Cost | Where |
|---|---|---|---|
| **Source check** | What did the framework change between the two pins for this op, on every SM? | no GPU | the agent, reading the two releases; the workflow says what to read |
| **Path gate** | Does a planned collector case run the kernels serving runs for that model? | one small GPU job per (backend, release, SM) | `components/gates.yaml` + `gates.py`, on a serving record from `probe_driver.py` |
| **Run diff** | Did the collected tables change against the previous release in a way nobody explained? | no GPU | `components/run_diff.py` |

Each catches what the others cannot. The source check finds API breaks and
paths the collector builds differently from serving, before any GPU hour. A
gate proves kernel identity for one case; it does not see an operand width, a
weight value, a slow node or a case it has no row for. The run diff sees every
table and every failure, but only after the run. Anything a collector tool
already answers (`collect.py --plan-only`, `--plan-table`, `--case-filter`) is
used directly, not wrapped.

## What is here

| Path | What it does |
|---|---|
| `targets.yaml` | the probe roster's overlay: per-model generate args (each with its fact), dummy overrides, owner exclusions, topologies. Versions and images come from `collector/framework_manifest.yaml` (`probe_driver.load_targets`) |
| `components/probe_driver.py` | a campaign for one system: golden `cli generate` render -> `probe.serving` shards -> raws -> `records.jsonl` |
| `components/probes/` | in-container probes: in-process for tp1, the real server (`probe_*_server.py`) for tp>1 |
| `components/dummies.py`, `fetch_inputs.py` | depth-cut, width-true dummy checkpoints from fetched configs |
| `components/kernel_taxonomy.yaml` | kernel name -> backend label and role, one file for every SM (`sms:` scopes a rule) |
| `components/gates.yaml` + `gates.py` | the path gates: each names one planned collector case; `gates.py plan` writes the `probe.trace_case` shards the collection pipeline runs, `gates.py grade` grades their kernels against the serving record. `ungated` gives the reason an op has no gate; `gates.py paths` reads from the op plans which paths the gates name |
| `components/path_diff.py` | a collector case's kernels vs the serving record, through the same taxonomy |
| `components/run_diff.py` | a run's tables against the previous release (`tables`) and its failed tasks per shard (`failures`) |
| `workflows/` | `upgrade_op`, `onboard_model`, `new_op_collector`: the step tables |

## Driving a workflow

A workflow is its step table; one command says what the gates cover and two say
whether a step is done:

```
python3 components/gates.py paths --sm sm100 --op <op>                               # no GPU: each planned path and its gate
python3 components/gates.py grade --sm sm100 --op <op> --results <job result dirs>   # exit 0: every gate in scope aligned
python3 components/run_diff.py tables <packaged tables> --base <shipped data>        # leads to explain before integrating
```

Steps that need judgment (the source check, triage, a deviation note) are
prose in the .md and land where the collector reads them: a
`collector/skips.py` entry, the op plan's `dispatch`, a note at the code, a
fix. There is no findings file and no checklist file. Commit evidence only
when a test or script reads it; otherwise cite the run at the code.

## Running

Everything runs from one control machine; GPU work goes through the collection
pipeline (the `aisim-campaign` skill). Nothing generated is committed.

- **Workspace.** `AIS_PROBE_WORKSPACE` (default `collector/opharness/workspace/`,
  git-ignored) holds `configs/`, `dummy_models/`, `campaigns/<name>/` and `results/`.
- **Releases.** A gate runs at the release its op is collected with
  (`collector/framework_manifest.yaml`: the framework default or a family pin, e.g. vLLM
  gemm/moe/kda at 0.31.0 next to a 0.30.0 default). `probe.trace_case` refuses any other
  installed release, and its serving record is probed at that release; every plan is one
  job per (backend, release).
- **Serving probes.** A campaign is one system: `probe_driver.py --system <system>
  --gates [--op <op>] --emit-shards --url-base <share dir>` renders a golden config for
  each serving record the gates in scope read (repo, kv, prompt length, release; or
  `--backends <fw> --only <repos> [--isl N] [--version V]` for a roster cut), uploads its
  inputs (dummy model dir and rendered config) and writes `probe.serving` shard plans under
  `campaigns/<name>/shards/tp<N>/<backend>-<version>.yaml` (`collector/probes/serving.py`
  runs `probes/probe_<fw>.py` in the job). Then `--import-raws <job result dir>` (only
  shards that ran this emit) and `--records` (kernels labeled with the rules of the SM each
  raw ran on; one line per run says `ok` or why it gave no serving evidence).
- **Gates.** `gates.py plan --sm <sm> [--op <op> | --fw <fw>] --serving <campaign dir>
  --out <dir>` writes one plan per (backend, release): the campaign's serving shards and a
  `probe.trace_case` shard per gate (collect.py runs each planned case once under the
  profiler, its measurement child processes included, keeping the kernels of
  `benchmark_with_power`'s timed region). `gates.py grade --sm <sm> [--op <op>] --results
  <job result dirs>` grades each gate against the serving record of its repo, kv, prompt
  length and release on the same SM, on the kernels of the roles and phase its op decides;
  every other kernel of the timed runs alone (the trace's `measured_kernels`, warmups left
  out) must be one serving launches under its role, or the gate diverges with `aux_drift`.
  A kernel that differs from serving's only in its tactic (tile, stages) is the same path:
  it shows under `tactic_drift` and does not diverge the gate. A gate compares kernels, not
  time: time is compared table against table by the run diff.
  It writes the verdicts to `results/pathdiff/<sm>/<backend>-<version>/`. A gate exists on
  an SM where its case is planned; `no_serving` marks an SM where the model cannot be
  served, which gets a `platform-floor` verdict.
- **Paths.** `gates.py paths --sm <sm> [--fw <fw>] [--op <op>]` lists what the plans
  collect on that SM as paths: the cases of one op that share a phase, the op plan's dtype
  fields and its `dispatch`, which is what selects the kernels. Each path shows the gates
  naming a case on it, the `ungated` reason, the gate that can be derived for it, or `NO
  GATE`. A source-check finding names a path; if no gate is on it, the gates say nothing
  about that change. `--derive` (on `plan`, `grade` and `probe_driver.py --gates`) adds a
  gate for each unnamed path whose cases name a roster checkpoint: the case nearest the
  probe point (one request of the probe's prompt length, one GPU), graded against that
  checkpoint at the case's KV dtype. A path whose cases name no model (gemm, attention)
  needs a row in `gates.yaml`.
- **Run diff.** `run_diff.py tables <new root> --base <shipped data> [--system S]
  [--backend B] [--family F] [--detail]` reads the packaged layout
  (`<system>/<family>/<backend>/<version>/<table>.parquet`) on both sides and compares
  each table at its manifest pin with the same table at the greatest older version: rows,
  cases shared, lost and gained, key buckets lost, and the median new/old latency over the
  launch-bound half (`small`) and the compute-bound fifth (`large`) of the shared cases.
  `large` up with `small` unchanged is a slow node or a slower kernel; both moved is a
  change of the timed region; a lost bucket is a skip, a plan change or a key the release
  no longer serves. `run_diff.py failures <result dirs>` gives each shard's failed tasks,
  their share and their most frequent messages. Every lead is explained (in the commit or
  the PR) or fixed before the table is integrated.
- **SM.** A campaign's SM is its system's (`src/aisim_core/systems/<system>.yaml`);
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
