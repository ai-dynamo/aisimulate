---
name: opharness
description: Use when running or extending the op-path harness under collector/opharness (upgrade_op / onboard_model / new_op_collector, serving probes, path_diff gates). Lists every environment knob the components read and where each per-SM artifact lives; procedure lives in collector/opharness/README.md and workflows/*.md.
---

# opharness — how a session drives it

Runbook pointers only; policy is in `.claude/rules/` (collector rules apply to
anything under `collector/`). Read `collector/opharness/README.md` first, then
the workflow you are on (`collector/opharness/workflows/<workflow>.md` + `.yaml`).

## Environment knobs (the complete list — nothing else is read from the env)

| variable | read by | meaning / default |
|---|---|---|
| `AIS_PROBE_WORKSPACE` | every component | workspace on the control machine: `configs/`, `dummy_models/`, `campaigns/<name>/`, `results/`. Default `collector/opharness/workspace/` (git-ignored). |
| `AIS_GENERATOR_CLI` | probe_driver.render_golden | the `aiconfigurator` CLI for golden renders (default `<ws>/venv_ais/bin/aiconfigurator`, built with `uv sync`). |
| `AIS_GENERATOR_SRC` | probe_driver | generator source tree the renders run (default this checkout's `src`). |
| `AIS_KDA_DECODE_PATHS` | collector/vllm/collect_kda.py (capture-only) | `fused,packed` default; the kda decode gate passes `env.AIS_KDA_DECODE_PATHS=fused` to probe.trace_case because the serving probe is non-spec. |
| `AIS_DEBUG_IDXMETA` | collector/vllm/collect_mla_module.py (DSA debugging) | developer switch, off by default. |

## Where things live

A serving-probe campaign is one system: `campaigns/<name>/` holds its plan, golden renders,
bundles and shard plans, raws, `records.jsonl` and `matrix-<fw>-<version>.yaml`. Records carry
the SM their raw ran on and are labeled with that SM's rules (`sms:` in
`components/kernel_taxonomy.yaml`). Gate verdicts are `results/pathdiff/<sm>/<fw>-<ver>/`,
case plans `results/<sm>/cases/`, rule-7 rechecks `results/<sm>/rechecks/`. A gate in
`components/gates.yaml` may carry `no_serving: {<sm>: <why>}`; customizations in
`targets.yaml` may carry `sms: [sm120]` to apply on those SMs only.

## Commands

```
python3 components/workflow_check.py <workflow> --param fw=vllm --param version=0.30.0 --param sm=sm100
python3 components/probe_driver.py --system b200_sxm --backends vllm --only <repos> --emit-shards --url-base <share>
python3 components/probe_driver.py --campaign b200_sxm --import-raws <job result dir> --records --matrix
python3 components/gates.py plan --fw <fw> --sm <sm> --out <dir>                       # probe.trace_case shard plan
python3 components/gates.py grade --fw <fw> --sm <sm> --results <job result dir>       # verdicts per gate
python3 components/case_inventory.py --fw <fw> --version <ver> --sm <sm> --base <rev>    # CPU plan diff (no image)
python3 components/case_inventory.py --fw <fw> --skips [--ops a,b]                      # rule-7 recheck skeleton
collect.py --backend <fw> --ops probe.trace_case --case-filter op=<op>,<field>=<value>,...  # one planned case, in the image
```

The collector's vocabulary (Pin, Plan, Record, Note, Rule) is in
`.claude/rules/collector/overview.md`; a campaign's conclusions go where it
says (skips.py entries, the op plan's dispatch, deviation notes, fixes), never
into a findings file; cite the run (pipeline id, system, version) at the code
instead of committing its evidence.

Smoke, collection and before/after diffs are not opharness components: use
`collect.py --smoke` / `--limit` / `--case-filter` through the collection pipeline
(the `aic-auto-collect` skill) and `tools/perf_database/parquet_diff.py`.

Serving probes and traces run inside the framework image of the collection pipeline;
a tp>1 serving probe needs a job with that many GPUs (`AISIM_GPU_NUM`).
