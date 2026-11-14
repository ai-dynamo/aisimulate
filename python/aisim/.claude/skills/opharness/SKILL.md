---
name: opharness
description: Use when running or extending the op-path harness under collector/opharness (upgrade_op / onboard_model / new_op_collector, serving probes, path_diff gates). Lists every environment knob the components read and where each per-SM artifact lives; procedure lives in collector/opharness/README.md and workflows/*.md.
---

# opharness — how a session drives it

Runbook pointers only; policy is in `.claude/rules/` (collector rules apply to
anything under `collector/`). Read `collector/opharness/README.md` first, then
the workflow you are on (`collector/opharness/workflows/<workflow>.md`).

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
bundles and shard plans, raws and `records.jsonl`. Records carry
the SM their raw ran on and are labeled with that SM's rules (`sms:` in
`components/kernel_taxonomy.yaml`). Gate verdicts are `results/pathdiff/<sm>/<fw>-<ver>/`. A gate in
`components/gates.yaml` may carry `no_serving: {<sm>: <why>}`; customizations in
`targets.yaml` may carry `sms: [sm120]` to apply on those SMs only.

## Commands

```
python3 components/probe_driver.py --campaign <c> --system b200_sxm --gates --op <op> --emit-shards --url-base <share>
python3 components/gates.py plan --sm sm100 --op <op> --serving <campaign dir> --out <dir>  # one plan per (backend, release)
python3 components/probe_driver.py --campaign <c> --import-raws <job result dir> --records  # per job
python3 components/gates.py grade --sm sm100 --op <op> --results <job result dirs>      # verdicts per gate; exit 0 = aligned
python3 components/run_diff.py tables <packaged tables> --base <shipped data> [--detail] # new tables vs the previous release
python3 components/run_diff.py failures <job result dirs>                                # failed tasks per shard
collect.py --backend <fw> --ops probe.trace_case --case-filter op=<op>,<field>=<value>,...  # one planned case, in the image
```

The collector's vocabulary (Pin, Plan, Record, Note, Rule) is in
`.claude/rules/collector/overview.md`; a campaign's conclusions go where it
says (skips.py entries, the op plan's dispatch, deviation notes, fixes), never
into a findings file; cite the run (pipeline id, system, version) at the code
instead of committing its evidence.

Collection and the plan diff are not opharness components: use `collect.py`
(`--limit`, `--case-filter`, `--plan-only`, `--plan-table`) through the collection
pipeline (the `aic-auto-collect` skill). `tools/perf_database/parquet_diff.py`
summarizes a data PR for review; `run_diff.py` compares a release with the one before.

Serving probes and traces run inside the framework image of the collection pipeline;
a tp>1 serving probe needs a job with that many GPUs (`AISIM_GPU_NUM`).
