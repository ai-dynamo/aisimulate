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
| `AIS_PROBE_WORKSPACE` | every component | GPU workspace root (dummy_models/, archive/, configs/, facts/). Default: cwd. May live outside the checkout. Legacy alias `AIC_PROBE_WORKSPACE`. |
| `AIS_SM` | probe_driver.current_sm, workflow_check | OVERRIDE of the SM; default is `targets.yaml platform.sm`. Export only to grade another SM's imported evidence; never leave it set. |
| `AIS_GENERATOR_CLI` | probe_driver.render_golden | path of the `aiconfigurator` CLI for golden renders (default `<ws>/venv_ais/bin/aiconfigurator`, built with `uv sync`). |
| `AIS_GENERATOR_SRC` | probe_driver | generator source tree stamped into renders (default this checkout's `src`). |
| `AIS_PROBE_ISL` | probes/probe_vllm.py `--isl` default | prompt length of a probe (default 4096; shape-conditional paths such as DSA sparse thresholds are invisible below theirs). |
| `AIS_KDA_DECODE_PATHS` | collector/vllm/collect_kda.py (capture-only) | `fused,packed` default; the kda decode gate passes `env.AIS_KDA_DECODE_PATHS=fused` to probe.trace_case because the serving probe is non-spec. |
| `AIS_DEBUG_IDXMETA` | collector/vllm/collect_mla_module.py (DSA debugging) | developer switch, off by default. |
| `CUDA_MPS_PIPE_DIRECTORY=/nonexistent-no-mps` | every probe container | mandatory on boxes with a host MPS daemon (queues set it). |
| `HF_HUB_OFFLINE=1`, `TRITON_CACHE_DIR`, `DG_JIT_CACHE_DIR` | containers | offline + shared JIT cache (`<ws>/jitcache`), set by the emitted queues. |

## Per-SM artifacts

Rules scoped by `sms:` in `components/kernel_taxonomy.yaml`, `results/<sm>/<fw>-<ver>.yaml`,
`results/pathdiff/<sm>/<fw>-<ver>/`. A gate in `components/gates.yaml` may carry
`no_serving: {<sm>: <why>}` (the probe box cannot serve the model there) or `serving_raw`
(graded against a dedicated probe raw). Customizations in `targets.yaml` may carry `sms: [sm120]` to apply
on those SMs only. Run ids include the platform name.

## Commands

```
python3 components/workflow_check.py <workflow> --param fw=vllm --param version=0.30.0 [--param sm=sm120]
python3 components/probe_driver.py --emit-shards <dir> --url-base <share> [--only ...]   # serving probes as probe.serving shards
python3 components/probe_driver.py --import-raws <job result dir> | --records | --matrix
python3 components/probe_driver.py --plan | --emit-queues                               # dry plan / per-GPU docker route on a probe box
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

Serving probes run inside the framework image with the checkout mounted and
`PYTHONPATH=<checkout>/python/aisimulate`; probes must not share a GPU.
