---
name: opharness
description: Use when running or extending the op-path harness under collector/opharness (upgrade_op / onboard_model / new_op_collector campaigns, probes, path_diff gates, per-SM results). Lists every environment knob the components read and where each per-SM artifact lives; procedure lives in collector/opharness/README.md and workflows/*.md.
---

# opharness — how a session drives it

Runbook pointers only; policy is in `.claude/rules/` (collector rules apply to
anything under `collector/`). Read `collector/opharness/README.md` first, then
the workflow you are on (`collector/opharness/workflows/<workflow>.md` + `.yaml`).

## Environment knobs (the complete list — nothing else is read from the env)

| variable | read by | meaning / default |
|---|---|---|
| `AIS_PROBE_WORKSPACE` | every component, verdict scripts, evidence_bundle | GPU workspace root (dummy_models/, archive/, configs/, facts/). Default: cwd. May live outside the checkout. Legacy alias `AIC_PROBE_WORKSPACE`. |
| `AIS_SM` | probe_driver.current_sm, workflow_check, captures/verdicts_*.sh | OVERRIDE of the SM; default is `targets.yaml platform.sm`. Export only to grade another SM's imported evidence; never leave it set. |
| `AIS_GENERATOR_CLI` | probe_driver.render_golden | path of the `aiconfigurator` CLI for golden renders (default `<ws>/venv_ais/bin/aiconfigurator`, built by `components/build_images.sh`). |
| `AIS_GENERATOR_SRC` | probe_driver | generator source tree stamped into renders (default this checkout's `src`). |
| `AIS_PROBE_ISL` | probes/probe_vllm.py `--isl` default | prompt length of a probe (default 4096; shape-conditional paths such as DSA sparse thresholds are invisible below theirs). |
| `AIS_PROBE_OUT` | probes | raw JSON output path inside the container (set by the emitted queues). |
| `AIS_KDA_DECODE_PATHS` | collector/vllm/collect_kda.py (capture-only) | `fused,packed` default; the kda decode gate capture sets `fused` (captures/kda_gen.py does it) because the serving probe is non-spec. |
| `AIS_KV_BLOCK_SIZE` | collector/vllm/collect_attn.py, collect_mla_module.py (A/B only) | override of the per-SM KV page size table (`collector/vllm/utils.kv_block_size`); never set in collection runs. |
| `AIS_DSA_LEGACY_MODULE`, `AIS_DEBUG_IDXMETA` | probes / dsa collector debugging | developer switches, off by default. |
| `CUDA_MPS_PIPE_DIRECTORY=/nonexistent-no-mps` | every probe/capture container | mandatory on boxes with a host MPS daemon (queues set it). |
| `HF_HUB_OFFLINE=1`, `TRITON_CACHE_DIR`, `DG_JIT_CACHE_DIR` | containers | offline + shared JIT cache (`<ws>/jitcache`), set by the emitted queues. |

## Per-SM artifacts (never shared across SMs)

`components/kernel_taxonomy_<sm>.yaml`, `results/<sm>/<fw>-<ver>.yaml`,
`results/pathdiff/<sm>/<fw>-<ver>/`, `results/retests/<sm>/`. Gate lines in
`components/captures/verdicts_*.sh` may carry `FLOOR_SM=<sm> FLOOR_NOTE="..."`
(platform floor on that SM) or `SERVING_RAW=<raw>` (graded against a dedicated
probe raw). Customizations in `targets.yaml` may carry `sms: [sm120]` to apply
on those SMs only. Run ids include the platform name.

## Commands

```
python3 components/workflow_check.py <workflow> --param fw=vllm --param version=0.30.0 [--param sm=sm120]
python3 components/probe_driver.py --plan | --emit-queues | --records | --matrix        # never --plan with --emit-queues
AIS_SM=<sm> bash components/captures/verdicts_vllm_0300.sh                              # grade every declared gate
python3 components/evidence_bundle.py --campaign <date>_<sm>                            # pack evidence, index in results/
```

Full perf-data collection after a campaign is NOT an opharness component: use
`tools/perf_database/collect_campaign.py` (build-image / run / status / finalize) as
described in the `aic-auto-collect` skill — never a hand-written docker loop.

Captures run inside the framework image with the checkout mounted and
`PYTHONPATH=<checkout>/python/aisimulate`; one script per gate under
`components/captures/`. Probes and captures must not share a GPU.
