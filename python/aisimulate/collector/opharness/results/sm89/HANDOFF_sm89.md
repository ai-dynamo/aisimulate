# opharness / collector on sm89 (L40) — handoff to upstream

Date: 2026-10-04. Box: 8 x NVIDIA L40 (CC 8.9, 46068 MiB / 44.4 GiB usable), driver 595.58.03, docker, no sudo.
Branch: `tianhaox/aisimulate:feat/op-probe-harness-sm89`, based on PR #228 head `2a7ad55b`. Everything below is committed and pushed; nothing is merged.
Workspace (data, box-local, **recycled ~8 h after 2026-10-04 06:23**): `/tmp/tianhaox/ws_sm89` and checkout `/tmp/tianhaox/sm89`. Nothing was placed on /raid.
Persistent copies: `debug/opharness/sm89/` on the scratch NFS (evidence bundle `2026-10-04_sm89.*.tar.gz`, indexed in `results/evidence_index.yaml`, plus `evidence_sm89_extra_*.tgz`:
A/B raws, sampling logs, scripts). Scratch NFS had 5 GB free; only these ~3.4 MB live there.

Scope asked: (1) probe, (2) upgrade the sglang / vllm / trtllm collectors to the pinned versions (same as sm90; sm100 in progress), harness + collector + generator fixes may go in the code,
probe facts may go in. Pins are the PR head's: vllm 0.30.0, sglang 0.5.21, trtllm 1.3.0rc29 (digests match `framework_manifest.yaml`).

## 1. Result

| framework | identity matrix (`results/sm89/`) | real gates | declared sm89 floors | `workflow_check upgrade_op --param sm=sm89` |
|---|---|---|---|---|
| vllm 0.30.0 | 67 pass / 4 pass+custom / 53 fail (sm90 110/1/13, sm120 98/7/19) | 13 / 13 ALIGNED | 18 | **all 7 steps green** |
| sglang 0.5.21 | 65 / 0 / 59 (sm90 101/4/19) | 6 / 6 ALIGNED (encoder_attn closed by the vision probe) | 23 | **all 7 steps green** |
| trtllm 1.3.0rc29 | 54 / 0 / 70 (the 7 NVFP4 MoE cells pass under the generator's own MARLIN render) | 9 / 9 ALIGNED (encoder_attn closed by the vision probe) | 15 (incl. the 2 msa framework-gap floors) | **all 7 steps green** |

After the probe change (vision forward) sglang and trtllm were re-probed in full; the matrices came back with identical pass/fail counts and causes except five NVFP4 DSA trtllm cells (DeepSeek-V3.2-NVFP4, GLM-5/5.1/5.2/5.3-NVFP4) whose first wall moved from the MoE selection (now MARLIN via the generator) to the next one, the DeepGEMM sparse-attention arch assert. All 124 cells of every matrix carry a cause; every fail is root-caused in `results/findings.yaml` (5 new entries `sm89_*`).
Fail classes in short: capacity (44 GiB), sparse-MLA has no backend below SM90 (vllm 14 / sglang 14), DeepGEMM arch (DeepSeek-V4), FP8-block / MXFP4 MoE has **no trtllm implementation on SM89** (26 cells),
flashinfer's cutlass fused-MoE JIT does not compile for sm_89 (sglang, 5 Nemotron-3 cells), shared 101376 B smem (4 NVFP4 MLA cells, same as sm120), image gaps (transformers/fla), generator rejects (12, same as sm90/sm120).

## 2. What changed, by commit (and what to drop when absorbing)

| commit | content | absorb note |
|---|---|---|
| bd4cbfa9 (+review fix) | generator `hardware.yaml` `l40s` profile (product-keyed like h100/h200; first committed as an SM-keyed `sm89` with a phantom `l40` system, renamed in review) + explicit `request_resolution` entry. Without it `resolve_facts` raised and the pipeline swallowed it: no model fact reached l40s renders | keep |
| 781a7435 | `targets.yaml` platform pin -> `l40_sm89`, `kernel_taxonomy_sm89.yaml` seed | **box-local pin: drop that hunk on a shared branch** |
| 05b10973 | collector: DSV4 attention module ops `op_min_sm: 90` (were the only sparse family without a floor) | keep |
| d4e567b2 | sglang 0.5.21: attention mock `prefill/decode_attention_backend_str`; encoder Triton kernel moved module; MLA mock `hf_config` | keep — lanes Hopper/Blackwell never reach; **sm120 will hit the same three** |
| e7c9e56f, 465f44e0 | `_fail_cause` on the full line + deep traceback pass; comma-list `FLOOR_SM`; sm89 vllm gate floors | keep |
| 955e1b73 | Gemma4 + Llama4 case maps: `sm89: triton` (framework forces triton off SM90/SM100) | keep |
| bdebc044 | sm89 vocabulary; floor mechanism ported to sglang/trtllm verdict scripts; 4 missing sglang gate recipes; golden-render cache keyed on the dirty generator diff | keep |
| a62123a1 | sm89 matrices (vllm, sglang), pathdiff verdicts, retests, findings, targets customizations | data; `results/sm89/trtllm-*` in the next commit; the 10 trtllm MARLIN customizations were later replaced by the generator fact (review pass) |
| later | trtllm matrix/pathdiff/retests, trt gate recipes, vllm gdn FIXME re-verification, probe console tails, final findings, this file | |
| review pass | generator: `l40s` quant-keyed trtllm MoE backend + `moe_backend` param honoured on trtllm; collector: sglang `attention_context` no longer plans the fp8-context-FMHA case on the flashinfer lane (sm89/sm120; it refused it at run time, 13/40 sampled tasks); probe_driver deciding-line / stamp tweaks; evidence bundle packs console tails | keep |

Arch-neutral, please propagate to sm90/sm120/sm100 branches: golden cache stamp, console-tail capture + deep classification, `FLOOR_SM` lists + floor support in the sglang/trtllm verdict scripts, the 8 gate recipes
(sgl_attn_ctx/gen/gemma4_hd512, sgl_gdn_gen, trt_attn_ctx/gen + fp8kv — the sm100 notes list the same gaps), the three sglang collector lane fixes.

## 3. Owner decisions / open items

1. **trtllm MoE default was platform-blind** (same as the B200 notes): CUTLASS cannot serve FP8-block below SM90/SM120 nor NVFP4 below SM100. **Fixed in the generator** (review pass): `hardware.yaml` `l40s` keys the trtllm MoE backend by the artifact's quantization family (`nvfp4: MARLIN`; `facts/apply.quant_family_of`), and the trtllm template now honours the generic `moe_backend` param, which the hardware fill used to override silently. The 10 NVFP4 roster cells were re-probed under the plain render (no targets.yaml customization left for them). A/B on an empty L40: AUTO resolves to CUTLASS (a hardware-wide AUTO fact was tried and **reverted**),
   TritonFusedMoE is SM90-only, DEEPGEMM requires SM100/103/107, VanillaMoE dies in CUDA-graph capture, **Marlin serves NVFP4 only** (weight-only). Raws for every backend x cell in `facts/ab_trt_moe_sm89/raw/` (evidence archive `evidence_sm89_extra_*.tgz`), the AUTO/DEEPGEMM ones re-taken on a clean box in the review pass. FP8-block / MXFP4 MoE have no trtllm implementation on SM89 at all (every backend turned down), so those cells stay platform floors.
2. **encoder_attn gates (sglang, trtllm): closed.** `probe_sglang` / `probe_trtllm` now run one synthetic image through the vision module (`model.visual` / `mm_encoder.visual`, HF pixel_values layout) under the device profiler and record it as `profile_run_kernels`, the table the gate's phase reads; both gates ALIGNED on sm89 (sglang Triton `_fwd_kernel`, trtllm `fmha_v2 ... sm89`). Cost: the probe file is part of the execution fingerprint, so sglang + trtllm were fully re-probed. sm90's earlier aligned verdicts rest on records this checkout cannot explain — re-grade with the new probe on the H20 box. The trtllm **msa gates** the sm90 session left undeclared are now declared (`trt_msa_ctx/gen` recipes) as a floor on every probed SM with the framework-gap note, so `gates_declared` is green everywhere and the gate grades for real once an rc serves MiniMax-M3.
3. **FP8 context-FMHA plan hole**: sglang `attention_context` plans `fp8_kv_cache_and_context_fmha` for FlashInfer SMs (sm89/sm120) where the collector refuses it on purpose — 13 of 40 sampled cases. The generator comment says backend rejections are runtime failures by policy; a per-SM precision gate would remove a third of the op's failures. Also the refusal text still says "0.5.14".
4. trtllm `attn_ctx_fp8`: serving's fp8-KV prefill runs `fmha_v2 ... bf16 ...` over the fp8 cache; the collector's fp8-KV + fp8-context-FMHA case runs the `e4m3` kernel, which serving does not select on sm89. The gate recipe captures the bf16-compute combo (ALIGNED). Whether the SDK should bill the e4m3 case on sm89 is yours.
5. trtllm MLA wall (`no MLA FMHA below Hopper`) **re-verified on rc29** (attentionOp.cpp:3234); collector guards kept. vllm GDN context IMA on SM89 **re-verified on 0.30.0** (17 of 21 failed sampled tasks are CUDA faults).
6. OPEN harness items: tp>1 cells hard-code `device=0,1,2,3` (no group reservation; rerun alone: sglang Qwen3.8-2.4T = capacity, trtllm = `Executor worker returned error` with no cause in the raw); `fetch_inputs.py` bundled fallback does not stage Llama-4 `preprocessor_config.json` / tokenizers (workspace `stage_tok.py` pulls them from unsloth mirrors);
   `workflow_check gates_declared` for trtllm rc29 lacks msa gates on every SM; trt `kda` is not a trtllm op (my smoke list included it, exit 2 is mine).
7. Not done: the **full data collection** (sm120's vllm run was ~12 h on 8 GPUs) and publication of any perf data; SDK consumer tests against new `l40s` data. **Delivery blocker if anyone collects perf data here**: the SDK system `l40s` carries device `NVIDIA L40S` in its parquet tables; this box is an **L40** (same AD102 die, lower TDP/clocks). Playbook §8 forbids relabelling a nearby product with the same SM, so nothing measured here may be published under `l40s` without an `l40` system definition (the sm120 RTX PRO 5000 campaign hit the same wall). The identity probes are unaffected: kernel/backend selection is by SM, not by product.

8. **Boundary-shape CUDA faults are not being researched (owner decision 2026-10-04 evening)**: the sampled IMAs / SIGABRTs are all at large-token corners — trtllm `attention_context` hd256 at 41k–131k tokens (`[16, 8192, 64, 2, 256]` bf16, `[4, 10240, 96, 96, 256]` fp8), vLLM GDN at ~1M-token sub-points (known, FIXME re-verified), sglang `mla_generation` at kv 65535 × batch 128. A full hd256 `attention_context` collection (17,547 cases, ~2 h on 7 GPUs) was prepared and then dropped on that decision. Also noted, not fixed: trtllm fp8 `gemm` with N ∈ {4, 8} fails torch `_scaled_mm`'s stride check (`out.strides()[0] % 16`) — arch-independent, worth checking on sm90 before any guard.

## 4. Step 2 evidence: collector smoke + 40-case sample (sm89, one container per op, `--limit 40 --shuffle`; `kda` smoke on trtllm is not an op)

| framework | op | cases | errors | classes |
|---|---|---|---|---|
| sglang | attention_context | 40 | 13 | 13 refused combo (fp8 ctx FMHA on flashinfer) |
| sglang | attention_generation | 40 | 0 | - |
| sglang | compute_scale | 40 | 0 | - |
| sglang | encoder_attention | 40 | 0 | - |
| sglang | gdn | 40 | 21 | 21 grid task: partial points failed (see detail) |
| sglang | gemm | 40 | 0 | - |
| sglang | kda | 12 | 5 | 5 framework int32 index limit |
| sglang | mla_bmm_gen_post | 40 | 0 | - |
| sglang | mla_bmm_gen_pre | 40 | 0 | - |
| sglang | mla_context | 40 | 0 | - |
| sglang | mla_generation | 40 | 4 | 3 capacity (44 GiB), 1 CUDA fault (kernel limit) |
| sglang | moe | 40 | 1 | 1 other |
| sglang | msa_context_module | 40 | ? | running/killed |
| sglang | msa_generation_module | 5 | 0 | - |
| trtllm | attention_context | 40 | 6 | 4 CUDA fault (kernel limit), 2 platform guard |
| trtllm | attention_generation | 40 | 2 | 2 platform guard |
| trtllm | compute_scale | 40 | 0 | - |
| trtllm | encoder_attention | 40 | 0 | - |
| trtllm | gdn | 40 | 0 | - |
| trtllm | gemm | 40 | 2 | 2 alignment guard |
| trtllm | mamba2 | 12 | 0 | - |
| trtllm | mhc_module | 40 | 2 | 2 capacity (44 GiB) |
| trtllm | mla_bmm_gen_post | 40 | 0 | - |
| trtllm | mla_bmm_gen_pre | 40 | 0 | - |
| trtllm | moe | 40 | 8 | 7 alignment guard, 1 capacity (44 GiB) |
| trtllm | msa_context_module | 40 | 1 | 1 capacity (44 GiB) |
| trtllm | msa_generation_module | 40 | 1 | 1 capacity (44 GiB) |
| vllm | attention_context | 40 | 0 | - |
| vllm | attention_generation | 40 | 1 | 1 capacity (44 GiB) |
| vllm | compute_scale | 40 | 0 | - |
| vllm | encoder_attention | 40 | 0 | - |
| vllm | gdn | 40 | 21 | 17 CUDA fault (kernel limit), 3 other, 1 capacity (44 GiB) |
| vllm | gemm | 40 | 0 | - |
| vllm | kda | 12 | 5 | 4 capacity (44 GiB), 1 grid task: partial points failed (see detail) |
| vllm | mla_bmm_gen_post | 40 | 0 | - |
| vllm | mla_bmm_gen_pre | 40 | 0 | - |
| vllm | mla_context_module | 40 | 0 | - |
| vllm | mla_generation_module | 40 | 0 | - |
| vllm | moe | 40 | 2 | 1 capacity (44 GiB), 1 alignment guard |
| vllm | msa_context_module | 40 | 0 | - |
| vllm | msa_generation_module | 40 | 0 | - |

(`running/killed`: the sglang `msa_context_module` sample was still running after ~1h20 and was stopped by hand; the trtllm one needed about an hour to finish. MSA context grids are by far the slowest op on L40.)
Plan audit (`collect.py --model-cases-full --sm 89 --plan-only`): vllm 20 ops, sglang 27, trtllm 23; DSA / DSV4-attention / GLM-5 sparse modules floor to 0 cases at SM90.
Smoke (4 cases/op): vllm 14 ops with data, sglang 14 (mhc/dsv4 are deliberate UnverifiedCollector refusals), trtllm 13 (MLA modules refuse by the verified guard).

**Tests**: `tests/unit/collector` + `tests/unit/generator` 2186 passed / 13 skipped on this branch. `tests/unit/collector/test_collect_provenance_writer.py` has 26 failures (`FileNotFoundError: all_<ts>/collector_profile_sglang.prof`) that are **identical on the untouched base branch** (verified in a worktree) — pre-existing, deselected here.

## 5. Reproduce

```
export AIS_PROBE_WORKSPACE=/tmp/tianhaox/ws_sm89     # fetch_inputs --from-roster, dummies.py, then:
python3 components/probe_driver.py --emit-queues --backends <fw> --gpu-list 0,1,2,3,4,5,6,7   # run archive/queues/gpuN.sh, then --records / --matrix
bash components/captures/verdicts_<fw>.sh            # after `path_diff --capture` recipes in the framework image (collect/run_captures.sh)
python3 components/workflow_check.py upgrade_op --param fw=<fw> --param version=<ver> --param sm=sm89
python3 tools/perf_database/collect_campaign.py build-image --backend <fw> --op gemm      # collector images; collect.py --sm 89 --ops <op> --smoke
```
Wall clock (rough, from this session): a 224-run probe pass took on the order of one hour on 7 GPUs per framework; a full `--emit-queues` re-renders every golden serially (~2 min when cached, tens of minutes cold) — warm the cache with parallel `--only` emits.
