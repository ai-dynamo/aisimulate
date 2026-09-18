# TRT-LLM profiles for the 62 remaining Gym failures

## Scope

The `max_model_len` replay at AISimulate `89d2051137b772944a3e303c452108811790bef4`
left 62 missing-profile failures: 31 DeepSeek-R1 points on B200, 10 on H200,
and 21 GPT-OSS-120B points on B200. All use the original TRT-LLM database
query version `1.3.0rc20`.

This campaign measures TRT-LLM directly. It does not borrow SGLang timings.
The existing row schemas and consumer keys are unchanged.

The refreshed context MLA tables and their `op_backend_facts.yaml` entries
cover B200 and H200 only. The B300, GB200, GB300, H100 and RTX Pro 6000
Server rc20 tables retain their historical labels and have not been remeasured
or certified by this campaign. The facts registry joins persisted table keys;
it is not a prediction of the current collector's output on every system.
The source-backed collector correction applies to future runs on the listed
SMs. Refreshing each remaining system requires matching collection evidence
and an accompanying table/facts update; rewriting old labels alone would not
establish that evidence. `tools/perf_database/backend_facts.py --check` checks
all persisted slices, including the two refreshed systems.

## Results

| Production run | Planned / passed cases | Failed / unattempted | Measured rows |
| --- | ---: | ---: | ---: |
| B200, job `1993260`, 4 GPUs | 3,083 / 3,083 | 0 / 0 | 3,080 MLA + 81 MoE |
| H200, job `4375636`, 2 GPUs | 3,080 / 3,080 | 0 / 0 | 3,080 MLA |

Both jobs completed with exit `0:0`. Each MoE case produces 27 token-count
rows. All new physical keys are unique and all latencies are finite and positive.
See [collection evidence](gym-profiles-20260917-collection.json) for plans,
source/image/raw/parquet hashes, runtime kernel names, and observed telemetry.

Across overlapping MLA shapes, new/old median latency ratios are 0.993–0.996.
Six H200 shapes have much larger old timings. Job `4375895` repeated each
three times on another H200 node (`viking-prod-216`): all 18 passed, within
2.2% of the new production measurements. The cause of the old values is not
established. These diagnostic repeats remain separate from the published rows.

- Reproduced all 62 original missing-profile failures before replacing data.
- Recovered all 62 points, and reran the other 201 points: **263/263 passed**.
- Completed **293,180 requests**, including 35,520 in the recovered points,
  with zero truncated requests and zero context-limit violations.
- All 62 recovered native request specs are byte-identical to the failing
  baseline. Original workload fields are unchanged across all 263 points.
- Checked 1,773 installed package files: only the three published parquet
  files differ; no source or native binary changed, and no files were added
  or removed (excluding Python bytecode caches).

The replay uses the frozen `89d2051137b772944a3e303c452108811790bef4` build from
[PR #261](https://github.com/ai-dynamo/aisimulate/pull/261), with the measured
data overlaid. [Per-point replay evidence](gym-profiles-20260917-replay.json)
includes metrics, native spec hashes, original errors, and package hashes.
This verifies prediction availability and request completion; it does not
establish matched-silicon latency accuracy. Original silicon recipe image
versions are retained; the collector and database query both use `1.3.0rc20`.

Validation: 44 focused collector tests, Ruff check/format for changed Python
files, all seven collector-data checks, and the backend-facts registry check
passed. The latter matches 1,922 fact slices; its existing unrelated vLLM/FPM
warnings remain.

## Collector corrections

- Context MLA with FP8 KV internally quantizes Q/K/V to FP8 on
  SM90/100/103/120. The Python inputs remain BF16; logging their dtype as
  compute precision mislabels the measurements. Record FP8/FP8 for that path
  and BF16/BF16 for the control. The SM list mirrors the upstream
  `mFP8ContextMLA` condition, whose false branch leaves context compute BF16;
  it is not a list of the only architectures audited by this campaign.
  Generation tables are outside this refresh.
- Blackwell MXFP4 MoE pads weights before TP sharding through its native
  weight loader. Do not reject the logical intermediate dimension using
  the CUTLASS plugin's physical-weight alignment rule. Keep the original
  GPT-OSS shape (`hidden=2880`, `intermediate=2880`, 128 experts, top-k 4)
  and record TP=2/4/8, EP=1 under their original keys.

The measured runtime sources match TensorRT-LLM commit
[`c25c23f71786bad54d192893d696ce8043426eca`](https://github.com/NVIDIA/TensorRT-LLM/tree/c25c23f71786bad54d192893d696ce8043426eca):

- [MLA precision dispatch](https://github.com/NVIDIA/TensorRT-LLM/blob/c25c23f71786bad54d192893d696ce8043426eca/cpp/tensorrt_llm/thop/attentionOp.cpp#L1228-L1231).
- [MXFP4 weight creation and padding](https://github.com/NVIDIA/TensorRT-LLM/blob/c25c23f71786bad54d192893d696ce8043426eca/tensorrt_llm/_torch/modules/fused_moe/quantization.py#L5634-L5735).

## Reproduction

- OCI index: `sha256:1532b38814b3faf2affdb5ef01ca91468685d314ffb7e8926a0567595355ed88`.
- Pinned amd64 image: `nvcr.io/nvidia/tensorrt-llm/release@sha256:9b3b4dfb811caa9420fa99a6f958155f6a1f727ffc2b5a5c2d9d2ce51fdc323d`.
- Source snapshot: `c0317fccb25385093f9e8948675f765e4a6a0e96` plus the two
  collector corrections above; the source archive hash identifies the exact bytes.
- MLA uses the full existing `get_context_mla_test_cases()` grid.
- MoE selects the three matching `get_moe_test_cases()` tuples at runtime,
  preserving all 27 canonical token counts, through 16,384 tokens, and the
  `power_law_1.2` distribution.
- Both MLA precision slices are recollected; the fresh grids preserve every
  previously stored KV-storage/shape combination. Other GPUs are outside
  this data refresh.
- The 81 new MoE rows are finalized separately and appended using the old
  Arrow schema. All 218,295 existing rows retain their exact values, including
  power fields. New rows have null power fields; node telemetry is separate.
- Production invokes `run_mla` / `run_moe_torch` without profiler wrappers.
  MoE is the last case in each worker; `TRTLLM_MOE_RESTART_WORKER=0` lets the
  worker write its outcome before exiting normally.
- Smoke runs are separate. The original smoke wrapper labeled MoE exit 10
  as a crash; the collector explicitly uses that code to request worker
  recycling after finishing. The smoke CSVs retain all three expected token
  counts. No failed measurement is treated as a successful production case.

Artifact location aliases below are resolved through the campaign owner's
access-controlled retention manifest. Measurement hashes and artifact filenames
are unchanged. Artifacts are retained under `results/trt-62-20260917` on:

- B200: `${B200_CLUSTER}`, storage root
  `${B200_ARTIFACT_ROOT}`.
- H200: `${H200_CLUSTER}`, storage root `${H200_ARTIFACT_ROOT}`.
- Initial H200 preparation and failed launch: `${H200_PREPARATION_CLUSTER}`,
  storage root `${H200_PREPARATION_ARTIFACT_ROOT}`.

Local evidence, runners, source archives, native replay reports, and package
hashes are under `${CAMPAIGN_ARTIFACT_ROOT}/`.

## Infrastructure retries

- Aria image job `8899` failed registry authentication; a separate public
  Enroot configuration succeeded in job `8900`. Smoke job `8905` then failed
  before the runner started on `neb-cdg-slurm-1-gpu-11` (signal 53), and its
  dependent production job `8908` did not run.
- ComputeLab import job `4374730` hit unsupported overlay whiteout operations
  on NFS. It and its pending dependent GPU job `4374761` were cancelled.
  The completed B200 squashfs was transferred in eight parallel streams
  through the user's desktop, preserving SSH host verification. CPU job
  `4375480` verified its full SHA-256 before GPU use (exit `0:0`).
- ComputeLab job `4375492` failed during MPI initialization before smoke
  measurements. B200 defaults to `pmix`; ComputeLab has no MPI default.
  Retry `4375636` explicitly selects `srun --mpi=pmix` with the same image
  and collector source.
- These are preparation/launch failures, separate from measured case results.
  Their logs remain in each cluster's `logs/trt62-*` paths.

## GPT-OSS TPOT follow-up

The coverage replay exposed a large GPT-OSS/B200 TPOT gap. An independent
same-input graph benchmark confirmed that explicit MXFP4 autotuning lowers
TP1/tokens256 MoE latency from 1.7934 to 1.3227 ms (-26.25%). This is an
operator measurement, not an E2E accuracy result.

The collector now matches native GPT-OSS expert bias and routing-weight dtype,
and warms TRTLLMGen MXFP4 tactics instead of skipping them. GPT-OSS cache names
are separate from the earlier unbiased configuration. Native source evidence:

- [GPT-OSS bias and routing dtype, rc14](https://github.com/NVIDIA/TensorRT-LLM/blob/93cb6518b6d6dbd6095748189e626db731f44545/tensorrt_llm/_torch/models/modeling_gpt_oss.py#L159-L190).
- [Serving autotuner warmup, rc14](https://github.com/NVIDIA/TensorRT-LLM/blob/93cb6518b6d6dbd6095748189e626db731f44545/tensorrt_llm/_torch/pyexecutor/model_engine.py#L827-L842).
- [Autotuner enabled by default](https://github.com/NVIDIA/TensorRT-LLM/blob/93cb6518b6d6dbd6095748189e626db731f44545/tensorrt_llm/llmapi/llm_args.py#L3715-L3719).

The original coverage and collection results above remain historical evidence.
The follow-up refresh is scoped to GPT-OSS-120B/B200 MXFP4; other hardware
corpora are not implicitly recollected by this source change. Source-level
parity with the recipe does not replace an original serving kernel trace.

### Refreshed B200 profiles and replay

- Source `3830b5a4`; B200 job `1994597` completed `0:0` in 1m46s on two GPUs.
  All six canonical TP/EP cases passed: `(1,1)`, `(2,1)`, `(4,1)`, `(8,1)`,
  `(1,2)`, `(1,4)`, each with 27 token counts. All 162 timings used CUDA graphs.
- Replaced exactly those 162 MXFP4/power_law_1.2 rows. The other **218,214**
  MoE rows retain their exact values. Shape identities and Arrow schema are
  unchanged. This supersedes the earlier 81-row append for these GPT-OSS keys.
- Reran all **263 native specs**, with byte-identical spec serialization and
  the frozen runtime. Only one of 1,773 installed package files changed:
  B200/TRT-LLM rc20 `moe_perf.parquet`. All **293,180 requests** completed with
  identical sampled lengths, zero truncations, and zero context violations.
- All 232 points outside GPT-OSS/B200 retain exactly the same TTFT and TPOT.
  Among the 31 affected points, nine have worse TPOT APE; the largest increase
  is config283/ISL1024/concurrency64, **191.71% to 200.14%**. No rows were
  selected or discarded based on improved error.

| Backend / subset | Points | TTFT MAPE before / after | TPOT MAPE before / after |
| --- | ---: | ---: | ---: |
| TRT-LLM, all | 166 | 66.25% / 66.00% | 68.16% / 66.72% |
| TRT-LLM, GPT-OSS/B200 | 31 | 86.99% / 85.67% | 176.52% / 168.78% |
| SGLang, all | 97 | 69.61% / 69.61% | 12.55% / 12.55% |

MAPE is the unweighted mean of per-point absolute percentage errors against
positive silicon measurements. This is a data-only E2E ablation; it does not
claim exact runtime-version parity or that the remaining large gap is fixed.
Silicon uses rc14/rc18, while these profiles use rc20.

The controlled four-arm job `1994556` keeps routing hashes identical. At
TP1/tokens256, baseline/tuned/bias+tuned/full-contract medians are
1.82016/1.34733/1.34962/1.34993 ms. Kernel traces show a `128x8` to `128x16`
Bfloat16/MXFP4 GEMM tactic change. Bias and routing dtype fix correctness but
have negligible timing impact in this experiment.

[Follow-up evidence](gym-tpot-20260917.json) contains each paired point,
source/data hashes, coverage, and validation provenance. Raw reports, scripts,
and diagnostics remain under `trt-62-20260917/tpot-fix` in the local cache and
B200 storage. Validation: 52 focused tests, Ruff, and seven data checks passed.

An independent EP4 check (`1994626`, one B200, exit `0:0`) also shows that
alignment need not improve every shape. With identical routing hashes,
baseline/full-contract TP1/EP4 latencies at tokens160/192/256 are
0.29861/0.33427, 0.30739/0.34978, and 0.35100/0.42240 ms. The slowdown is
reproducible in direction, but does not isolate the full production change
(up to +51.3%): production uses balanced maximum-shape warmup, while this
controlled probe tunes the measured routing inputs. Neither result is used
to discard higher timings or select a better MAPE.

Original GPT-OSS config131 server artifacts from InferenceX run
`26016885799` are expired. Source-level recipe alignment and new same-version
microbenchmarks cannot establish its historical executed kernel/tactic.

The pinned original [GPT-OSS/B200 launch script](https://github.com/SemiAnalysisAI/InferenceX/blob/2baba8e27be8529b4453afc953f9859ec092c73d/benchmarks/single_node/gptoss_fp4_b200_trt.sh)
sets `TRTLLM_ENABLE_PDL=1`. The native environment helper defaults PDL to true
in both [rc14](https://github.com/NVIDIA/TensorRT-LLM/blob/93cb6518b6d6dbd6095748189e626db731f44545/cpp/tensorrt_llm/common/envUtils.cpp#L249-L275)
and [rc20](https://github.com/NVIDIA/TensorRT-LLM/blob/c25c23f71786bad54d192893d696ce8043426eca/cpp/tensorrt_llm/common/envUtils.cpp#L234-L260).
No PDL-default mismatch was identified. This source check does not establish
full serving-environment parity or a historical executed-kernel match.

### rc14 runtime cross-check

CPU image import `1994625` and one-B200 probe `1994650` both completed `0:0`.
The latter produced 18/18 graph timings (two arms, three shapes, three repeats)
in 1m33s. Its pinned amd64 image is
`sha256:fe2f17d0c9698bafeb9f437929003c6badc56d17e8382a02505233145a07f61f`.
The runner verifies runtime version `1.3.0rc14` and SM100. Routing hashes match
both arms and the rc20 probe. Each timed closure contains five forwards;
the following medians divide its elapsed time by five, like the collector.

| Tokens | rc14 baseline / aligned (ms) | rc20 baseline / aligned (ms) |
| ---: | ---: | ---: |
| 64 | 0.95476 / 0.93937 | 0.89840 / 0.89107 |
| 256 | 1.92882 / 1.40442 | 1.82016 / 1.34993 |
| 16,384 | 12.52036 / 12.56028 | 11.35655 / 11.26267 |

At 256 tokens, rc14 also switches the native BF16/MXFP4 GEMM tactic from
`128x8` to `128x16`. The aligned rc14 timing is 4.04% higher than rc20;
the other two shapes are 5.42% and 11.52% higher. These observations do not
support the runtime version alone explaining the original roughly 5x E2E
overprediction. Different B200 nodes and image dependencies prevent treating
this as an isolated version-only effect. The tag's resolved image is not
proof of the historical May serving image digest. No rc14 timings replace
rc20 production data.

The remaining attribution requires a matched full-model serving trace:
actual expert routing and decode shapes, MoE tactics, and per-step timing.
The expired original server artifacts prevent checking those historical facts.
A new matched serving run can test the mechanism but cannot recover them.

## Full serving reproduction: activation mapping correction

The [GPT-OSS/B200 reproduction](gym-tpot-serving-20260917.md) supersedes the
assumption that W4A16 was the effective serving precision. The real rc14 server
selects **W4A8 MXFP4/MXFP8** by its Blackwell runtime default. Source, live model
metadata, and executed GPU kernels agree. The earlier W4A16 collector ablation
remains valid for that operator, but did not establish full serving parity.

At config131/ISL1024/OSL1024/C256, uninstrumented TPOT is **16.54925 ms**
(historical **16.79212 ms**). Replaying the exact client lengths changes TPOT
**65.16758 → 17.17212 ms** when only `aic_moe_dtype` changes from W4A16 to W4A8;
APE falls **293.78% → 3.76%**. All 2,560 requests complete in both arms.
This is a one-point verification, not a replacement 31/263-point MAPE.
The automatic Gym mapping and missing W4A8 TP2/4/8 profiles still need correction;
the earlier W4A16 timings must not be relabeled as W4A8.

## Full correction follow-up

The mapping, W4A8 collection, and complete 263-point rerun are recorded in
[the 2026-09-18 report](gym-w4a8-20260918.md). Earlier metrics above remain
historical ablations.
