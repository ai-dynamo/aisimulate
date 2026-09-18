# TRT-LLM profiles for the 62 remaining Gym failures

## Scope

The `max_model_len` replay at AISimulate `89d2051137b772944a3e303c452108811790bef4`
left 62 missing-profile failures: 31 DeepSeek-R1 points on B200, 10 on H200,
and 21 GPT-OSS-120B points on B200. All use the original TRT-LLM database
query version `1.3.0rc20`.

This campaign measures TRT-LLM directly. It does not borrow SGLang timings.
The existing row schemas and consumer keys are unchanged.

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
  and BF16/BF16 for the control. Generation tables are outside this refresh.
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

Artifacts are retained under `results/trt-62-20260917` on:

- B200: `nsc-svg-slurm-1-login-02.nvidia.com`, storage root
  `/lustre/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/users/simonec`.
- H200: `computelab-sc-01`, storage root `/home/scratch.simonec_gpu`.
- Initial H200 preparation and failed launch: `neb-cdg-slurm-1-login-02.nvidia.com`,
  storage root `/lustre/fsw/portfolios/coreai/projects/coreai_comparch_lights-out-inf/users/simonec`.

Local evidence, runners, source archives, native replay reports, and package
hashes are under `/Users/simonec/.cache/aisim-e2e-gym/trt-62-20260917/`.

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
