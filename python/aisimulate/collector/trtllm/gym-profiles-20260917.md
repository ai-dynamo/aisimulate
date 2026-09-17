# TRT-LLM profiles for the 62 remaining Gym failures

## Scope

The `max_model_len` replay at AISimulate `89d2051137b772944a3e303c452108811790bef4`
left 62 missing-profile failures: 31 DeepSeek-R1 points on B200, 10 on H200,
and 21 GPT-OSS-120B points on B200. All use the original TRT-LLM database
query version `1.3.0rc20`.

This campaign measures TRT-LLM directly. It does not borrow SGLang timings.
The existing row schemas and consumer keys are unchanged.

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

The runtime sources match TensorRT-LLM commit
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
- H200: `neb-cdg-slurm-1-login-02.nvidia.com`, storage root
  `/lustre/fsw/portfolios/coreai/projects/coreai_comparch_lights-out-inf/users/simonec`.

Local evidence, runners, source archives, native replay reports, and package
hashes are under `/Users/simonec/.cache/aisim-e2e-gym/trt-62-20260917/`.
