<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GLM-5.3-Flash sparse-MLA + IndexPool attention collection

Standalone collectors that load the real GLM-5.3-Flash checkpoint through the
framework's own model builder and time one NoPE sparse-MLA attention layer over
real KV/IndexPool state. They publish `glm53_attention_module_perf.parquet`,
the single GLM-specific measured table (every other GLM operator uses the
generic GEMM/MoE/KDA/mHC/collective tables). The collectors follow the
DeepSeek-V4.1 precedent in `sglang/README.dsv41.md`.

## Pinned runtimes

| Backend | Version directory | Image | Notes |
| --- | --- | --- | --- |
| vLLM | `0.31.0` | `vllm/vllm-openai@sha256:3f7dd5b7…b2971` (`v0.31.0-aarch64`; tag commit `db9527a4`) | Stock, no overlay. Its pooled-index prefill write still assumes pool-aligned chunk starts (see "IndexPool alignment"). |
| SGLang | `0.5.20` | `lmsysorg/sglang@sha256:b0d8718a…c8862` | Stock. |

Checkpoints: `zai-org/GLM-5.3-Flash@eb9eb208…` (FP8 block-128) and
`nvidia/GLM-5.3-Flash-NVFP4@09b04e5e…`.

## Measured boundary

`Glm5NextMLAAttention.forward` (vLLM) / `DeepseekV2AttentionMLA.forward`
(SGLang) of layer 3: fused q_a/kv_a latent projection, q_a/kv_a RMSNorm, q_b
projection, absorbed kv_b BMMs, the complete IndexPool indexer (wq_b,
wk/weights projections, k-norm, Hadamard + FP8 quantization, pool compress
gate, pooled index-K write, retained-tail update, MQA logits and top-k, or the
causal fill of the short-prefix regime), FP8 latent KV write, the NoPE sparse
MLA kernel and the local `o_proj` GEMM.

Excluded: the attention-output all-reduce (SGLang builds attention with
`reduce_results=False` and reduces in `MHCLayerCommunicator.prepare_mlp`;
vLLM's `o_proj.reduce_results` is switched off only for the measured calls,
exactly as the model does for sequence parallelism), mHC pre/post and the input
norm (separate ops), and the framework's per-forward attention-metadata build.
The model adds `attention_allreduce_*` separately.

Layer 3 represents all 11 sparse-MLA layers (3, 7, …, 43): same module class
and dimensions, own indexer (`indexer_types == "full"`, no top-k reuse) and
identical per-layer quantization exclusions in both checkpoints. The contract
(`representative_layer_is_uniform`) rejects a checkpoint where this fails.

Projection precision is observed from the loaded modules and must match the
key: SGLang FP8 serves q_a/kv_a, q_b and o_proj as FP8 block-128 (kv_b and the
indexer BF16); SGLang NVFP4 and every vLLM deployment serve BF16 MLA
projections (vLLM builds MLA with `quant_config=None`).

## Execution mode and timing

Both phases run under the frameworks' serving CUDA graphs (clean-truth server
flags):

- SGLang: `--cuda-graph-backend-prefill breakable --cuda-graph-max-bs-prefill
  8192` (58 token buckets) and full decode graphs for bs 1..32;
  `attention_backend=dsa`, DSA prefill/decode `trtllm`, page 64, FP8 KV,
  chunked prefill 8192.
- vLLM: default `CompilationMode.NONE` + `FULL_AND_PIECEWISE` with the 62
  `--cudagraph-capture-sizes` up to 8192 (breakable piecewise prefill graphs,
  FULL uniform-decode graphs), `FLASHINFER_MLA_SPARSE`, FP8 KV, 8192 batched
  tokens.

How the frameworks execute the module:

- SGLang prefill (`runner/prefill_cuda_graph_runner.py`,
  `runner_backend/breakable_cuda_graph_backend.py`): the transformer body is
  captured per token bucket as graph segments separated by `eager_on_graph`
  breaks. For this module the absorbed MLA method is pinned in graph mode
  (`deepseek_common/attention_backend_handler.py`), and the pooled-key indexer
  (`attention/dsa/kpool_prefill_cuda_graph.py`, reading the live batch from the
  TcPiecewise context) and the MLA BMM + attention core
  (`attention_forward_methods/forward_mla.py` `bcg_mla_bmm_then_unified_attention`)
  are eager breaks; projections, norms and o_proj replay from segments.
- vLLM prefill (`v1/worker/gpu/cudagraph_utils.py:552-557` `run_pw_graph`,
  `compilation/breakable_cudagraph.py`): ops decorated with
  `eager_break_during_capture` -- the IndexPool indexer
  (`models/glm5next/nvidia/sparse_indexer.py:93`) and the MLA attention op
  (`model_executor/layers/attention/mla_attention.py:1466`) -- run eagerly
  against the step's forward context; everything else replays from captured
  segments.
- Decode (both): FULL graphs (`DecodeCudaGraphRunner.execute`;
  vLLM `ModelCudaGraphManager.run_fullgraph`, cudagraph_utils.py:751-764).

The probe records the module's capture-time inputs for every graph size while
the framework captures, executes the real planned step (the framework replays
its own graph for the padded size; witnessed), then captures the module alone
with the framework's own capture mechanism (SGLang/vLLM
`BreakableCUDAGraphCapture` for prefill, `torch.cuda.graph` under the
framework's capture mode for decode) under that step's live contexts.
Padding to the framework bucket is therefore included.

### Timing: GPU kernel time only

`timing_method` `cupti_gpu_busy_union_framework_breakable_module_graph_replay`
(prefill) and `cupti_gpu_busy_union_captured_module_graph_replay` (decode),
`used_cuda_graph=true` (`glm53flash_attention_runtime.KernelTimer`):

1. The module graph is replayed `warmup + iterations` (3 + 10) times under the
   torch profiler (kineto/CUPTI, CPU + CUDA activities). Each repetition is a
   `record_function` range that encloses the replay (graph launches and the
   eager breaks' kernel launches) and a trailing `torch.cuda.synchronize()`,
   so the GPU work of two repetitions never interleaves and the profiler's
   start/stop work lies outside every range.
2. Every CUPTI GPU activity on the rank's device -- kernels (CUDA-graph kernels
   are reported per node), device memcpys and memsets -- is attributed to the
   repetition whose range contains the CUDA runtime/driver call with the same
   correlation id. An activity without a recorded launch call is attributed
   only if it executes entirely inside one range (counted as `time_only`); a
   correlation/time contradiction, an activity of a call outside every range
   that executes inside one, or a repetition without kernels fails the target.
3. Repetition latency = the length of the union of its activities' GPU-busy
   intervals (overlapping streams count once, host launch gaps never count).
4. Row latency = median over the 10 timed repetitions of the maximum across TP
   ranks (as before).

Per-target diagnostics in the evidence JSON: per repetition the busy union,
the plain kernel-duration sum, kernel/memcpy/memset counts, the GPU span, the
CUDA-event interval of the profiled repetition and of an unprofiled
back-to-back replay of the same graph (the previous host-inclusive method),
the host enqueue time, and the CV of the rank maxima.

Validation (smoke rows vs in-serving nsys node traces of the module's kernels,
per MLA layer, rank maximum):

- SGLang 0.5.20, 8 holdout geometries x 4 deployments: 0.93-1.02x (one point,
  fp8-tp4 B32 prefix 98048, 1.105x).
- vLLM 0.31.0 (stock `vllm serve`, fp8-tp2 and nvfp4-tp4, 7 points each):
  1.02-1.12x, median 1.06x. **Known bias, accepted:** per kernel, the cuBLAS
  (nvjet) projection GEMMs run 5-18% slower in the standalone module replay
  while the sparse fmha and the FWHT quantization match; 200 warmup
  repetitions do not change it. This is the standalone-op method of the other
  operator tables; `collection_meta.yaml` records it.
- Decode: kernel-only = 0.87-0.98x the CUDA-event interval of the same
  full-graph replay (the plain kernel-duration sum is 1.01-1.26x, because
  kernels overlap; hence the union).

Previous revisions (`0.30.0+glm53tail` tables) bracketed the replay with CUDA
events, so their prefill rows also contained the eager breaks' host launch
gaps; their decode rows were event-timed full-graph replays.

### IndexPool alignment (vLLM)

Stock vLLM 0.31.0 leaves the boundary pool of a prefill chunk that starts off
the 4-token pool grid unwritten, or fills it with another request's tokens
(`models/glm5next/nvidia/sparse_indexer.py:47-90` `_kpool_compress_insert`).
Every planned vLLM chunk therefore starts on a multiple of 4: prefill targets
(prefix and new tokens) are multiples of 4, seeding chunks are the batch's
share of the 8192-token budget floored to a multiple of 4
(`contract.seed_chunk`: B=3 2728, B=6 1364, B=12 680, B=24 340), and
requests advance in lockstep. The launcher refuses an unaligned vLLM plan, and
the runner checks every scheduled prefill chunk before the step
(`KpoolAlignmentError`) and records the chunk starts per row
(`prefill_chunk_starts`). A decode reads L tokens after a prompt of L-1
tokens; only its last seeding chunk may end off the grid, which completes in
the tail as in serving. Decode keys therefore keep their true L: the rule is
enforced where the defect triggers, at prefill chunk starts. No planned geometry moved (`kpool_align4` would be the
recorded reason). SGLang is unaffected; its smoke validation rows keep the
serving geometry.

## Workload and state

`cases/base_ops/glm53flash_attention.yaml` (463 points):

- regular class (454, server context limit 131079 as in serving): prefill
  batch 1–32 × new tokens 8–8192 × cached prefix 0–98304 (every step within the
  8192-token serving budget; up to 106496 context), decode batch 1–32 ×
  absolute length 64–131072;
- long class (9, server context limit 1048576 = the model's
  `max_position_embeddings`): B=1 prefill with prefix 262144, 524288, 1032192
  and new tokens 1024, 8192, and B=1 decode at 262144, 524288 and 1048575 (a
  prompt must stay below the limit and vLLM caps a request at it, so the
  largest decode reads 1048575 tokens).

One attempt measures one context class (`--context-class`), so the server
context limit (`max_model_len` in the manifest and `collection_meta.yaml`) is
uniform per attempt. Regular rows use the serving limit 131079; the
long-context rows exist for table extrapolation coverage only (a 1M limit
changes graph-captured decode host bounds; measured effect on regular-length
decode rows <= 3%).

Memory feasibility (the sanctioned generation-time filter): an attempt may
carry its deployment's measured capacity (`--kv-token-capacity`,
`--transient-gib`, `--device-gib`, `--memory-evidence`). A target is never
queued when `batch * context` exceeds the KV pool or one step's IndexPool MQA
logits (`8192 * batch * ceil(context / 4) * 4` bytes) exceed the transient
headroom; the launcher logs `glm53flash_attention: dropped N/M cases (memory
budget, device=<GB>)`, and the drops and budget are frozen in the manifest and
listed in `collection_meta.yaml`. Used for SGLang NVFP4 TP1 (KV pool 3396928
tokens and 18 GiB headroom at `--mem-fraction-static 0.74`; 0.82 leaves too
little for SGLang's own prefill-graph capture): B32 prefill at prefix 98304
and B32 decode at 98304/131072 do not fit. TP1 is collected for the NVFP4 checkpoint only (the FP8
weights do not fit one GB300).

Grid points bracket the IndexPool boundary (`prefix + x <= 2048` short regime
selects every pool without scoring; otherwise pooled + retained tail). Request
token ids come from the seeded random-token generator
`collector/glm53flash_attention_tokens.py` (seed 53, 1179743 ids for the full
plan, ordinary tokenizer vocabulary only); KV, IndexPool and tail state are
produced by the framework's own allocation and forward helpers:

- SGLang: `one_batch.load_model`, `prepare_synthetic_inputs_for_latency_test`,
  and `extend`/`decode`; continuations re-extend each request from its own
  `req_to_token` prefix (as `prepare_extend_inputs_for_correctness_test`).
  Prefix state is seeded eagerly; only the measured step replays the serving
  prefill graph (witnessed).
- vLLM: the in-process `LLM` engine and its scheduler. Before each step the
  driver sets `max_num_scheduled_tokens` and `long_prefill_token_threshold`
  so B homogeneous requests advance in lockstep; this only decides batching.

Memory: per-target module graphs make reserved memory grow across targets
(vLLM: one shared private pool, the previous graph kept alive until the next
capture; SGLang: one `torch.cuda.MemPool` per prefill target, deleted after
it). Headroom therefore comes from capacity-only knobs (vLLM
`--gpu-memory-utilization`, SGLang `--mem-fraction-static`, the allocator
split), and a deployment's plan may be split across attempts whose manifests
carry disjoint `only_sets`; `finalize` admits split attempts only if they
cover the planned keys exactly once. Each attempt's knobs are frozen in its
manifest and listed in `collection_meta.yaml`.

## Published tables (2026-10-09)

| Backend | Rows | Not measured |
| --- | --- | --- |
| vLLM `0.31.0` | 2315 = 5 deployments x 463 | none |
| SGLang `0.5.20` | 2292 = 4 x 463 + 440 (nvfp4-tp1) | nvfp4-tp1: 19 memory-feasibility drops (KV pool 3396928 tokens, 10.5 GiB transient at `--mem-fraction-static 0.74`) and 4 classified failures (B32 x256 at prefix 16384/32768, B16 x512 at prefix 65536, B1 x8192 at prefix 1032192: reproducible OOMs in fresh processes), listed in `collection_meta.yaml` |

Every GPU was a GB300. Long collections fragment device memory: many
attempts were split across fresh processes (capacity only), and failed
attempts contribute their completely measured targets through
`finalize --partial` (a complete attempt's copy of a key wins). Each table
directory carries `glm53_attention_module_perf.evidence.json.gz` (per-row
rank-max medians of the timing diagnostics); the full per-repetition evidence
is kept with the collection audit.

## Allocator policy

At batch 32 and 98K–131K context the ragged IndexPool MQA-logits buffer of a
seeding extend (`deep_gemm.fp8_mqa_logits`, ~24 GiB) failed on fragmentation
(24 GiB reserved-but-unallocated) in all four SGLang deployments. The
published SGLang attempts therefore set `PYTORCH_CUDA_ALLOC_CONF=
max_split_size_mb:16384`, the split the qualified SGLang FP8 TP2 FPM campaign
uses. It changes allocation only, not kernels; it is frozen in each manifest.

## Data contract

`glm53_attention_module_perf.parquet` in
`systems/profiles/glm53flash/data/gb300/glm53_attention/<backend>/<version>/`,
read by `crates/core/src/perfmodel/perf_database/glm53flash.rs`:

| Column | Contract |
| --- | --- |
| `component` | `attention` |
| `geometry` | canonical sorted compact JSON of the model's `Glm53Attention` body without `name`/`measured` (backend, checkpoint, TP, local heads, phase, projection precision, FP8 KV, dimensions) |
| `batch_size`, `prefix`, `x` | prefill: requests, cached tokens and new tokens per request; decode: requests, 0 and absolute sequence length |
| `latency` | ms, GPU kernel time only; median over repetitions of the maximum across TP ranks of the repetition's GPU-busy union |
| `kernel_source` | observed module/quant-method/attention-backend witness |
| `measurement_scope`, `kv_seed_regime`, `execution_profile` | `local_compute`, `real_kv`, `full` |
| `source_sha256`, `config_sha256`, `runtime_digest`, `used_cuda_graph`, `sample_count` | provenance |

The writer (`glm53flash_attention_contract.write_parquet`) rejects duplicate
physical keys, a second runtime/source identity, a checkpoint with two
configurations and a phase that mixes CUDA-graph modes. Per-row sample ranges,
CV, IndexPool regime, repetition maxima and the timing diagnostics are kept in
the adjacent evidence JSON; `collection_meta.yaml` names the timing method per
phase and each attempt's context limit and capacity knobs.

## Running

```bash
python -m collector.glm53flash_attention_launch --backend sglang --checkpoint nvfp4 --tp 1 \
  --config src/aisimulate_core/model_configs/nvidia--GLM-5.3-Flash-NVFP4_config.json \
  [--context-class long] [--only-sets ...] [--smoke validation-sglang|validation-vllm|long] \
  --attempt <local dir> --remote-attempt <shared dir> --remote-source <shared collector copy> \
  --remote-model <checkpoint dir> --image <sqsh> --source-commit <sha>
sbatch <shared dir>/dryrun.sbatch   # CPU: framework, manifest, geometry, native args
sbatch <shared dir>/run.sbatch
python -m collector.glm53flash_attention_contract finalize <attempt>... \
  --output .../glm53_attention_module_perf.parquet --evidence <evidence.json>
```

Sources: vLLM `models/glm5next/common/{attention,model}.py`,
`models/glm5next/nvidia/sparse_indexer.py`, `model_executor/layers/mla.py`,
`model_executor/layers/attention/mla_attention.py`,
`v1/worker/gpu/{model_runner,cudagraph_utils}.py`,
`compilation/breakable_cudagraph.py`, `v1/core/sched/scheduler.py` (all at
v0.31.0, `db9527a4`); SGLang `models/glm5_next.py`, `models/deepseek_v2.py`,
`layers/attention/dsa/dsa_indexer_kpool.py`, `layers/communicator*.py`,
`benchmark/one_batch.py`, `model_executor/runner/decode_cuda_graph_runner.py`.
The adapters are original code that call these Apache-2.0 sources; no code is
copied.
