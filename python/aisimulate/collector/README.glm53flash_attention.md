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
| vLLM | `0.30.0+glm53tail.eb4704514fdf` | `vllm/vllm-openai@sha256:4864d466…dfb56` | Reviewed retained-tail overlay mounted first on `PYTHONPATH`. It changes `model_executor/layers/sparse_attn_indexer_kpool.py` (`_kpool_compress_insert` borrows each request's retained tail) and `v1/kv_cache_interface.py` (`KpoolTailSpec.uses_slot_mapping=False`), i.e. exactly the pooled-index/tail path measured here; stock 0.30.0 is never mixed in. |
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

## Execution mode (evidence)

The pinned serving configurations (FPM calibration `sglang-resolved-config.json`
and vLLM `resolved-config-node0.json`) run:

- SGLang: `cuda_graph_config.decode.backend=full` for bs 1..32 and
  `prefill.backend=disabled`, `attention_backend=dsa`, DSA prefill/decode
  `trtllm`, top-k `sgl-kernel`, page 64, FP8 KV, chunked prefill 8192.
- vLLM: `CompilationMode.NONE`, `cudagraph_mode=FULL_AND_PIECEWISE`, capture
  sizes up to 64 tokens, `FLASHINFER_MLA_SPARSE`, FP8 KV, 8192 batched tokens.

Therefore prefill is timed eagerly and decode through a CUDA graph:

- **Prefill** (`used_cuda_graph=false`): the probe fires inside the real
  framework forward of the planned step, repeats the module call (3 warmups +
  10 timed, back-to-back CUDA events; SGLang recreates `AttentionInputs` so the
  latent projection is recomputed every time) and then performs the real call.
- **Decode** (`used_cuda_graph=true`): the probe records the module's
  arguments and forward context while the framework captures its own decode
  graphs. After a real decode step replays the framework graph (witnessed via
  `DecodeCudaGraphRunner.execute` / `ModelCudaGraphManager.run_fullgraph`),
  the module is captured into its own CUDA graph from those capture-time
  arguments under the framework's capture mode, so it reads the persistent
  metadata buffers the replay just refreshed, and is replayed 3+10 times.

vLLM runs with `cudagraph_mode=FULL_DECODE_ONLY`: the same FULL uniform-decode
graphs as serving, but every prefill step eager. Serving replays prefill steps
of at most 64 tokens through breakable piecewise graphs, so vLLM prefill rows
with `batch * x <= 64` include launch overhead serving partly hides.

### Revision 2: prefill under the serving prefill CUDA graphs

On 2026-10-01 the serving deployment switched prefill to the frameworks'
breakable CUDA graphs (evidence: GLM prefill-graph A/B smoke v4): SGLang
`--cuda-graph-backend-prefill breakable --cuda-graph-max-bs-prefill 8192`
(58 token buckets), vLLM default `FULL_AND_PIECEWISE` with 62
`--cudagraph-capture-sizes` up to 8192. Revision 2 re-collects only the prefill
rows under exactly that mechanism; decode rows are carried from revision 1.

How the frameworks execute the module in graph-mode prefill:

- SGLang (`runner/prefill_cuda_graph_runner.py`,
  `runner_backend/breakable_cuda_graph_backend.py`): the transformer body is
  captured per token bucket as graph segments separated by `eager_on_graph`
  breaks. For this module the absorbed MLA method is pinned in graph mode
  (`deepseek_common/attention_backend_handler.py`), and the pooled-key indexer
  (`attention/dsa/kpool_prefill_cuda_graph.py`, reading the live batch from the
  TcPiecewise context) and the MLA BMM + attention core
  (`attention_forward_methods/forward_mla.py` `bcg_mla_bmm_then_unified_attention`)
  are eager breaks; projections, norms and o_proj replay from segments.
- vLLM (`v1/worker/gpu/cudagraph_utils.py` `run_pw_graph`,
  `compilation/breakable_cudagraph.py`): ops decorated with
  `eager_break_during_capture` -- the IndexPool indexer
  (`sparse_attn_indexer_kpool.py`) and the MLA attention op
  (`attention/mla_attention.py`) -- run eagerly against the step's forward
  context; everything else replays from captured segments.

The probe records the module's capture-time inputs for every bucket while the
framework captures, executes the real planned step (the framework replays its
own prefill graph for the padded bucket; witnessed), then captures the module
alone with the framework's own `BreakableCUDAGraphCapture` under that step's
live contexts and replays it 3+10 times (`timing_method`
`cuda_events_framework_breakable_module_graph_replay`, `used_cuda_graph=true`).
Padding to the framework bucket is therefore included. The eager breaks keep
their host launch cost, as in serving. Memory headroom for the per-target
module graphs comes from a smaller static pool (SGLang
`--mem-fraction-static 0.60`, vLLM `--gpu-memory-utilization 0.70`), a capacity
change only.

## Workload and state

`cases/base_ops/glm53flash_attention.yaml` (454 points): prefill batch
1–32 × new tokens 8–8192 × cached prefix 0–98304 (every step within the
8192-token serving budget; up to 106496 context), decode batch 1–32 × absolute
length 64–131072. Grid points bracket the IndexPool boundary (`prefix + x <=
2048` short regime selects every pool without scoring; otherwise pooled +
retained tail). Requests are real tokenized text; KV, IndexPool and tail state
are produced by the framework's own allocation and forward helpers:

- SGLang: `one_batch.load_model`, `prepare_synthetic_inputs_for_latency_test`,
  and `extend`/`decode`; continuations re-extend each request from its own
  `req_to_token` prefix (as `prepare_extend_inputs_for_correctness_test`).
- vLLM: the in-process `LLM` engine and its scheduler. Before each step the
  driver sets `max_num_scheduled_tokens` and `long_prefill_token_threshold`
  so B homogeneous requests advance in lockstep; this only decides batching.

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
| `latency` | ms; median over repetitions of the maximum across TP ranks |
| `kernel_source` | observed module/quant-method/attention-backend witness |
| `measurement_scope`, `kv_seed_regime`, `execution_profile` | `local_compute`, `real_kv`, `full` |
| `source_sha256`, `config_sha256`, `runtime_digest`, `used_cuda_graph`, `sample_count` | provenance |

The writer (`glm53flash_attention_contract.write_parquet`) rejects duplicate
physical keys, a second runtime/source identity, a checkpoint with two
configurations and a phase that mixes CUDA-graph modes. Per-row sample ranges,
IndexPool regime and repetition maxima are kept in the adjacent evidence JSON.

## Running

```bash
python -m collector.glm53flash_attention_launch --backend sglang --checkpoint fp8 --tp 2 \
  --config src/aisimulate_core/model_configs/zai-org--GLM-5.3-Flash_config.json \
  --attempt <local dir> --remote-attempt <shared dir> --remote-source <shared collector copy> \
  --remote-model <checkpoint dir> --image <sqsh> --source-commit <sha> [--smoke]
sbatch <shared dir>/run.sbatch
python -m collector.glm53flash_attention_contract finalize <attempt>... \
  --output .../glm53_attention_module_perf.parquet --evidence <evidence.json>
```

Sources: vLLM `models/glm5next/nvidia/{attention,model}.py`,
`model_executor/layers/{mla,sparse_attn_indexer_kpool}.py`,
`v1/worker/gpu/{model_runner,cudagraph_utils}.py`; SGLang
`models/glm5_next.py`, `models/deepseek_v2.py`,
`layers/attention/dsa/dsa_indexer_kpool.py`, `layers/communicator*.py`,
`benchmark/one_batch.py`, `model_executor/runner/decode_cuda_graph_runner.py`.
The adapters are original code that call these Apache-2.0 sources; no code is
copied.
