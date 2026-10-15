<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GLM-5.3-Flash sparse-MLA + IndexPool attention collection

Standalone collectors that load the real GLM-5.3-Flash checkpoint through the
framework's own model builder and time one NoPE sparse-MLA attention layer over
real KV/IndexPool state. They produce `glm53_attention_module_perf.parquet`,
the only GLM-specific measured table. Every other GLM operator uses the generic
GEMM/MoE/KDA/mHC/collective tables. The flow follows the DeepSeek-V4.1
precedent in `sglang/README.dsv41.md`.

## Pinned runtimes

| Backend | Version directory | Runtime |
| --- | --- | --- |
| vLLM | `0.31.0` | stock `vllm/vllm-openai:v0.31.0` (tag commit `db9527a4`) |
| SGLang | `0.5.20` | stock `lmsysorg/sglang:v0.5.20` (tag commit `94602c9c`) |

Image digests are pinned in `framework_manifest.yaml`. Checkpoints are
`zai-org/GLM-5.3-Flash` (FP8 block-128) and `nvidia/GLM-5.3-Flash-NVFP4`.
Deployments are TP2/TP4 on both backends plus vLLM NVFP4 TP1 (the FP8 weights
do not fit one GB300).

## Measured boundary

`Glm5NextMLAAttention.forward` (vLLM) / `DeepseekV2AttentionMLA.forward`
(SGLang) of layer 3: latent projections and norms, q_b projection, absorbed
kv_b BMMs, the complete IndexPool indexer, FP8 latent KV write, the sparse MLA
kernel and the local `o_proj` GEMM. The attention-output all-reduce, mHC, the
input norm and the per-forward metadata build are separate model ops. Layer 3
represents all 11 sparse-MLA layers;
`contract.representative_layer_is_uniform` rejects a checkpoint where that does
not hold.

## Execution and timing

Both phases run under the frameworks' serving CUDA graphs: breakable prefill
graphs up to 8192 tokens and full decode graphs for batch 1-32. The probe
records the module's capture-time inputs while the framework captures, executes
the real planned step, then captures the module alone with the framework's own
capture mechanism under that step's live contexts, so bucket padding is
included.

Latency is GPU kernel time only (`glm53flash_attention_runtime.KernelTimer`):
the module graph is replayed `warmup + iterations` (3 + 10) times under the
torch profiler; each repetition's latency is the union of its CUPTI GPU-busy
intervals; the row latency is the median over repetitions of the maximum
across TP ranks.

Stock vLLM 0.31.0 mishandles prefill chunks that start off the 4-token
IndexPool grid (`models/glm5next/nvidia/sparse_indexer.py`
`_kpool_compress_insert`). Every planned vLLM prefill chunk therefore starts on
a multiple of 4. The launcher refuses an unaligned plan and the runner raises
`KpoolAlignmentError` before such a step.

## Workload

`cases/base_ops/glm53flash_attention.yaml` (463 points). The regular class uses
the serving context limit 131079: prefill batch 1-32 x new tokens 8-8192 x
cached prefix 0-98304, and decode batch 1-32 x absolute length 64-131072. The
long class uses limit 1048576 for B=1 rows at 256K, 512K and about 1M. One
attempt measures one context class. Request token ids come from the seeded
generator `glm53flash_attention_tokens.py`. KV, IndexPool and tail state are
produced by the framework's own forward helpers.

A deployment's plan may be split across attempts with disjoint `--only-sets`
(capacity knobs such as `--vllm-gpu-memory-utilization`,
`--sglang-mem-fraction` and `--allocator-max-split-mb` are frozen per
manifest); `finalize` admits them only if they cover every planned key exactly
once, with `--partial` for a failed attempt's completed targets and
`--classified-failures` for keys that failed reproducibly.

## Data contract

`data/<system>/glm53_attention/<backend>/<version>/glm53_attention_module_perf.parquet`
under a systems root, read by `crates/core/src/perfmodel/perf_database/glm53flash.rs`
from the exact requested runtime only:

| Column | Contract |
| --- | --- |
| `component` | `attention` |
| `geometry` | canonical sorted compact JSON of the model's `Glm53Attention` body without `name`/`measured` |
| `batch_size`, `prefix`, `x` | prefill: requests, cached tokens, new tokens per request; decode: requests, 0, absolute sequence length |
| `latency` | ms, GPU kernel time as above |
| `kernel_source` | observed module/quant-method/attention-backend witness |
| `measurement_scope`, `kv_seed_regime`, `execution_profile` | `local_compute`, `real_kv`, `full` |
| `source_sha256`, `config_sha256`, `runtime_digest`, `used_cuda_graph`, `sample_count` | provenance |

The writer (`glm53flash_attention_contract.write_parquet`) rejects duplicate
physical keys, mixed runtime/source identities and mixed CUDA-graph modes
within a phase.

## Running

```bash
python -m collector.glm53flash_attention_launch --backend vllm --checkpoint nvfp4 --tp 1 \
  --config src/aisimulate_core/model_configs/nvidia--GLM-5.3-Flash-NVFP4_config.json \
  [--context-class long] [--only-sets ...] \
  --attempt <local dir> --remote-attempt <shared dir> --remote-source <shared collector copy> \
  --remote-model <checkpoint dir> --image <sqsh> --source-commit <sha>
sbatch <shared dir>/run.sbatch
python -m collector.glm53flash_attention_contract finalize <attempt>... \
  --output .../glm53_attention_module_perf.parquet
```

Sources: vLLM `models/glm5next/common/{attention,model}.py`,
`models/glm5next/nvidia/sparse_indexer.py`, `model_executor/layers/mla.py`,
`model_executor/layers/attention/mla_attention.py`,
`v1/worker/gpu/{model_runner,cudagraph_utils}.py`,
`compilation/breakable_cudagraph.py` (v0.31.0); SGLang `models/glm5_next.py`,
`models/deepseek_v2.py`, `layers/attention/dsa/dsa_indexer_kpool.py`,
`benchmark/one_batch.py`, `model_executor/runner/decode_cuda_graph_runner.py`
(v0.5.20). The adapters are original code that call these Apache-2.0 sources;
no code is copied.
