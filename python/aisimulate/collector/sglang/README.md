# SGLang Operator Performance Collection Tools

This directory contains scripts for collecting performance data of **Prefill-Decode (PD) disaggregated** DeepSeek model operators for the SGLang framework.

## Purpose

These scripts are designed to collect operator-level performance data for DeepSeek models in a PD-disaggregated serving architecture. They focus on the three largest modules in DeepSeek models:

1. **Attention (MLA)**: Multi-head Latent Attention mechanism
2. **MoE**: Mixture of Experts layers
3. **Shared Expert (MLP)**: Shared Multi-Layer Perceptron layers

The collected performance data can be used for performance modeling, scheduling optimization, and resource allocation in disaggregated serving systems.

## Overview

- **collect_mla_module.py**: Collects performance data for MLA and DSA attention module operators
- Large-EP expert compute is measured by
  `../wideep/sglang/collect_deepep_moe.py` into the `moe_ep` table; measured
  communication is collected by `../wideep/sglang/collect_moe_a2a.py`.
  Stock `moe_perf` remains only the modeled fallback when no matching
  dedicated `moe_ep` row is available.

## Requirements

- Stock SGLang collectors: v0.5.21

```bash
docker run -itd --shm-size 32g --gpus all --ipc=host --network=host --name sglang lmsysorg/sglang:v0.5.21-cu130
```

- WideEP MoE collector: the independent v0.5.10 image declared in
  `../framework_manifest.yaml`; WideEP MLA is not registered for stock v0.5.21.
- DeepSeek model config (or use dummy weights)

## Module collection contracts

MLA/DSA decode getters resolve an exact inner-shape manifest before queueing,
following the MSA collector pattern. A counted memory filter rejects shapes
whose page-rounded KV footprint exceeds 90% of live device memory. The estimate
includes every loaded layer, pool padding, native raw TRTLLM or scaled FlashMLA
FP8 storage, and index-K buffers. Index buffers are conservatively counted for
all loaded layers, including skip-indexer runs: their timed-layer toggle does
not remove the runner's cache allocation. Without a visible CUDA device,
planning retains all shapes. The retained manifest is part of task identity;
workers execute that plan without rebuilding or filtering it. Dense MLA context
now carries the same exact manifest, with its expanded full-history range and
visible activation estimate. DSA prefill keeps its existing separate contract.

This filter is an admission estimate, not a guarantee of runtime fit. Model
weights, request tables, graphs and workspaces also consume memory. SGLang
0.5.21 [caps the requested KV pool at profiled capacity](https://github.com/sgl-project/sglang/blob/e00930c5489053f26d86b179cee0d087f846acbb/python/sglang/srt/mem_cache/kv_cache_configurator.py#L2264),
and each queued decode point checks its page-rounded requirement against that
actual pool before allocating requests. Remaining capacity or model-context
errors stay visible and fail the group's completeness check. GPU qualification
is still required for new coverage.

The module collector uses the pinned SGLang **0.5.21** runtime in
[framework_manifest.yaml](../framework_manifest.yaml), including native input
relay, projection and cache-history handling. DSA prefill is measured eagerly;
one-token prefill extensions remain excluded pending native graph support.

Decode uses an exact-shape module CUDA graph when the requested batch is within
the serving graph coverage. Batch 52 is eligible between the serving buckets 48
and 56; the microbenchmark captures 52 directly. Serving may execute the padded
56-request bucket, whose padding overhead is not included in the row keyed 52.
Disabling graph padding requires explicit bucket membership. This follows the
[native decode eligibility check](https://github.com/sgl-project/sglang/blob/e00930c5489053f26d86b179cee0d087f846acbb/python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py#L714).

## Agentic MLA and sparse-module coverage

`mla_module.yaml` declares a shared dense-MLA context sequence axis
(`mla_context_sequence_lengths`) and token budget (`context.mla_max_tokens`).
They declare full-history queries through 1,048,576 tokens on vLLM, SGLang
and TRT-LLM; native model and backend limits still apply. Sparse modules retain their existing new-token/chunk axes; their
cached histories use the prefix and decode axes instead. TRT-LLM DSA and
TRT-LLM/vLLM MSA now include 4K through 1M prefix anchors, and their decode
axes reach 1,048,575 cached tokens. SGLang's existing DSA/MSA history coverage
and measured GLM decode execution path remain unchanged.

For an exact dense-MLA context manifest, SGLang sizes its native prefill chunk
capacity from the largest planned query-token batch. This permits full-history
module forwards beyond the default serving chunk; an explicit chunk override
still takes precedence. DSA keeps its existing chunked-prefill configuration.
Memory planning resolves the same bundled or local model config as execution.

The standalone MLA adapters share the expanded `mla.yaml` grid. SGLang's
mock runner now advertises the actual case length. TRT-LLM's standalone
harness still uses its fixed DeepSeek Scenario RoPE capacity (163,840); larger
queued shapes raise a classified native-limit error. Full TRT-LLM modules use
the real model's positional limit, size token/sequence capacities for each
case, and evict the old cached layer when those capacities change. Expanding
a sweep never extends a checkpoint's RoPE model definition.

Admission uses 90% of the collecting GPU's total memory. Single-layer
TRT-LLM/vLLM estimates count page-rounded caches, cached source tensors and
visible attention activations and top-k buffers; TRT-LLM estimates also include
DSA logits tiles and MSA index-score buffers. Its MSA collector retains the
native double-pool allocation and spare pages. SGLang estimates all loaded KV
layers. The native workspace, model weights, allocator fragmentation and backend limits can still
reject an admitted case. CPU coverage and memory-boundary tests establish the
plan only; new long-history paths require GPU qualification. The larger
context budget also admits larger batches at existing shorter anchors, so
collection-cost estimates must count those cells as well as new lengths.

## Long-history decode defaults

The shared MLA/DSA grid preserves existing batches and lengths, adds batch
`52` and past-KV lengths `262144, 393216, 524288, 655360, 720896`, and raises
the aggregate token budget to `37486592` (`52 * 720896`). MSA shares those
SGLang history axes but keeps its `33554432` budget. These defaults apply to
all declared models and precisions. Model-context violations remain visible
runtime errors, and a plan admitted by the memory estimate still needs native
GPU qualification. Model-specific campaigns and packaged measurements are
maintained separately from these shared collector defaults.

## Execution Modes

Stock diagnostics and the independent WideEP MoE collector use these modes:

### Mode 1: Direct Diagnostic Execution

Run scripts directly with command-line arguments for single GPU diagnostics.
The MLA command exercises the stock 0.5.21 module implementation; it is not a
registered or accepted WideEP MLA collector:

```bash
# Attention (MLA/DSA Module)
python collect_mla_module.py --mode context --attn-type mla

```

**Arguments:**
- `--device`: CUDA device (e.g., `cuda:0`, `cuda:1`)
- `--output-path`: Directory to save performance data files

### Mode 2: Framework Execution (collect.py)

Use the `collect.py` framework for integrated collection with other operators:

```bash
cd /path/to/collector/

# Run all stock SGLang operators
python collect.py --backend sglang

# Run stock DSA module operators in the v0.5.21 container
python collect.py --backend sglang --ops dsa_context_module dsa_generation_module

# Run the large-EP MoE compute operator separately in the v0.5.10 container
python collect.py --backend sglang --ops moe_ep

# Mixed: operations run in order; cases within each operation use the GPU pool
python collect.py --backend sglang --ops mla_bmm_gen_pre dsa_context_module
```

**Selected operators (`moe_ep` requires its separate v0.5.10 run):**

| Category | Operator | Description |
|----------|----------|-------------|
| Kernel | `gemm` | GEMM matrix multiplication |
| Kernel | `mla_context` | MLA prefill phase |
| Kernel | `mla_generation` | MLA decode phase |
| Kernel | `mla_bmm_gen_pre` | MLA BMM gen pre |
| Kernel | `mla_bmm_gen_post` | MLA BMM gen post |
| Kernel | `moe` | MOE operator |
| Kernel | `attention_context` | Standard Attention prefill |
| Kernel | `attention_generation` | Standard Attention decode |
| Module | `dsa_context_module` | DSA module prefill (DeepSeek-V3.2, GLM-5) |
| Module | `dsa_generation_module` | DSA module decode (DeepSeek-V3.2, GLM-5) |
| Wideep | `moe_a2a` | Large-EP MoE dispatch/combine communication |

**Note:** Requested operators run sequentially. Cases within one operator are
distributed across the selected GPU workers. Module-level operators use
subprocess-based GPU isolation (via `CUDA_VISIBLE_DEVICES`) to prevent
NCCL/distributed initialization conflicts.

## General Configuration

Direct scripts save staging results to one output directory. Keep that staging
directory outside the packaged perf-data tree; `collect.py` finalizes accepted
files as parquet, and each file is then published to the family selected by
`collector/op_backend_catalog.yaml`.

Modify `output_path` in each direct script to your desired staging location:
```python
output_path = "/aiconfigurator/collector_output/h100_sxm/sglang/0.5.21/"
```


## 1. Stock Attention Module Diagnostics (`collect_mla_module.py`)

This direct script retains historical `wideep_*` output filenames for local
diagnostics, but those files are not supported WideEP 0.5.10 artifacts. Use the
registered stock MLA/DSA ops through `collect.py` for accepted collection.

### Features
- Unified MLA (DeepSeek-V3) and DSA (DeepSeek-V3.2, GLM-5) benchmarking
- SM-gated precision sweep (bfloat16 + fp8 on Hopper+)
- Tests various batch sizes, sequence lengths, and head numbers
- Supports both prefill and decode phases
- Optional dummy weights mode for fast testing

### Usage

#### Direct Mode
```bash
# MLA context phase
SGLANG_LOAD_FORMAT=dummy SGLANG_TEST_NUM_LAYERS=2 \
    python collect_mla_module.py --mode context --attn-type mla

# DSA generation phase
SGLANG_LOAD_FORMAT=dummy SGLANG_TEST_NUM_LAYERS=2 \
    python collect_mla_module.py --mode generation --attn-type dsa
```

#### Environment Variables
- `DEEPSEEK_MODEL_PATH`: Path to DeepSeek model 
- `SGLANG_LOAD_FORMAT`: Load format, set to `dummy` to skip weight loading
- `SGLANG_TEST_NUM_LAYERS`: Load only specified number of layers (with dummy mode)
- `SGLANG_TEST_LAYER`: Layer index to test (default: 0)

### Test Parameters
The script automatically tests the following configuration combinations:
- Attention backends: `flashinfer`, `fa3`
- Head numbers: 128, 64, 32, 16
- Batch sizes: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024
- Sequence lengths: 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384

### Output
Direct diagnostic results use the historical filenames:
- `wideep_context_mla_perf.txt` / `dsa_context_module_perf.txt`: Prefill phase performance data
- `wideep_generation_mla_perf.txt` / `dsa_generation_module_perf.txt`: Decode phase performance data

Output format:
```csv
framework,version,device,op_name,kernel_source,model,architecture,mla_dtype,kv_cache_dtype,gemm_type,num_heads,batch_size,isl,tp_size,step,latency
```

## 2. Large-EP communication collection

Use `collector/wideep/sglang/collect_moe_a2a.py` for measured dispatch/combine
communication. Use `collector/wideep/sglang/collect_deepep_moe.py` for
Large-EP local expert compute; it publishes the dedicated `moe_ep` table.
Stock `moe_perf` is a modeled fallback only when a matching `moe_ep` row is
not available.

## GLM agentic decode campaign

`agentx_dsa_shards.yaml` selects the full-indexer and skip-indexer DSA module
groups for `nvidia/GLM-5.2-NVFP4`, TP8, BF16 projections and FP8 KV on B300.
The auto-collector also requires `AISIM_MODEL_FILTER=nvidia/GLM-5.2-NVFP4`
and `AISIM_SHARDS_PER_JOB=1`. Shared sweep defaults belong in
`cases/base_ops/mla_module.yaml`; this campaign file only selects the groups.

The SGLang 0.5.21 collection used AISimulate commit
[`0bba6709ed4ef3a0fcae878d2ad2fb61bbd11a04`](https://github.com/ai-dynamo/aisimulate/commit/0bba6709ed4ef3a0fcae878d2ad2fb61bbd11a04),
with 250 shapes per group, through past-KV length 720,896. Reproduction must
pin that collector revision, the runtime image digest recorded in the data
sidecar, and the campaign's B300 capacity. Current defaults may generate a
different plan. See collection jobs
[full-indexer](https://gitlab-master.nvidia.com/dl/ai-dynamo/aisim-auto-collector/-/jobs/523211485)
and [skip-indexer](https://gitlab-master.nvidia.com/dl/ai-dynamo/aisim-auto-collector/-/jobs/523211486).
The packaged 0.5.21 table preserves previous rows and appends both collection
events with their original source revision and plan hashes.
