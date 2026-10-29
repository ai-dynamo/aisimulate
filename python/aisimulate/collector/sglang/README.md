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

- Stock SGLang collectors: v0.5.14

```bash
docker run -itd --shm-size 32g --gpus all --ipc=host --network=host --name sglang lmsysorg/sglang:v0.5.14-cu130
```

- WideEP MoE collector: the independent v0.5.10 image declared in
  `../framework_manifest.yaml`; WideEP MLA is not registered for stock v0.5.14.
- DeepSeek model config (or use dummy weights)

## GLM AgentX collection branch

`simonec/glm-agentx-op-collection` is a collection-only development branch.
Its SGLang generation sweep replaces the general sweep with the AgentX target
below. It preserves the prefill batch and context grids, other backends, and
the output schema. Do not merge this narrowed sweep as a general default.

- Checkpoint: `nvidia/GLM-5.2-NVFP4`; B300 (SM103), TP8 geometry with 8 local
  attention heads, FP8 KV, BF16 attention/projections, TRTLLM DSA backend.
- Runtime: SGLang 0.5.20, source
  [`94602c9c2b7cbdb8efd5c52802dac6a1c180089e`](https://github.com/sgl-project/sglang/tree/94602c9c2b7cbdb8efd5c52802dac6a1c180089e).
  The model-scoped manifest pin and image digest must agree with the pipeline.
- Decode op batches: `4, 8, 16, 32, 52`. These bracket the sampled AgentX
  running-request maximum of 13 with the replay's `nextn=3` approximation.
- Past-KV lengths: `1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096,
  8192, 16384, 32768, 65536, 131072, 262144, 393216, 524288, 655360, 720896`.
- Both `dsa_generation_module` and `dsa_generation_module_skip_indexer`:
  **110 measurements per variant, 220 total**. Persisted rows use `isl=1`
  and `step=past_KV`, so the predictor's context coordinate is `past_KV+1`.
- Generation budget: `52 * 720896 = 37486592` tokens. The declaration admits
  these cases; GPU execution must establish feasibility and timing validity.
  Retain classified failures rather than treating absent rows as successful.

The checked SGLang APIs changed between 0.5.14 and 0.5.20. The module collector
uses the latter's published runtime context, resolved config, `ParallelState`,
GEMM initialization, request extension range, and explicit decode token input.
Source citations at those calls identify the serving contracts; the older
0.5.14 path remains available. This collection does not validate native MTP
verification or the replay's equivalent autoregressive approximation.

Decode measurements capture the module's **exact requested shape** in a CUDA
graph when that batch is within SGLang's graph-covered range. Batch 52 is
eligible even though the non-speculative serving buckets bracket it with
48 and 56. The microbenchmark creates its own graph at 52; it does not pad the
tensors or label a padded batch as 52. Disabling graph padding restricts
eligibility to the explicit capture list, and disabling graphs selects eager
timing. This follows the serving eligibility check in
[SGLang's decode runner](https://github.com/sgl-project/sglang/blob/94602c9c2b7cbdb8efd5c52802dac6a1c180089e/python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py#L675).

Use [agentx_dsa_shards.yaml](agentx_dsa_shards.yaml) with auto-collector commit
`f3ce58a3e20c29c9b31221073182956efd556324`, which includes the model-filter
launcher fix, frozen on a run branch. Set
`AISIM_SHARD_PLAN` to this file's SHA-pinned raw GitHub URL, and set:

```yaml
AISIM_REPO: ai-dynamo/aisimulate
AISIM_REVISION: <signed source commit>
AISIM_FRAMEWORKS: sglang:0.5.20
AISIM_SYSTEMS: b300_sxm
AISIM_OPS: dsa_generation_module,dsa_generation_module_skip_indexer
AISIM_MODEL_FILTER: nvidia/GLM-5.2-NVFP4
AISIM_GPU_NUM: "1"
AISIM_CLUSTER_OVERRIDE: computelab-aus-01.nvidia.com
AISIM_SHARDS_PER_JOB: "1"
AISIM_BACKEND_DOCKER_IMAGE: lmsysorg/sglang:v0.5.20-cu130@sha256:06e4f2ed21afde4ff513cda65070124e727ba23ccaeff7712b8c40e1097d611f
RUN_JOBS_FOR_MISSING_DATA_ONLY: "0"
CREATE_GITHUB_PR: "0"
```

Each shard reserves one B300 and measures one attention module with local TP8
dimensions; this is not an eight-rank communication benchmark. Separate shard
jobs preserve `--model-path` before model deduplication. The exact case filter
selects only the requested heads and precision. Leave collection limits empty.
Validate the two collection summaries and all 220 row identities before data
publication; a green parent pipeline is not evidence of complete coverage.

## Execution Modes

Stock diagnostics and the independent WideEP MoE collector use these modes:

### Mode 1: Direct Diagnostic Execution

Run scripts directly with command-line arguments for single GPU diagnostics.
The MLA command exercises the stock 0.5.14 module implementation; it is not a
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

# Run stock DSA module operators in the v0.5.14 container
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
output_path = "/aiconfigurator/collector_output/h100_sxm/sglang/0.5.14/"
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
