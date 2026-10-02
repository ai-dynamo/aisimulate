<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DeepSeek-V4.1 operator profiles on vLLM

Two standalone torchrun producers measure the DSV4.1 data contract defined for SGLang
(`collector/sglang/dsv41_contract.py`, `collector/sglang/README.dsv41.md`) on vLLM's native
modules, so one perf-database table (`dsv41_module_perf`) describes both runtimes with the
same physical keys:

| producer | components | native boundary |
|---|---|---|
| `collector/vllm/dsv41_isolated_runner.py` | `linear`, `engram`, `mhc`, gemm/moe/nccl baselines | `DeepseekV4MLP.{gate_up_proj,down_proj}`; `ParallelEngramEmbedding.lookup` + `Engram.forward` with the TP all-gather excluded; both `mhc_shifted_post_pre` sites of layer 2; `GateLinear`, `ParallelLMHead`, `RoutedExperts.forward_modular`, NCCL all-reduce |
| `collector/vllm/dsv41_attention_runner.py` | `attention` | forty `DeepseekV4FlashMLAAttention` layers with their serving KV specs/builders; `attn.forward` timed per representative layer with the `wo_b` TP all-reduce after the end event; real prefix KV for cached prefill and decode |

Pin: vLLM tag `v0.30.0` (commit `ced6857a`, image `vllm/vllm-openai:v0.30.0`); perf-database
version key `0.30.0` under backend `vllm`. Only the `full` execution profile exists on vLLM
(`decoder_bounded` is an SGLang-only replay). Rows carry the SDK's geometry for backend
`vllm` (`build_manifest(tp, False, "vllm")`): identical to the SGLang geometry except the
SDK assigns `fmha_quant_mode="bfloat16"` to vLLM attention — the runtime actually serves
`kv_cache_dtype=fp8_ds_mla` (`models/deepseek_v41/attention.py:136-152`); the receipt records
the runtime identity, the SDK key is what the consumer looks up.

## Serving identity on sm90 (H20, 2026-10-02)

Recorded in every receipt (`baseline_identity`, `linear_identity`, `engram_identity`,
`native_pool`) and in the rows' `kernel_source`:

- FP8 block[32,32] linears → `ModelOptLinearMethod[MarlinMxfp8LinearKernel]` (W8A16 Marlin;
  the FlashInfer/DeepGEMM MXFP8 kernels need SM100). `wo_a` (bmm) → `EmulationMxfp8LinearKernel`
  (dequantized BF16 `torch.bmm`).
- Experts (`expert_dtype=fp4`) → `Mxfp4MoEMethod` backend `MARLIN` / `MarlinExperts` (W4A16);
  the per-rank intermediate is rounded to 128 (TP4: 576 → 640). Baseline `moe_dtype` is
  `w4a16_mxfp4_marlin`.
- Engram tables GPU-resident (`EngramConfig(cpu_offload=False)`; serving default is host-resident
  via `VLLM_PLE_CPU_OFFLOAD=1`), TP **head**-sharded with an all-gather (SGLang row-shards and
  all-reduces); the lookup is the Triton `_engram_lookup_kernel`, the gate
  `_fused_engram_post_wkv_kernel`.
- mHC: one fused post + pre-mix + RMSNorm op per site (`mhc_shifted_post_pre` → TileLang
  `mhc_fused_post_pre_delayed_tilelang` on sm90; DeepGEMM `mega_mhc` only on SM100). The two
  sites of layer 2 are summed, comparable to SGLang's two `_hc_mix_and_combine` + `hc_post`
  sites with the norm fused (v0.5.21).
- Attention backend `FLASHMLA_SPARSE_DSV41`: compressed KV `MLAAttentionSpec` (block 64, 584 B/token,
  alignment 576), indexer K cache (`DEEPSEEK_V41_INDEXER`, block 64), SWA
  `SlidingWindowMLASpec` (block 32), compressor state `CircularBufferSpec` (block 8).
  The indexer skips scoring when `max_seq_len // compress_ratio <= index_topk`
  (`attention.py:1195-1217`): rows with ≤512 (ratio 1) / ≤1024 (ratio 2) tokens measure no
  top-k kernels — a serving property, not a collector choice.

Dummy weights: `initialize_dummy_weights` runs vLLM's native float initializer and fills what
the dummy loader leaves uninitialized (`weight_utils.py:1348-1354`): e8m0 scales = 127, packed
MXFP4 experts and the fp8 Engram tables = NaN-free random bytes (`WEIGHT_INITIALIZER`).

## Running (H20 box, docker)

```bash
# inside vllm/vllm-openai:v0.30.0 with the checkout at /harness and the workspace at /work
export CUDA_MPS_PIPE_DIRECTORY=/nonexistent-no-mps VLLM_USE_V2_MODEL_RUNNER=1 PYTHONPATH=/harness/python/aisimulate
export DSV41_PRIVATE_CACHE=/work/vllm/cache DSV41_ALLOCATION_ID=<run> DSV41_LAUNCH_IMAGE_SHA256=<index digest hex>
python3 -m torch.distributed.run --standalone --nproc-per-node=2 -m collector.vllm.dsv41_isolated_runner \
  --plan plan.json --manifest manifest-tp2-full.json --model-path /work/ckpt-meta \
  --prompt-file /work/corpus.txt --output raw/iso-tp2 --runtime-digest sha256:<index digest>
python3 -m torch.distributed.run --standalone --nproc-per-node=2 -m collector.vllm.dsv41_attention_runner \
  --plan attn-plan.json --manifest manifest-tp2-full.json --workloads workloads.json \
  --model-path /work/ckpt-meta --prompt-file /work/corpus.txt --output raw/attn-tp2 --runtime-digest sha256:<index digest>
# admission (no GPU) and publication
python -m collector.vllm.dsv41_attention_runner --plan … --manifest … --workloads … --output raw/attn-tp2 \
  --runtime-digest … --admit raw/attn-tp2/admitted.parquet
python -m collector.sglang.dsv41_publish --backend vllm --system h20_3e --systems-root src/aisimulate_core/systems \
  --isolated raw/iso-tp2 raw/iso-tp4 --attention raw/attn-tp2 raw/attn-tp4 \
  --image vllm/vllm-openai:v0.30.0 --torch 2.13.0+cu130 --nccl 2.30.7
```

Plans use the SGLang schemas (`dsv41.isolated-collection.v1` with `moe_backend: marlin`,
`dsv41.attention-collection.v1` with this module's `INPUT_METHOD`); `source_pins` cover
`REQUIRED_SOURCES` / `ATTENTION_SOURCES` hashed inside the image. The runtime NCCL
(`ncclGetVersion` of the mapped `libnccl.so.2`) is 2.30.7 while the torch header reports 2.29.7.

## Measurement notes

- The recorder times one representative layer per (phase, geometry) — the first layer carrying
  it — like the SGLang recorder; rows are never sums over layers that share a key.
- The output all-reduce is issued as plain NCCL `torch.distributed.all_reduce` after the end
  event. vLLM's own device communicator uses a FlashInfer two-shot all-reduce
  (`VLLM_ALLREDUCE_USE_FLASHINFER=1`) that spins in-kernel on the peer rank; left inside the
  layer sequence it smeared rank skew into neighbouring intervals (28 ms on a 0.6 ms layer).
- Small shapes are host-bound on H20 (≈1.3 ms of Python per layer vs 0.6 ms of kernels); the
  event interval includes that idle, exactly as in the SGLang producer.
- These producers are standalone (no `collect.py` OpEntry); their hash closures live in
  `collector/hash_closures.yaml` and `provenance.STANDALONE_COLLECTOR_MODULES`.
