<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Performance-model limitations

## TensorRT-LLM eager host execution

Op-level GPU costs do not represent all host-side submission gaps and cross-rank
waiting in eager prefill or mixed execution. The current ordinary op-level
configuration does not carry a generic measured eager/graph CPU-dispatch timing
contract. Supplying `cuda_graph_reserved_bytes` changes cache capacity; it does
not select or measure host execution latency.

The [GPT-OSS trace evidence](../../python/aisimulate/collector/trtllm/gym-gpt-eager-20260918.md)
records this gap for its pinned runtime and workload. Its observed client TTFT
residual cannot be treated as a universal constant or charged to the shared MoE
kernel table: graph decode has a different execution boundary, and queue/frontend
time is separate. Preserve the exact runtime, phase, capture coverage, and timing
boundary when interpreting those measurements.

Whole-forward measurements can encode their recorded execution regime, but do
not establish behavior after changing that regime. Likewise, a regression may
fit observed aggregate behavior without an explicit graph/capture input. Neither
route makes unmeasured execution-policy changes qualified.

## Mistral3 images

Mistral3/Pixtral modeling supports images and the final vision feature layer
(`vision_feature_layer=-1`). Video inputs and other feature-layer selections
raise an error. Requests with no image dimensions or image-token override remain
text-only, including with the default image count of one. Image workloads require
both `image_height` and `image_width`;
an image-token count alone cannot determine the row separators in the prompt.

The Mistral Medium 3.5 configuration is not bundled. Supply its Hugging Face model
ID to download the configuration, or a local directory containing `config.json`.

Images larger than the configured `vision_config.image_size` are resized to
that maximum side while preserving aspect ratio. Patch grids round up to the
patch/merge stride. Each merged row adds one prompt separator or end token,
which contributes to the decoder context but not the encoder embeddings.
For example, a 70×70 image with patch size 14 and merge size 2 gives 36 encoder
patches, 9 embeddings, and 12 context tokens.

## Per-layer checkpoint precision

For Nemotron-H, the SDK reads the precision of each GEMM projection (Mamba,
attention, dense MLP, shared experts, latent projections) and of the routed
experts from the checkpoint's per-layer metadata (`quantized_layers`,
`config_groups`, exclusions). Each precision is priced for its own count of
layers, and the MoE dispatch ops follow the routed-expert split. Other model
families keep one precision per op kind.

- A projection that a MIXED_PRECISION listing leaves out is BF16, the rule
  vLLM and SGLang apply. Other metadata leaves it on the global mode.
- An excluded module, or a module inside an excluded parent, is BF16.
- On non-Blackwell systems, NVFP4 runs weight-only (`nvfp4_wo`).
- An explicit `gemm_quant_mode` or `moe_quant_mode` prices every projection
  or routed expert with that one mode.
- Loading fails when a module has conflicting or unsupported algorithms, or
  when the modules of one fused op, such as Q, K and V, resolve to different
  modes.

Known limits:

- The routed-expert precision of a layer comes from expert 0.
- An FP8 projection is dynamic FP8 only when the global mode is dynamic FP8.
- A checkpoint that lists `W4A16_NVFP4` layers, such as Nemotron 3.5
  Lightning, keeps the global modes. No packaged table has `w4a16_nvfp4` MoE
  rows yet. Only the vLLM 0.30.0 b200 and gb200 tables and the vLLM 0.31.0
  b200 table have `w4a16_nvfp4` GEMM rows.

## Other scoped boundaries

- [Support](support-matrix.md) distinguishes engine queries from Replay and accuracy.
- [Whole-forward](methods/whole-forward.md) documents exact identity and query-domain failures.
- [Memory](memory.md) distinguishes physical payload/resource estimates from observed allocations.
- [DeepSeek-V4.1](models/deepseek-v41.md) separates analytical and measured execution contracts.
- [Replay features](../replay/features.md) records supported scheduler/cache/topology combinations.

These are current modeling boundaries. Completed implementation plans and
historical PR checklists are not part of this guide.
