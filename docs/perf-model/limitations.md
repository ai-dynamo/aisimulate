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

For Nemotron-H, Qwen3.5 (including Qwen3.6 and Qwen3.8) and Gemma 4, the SDK
reads the precision of each GEMM projection and of the routed experts from the
checkpoint's per-layer metadata (`quantized_layers`, `config_groups`,
exclusions). The result depends on the checkpoint and not on the model name.
Other model families keep one precision per op kind, with family-specific BF16
exclusions.

| Family | Projections priced per layer |
| --- | --- |
| Nemotron-H | Mamba `in_proj` and `out_proj`, attention QKV and output, dense MLP, shared experts, `fc1`/`fc2` latent projections |
| Qwen3.5, 3.6, 3.8 | GDN `in_proj_qkvz`, `in_proj_ba` and `out_proj`, attention QKV and output, dense MLP, shared experts |
| Gemma 4 | sliding and global attention QKV and output, shared dense MLP |

All three families also price their routed experts per layer.

- A projection that a MIXED_PRECISION listing leaves out is BF16, the rule
  vLLM and SGLang apply. Metadata that is not a MIXED_PRECISION listing leaves
  it on the global mode.
- An excluded projection or routed-expert layer is BF16 in every metadata
  format. An exclusion of a parent module, such as one decoder layer, also
  excludes every projection inside it.
- On non-Blackwell systems, an NVFP4 projection or expert runs weight-only
  (`nvfp4_wo`), as the global NVFP4 mode does.
- A `W4A16_NVFP4` projection is priced as `w4a16_nvfp4`: NVFP4 weights with
  BF16 activations. Measured vLLM `w4a16_nvfp4` GEMM rows exist only for the
  Nemotron 3.5 Lightning shared-expert shapes, on b200_sxm and gb200 at vLLM
  0.30.0 and on b200_sxm at 0.31.0. Those rows measure the FlashInfer CuTe-DSL
  W4A16 kernel. vLLM 0.24.0 runs Marlin, and the vLLM source selects Marlin
  below SM100 (not measured). On any other system, version or shape, strict
  SILICON prediction fails for such a projection, and HYBRID estimates it
  from another quantization profile.
- Routed experts of a ModelOpt export that are listed `W4A16_NVFP4` are
  priced as `w4a16_nvfp4`, in uniform and mixed listings. The vLLM source
  runs them weight-only on Marlin. The vLLM MoE collector has a
  `w4a16_nvfp4` mode for Nemotron 3.5 Lightning, but no MoE table has
  `w4a16_nvfp4` rows yet. Strict SILICON prediction fails for them, and
  HYBRID estimates them from another quantization profile.

If layers give a projection or the routed experts different precisions, the
SDK prices each precision for its own count of layers. The MoE dispatch ops
follow the routed-expert split. Loading fails when one module path has
conflicting or unsupported algorithms, or when the checkpoint modules that one
op fuses disagree (for example Q, K and V, or Qwen's `in_proj_qkv` and
`in_proj_z`). The MTP layers use the decoder layers' precision.

Known limits:

- The routed-expert precision of a layer comes from expert 0 (Nemotron-H,
  Gemma 4) or from the listed experts block (Qwen). Different precisions for
  experts of one layer are not modeled.
- An FP8 projection is dynamic FP8 only when the global mode is dynamic FP8.
  Otherwise it is static FP8.

An explicit `gemm_quant_mode` prices all projections with one mode. An
explicit `moe_quant_mode` prices all routed experts with one mode. Metadata
without per-layer information keeps the global modes.

## Other scoped boundaries

- [Support](support-matrix.md) distinguishes engine queries from Replay and accuracy.
- [Whole-forward](methods/whole-forward.md) documents exact identity and query-domain failures.
- [Memory](memory.md) distinguishes physical payload/resource estimates from observed allocations.
- [DeepSeek-V4.1](models/deepseek-v41.md) separates analytical and measured execution contracts.
- [Replay features](../replay/features.md) records supported scheduler/cache/topology combinations.

These are current modeling boundaries. Completed implementation plans and
historical PR checklists are not part of this guide.
