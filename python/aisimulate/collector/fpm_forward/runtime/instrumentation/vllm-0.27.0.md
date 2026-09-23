<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# vLLM 0.27.0 source mapping

Read [the common field and lifecycle contract](README.md) with this version-specific audit. The manifest pins vLLM revision `4bdc8a788d2e2ce9165d552b3d4d8b72604626bf` and Dynamo scheduler revision `41882ae9b07232eed4850fb1daf8c958abb2556a` through exact installation-file hashes. The source notes and adapter files are hashed into each campaign bundle.

| Interpretation | Immutable source |
| --- | --- |
| Native profile allowance, initialization and post-warm-up boundary | [GPU worker](https://github.com/vllm-project/vllm/blob/4bdc8a788d2e2ce9165d552b3d4d8b72604626bf/vllm/v1/worker/gpu_worker.py): `determine_available_memory`, `initialize_from_config`, `compile_or_warm_up_model`. |
| Actual backing allocation, packed slot metadata and resolved kernel blocks | [GPU model runner](https://github.com/vllm-project/vllm/blob/4bdc8a788d2e2ce9165d552b3d4d8b72604626bf/vllm/v1/worker/gpu_model_runner.py): `_allocate_kv_cache_tensors`, `_reshape_kv_cache_tensors`, `initialize_kv_cache`. |
| KV-plane/packed/padded affine views and kernel block subdivision | [GPU attention utilities](https://github.com/vllm-project/vllm/blob/4bdc8a788d2e2ce9165d552b3d4d8b72604626bf/vllm/v1/worker/gpu/attn_utils.py): `_reshape_attention_kv_cache`, `_reshape_kv_cache`. |
| Backend-provided block axis and layout | [backend interface](https://github.com/vllm-project/vllm/blob/4bdc8a788d2e2ce9165d552b3d4d8b72604626bf/vllm/v1/attention/backend.py): `get_kv_cache_block_dim`; [FlashAttention](https://github.com/vllm-project/vllm/blob/4bdc8a788d2e2ce9165d552b3d4d8b72604626bf/vllm/v1/attention/backends/flash_attn.py) and [FlashInfer](https://github.com/vllm-project/vllm/blob/4bdc8a788d2e2ce9165d552b3d4d8b72604626bf/vllm/v1/attention/backends/flashinfer.py): `get_kv_cache_shape`, `get_kv_cache_stride_order`. Physical tensor strides retain the selected HND/NHD layout. |
| Attention group/kernel-block preparation and binding views to layers | [worker utilities](https://github.com/vllm-project/vllm/blob/4bdc8a788d2e2ce9165d552b3d4d8b72604626bf/vllm/v1/worker/utils.py): `AttentionGroup`, `prepare_kernel_block_sizes`, `bind_kv_cache`. |
| Raw spec classes, dtype, page bytes and retention | [KV cache interface](https://github.com/vllm-project/vllm/blob/4bdc8a788d2e2ce9165d552b3d4d8b72604626bf/vllm/v1/kv_cache_interface.py): `FullAttentionSpec`, `SlidingWindowSpec`, allocation tensor fields. |
| Shared pool, null reservation, free queue and retention manager | [cache manager](https://github.com/vllm-project/vllm/blob/4bdc8a788d2e2ce9165d552b3d4d8b72604626bf/vllm/v1/core/kv_cache_manager.py), [block pool](https://github.com/vllm-project/vllm/blob/4bdc8a788d2e2ce9165d552b3d4d8b72604626bf/vllm/v1/core/block_pool.py), [cache utilities](https://github.com/vllm-project/vllm/blob/4bdc8a788d2e2ce9165d552b3d4d8b72604626bf/vllm/v1/core/kv_cache_utils.py), [single-type managers](https://github.com/vllm-project/vllm/blob/4bdc8a788d2e2ce9165d552b3d4d8b72604626bf/vllm/v1/core/single_type_kv_cache_manager.py). The scheduler observation precedes request scheduling. |
| Loaded HF configuration source | [model configuration](https://github.com/vllm-project/vllm/blob/4bdc8a788d2e2ce9165d552b3d4d8b72604626bf/vllm/config/model.py): `get_config(hf_config_path or model, ..., revision, ...)`; [configuration parser](https://github.com/vllm-project/vllm/blob/4bdc8a788d2e2ce9165d552b3d4d8b72604626bf/vllm/transformers_utils/config.py): `HFConfigParser.parse`. |
| Scheduler constructor, DP rank and native self-benchmark | [Dynamo instrumented scheduler](https://github.com/ai-dynamo/dynamo/blob/41882ae9b07232eed4850fb1daf8c958abb2556a/components/src/dynamo/vllm/instrumented_scheduler.py): native initialization and `_fpm_dp_rank`. |
| Benchmark scheduler argument acceptance | [Dynamo argument consumer](https://github.com/ai-dynamo/dynamo/blob/41882ae9b07232eed4850fb1daf8c958abb2556a/components/src/dynamo/vllm/args.py#L354): `update_engine_config_with_dynamo` requires `InstrumentedScheduler` in the scheduler-class string before import. The exported class is `ObservedInstrumentedScheduler`, a native subclass. |

`SlidingWindowSpec` has no `extra_retained_tokens` field at this revision. Its absence maps explicitly to zero. An unexpected nonzero value remains unsupported.

Both pins reject attention chunk retention, speculative groups, unknown cache spec classes and unsupported alias layouts. Runtime source checks include the selected backend module; selecting another backend requires auditing and declaring that implementation rather than claiming that these inspected backends cover it. A vendor patch changing a declared file produces unresolved evidence with the actual hash retained. Update source notes, mapping tests and the manifest in a fresh campaign attempt after inspecting such a patch.
