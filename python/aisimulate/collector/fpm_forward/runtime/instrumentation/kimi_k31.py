# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read initialized K3.1 runtime objects; source contract: vllm-kimi-k31.md.

This observer does not construct cache views, quantizers, or model state.
"""

from __future__ import annotations

import math

import fpm_memory_observer as legacy

VERSION = "0.1.dev24216+g266e88c05"
LAYOUT = "kimi-k31-block-outer/v1"
SPEC = "vllm.v1.kv_cache_interface."
KDA = "vllm.models.kimi_k3.nvidia.kda.KimiK3DeltaAttention"


def _spec(spec, vllm_config):
    name = legacy._class_name(spec)
    if name == SPEC + "UniformTypeKVCacheSpecs":
        children = {key: _spec(value, vllm_config) for key, value in spec.kv_cache_specs.items()}
        return {
            "spec_type": name,
            "block_size_tokens": spec.block_size,
            "spec_page_size_bytes": spec.page_size_bytes,
            "sliding_window": None,
            "layer_specs": children,
        }
    if name not in {SPEC + "MambaSpec", SPEC + "MLAAttentionSpec"}:
        raise ValueError(f"K3.1 cache spec has no audited mapping: {name}")
    group = {
        "spec_type": name,
        "block_size_tokens": spec.block_size,
        "spec_page_size_bytes": spec.page_size_bytes,
        "sliding_window": getattr(spec, "sliding_window", None),
        "attention_chunk_size": getattr(spec, "attention_chunk_size", None),
        "extra_retained_tokens": getattr(spec, "extra_retained_tokens", 0),
    }
    if name == SPEC + "MambaSpec":
        group.update(
            state_shapes=[list(shape) for shape in spec.shapes],
            state_dtypes=[str(dtype) for dtype in spec.dtypes],
            state_content_size_bytes=spec.state_content_size_bytes,
            page_size_padded=spec.page_size_padded,
            mamba_cache_mode=spec.mamba_cache_mode,
            num_speculative_blocks=spec.num_speculative_blocks,
            num_prefill_checkpoint_blocks=spec.num_prefill_checkpoint_blocks,
            prefill_checkpoint_alignment=spec.prefill_checkpoint_alignment,
            max_model_len=vllm_config.model_config.max_model_len,
            blocks_per_request=spec.max_num_blocks_per_req(vllm_config, vllm_config.model_config.max_model_len),
        )
    else:
        group.update(dtype=str(spec.dtype), cache_dtype_str=spec.cache_dtype_str)
    return group


def groups(config, vllm_config):
    result = []
    for raw in config.kv_cache_groups:
        if raw.host_resident or raw.is_eagle_group:
            raise ValueError("K3.1 observer requires device-only non-speculative groups")
        group = {"layer_names": list(raw.layer_names), **_spec(raw.kv_cache_spec, vllm_config)}
        result.append(group)
    return result


def _require_baseline(owner):
    config = owner.vllm_config
    if list(config.model_config.hf_config.architectures) != ["KimiK31ForConditionalGeneration"]:
        raise ValueError("K3.1 adapter requires the native KimiK31 architecture")
    cache = config.cache_config
    if cache.enable_prefix_caching or cache.mamba_cache_mode != "none" or config.speculative_config is not None:
        raise ValueError("K3.1 adapter requires mamba none, prefix caching off and no speculation")


def scheduler_memory(owner, config):
    _require_baseline(owner)
    manager = owner.kv_cache_manager
    pool = manager.block_pool
    managers = manager.coordinator.single_type_managers
    if len(managers) != len(config.kv_cache_groups) or any(item.block_pool is not pool for item in managers):
        raise ValueError("K3.1 scheduler groups do not share the observed block pool")
    if pool.num_gpu_blocks != config.num_blocks or manager.watermark_blocks != 0:
        raise ValueError("K3.1 scheduler pool geometry or watermark is unsupported")
    free = pool.get_num_free_blocks()
    if not 0 < free < config.num_blocks or pool.null_block.is_null is not True:
        raise ValueError("K3.1 scheduler reservation is unresolved")
    return {
        "allocation_layout": LAYOUT,
        "num_blocks": config.num_blocks,
        "initial_free_blocks": free,
        "reserved_blocks": config.num_blocks - free,
        "null_block_id": pool.null_block.block_id,
        "watermark_blocks": manager.watermark_blocks,
        "pool_count": 1,
        "groups": groups(config, owner.vllm_config),
    }


def _view(tensor, ids):
    key, _ = legacy._storage(tensor)
    size = tensor.element_size()
    return {
        "storage_id": ids[key],
        "storage_offset_bytes": tensor.storage_offset() * size,
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "stride_bytes": [stride * size for stride in tensor.stride()],
        "element_size_bytes": size,
    }


def worker_memory(owner):
    _require_baseline(owner)
    if not getattr(owner, "_fpm_cache_initialized", False):
        raise ValueError("K3.1 cache initialization was not observed")
    runner = owner.model_runner
    if getattr(runner, "shared_kv_cache_layers", None):
        raise ValueError("K3.1 cross-layer KV sharing is unsupported")
    config = runner.kv_cache_config
    observed_groups = groups(config, owner.vllm_config)
    storages = legacy._tensor_storages(runner.kv_caches)
    if len(storages) != 1:
        raise ValueError("K3.1 native allocator must expose exactly one CUDA backing storage")
    ids = {key: f"storage_{index}" for index, key in enumerate(sorted(storages))}
    context = owner.vllm_config.compilation_config.static_forward_context
    layers = {name: context[name] for group in observed_groups for name in group["layer_names"]}
    views = {}
    for group in observed_groups:
        spec = next(iter(group["layer_specs"].values())) if "layer_specs" in group else group
        group["kind"] = "state" if spec["spec_type"] == SPEC + "MambaSpec" else "attention"
        group["layer_classes"] = {name: legacy._class_name(layers[name]) for name in group["layer_names"]}
        for name in group["layer_names"]:
            tensors = layers[name].kv_cache
            components = tensors if isinstance(tensors, tuple) else (tensors,)
            if legacy._tensor_storages(components) != storages:
                raise ValueError("K3.1 bound layer state does not share the native backing allocation")
            views[name] = [_view(tensor, ids) for tensor in components]
    allocations = []
    for tensor in config.kv_cache_tensors:
        allocations.append(
            {
                "size": tensor.size,
                "layers": list(tensor.layers),
                "layer_stride": tensor.layer_stride,
                "block_stride": tensor.block_stride,
                "offset": tensor.offset,
                "host_resident": tensor.host_resident,
                "storage_id": next(iter(ids.values())),
            }
        )
    return {
        "allocation_layout": LAYOUT,
        "num_blocks": config.num_blocks,
        "available_cache_bytes": owner._fpm_available_cache_bytes,
        "allocated_cache_bytes": sum(storages.values()),
        "storages": [
            {"storage_id": ids[key], "device": key[0], "pointer": key[1], "size_bytes": size}
            for key, size in storages.items()
        ],
        "tensor_allocations": allocations,
        "layer_components": views,
        "groups": observed_groups,
        "diagnostics": {},
    }


def attention_groups(owner, cache, source_files):
    runner = owner.model_runner
    source = type(runner).__module__.replace(".", "/") + ".py"
    if source not in source_files:
        raise ValueError("K3.1 runner source is not pinned")
    size_field = {
        "vllm.v1.worker.gpu_model_runner.GPUModelRunner": "_kernel_block_sizes",
        "vllm.v1.worker.gpu.model_runner.GPUModelRunner": "kernel_block_sizes",
    }.get(legacy._class_name(runner))
    if size_field is None:
        raise ValueError("K3.1 runner class has no audited kernel block mapping")
    sizes = getattr(runner, size_field)
    if list(sizes) != [group["block_size_tokens"] for group in cache["groups"]]:
        raise ValueError("K3.1 block-outer adapter does not support subdivided kernel blocks")
    cache["model_runner"] = {"class": legacy._class_name(runner), "kernel_block_sizes": list(sizes)}
    observed, names, covered = [], set(), set()
    for group_id, entries in enumerate(runner.attn_groups):
        for entry in entries:
            backend = entry.backend
            source = backend.__module__.replace(".", "/") + ".py"
            if source not in source_files:
                raise ValueError(f"K3.1 backend source is not pinned: {source}")
            layers = list(entry.layer_names)
            if covered.intersection(layers) or not set(layers) <= set(cache["groups"][group_id]["layer_names"]):
                raise ValueError("K3.1 backend layer coverage is inconsistent")
            covered.update(layers)
            names.add(backend.get_name())
            observed.append({"group_id": group_id, "layer_names": layers, "backend": backend.get_name()})
    if covered != set(cache["layer_components"]):
        raise ValueError("K3.1 backend groups do not cover all bound layer states")
    cache["attention_groups"] = observed
    cache["attention_backends"] = sorted(names)


def initialized_moe(owner):
    # Read post-load RoutedExperts, not the checkpoint quantization declaration.
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.models.kimi_k3.nvidia.model import KimiMoE

    result = []
    modules = dict(owner.model_runner.model.named_modules())
    expected = []
    for name, module in modules.items():
        if isinstance(module, KimiMoE):
            matches = [
                key for key, child in modules.items() if key.startswith(name + ".") and isinstance(child, RoutedExperts)
            ]
            if len(matches) != 1:
                raise ValueError(f"K3.1 MoE layer does not own exactly one audited RoutedExperts: {name}")
            expected.extend(matches)
    for name, layer in modules.items():
        if not isinstance(layer, RoutedExperts):
            continue
        method = layer.quant_method
        config = getattr(method, "moe_quant_config", None)
        kernel = getattr(method, "moe_kernel", None)
        if config is None or kernel is None:
            raise ValueError(f"initialized MoE precision/kernel is missing: {name}")
        result.append(
            {
                "name": name,
                "layer_type": legacy._class_name(layer),
                "method_type": legacy._class_name(method),
                "backend": method.mxfp4_backend.name,
                "kernel_type": legacy._class_name(kernel),
                "descriptors": {
                    field: {"dtype": str(getattr(config, field).dtype)} for field in ("_a1", "_a2", "_w1", "_w2")
                },
                "weight_shapes": {field: list(getattr(layer, field).shape) for field in ("w13_weight", "w2_weight")},
            }
        )
    if not result or any(math.prod(shape) <= 0 for item in result for shape in item["weight_shapes"].values()):
        raise ValueError("initialized MoE layer/weight coverage is empty")
    return {"enumeration": "model.named_modules:RoutedExperts", "expected_layer_names": expected, "layers": result}
