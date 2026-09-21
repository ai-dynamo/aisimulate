# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observe initialized vLLM memory without changing execution or profiling.

Original collector adapters against vLLM 0.27.0, commit
4bdc8a788d2e2ce9165d552b3d4d8b72604626bf: vllm/v1/worker/gpu_worker.py
(determine_available_memory, initialize_from_config, compile_or_warm_up_model),
vllm/v1/worker/gpu_model_runner.py (initialize_kv_cache_tensors), and
vllm/v1/core/{kv_cache_manager,block_pool}.py. Cache specs are read, not rebuilt.
https://github.com/vllm-project/vllm/tree/4bdc8a788d2e2ce9165d552b3d4d8b72604626bf/vllm/v1

Unsupported observation APIs write unresolved evidence. Exceptions from the
runtime itself are never caught by these observation helpers or wrappers.
"""

from __future__ import annotations

import importlib.metadata
import json
import logging
import os
from enum import Enum
from pathlib import Path
from typing import Any

SCHEMA_NAME = "aisimulate_fpm_runtime_memory"
SCHEMA_VERSION = 1
SUPPORTED_VERSION = "0.27.0"
RESULTS_DIR = Path("/results")
_SPEC_MODULE = "vllm.v1.kv_cache_interface."
_CONV_CLASS = "vllm.models.inkling.nvidia.sconv_swa_attn.InklingConvState"
_CONFIG_FIELDS = {
    "model_config": ("model", "revision", "dtype", "quantization", "max_model_len", "enforce_eager"),
    "cache_config": (
        "cache_dtype",
        "gpu_memory_utilization",
        "kv_cache_memory_bytes",
        "enable_prefix_caching",
        "num_gpu_blocks_override",
    ),
    "scheduler_config": ("max_num_batched_tokens", "max_num_seqs", "async_scheduling"),
    "parallel_config": (
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "data_parallel_size",
        "data_parallel_size_local",
        "data_parallel_external_lb",
        "enable_expert_parallel",
        "enable_eplb",
        "decode_context_parallel_size",
        "prefill_context_parallel_size",
    ),
    "compilation_config": ("mode", "cudagraph_mode", "cudagraph_capture_sizes", "max_cudagraph_capture_size"),
    "kernel_config": ("moe_backend",),
}


def _class_name(value: object) -> str:
    return f"{type(value).__module__}.{type(value).__name__}"


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.name
    if value is None or type(value) in (str, int, float, bool):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return str(value)


def compilation_config(config: Any) -> dict[str, Any]:
    """Snapshot the graph settings before the worker resolves backend support."""
    compilation = config.compilation_config
    return {key: _json_value(getattr(compilation, key)) for key in _CONFIG_FIELDS["compilation_config"]}


def resolved_config(config: Any) -> dict[str, Any]:
    result = {
        section: {
            key: _json_value(getattr(getattr(config, section), key))
            for key in fields
            if hasattr(getattr(config, section), key)
        }
        for section, fields in _CONFIG_FIELDS.items()
        if getattr(config, section, None) is not None
    }
    hf_config = getattr(config.model_config, "hf_config", None)
    if hf_config is not None and getattr(hf_config, "_commit_hash", None) is not None:
        result["model_config"]["loaded_config_commit_hash"] = hf_config._commit_hash
    # vLLM 0.27.0 config/offload.py owns weight offload; cache_config does not.
    offload = config.offload_config
    result["offload_config"] = {
        "offload_backend": offload.offload_backend,
        "uva": {"cpu_offload_gb": offload.uva.cpu_offload_gb},
        "prefetch": {
            key: getattr(offload.prefetch, key)
            for key in ("offload_group_size", "offload_num_in_group", "offload_prefetch_step")
        },
    }
    quant = config.quant_config
    result["quantization_config"] = (
        {
            "type": _class_name(quant),
            **{
                key: _json_value(getattr(quant, key))
                for key in ("quant_method", "activation_scheme", "weight_block_size")
                if hasattr(quant, key)
            },
        }
        if quant is not None
        else None
    )
    return result


def _offloader_type() -> str:
    # vLLM 0.27.0 model_executor/offloader/base.py:get_offloader is read-only.
    from vllm.model_executor.offloader import get_offloader

    return _class_name(get_offloader())


def _positive(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer, got {value!r}")
    return value


def cache_groups(cache_config: Any) -> list[dict[str, Any]]:
    groups = []
    all_names: set[str] = set()
    for group in cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        spec_type = _class_name(spec)
        if spec_type not in {_SPEC_MODULE + "FullAttentionSpec", _SPEC_MODULE + "SlidingWindowSpec"}:
            raise ValueError(f"unsupported runtime cache spec: {spec_type}")
        if getattr(group, "is_eagle_group", False):
            raise ValueError("speculative cache groups are unsupported")
        names = list(group.layer_names)
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("runtime cache groups require explicit layer names")
        if len(names) != len(set(names)) or all_names.intersection(names):
            raise ValueError("runtime cache group layer aliases are unsupported")
        all_names.update(names)
        window = getattr(spec, "sliding_window", None)
        if spec_type.endswith(".SlidingWindowSpec"):
            _positive(window, "sliding_window")
        elif window is not None or getattr(spec, "attention_chunk_size", None) is not None:
            raise ValueError("full-attention cache with alternate retention is unsupported")
        groups.append(
            {
                "layer_names": names,
                "spec_type": spec_type,
                "block_size_tokens": _positive(spec.block_size, "cache block size"),
                "spec_page_size_bytes": _positive(spec.page_size_bytes, "cache spec page size"),
                "sliding_window": window,
                "dtype": str(spec.dtype),
            }
        )
    if not groups:
        raise ValueError("runtime emitted no cache groups")
    return groups


def _storage(tensor: Any) -> tuple[tuple[str, int], int]:
    storage = tensor.untyped_storage()
    device = str(storage.device)
    if not device.startswith("cuda:"):
        raise ValueError(f"cache storage is not on a CUDA device: {device}")
    return (device, _positive(storage.data_ptr(), "storage pointer")), _positive(storage.nbytes(), "storage bytes")


def _tensor_storages(tensors: Any) -> dict[tuple[str, int], int]:
    result: dict[tuple[str, int], int] = {}

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        else:
            key, size = _storage(value)
            if key in result and result[key] != size:
                raise ValueError("aliased cache storage reports inconsistent sizes")
            result[key] = size

    visit(tensors)
    if not result:
        raise ValueError("runtime emitted no physical cache storage")
    if len({device for device, _pointer in result}) != 1:
        raise ValueError("rank cache spans multiple devices")
    return result


def worker_memory(worker: Any) -> dict[str, Any]:
    if not getattr(worker, "_fpm_cache_initialized", False):
        raise ValueError("worker cache initialization was not observed")
    runner = worker.model_runner
    if getattr(runner, "shared_kv_cache_layers", None):
        raise ValueError("cross-layer KV sharing is not yet supported by memory finalization")
    config = runner.kv_cache_config
    groups = cache_groups(config)
    storages = _tensor_storages(runner.kv_caches)
    context = worker.vllm_config.compilation_config.static_forward_context
    layers = {name: context[name] for group in groups for name in group["layer_names"]}
    layer_storages = {name: _storage(layer.kv_cache) for name, layer in layers.items()}
    if dict(layer_storages.values()) != storages:
        raise ValueError("cache group layers do not account for every model-runner storage")

    # Every observed alias must be declared by the runtime allocation plan.
    # Packed tensors explicitly share one backing allocation through block_stride.
    expected: dict[tuple[str, int], int] = {}
    covered: set[str] = set()
    packed_key = None
    allocations = []
    for tensor in config.kv_cache_tensors:
        names = list(tensor.shared_by)
        if not names or covered.intersection(names) or any(name not in layers for name in names):
            raise ValueError("runtime tensor allocation has missing or duplicate cache layers")
        keys = {layer_storages[name][0] for name in names}
        if len(keys) != 1:
            raise ValueError("runtime-declared shared tensor uses different physical storages")
        key = next(iter(keys))
        size = _positive(tensor.size, "configured cache tensor bytes")
        stride = getattr(tensor, "block_stride", 0)
        if type(stride) is not int or stride < 0:
            raise ValueError("invalid packed cache block stride")
        if stride:
            if packed_key is None:
                if key in expected:
                    raise ValueError("packed cache aliases an unpacked allocation")
                packed_key = key
            elif key != packed_key:
                raise ValueError("packed cache tensors use multiple backing allocations")
        elif key in expected:
            raise ValueError("undeclared cache tensor alias")
        if key in expected and expected[key] != size:
            raise ValueError("packed allocation sizes disagree")
        expected[key] = size
        covered.update(names)
        allocations.append(
            {"size": size, "shared_by": names, "block_stride": stride, "offset": getattr(tensor, "offset", 0)}
        )
    if covered != set(layers) or expected != storages:
        raise ValueError("runtime allocation plan disagrees with physical cache storage")
    for group in groups:
        classes = {name: _class_name(layers[name]) for name in group["layer_names"]}
        conv = {value == _CONV_CLASS for value in classes.values()}
        if len(conv) != 1:
            raise ValueError("mixed convolution and attention cache group is unsupported")
        group["kind"] = "convolution" if conv == {True} else "attention"
        group["layer_classes"] = classes
        if group["kind"] == "convolution" and group["sliding_window"] != 4:
            raise ValueError("unsupported Inkling convolution retention")
    diagnostics = {
        name: _json_value(getattr(worker, name))
        for name in (
            "available_kv_cache_memory_bytes",
            "peak_activation_memory",
            "non_torch_memory",
            "total_consumed",
            "requested_memory",
            "cudagraph_memory_estimate",
        )
        if hasattr(worker, name)
    }
    return {
        "num_blocks": _positive(config.num_blocks, "cache block count"),
        "available_cache_bytes": _positive(worker._fpm_available_cache_bytes, "profiled cache budget"),
        "allocated_cache_bytes": sum(storages.values()),
        "storages": [
            {"device": key[0], "pointer": key[1], "size_bytes": size} for key, size in sorted(storages.items())
        ],
        "tensor_allocations": allocations,
        "groups": groups,
        "diagnostics": diagnostics,
    }


def scheduler_memory(scheduler: Any, config: Any) -> dict[str, Any]:
    manager = scheduler.kv_cache_manager
    pool = manager.block_pool
    managers = manager.coordinator.single_type_managers
    if not managers or any(item.block_pool is not pool for item in managers):
        raise ValueError("runtime cache groups must share one block pool")
    num_blocks = _positive(config.num_blocks, "scheduler cache block count")
    if pool.num_gpu_blocks != num_blocks:
        raise ValueError("scheduler and configured block pool sizes disagree")
    free = _positive(pool.get_num_free_blocks(), "initial free cache blocks")
    if free >= num_blocks or getattr(pool.null_block, "is_null", False) is not True:
        raise ValueError("scheduler null block reservation is not observable")
    if type(manager.watermark_blocks) is not int or manager.watermark_blocks != 0:
        raise ValueError("nonzero scheduler cache watermark is unsupported")
    return {
        "num_blocks": num_blocks,
        "initial_free_blocks": free,
        "reserved_blocks": num_blocks - free,
        "null_block_id": pool.null_block.block_id,
        "watermark_blocks": manager.watermark_blocks,
        "pool_count": 1,
        "groups": cache_groups(config),
    }


def observe(
    kind: str,
    owner: Any,
    *,
    dp_rank: int,
    tp_rank: int | None = None,
    pp_rank: int | None = None,
    cache_config: Any = None,
) -> None:
    """Write atomic evidence after successful initialization, never run a model."""
    provenance = json.loads((RESULTS_DIR / "collector-provenance.json").read_text())
    payload: dict[str, Any] = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "collector_provenance": provenance,
        "backend_version": importlib.metadata.version("vllm"),
        "dp_rank": dp_rank,
        "tp_rank": tp_rank,
        "pp_rank": pp_rank,
    }
    try:
        if payload["backend_version"] != SUPPORTED_VERSION:
            raise ValueError(
                f"memory observation currently supports vLLM {SUPPORTED_VERSION}; found {payload['backend_version']}"
            )
        if (
            getattr(owner.vllm_config, "speculative_config", None) is not None
            or getattr(owner.vllm_config, "kv_transfer_config", None) is not None
        ):
            raise ValueError("runtime memory observation supports non-speculative local cache only")
        payload["resolved_config"] = resolved_config(owner.vllm_config)
        offload = payload["resolved_config"]["offload_config"]
        if kind == "worker":
            payload["effective_offloader"] = _offloader_type()
            initial = getattr(owner, "_fpm_initial_compilation_config", None)
            if initial is not None:
                payload["initial_compilation_config"] = initial
        cache = owner.vllm_config.cache_config
        if (
            getattr(cache, "kv_cache_memory_bytes", None) is not None
            or getattr(cache, "num_gpu_blocks_override", None) is not None
            or offload["offload_backend"] != "auto"
            or offload["uva"]["cpu_offload_gb"] != 0
            or offload["prefetch"]["offload_group_size"] != 0
            or (
                kind == "worker"
                and payload["effective_offloader"] != "vllm.model_executor.offloader.base.NoopOffloader"
            )
        ):
            raise ValueError("manual KV allocation or offloaded cache is unsupported by runtime memory finalization")
        payload["cache"] = worker_memory(owner) if kind == "worker" else scheduler_memory(owner, cache_config)
        payload["status"] = "resolved"
    except Exception as error:
        payload.update(status="unresolved", error=f"{type(error).__name__}: {error}")
        logging.getLogger(__name__).warning("FPM runtime memory remains unresolved: %s", payload["error"])
    suffix = f"-tp{tp_rank}-pp{pp_rank}" if kind == "worker" else ""
    path = RESULTS_DIR / f"fpm-memory-{kind}-dp{dp_rank}{suffix}.json"
    if path.exists():
        raise RuntimeError(f"refusing duplicate runtime memory observation: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
