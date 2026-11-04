# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read native cache/input tensor metadata without replacing tensor objects."""

import contextlib
import ctypes
import functools
import hashlib
from pathlib import Path


def loaded_cuda_runtimes():
    """Query already loaded CUDA runtime libraries, without guessing a version."""
    maps = Path("/proc/self/maps")
    if not maps.exists():
        return []
    libraries = sorted({line.split()[-1] for line in maps.read_text().splitlines() if "/libcudart.so" in line})
    records = []
    for path in libraries:
        try:
            library = ctypes.CDLL(path)
            version = ctypes.c_int()
            status = library.cudaRuntimeGetVersion(ctypes.byref(version))
            records.append({"library": path, "runtime_version_code": version.value if status == 0 else None})
        except (OSError, AttributeError) as error:
            records.append({"library": path, "query_error": str(error)})
    return records


def tensor_observation(value):
    """Return primitive metadata, deduplicating aliased physical storage."""
    tensors = []
    storages = {}

    def visit(item, path):
        if hasattr(item, "untyped_storage") and hasattr(item, "shape"):
            storage = item.untyped_storage()
            device = str(item.device)
            identity = (device, int(storage.data_ptr()))
            storage_bytes = int(storage.nbytes())
            storage_id = hashlib.sha256(repr(identity).encode()).hexdigest()[:20]
            storages[identity] = max(storages.get(identity, 0), storage_bytes)
            tensors.append(
                {
                    "path": path,
                    "shape": list(item.shape),
                    "strides": list(item.stride()),
                    "dtype": str(item.dtype),
                    "device": device,
                    "logical_bytes": int(item.numel()) * int(item.element_size()),
                    "storage_bytes": storage_bytes,
                    "storage_id": storage_id,
                }
            )
        elif isinstance(item, dict):
            for name, child in item.items():
                visit(child, f"{path}.{name}")
        elif isinstance(item, (tuple, list)):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")

    visit(value, "tensor")
    return {"tensors": tensors, "unique_storage_bytes": sum(storages.values())}


def forward_inputs(kernel, backend):
    """Observe the exact Q/K/V arguments captured by each audited native call."""
    names = {
        "vllm": {"query_fwd", "key_vllm", "value_vllm"},
        "sglang": {"q", "k", "v"},
        "trtllm": {"forward_args"},
    }[backend]
    cells = dict(zip(kernel.__code__.co_freevars, kernel.__closure__ or (), strict=True))
    return tensor_observation({name: cell.cell_contents for name, cell in cells.items() if name in names})


@contextlib.contextmanager
def observe_native_caches(module, backend, emit):
    """Wrap only constructor aliases; restore them and return original objects."""
    originals = {}

    def wrap(name, kind, extract, attributes=()):
        original = getattr(module, name)
        originals[name] = original

        @functools.wraps(original)
        def observed(*args, **kwargs):
            result = original(*args, **kwargs)
            try:
                record = {"kind": kind, **tensor_observation(extract(result))}
                record["factory_attributes"] = {
                    name: value
                    for name in attributes
                    if isinstance((value := getattr(result, name, None)), (str, int, float, bool))
                }
                emit(record)
            except Exception as error:
                # A missing observation invalidates the fit; it must never alter
                # a native constructor's result or mask its own exception.
                emit({"kind": kind, "observation_error": str(error)})
            return result

        setattr(module, name, observed)

    try:
        if backend == "vllm":
            wrap("create_kv_cache_and_block_mappings", "vllm_initial_cache", lambda result: result[0])
            wrap("kv_cache_in_resolved_layout", "vllm_final_cache", lambda result: result)
        elif backend == "sglang":
            wrap(
                "MHATokenToKVPool",
                "sglang_cache_pool",
                lambda pool: {"k": pool.k_buffer, "v": pool.v_buffer},
                ("size", "page_size", "layer_num", "head_num", "head_dim"),
            )
        elif backend == "trtllm":
            wrap(
                "KVCacheManager",
                "trtllm_primary_cache_pool",
                lambda manager: manager.get_unique_primary_pool(),
                ("tokens_per_block", "blocks_in_primary_pool", "kv_factor", "head_dim"),
            )
        else:
            raise ValueError(f"unsupported cache observer backend: {backend}")
        yield
    finally:
        for name, original in originals.items():
            setattr(module, name, original)
