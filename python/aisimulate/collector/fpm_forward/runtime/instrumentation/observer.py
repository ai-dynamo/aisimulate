# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Original read-only adapters; see the bundled version-specific source notes.

The runner imports this file only inside the selected runtime. Inspection and
CPU import use the manifest, never this module. No model or timing code is run.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
from typing import Any

import fpm_memory_observer as legacy


def _json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(item) for item in value]
    return legacy._json_value(value)


def _installed_file(name: str) -> Path:
    package, *parts = name.split("/")
    module = importlib.import_module(package)
    paths = [Path(root).joinpath(*parts) for root in module.__path__]
    found = [path for path in paths if path.is_file()]
    if len(found) != 1:
        raise ValueError(f"runtime source path must resolve uniquely: {name}")
    return found[0]


def _runtime(payload: dict, manifest: dict, version: str) -> None:
    actual_version = importlib.metadata.version("vllm")
    build = importlib.import_module("vllm._version")
    commit = getattr(build, "__commit_id__", None)
    pinned = manifest["runtime"]
    observed = payload["runtime"] = {
        "framework": "vllm",
        "version": actual_version,
        "source_revision": pinned["source_revision"],
        "source_files": {},
        "image": payload["launch"]["deployment"]["image"],
    }
    payload["source_identity"] = {"vllm_commit_id": commit, "source_revision_basis": "declared source and file hashes"}
    errors = []
    if actual_version != version or pinned["version"] != version:
        errors.append(f"adapter requires vLLM {version}; installed={actual_version}, manifest={pinned['version']}")
    # Wheels may omit a VCS identifier. The exact inspected source files remain
    # mandatory; when a wheel supplies an identifier, it must also agree.
    if commit is not None:
        short = str(commit).removeprefix("g")
        if len(short) < 7 or not pinned["source_revision"].startswith(short):
            errors.append(f"installed vLLM commit {commit!r} contradicts source revision")
    for name, expected in pinned["source_files"].items():
        try:
            digest = hashlib.sha256(_installed_file(name).read_bytes()).hexdigest()
            observed["source_files"][name] = digest
            if digest != expected:
                errors.append(f"runtime source hash differs: {name}")
        except (OSError, ValueError, AttributeError, ImportError) as error:
            errors.append(f"runtime source {name}: {error}")
    if errors:
        raise ValueError("; ".join(errors))


def _loaded_config(owner: Any, payload: dict) -> None:
    model = owner.vllm_config.model_config
    config_format = getattr(model, "config_format", "auto")
    if config_format not in ("auto", "hf"):
        raise ValueError(f"loaded config format requires another source mapping: {config_format}")
    source = getattr(model, "hf_config_path", None) or model.model
    candidate = Path(source)
    if candidate.is_dir():
        if config_format == "auto" and (candidate / "params.json").exists():
            raise ValueError("automatic HF/Mistral config selection requires an explicit loaded-file mapping")
        candidate = candidate / "config.json"
    elif not candidate.is_file():
        # Resolve only the already-loaded immutable HF revision. Never download
        # another config during observation or hash a caller-supplied host copy.
        commit = getattr(getattr(model, "hf_config", None), "_commit_hash", None)
        if not isinstance(commit, str) or len(commit) != 40:
            raise ValueError("loaded remote config has no immutable cache commit")
        from huggingface_hub import try_to_load_from_cache

        if config_format == "auto" and isinstance(try_to_load_from_cache(source, "params.json", revision=commit), str):
            raise ValueError("automatic cached HF/Mistral config selection is ambiguous")
        cached = try_to_load_from_cache(source, "config.json", revision=commit)
        if not isinstance(cached, str):
            raise ValueError("actual loaded config.json is absent from the local HF cache")
        candidate = Path(cached)
    if candidate.name != "config.json":
        raise ValueError("this adapter requires an actual loaded HF config.json")
    raw = candidate.read_bytes()
    document = json.loads(raw)
    if document.get("configuration_files"):
        raise ValueError("version-selected HF configuration files require a separate audited mapping")
    digest = hashlib.sha256(raw).hexdigest()
    payload["model_config_sha256"] = digest
    payload["loaded_model_config"] = {"path": str(candidate.resolve()), "sha256": digest}
    expected_sources = payload["launch"].get("model_config", {}).get("source_files")
    if expected_sources is not None:
        observed = payload["model_config_source_files"] = {}
        for name in expected_sources:
            if (
                not isinstance(name, str)
                or name in ("", ".", "..")
                or Path(name).name != name
                or any(char in name for char in ("\\", ":", "\x00", "\n", "\r"))
            ):
                raise ValueError("model config sidecars must be adjacent relative filenames")
            observed[name] = hashlib.sha256((candidate.parent / name).read_bytes()).hexdigest()


def _resolved(owner: Any, *, hybrid_evidence: bool = False) -> dict:
    config = owner.vllm_config
    value = legacy.resolved_config(config, hybrid_evidence=hybrid_evidence)
    for key in ("speculative_config", "kv_transfer_config"):
        value[key] = _json(getattr(config, key, None))
    for key in ("hf_config_path", "hf_overrides", "config_format"):
        if hasattr(config.model_config, key):
            value["model_config"][key] = _json(getattr(config.model_config, key))
    return value


def _raw_cache(config: Any, *, hybrid_evidence: bool = False) -> dict:
    fields = (
        "block_size",
        "storage_block_size",
        "page_size_bytes",
        "page_size_padded",
        "dtype",
        "sliding_window",
        "extra_retained_tokens",
        "attention_chunk_size",
        "num_kv_heads",
        "head_size",
        "kv_quant_mode",
    )
    tensor_fields = ("size", "shared_by", "block_stride", "offset")
    if hybrid_evidence:
        # vLLM 0.31.0 token-state, MLA role and Mamba state spec fields
        # (kv_cache_interface.py:504, :668, :1043-1052) and layer placement (:1418-1438).
        fields += (
            "mamba_cache_mode",
            "num_speculative_blocks",
            "num_prefill_checkpoint_blocks",
            "tokens_per_state",
            "cache_role",
            "shapes",
            "dtypes",
        )
        tensor_fields += ("layers", "layer_stride")
    # A vLLM 0.31.0 UniformTypeKVCacheSpecs inherits the abstract KVCacheSpec.tokens_per_state, which
    # raises NotImplementedError (kv_cache_interface.py:185-187, :1233-1333). The layer specs of its
    # cache group record the value of each layer (fpm_memory_observer.py:476-485).
    uniform_fields = tuple(key for key in fields if key != "tokens_per_state")
    record = {
        "num_blocks": config.num_blocks,
        "groups": [
            {
                "layer_names": list(group.layer_names),
                "spec_type": legacy._class_name(group.kv_cache_spec),
                **{
                    key: _json(getattr(group.kv_cache_spec, key))
                    for key in (
                        uniform_fields if legacy._class_name(group.kv_cache_spec) == legacy._UNIFORM_SPEC else fields
                    )
                    if hasattr(group.kv_cache_spec, key)
                },
            }
            for group in config.kv_cache_groups
        ],
        "tensor_allocations": [
            {key: _json(getattr(tensor, key)) for key in tensor_fields if hasattr(tensor, key)}
            for tensor in config.kv_cache_tensors
        ],
    }
    if hybrid_evidence and hasattr(config, "kv_cache_layout"):
        # The layout name that the engine core resolves for every worker (engine/core.py:323-324).
        record["kv_cache_layout"] = _json(config.kv_cache_layout)
    return record


def _layer_specs(spec: Any) -> list[Any]:
    """The spec of every layer of a cache group: the inner specs of a uniform group."""
    return list(spec.kv_cache_specs.values()) if legacy._class_name(spec) == legacy._UNIFORM_SPEC else [spec]


def _retention(spec: Any) -> tuple[Any, Any]:
    # vLLM 0.31.0 SlidingWindowSpec and its subclasses declare extra_retained_tokens
    # with the default 0 (kv_cache_interface.py:838); its absence there is an
    # incompatible runtime API. Other spec classes have no such field.
    window = legacy._class_name(spec).removeprefix(legacy._SPEC_MODULE) in legacy._WINDOW_SPECS
    return getattr(spec, "extra_retained_tokens", None if window else 0), getattr(spec, "attention_chunk_size", None)


def _indexer(spec: Any) -> bool:
    # The sparse MLA indexer (SparseCacheRole.INDEXER, kv_cache_interface.py:150-152, :668).
    return (
        legacy._class_name(spec) == legacy._SPEC_MODULE + "MLAAttentionSpec"
        and getattr(getattr(spec, "cache_role", None), "name", None) == "INDEXER"
    )


def _block_pool_groups(cache: dict, config: Any, pool_id: str) -> None:
    """Apply the vLLM 0.31.0 retention, storage-block and placement rules."""
    errors = []
    for raw, group in zip(config.kv_cache_groups, cache["groups"], strict=True):
        names = group["layer_names"]
        specs = _layer_specs(raw.kv_cache_spec)
        retention = [_retention(spec) for spec in specs]
        extra, chunk = retention[0]
        group.update(pool_id=pool_id, extra_retained_tokens=extra, attention_chunk_size=chunk)
        if any(value != retention[0] for value in retention):
            errors.append(f"cache group layers disagree on retention: {names}")
        if type(extra) is not int or extra != 0 or chunk is not None:
            errors.append(f"unsupported cache retention for {names}: extra={extra!r}, chunk={chunk!r}")
        # Only the indexer stores its tokens in blocks other than the cache block.
        # The worker views it with that storage block (worker/utils.py:449-450).
        if any(
            getattr(spec, "storage_block_size", None) not in (None, spec.block_size) and not _indexer(spec)
            for spec in specs
        ):
            errors.append(f"compressed cache storage is unsupported: {names}")
        # HiSparse places groups in a host pool or outside the transferable state
        # (kv_cache_interface.py:1441-1465, v1/hisparse/layout.py). One device pool is modeled.
        host, role, transfer = (
            getattr(raw, "host_resident", None),
            _json(getattr(raw, "role", None)),
            getattr(raw, "enable_kv_transfer", None),
        )
        if host is not False or role != "DEFAULT" or transfer is not True:
            errors.append(
                f"unsupported cache group placement for {names}: "
                f"host_resident={host!r}, role={role!r}, enable_kv_transfer={transfer!r}"
            )
    for tensor in config.kv_cache_tensors:
        host = getattr(tensor, "host_resident", None)
        if host is not False:
            errors.append(
                f"unsupported cache tensor placement for {legacy._tensor_layers(tensor)}: host_resident={host!r}"
            )
    if errors:
        raise ValueError("; ".join(errors))


def _groups(cache: dict, config: Any, version: str, pool_id: str) -> None:
    if version == "0.31.0":
        _block_pool_groups(cache, config, pool_id)
        return
    errors = []
    for raw, group in zip(config.kv_cache_groups, cache["groups"], strict=True):
        spec = raw.kv_cache_spec
        # 0.27 lacks this field. 0.28 introduces it on SlidingWindowSpec; its
        # absence there is an incompatible runtime API, not a default of zero.
        if version == "0.28.0" and group["spec_type"].endswith(".SlidingWindowSpec"):
            extra = getattr(spec, "extra_retained_tokens", None)
        else:
            extra = getattr(spec, "extra_retained_tokens", 0)
        chunk = getattr(spec, "attention_chunk_size", None)
        group.update(pool_id=pool_id, extra_retained_tokens=extra, attention_chunk_size=chunk)
        if type(extra) is not int or extra != 0 or chunk is not None:
            errors.append(f"unsupported cache retention for {group['layer_names']}: extra={extra!r}, chunk={chunk!r}")
        if getattr(spec, "storage_block_size", spec.block_size) != spec.block_size:
            errors.append(f"compressed cache storage is unsupported: {group['layer_names']}")
    if errors:
        raise ValueError("; ".join(errors))


# vLLM 0.31.0 MLAAttentionSpec and its one subclass (kv_cache_interface.py:662, :739):
# the classes that pass the isinstance check of worker/utils.py:449.
_MLA_SPECS = frozenset(legacy._SPEC_MODULE + name for name in ("MLAAttentionSpec", "HiddenStateCacheSpec"))


def _view_kernel_block(spec: Any, kernel: int, block: int) -> int:
    """The kernel block of a vLLM 0.31.0 layer view.

    It is the group's kernel block, except for an MLAAttentionSpec with a storage
    block (the indexer), which the worker views with it (worker/utils.py:449-450),
    and for a Mamba layer, which keeps the group block size (worker/utils.py:500-502).
    """
    storage = getattr(spec, "storage_block_size", None)
    if legacy._class_name(spec) in _MLA_SPECS and storage is not None:
        return legacy._positive(storage, "storage block size")
    if legacy._class_name(spec) == legacy._MAMBA_SPEC:
        return block
    return kernel


def _block_first_views(name: str, kv_cache: Any, storage_ids: dict, kernel: int, blocks: int) -> list[dict]:
    """Record every view of one vLLM 0.31.0 layer: one tensor, or each tensor of a tuple.

    create_kv_cache_views gives each layer a [B, H, N, C] view, block axis first
    (kv_cache_interface.py:353-417). The layer's bind_kv_cache keeps it first: MLA
    and the indexer drop H (mla_attention.py:733-735, deepseek_v2.py:652-655), and
    a Mamba layer cuts one view per state from the page (mamba/abstract.py:28-42).
    AttentionBackend.get_kv_cache_block_dim no longer exists, so no axis is
    searched by its length. ``blocks`` is num_blocks * (block_size // kernel).
    """
    views = []
    for tensor in kv_cache if isinstance(kv_cache, (tuple, list)) else (kv_cache,):
        key, _ = legacy._storage(tensor)
        width = legacy._positive(tensor.element_size(), "tensor element size")
        shape = list(tensor.shape)
        if shape[0] != blocks:
            raise ValueError(f"cache view block axis does not hold num_blocks * (block_size // kernel): {name}")
        views.append(
            {
                "storage_id": storage_ids[key],
                "storage_offset_bytes": tensor.storage_offset() * width,
                "shape": shape,
                "stride_bytes": [stride * width for stride in tensor.stride()],
                "element_size_bytes": width,
                "block_axis": 0,
                "kernel_block_size_tokens": kernel,
            }
        )
    return views


def _layer_views(owner: Any, cache: dict, source_files: dict, version: str) -> None:
    runner = owner.model_runner
    runner_class = legacy._class_name(runner)
    # Worker.init_device selects either native runner. Each initialize_kv_cache
    # stores the sizes it passes to the tensor allocator under a different name;
    # missing state must not fall back to another runner's field or spec sizes.
    size_fields = {
        "vllm.v1.worker.gpu_model_runner.GPUModelRunner": "_kernel_block_sizes",
        "vllm.v1.worker.gpu.model_runner.GPUModelRunner": "kernel_block_sizes",
    }
    if runner_class not in size_fields:
        raise ValueError(f"model runner has no audited cache mapping: {runner_class}")
    runner_source = type(runner).__module__.replace(".", "/") + ".py"
    if runner_source not in source_files:
        raise ValueError(f"selected model runner source has not been audited: {runner_source}")
    kernel_block_sizes = getattr(runner, size_fields[runner_class])
    if not isinstance(kernel_block_sizes, list) or len(kernel_block_sizes) != len(cache["groups"]):
        raise ValueError("initialized kernel block sizes do not cover every cache group")
    kernel_block_sizes = [legacy._positive(size, "kernel block size") for size in kernel_block_sizes]
    cache["model_runner"] = {"class": runner_class, "kernel_block_sizes": kernel_block_sizes}
    layers = owner.vllm_config.compilation_config.static_forward_context
    storage_ids = {}
    for index, storage in enumerate(cache["storages"]):
        storage["storage_id"] = f"storage-{index}"
        storage_ids[(storage["device"], storage["pointer"])] = storage["storage_id"]
    for allocation in cache["tensor_allocations"]:
        # worker_memory proved that the layers of an allocation use one storage;
        # a vLLM 0.31.0 Mamba layer binds a tuple of state views on it.
        (key,) = legacy._layer_storages(layers[allocation["shared_by"][0]].kv_cache)
        allocation["storage_id"] = storage_ids[key]
    views = cache["layer_tensors"] = {}
    groups = []
    backends = set()
    for attention_groups in runner.attn_groups:
        for group in attention_groups:
            spec = group.kv_cache_spec
            kernel = kernel_block_sizes[group.kv_cache_group_id]
            if version == "0.31.0":
                block = cache["groups"][group.kv_cache_group_id]["block_size_tokens"]
                kernel = _view_kernel_block(spec, kernel, block)
            else:
                if getattr(spec, "storage_block_size", spec.block_size) != spec.block_size:
                    raise ValueError("compressed kernel cache blocks are unsupported")
                # Both audited runners use auto for an unquantized group's shape.
                # Read KVQuantMode by its public enum name; do not infer from dtype.
                mode = getattr(spec, "kv_quant_mode", None)
                if mode is None:
                    raise ValueError("cache spec has no observed KV quantization mode")
                dtype = (
                    "auto"
                    if getattr(mode, "name", None) == "NONE"
                    else (getattr(spec, "cache_dtype_str", None) or owner.vllm_config.cache_config.cache_dtype)
                )
            backend = group.backend
            backend_source = backend.__module__.replace(".", "/") + ".py"
            if backend_source not in source_files:
                raise ValueError(f"selected backend source has not been audited: {backend_source}")
            if version != "0.31.0":
                axis = backend.get_kv_cache_block_dim(kernel, spec.num_kv_heads, spec.head_size, cache_dtype_str=dtype)
                if type(axis) is not int or axis < 0:
                    raise ValueError("backend returned an invalid cache block dimension")
            backends.add(backend.get_name())
            groups.append(
                {
                    "kv_cache_group_id": group.kv_cache_group_id,
                    "backend_class": f"{backend.__module__}.{backend.__qualname__}",
                    "layer_names": list(group.layer_names),
                }
            )
            for name in group.layer_names:
                if name in views:
                    raise ValueError(f"duplicate attention group layer: {name}")
                if version == "0.31.0":
                    blocks = cache["num_blocks"] * (block // kernel)
                    views[name] = _block_first_views(name, layers[name].kv_cache, storage_ids, kernel, blocks)
                    continue
                tensor = layers[name].kv_cache
                key, _ = legacy._storage(tensor)
                width = legacy._positive(tensor.element_size(), "tensor element size")
                shape = list(tensor.shape)
                if axis >= len(shape):
                    raise ValueError("backend block axis is outside the physical tensor shape")
                views[name] = {
                    "storage_id": storage_ids[key],
                    "storage_offset_bytes": tensor.storage_offset() * width,
                    "shape": shape,
                    "stride_bytes": [stride * width for stride in tensor.stride()],
                    "element_size_bytes": width,
                    "block_axis": axis,
                    "kernel_block_size_tokens": kernel,
                }
    expected = {name for group in cache["groups"] for name in group["layer_names"]}
    if set(views) != expected:
        raise ValueError("attention backend views do not cover every physical cache layer")
    cache["attention_groups"] = groups
    cache["attention_backends"] = sorted(backends)


def _reservations(owner: Any, cache: dict) -> None:
    pool = owner.kv_cache_manager.block_pool
    free = [block.block_id for block in pool.free_block_queue.get_all_free_blocks()]
    ids = [block.block_id for block in pool.blocks]
    if len(set(ids)) != cache["num_blocks"] or set(ids) != set(range(cache["num_blocks"])):
        raise ValueError("block pool does not expose every unique block ID")
    if len(set(free)) != len(free) or len(free) != cache["initial_free_blocks"] or not set(free) <= set(ids):
        raise ValueError("free block queue disagrees with the observed initial count")
    reserved = sorted(set(ids) - set(free))
    cache["permanent_reserved_block_ids"] = reserved
    cache["group_pool_ids"] = [cache["pool_id"]] * len(cache["groups"])
    # Audited BlockPool constructors reserve only null_block. Do not classify
    # a live request allocation as a permanent reservation from its ref count.
    if reserved != [pool.null_block.block_id]:
        raise ValueError("additional initialized reservations lack an audited permanent-reservation mapping")


def _semantics(owner: Any, payload: dict) -> None:
    config = owner.vllm_config
    offload = payload["resolved_config"]["offload_config"]
    semantics = payload["cache"]["semantics"] = {
        "allocation": "shared_block_pool",
        "storage": "hbm",
        "retention": "full_or_window",
        "prefix_reuse": False,
        "offload": (
            offload["offload_backend"] != "auto"
            or offload["uva"]["cpu_offload_gb"] != 0
            or offload["prefetch"]["offload_group_size"] != 0
            or payload.get("effective_offloader") != "vllm.model_executor.offloader.base.NoopOffloader"
        ),
        "speculative": getattr(config, "speculative_config", None) is not None,
    }
    if (
        semantics["offload"]
        or semantics["speculative"]
        or getattr(config, "kv_transfer_config", None) is not None
        or getattr(config.cache_config, "kv_cache_memory_bytes", None) is not None
        or getattr(config.cache_config, "num_gpu_blocks_override", None) is not None
    ):
        raise ValueError("offload, speculative, transferred or manually overridden cache is unsupported")


def _hardware(owner: Any, gpu: str) -> dict:
    import torch

    properties = torch.cuda.get_device_properties(owner.device)
    return {
        "gpu": gpu,
        "sm": properties.major * 10 + properties.minor,
        "device_name": properties.name,
        "total_memory_bytes": properties.total_memory,
    }


def _prefix_caching(owner: Any, cache: dict, kind: str) -> dict:
    """The vLLM 0.31.0 prefix-caching settings of one process; None where absent.

    The scheduler adds its block size and hash block size (core/sched/scheduler.py:198, :310-312).
    """
    settings = owner.vllm_config.cache_config
    value = {key: _json(getattr(settings, key, None)) for key in ("enable_prefix_caching", "mamba_cache_mode")}
    if kind == "scheduler":
        value.update(block_size=cache.get("block_size"), hash_block_size=cache.get("hash_block_size"))
    return value


def observe(
    kind: str,
    owner: Any,
    *,
    version: str,
    dp_rank: int,
    tp_rank: int | None = None,
    pp_rank: int | None = None,
    cache_config: Any = None,
) -> None:
    """Publish raw independent observations after a successful runtime hook."""
    context = json.loads(Path(os.environ["AISIMULATE_RUNTIME_CONTEXT"]).read_text())
    manifest = json.loads(Path(os.environ["AISIMULATE_RUNTIME_INSTRUMENTATION"]).read_text())
    directory = Path(os.environ["AISIMULATE_RUNTIME_OBSERVATION_DIR"])
    payload = {
        "schema_version": "aisimulate-runtime-observation/v1",
        **{key: context[key] for key in ("attempt_id", "configuration", "phase", "bundle_sha256", "launch")},
        "identity": context["launch"]["identity"],
        "kind": kind,
        "dp_rank": dp_rank,
        "tp_rank": tp_rank,
        "pp_rank": pp_rank,
        "unresolved_fields": [],
        "lifecycle": {
            "measurement": "after_warmup",
            "cache_initialized": bool(getattr(owner, "_fpm_cache_initialized", False)),
            "warmup_completed": bool(getattr(owner, "_observation_warmup_completed", False)),
            "capture_completed": bool(getattr(owner, "_observation_warmup_completed", False)),
        }
        if kind == "worker"
        else {"measurement": "scheduler_initialized", "cache_initialized": True},
    }
    errors = []

    # CPU readiness is independent of physical-memory import. A missing or
    # partial CPU snapshot must not relabel otherwise valid cache observations.
    try:
        cpu = legacy.observe_cpu(kind, dp_rank=dp_rank, tp_rank=tp_rank, pp_rank=pp_rank, directory=directory)
        payload["cpu_affinity_observation"] = {"status": cpu["status"], "hostname": cpu["hostname"], "pid": cpu["pid"]}
    except Exception as error:
        payload["cpu_affinity_observation"] = {"status": "unavailable", "error": f"{type(error).__name__}: {error}"}
        logging.getLogger(__name__).warning("CPU affinity observation unavailable: %s", error)

    def capture(label, action):
        try:
            action()
        except Exception as error:
            payload["unresolved_fields"].append(label)
            errors.append(f"{label}: {type(error).__name__}: {error}")

    if "instrumentation_binding" in context:
        from fpm_runtime_instrumentation import observed_binding

        capture("instrumentation_binding", lambda: payload.update(instrumentation_binding=observed_binding()))
    # Only vLLM 0.31.0 records carry the hybrid cache evidence; older records keep their keys.
    hybrid = version == "0.31.0"
    capture("runtime", lambda: _runtime(payload, manifest, version))
    capture("model_config_sha256", lambda: _loaded_config(owner, payload))
    capture("resolved_config", lambda: payload.update(resolved_config=_resolved(owner, hybrid_evidence=hybrid)))
    pool_id = f"dp{dp_rank}-shared-pool"
    if kind == "worker":
        capture("hardware", lambda: payload.update(hardware=_hardware(owner, payload["identity"]["gpu"])))
        capture("effective_offloader", lambda: payload.update(effective_offloader=legacy._offloader_type()))
        if hasattr(owner, "_fpm_initial_compilation_config"):
            payload["initial_compilation_config"] = owner._fpm_initial_compilation_config
        capture("cache", lambda: payload.update(cache=legacy.worker_memory(owner, hybrid_evidence=hybrid)))
        config = getattr(getattr(owner, "model_runner", None), "kv_cache_config", None)
    else:
        config = cache_config
        capture("cache", lambda: payload.update(cache=legacy.scheduler_memory(owner, config, hybrid_evidence=hybrid)))
    capture("raw_cache_config", lambda: payload.update(raw_cache_config=_raw_cache(config, hybrid_evidence=hybrid)))
    if "cache" in payload:
        cache = payload["cache"]
        cache["pool_id"] = pool_id
        if hybrid:
            capture("cache.prefix_caching", lambda: cache.update(prefix_caching=_prefix_caching(owner, cache, kind)))
        capture("cache.groups", lambda: _groups(cache, config, version, pool_id))
        if kind == "worker":
            capture("cache.semantics", lambda: _semantics(owner, payload))
            capture(
                "cache.layer_tensors",
                lambda: _layer_views(owner, cache, manifest["runtime"]["source_files"], version),
            )
            if "attention_groups" in cache:
                payload["attention_groups"] = cache.pop("attention_groups")
                names = cache.pop("attention_backends")
                payload["attention_backend"] = names[0] if len(names) == 1 else names
        else:
            capture("cache.permanent_reserved_block_ids", lambda: _reservations(owner, cache))
    if kind == "worker" and ("collector_attempt_id" in context or (directory / "collector-provenance.json").exists()):
        # Retain the existing formal-collection execution contract unchanged.
        previous = legacy.RESULTS_DIR
        try:
            legacy.RESULTS_DIR = directory
            capture(
                "execution_evidence",
                lambda: legacy.observe_execution(owner, dp_rank=dp_rank, tp_rank=tp_rank, pp_rank=pp_rank),
            )
        finally:
            legacy.RESULTS_DIR = previous
    if errors:
        payload["error"] = "; ".join(errors)
        logging.getLogger(__name__).warning("Runtime observation incomplete: %s", payload["error"])
    directory.mkdir(parents=True, exist_ok=True)
    suffix = f"-tp{tp_rank}-pp{pp_rank}" if kind == "worker" else ""
    path = directory / f"runtime-observation-{kind}-dp{dp_rank}{suffix}.json"
    legacy._write_execution_file(path, payload)
