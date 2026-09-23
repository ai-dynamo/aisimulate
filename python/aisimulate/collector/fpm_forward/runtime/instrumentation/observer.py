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


def _resolved(owner: Any) -> dict:
    config = owner.vllm_config
    value = legacy.resolved_config(config)
    for key in ("speculative_config", "kv_transfer_config"):
        value[key] = _json(getattr(config, key, None))
    for key in ("hf_config_path", "hf_overrides", "config_format"):
        if hasattr(config.model_config, key):
            value["model_config"][key] = _json(getattr(config.model_config, key))
    return value


def _raw_cache(config: Any) -> dict:
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
    return {
        "num_blocks": config.num_blocks,
        "groups": [
            {
                "layer_names": list(group.layer_names),
                "spec_type": legacy._class_name(group.kv_cache_spec),
                **{
                    key: _json(getattr(group.kv_cache_spec, key)) for key in fields if hasattr(group.kv_cache_spec, key)
                },
            }
            for group in config.kv_cache_groups
        ],
        "tensor_allocations": [
            {
                key: _json(getattr(tensor, key))
                for key in ("size", "shared_by", "block_stride", "offset")
                if hasattr(tensor, key)
            }
            for tensor in config.kv_cache_tensors
        ],
    }


def _groups(cache: dict, config: Any, version: str, pool_id: str) -> None:
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


def _layer_views(owner: Any, cache: dict, source_files: dict) -> None:
    runner = owner.model_runner
    layers = owner.vllm_config.compilation_config.static_forward_context
    storage_ids = {}
    for index, storage in enumerate(cache["storages"]):
        storage["storage_id"] = f"storage-{index}"
        storage_ids[(storage["device"], storage["pointer"])] = storage["storage_id"]
    for allocation in cache["tensor_allocations"]:
        key, _ = legacy._storage(layers[allocation["shared_by"][0]].kv_cache)
        allocation["storage_id"] = storage_ids[key]
    views = cache["layer_tensors"] = {}
    groups = []
    backends = set()
    for attention_groups in runner.attn_groups:
        for group in attention_groups:
            spec = group.kv_cache_spec
            kernel = legacy._positive(runner._kernel_block_sizes[group.kv_cache_group_id], "kernel block size")
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

    def capture(label, action):
        try:
            action()
        except Exception as error:
            payload["unresolved_fields"].append(label)
            errors.append(f"{label}: {type(error).__name__}: {error}")

    capture("runtime", lambda: _runtime(payload, manifest, version))
    capture("model_config_sha256", lambda: _loaded_config(owner, payload))
    capture("resolved_config", lambda: payload.update(resolved_config=_resolved(owner)))
    pool_id = f"dp{dp_rank}-shared-pool"
    if kind == "worker":
        capture("hardware", lambda: payload.update(hardware=_hardware(owner, payload["identity"]["gpu"])))
        capture("effective_offloader", lambda: payload.update(effective_offloader=legacy._offloader_type()))
        if hasattr(owner, "_fpm_initial_compilation_config"):
            payload["initial_compilation_config"] = owner._fpm_initial_compilation_config
        capture("cache", lambda: payload.update(cache=legacy.worker_memory(owner)))
        config = getattr(getattr(owner, "model_runner", None), "kv_cache_config", None)
    else:
        config = cache_config
        capture("cache", lambda: payload.update(cache=legacy.scheduler_memory(owner, config)))
    capture("raw_cache_config", lambda: payload.update(raw_cache_config=_raw_cache(config)))
    if "cache" in payload:
        cache = payload["cache"]
        cache["pool_id"] = pool_id
        capture("cache.groups", lambda: _groups(cache, config, version, pool_id))
        if kind == "worker":
            capture("cache.semantics", lambda: _semantics(owner, payload))
            capture("cache.layer_tensors", lambda: _layer_views(owner, cache, manifest["runtime"]["source_files"]))
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
