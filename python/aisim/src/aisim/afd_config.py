# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Input-boundary adapters for common worker settings on an AFD pool.

These adapters do not select estimators or price operations. Public workers,
SDK searches and replay compilation share the same normalized worker settings.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy


def matches_integer_domain(value, domain):
    """Check a finite candidate without expanding a potentially large range."""
    if isinstance(domain, dict) and "range" in domain:
        bounds = domain["range"]
        return bounds["min"] <= value <= bounds["max"] and (
            bounds.get("scale", "linear") == "log" or (value - bounds["min"]) % bounds.get("step", 1) == 0
        )
    if isinstance(domain, dict) and "choices" in domain:
        return value in domain["choices"]
    if isinstance(domain, frozenset):
        return value in domain
    return value == domain


def normalize_afd_worker_input(value):
    if not isinstance(value, Mapping):
        return value
    value = dict(value)
    if hasattr(value.get("afd"), "model_dump"):
        value["afd"] = value["afd"].model_dump(mode="python", exclude_unset=True)
    if not isinstance(value.get("afd", {}), Mapping):
        return value
    afd = dict(value.get("afd", {}))
    hardware = value.pop("hardware", None)
    if hardware is not None:
        for name in ("a_system", "f_system"):
            if afd.get(name) is None:
                afd[name] = hardware
    parallel = value.get("parallelism")
    if hasattr(parallel, "model_dump"):
        parallel = parallel.model_dump(mode="python", exclude_unset=True)
    if isinstance(parallel, Mapping):
        parallel = dict(parallel)
        for name, target in (("tensor", "tp_a"), ("moe_expert", "f_moe_ep_size")):
            selected = parallel.pop(name, None)
            if selected is not None:
                if afd.get(target) is not None and afd[target] != selected:
                    raise ValueError(f"parallelism.{name} conflicts with afd.{target}")
                afd[target] = selected
        value["parallelism"] = parallel or None
    value["afd"] = afd
    return value


def validate_afd_worker(worker):
    parallel = worker.parallelism
    if parallel is not None:
        for name in ("replicas", "pipeline", "moe_tensor", "prefill_context", "decode_context"):
            if getattr(parallel, name, None) not in (None, 1):
                raise ValueError(f"AFD parallelism.{name} supports only 1")
        if getattr(parallel, "preset", "default") not in ("default", False, {}):
            raise ValueError("AFD parallelism uses independent fields, not preset lists")
    cache = worker.kv_cache
    if cache is not None:
        if cache.block_size not in (None, 1):
            raise ValueError("AFD kv_cache.block_size supports only 1")
        if cache.bytes_per_token != "auto":
            raise ValueError("AFD kv_cache.bytes_per_token requires auto")
        for name in ("host_offload", "hicache", "g3_offload", "kvcr", "state_cache", "prefix_match_unit"):
            if getattr(cache, name, None) is not None:
                raise ValueError(f"AFD kv_cache.{name} is unsupported")
        if cache.decode_radix_cache:
            raise ValueError("decode_radix_cache is unsupported for AFD")
        capacity = cache.capacity
        if capacity.memory_fraction is not None or getattr(capacity, "cuda_graph_reserved_bytes", 0):
            raise ValueError("AFD does not yet support memory_fraction or cuda_graph_reserved_bytes overrides")
        if getattr(capacity, "bytes", None) is not None:
            raise ValueError("AFD fixed capacity requires blocks, not bytes")
    if worker.timing is not None:
        timing = worker.timing
        if timing.type != "default" or timing.estimation_mode not in (None, "auto", "op_level"):
            raise ValueError("AFD timing requires default op-level timing")
        if timing.fpm_parquet_path is not None or timing.forward_model != "op_level":
            raise ValueError("AFD timing requires op_level")
    scheduler = worker.scheduler
    if scheduler is not None and (
        getattr(scheduler, "prefill_schedule_interval", 1) != 1
        or getattr(scheduler, "async_scheduling", None) is not None
    ):
        raise ValueError("AFD uses the SGLang scheduler; vLLM scheduling controls are unsupported")
    return worker


def afd_worker_runtime(settings):
    """Lower already validated, concrete common settings to flat replay args."""
    result = {}
    scheduler = settings.get("scheduler") or {}
    for source, target in (
        ("max_sequences", "max_num_seqs"),
        ("max_batched_tokens", "max_num_batched_tokens"),
        ("prefill_decode_interval", "prefill_decode_interval"),
    ):
        if scheduler.get(source) is not None:
            result[target] = scheduler[source]
    cache = settings.get("kv_cache") or {}
    if "prefix_caching" in cache:
        result["enable_prefix_caching"] = cache["prefix_caching"]
    if cache.get("block_size") is not None:
        result["block_size"] = cache["block_size"]
    capacity = cache.get("capacity") or {}
    if capacity.get("type") == "fixed":
        result["num_gpu_blocks"] = capacity["blocks"]
    return result


def afd_worker_timing(settings, *, quant, estimator, attention_backend=None):
    timing = settings.get("timing") or {}
    quant = dict(quant)
    estimator = dict(estimator)
    for name in ("gemm_quant_mode", "moe_quant_mode", "fmha_quant_mode", "kvcache_quant_mode", "comm_quant_mode"):
        if timing.get(name) is not None:
            quant[name] = timing[name]
    for name in (
        "database_mode",
        "systems_paths",
        "transfer_policy",
        "estimation_mode",
        "fallback_policy",
        "estimator_config",
    ):
        if timing.get(name) is not None:
            estimator[name] = deepcopy(timing[name])
    return {
        "quant": quant,
        "estimator": estimator,
        **(
            {"attention_backend": timing.get("attention_backend") or attention_backend}
            if timing.get("attention_backend") or attention_backend
            else {}
        ),
    }


def normalize_afd_search_input(value):
    """Consume explicit SDK role knobs before Pydantic supplies ordinary defaults."""
    if not isinstance(value, dict) or not isinstance(value.get("afd_roles"), dict) or not value["afd_roles"]:
        return value
    # Preserve the deployment-capability diagnostic before processing role knobs.
    if value.get("encoder") is not None:
        return value
    if value.get("agg_vision") or value.get("prefill_vision"):
        raise ValueError("image workloads are unsupported for AFD")
    value = dict(value)
    roles = deepcopy(value["afd_roles"])
    controls = value.get("role_estimator_controls", {})
    controls = deepcopy(controls) if isinstance(controls, dict) else None
    for role, worker in roles.items():
        if hasattr(worker, "model_dump"):
            worker = worker.model_dump(mode="python", exclude_none=True)
            roles[role] = worker
        if not isinstance(role, str) or not isinstance(worker, dict):
            return value
        legacy = "agg" if role == "aggregated" else role
        prefix = legacy + "_"
        if controls is not None and isinstance(controls.get(legacy), dict):
            timing = dict(worker.get("timing") or {})
            for name, selected in controls.pop(legacy).items():
                if name in timing and timing[name] != selected:
                    raise ValueError(f"role_estimator_controls.{legacy}.{name} conflicts with workers.{role}.timing")
                timing[name] = selected
            if timing:
                worker["timing"] = timing
        scheduler = dict(worker.get("scheduler") or {})
        cache = dict(worker.get("kv_cache") or {})
        for key in list(value):
            if not key.startswith(prefix):
                continue
            name = key[len(prefix) :]
            selected = value.pop(key)
            if selected is None:
                continue
            if name in ("max_num_seqs", "max_num_batched_tokens"):
                target = "max_sequences" if name == "max_num_seqs" else "max_batched_tokens"
                selected = {"choices": selected} if isinstance(selected, list) else selected
                if target in scheduler and scheduler[target] != selected:
                    raise ValueError(f"{key} conflicts with workers.{role}.scheduler.{target}")
                scheduler[target] = selected
            elif name in ("enable_prefix_caching", "block_size", "num_gpu_blocks"):
                target = {
                    "enable_prefix_caching": "prefix_caching",
                    "block_size": "block_size",
                    "num_gpu_blocks": "capacity",
                }[name]
                selected = {"type": "fixed", "blocks": selected} if name == "num_gpu_blocks" else selected
                if target in cache and cache[target] != selected:
                    raise ValueError(f"{key} conflicts with workers.{role}.kv_cache.{target}")
                cache[target] = selected
            elif name in ("context_length", "startup_time"):
                target = "context_length" if name == "context_length" else "startup_seconds"
                if target in worker and worker[target] != selected:
                    raise ValueError(f"{key} conflicts with workers.{role}.{target}")
                worker[target] = selected
            elif name == "hardware_sku":
                worker["hardware"] = selected
            elif name == "forward_model" and selected == "op_level":
                timing = worker.setdefault("timing", {})
                if timing.get("estimation_mode", "op_level") != "op_level":
                    raise ValueError(f"{key} conflicts with workers.{role}.timing.estimation_mode")
                timing["estimation_mode"] = "op_level"
            elif name == "forward_model":
                raise ValueError("role_estimator_controls require regular language workers; AFD requires op_level")
            elif name == "radix_cache" and role == "decode":
                if selected:
                    raise ValueError("decode_radix_cache is unsupported for AFD")
            else:
                raise ValueError(f"AFD {key} is unsupported; configure supported common worker fields")
        if scheduler:
            worker["scheduler"] = scheduler
        if cache:
            worker["kv_cache"] = cache
        integer_ranges = value.get("engine_integer_log_ranges", {})
        if isinstance(integer_ranges, dict):
            integer_ranges = dict(integer_ranges)
            for name, bounds in list(integer_ranges.items()):
                if not name.startswith(prefix):
                    continue
                targets = {"max_num_seqs": "max_sequences", "max_num_batched_tokens": "max_batched_tokens"}
                suffix = name[len(prefix) :]
                if suffix not in targets or not isinstance(bounds, list) or len(bounds) != 2:
                    raise ValueError(f"AFD {name} has an unsupported integer range")
                target = targets[suffix]
                domain = {"range": {"min": bounds[0], "max": bounds[1], "scale": "log"}}
                previous = scheduler.get(target)
                if previous is not None and previous not in (domain, {"choices": [bounds[0]]}, bounds[0]):
                    raise ValueError(f"{name} range conflicts with workers.{role}.scheduler.{target}")
                scheduler[target] = domain
                worker["scheduler"] = scheduler
                integer_ranges.pop(name)
            if "engine_integer_log_ranges" in value:
                value["engine_integer_log_ranges"] = integer_ranges
        for map_name in ("engine_float_ranges", "engine_log_ranges"):
            if any(name.startswith(prefix) for name in value.get(map_name, {})):
                raise ValueError(f"AFD {map_name} overrides are unsupported")
    value["afd_roles"] = roles
    if controls is not None and "role_estimator_controls" in value:
        value["role_estimator_controls"] = controls
    return value
