# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AISimulate-owned AIC KV-capacity materialization.

Both the engine-only and Dynamo replay compositions use this module so their
rank-local KV capacity is derived from the same defaults and AIC argument set.
"""

from __future__ import annotations

from functools import cache
from typing import Any

DEFAULT_BACKEND_VERSIONS = {
    "vllm": "0.19.0",
    "sglang": "0.5.10",
    "trtllm": "1.3.0rc10",
}
DEFAULT_GPU_MEMORY_UTILIZATION = 0.9
DEFAULT_MEM_FRACTION_STATIC = 0.88
DEFAULT_FREE_GPU_MEMORY_FRACTION = 0.9

_DEFAULT_AIC_SYSTEM = "h200_sxm"
_DEFAULT_MAX_NUM_BATCHED_TOKENS = 8192
_DEFAULT_MAX_NUM_SEQUENCES = 1
_DEFAULT_BLOCK_SIZES = {"vllm": 64, "sglang": 1, "trtllm": 32}


def materialize_aic_num_gpu_blocks(
    raw: dict[str, Any], *, memory_diagnostics: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Return engine arguments with rank-local AIC KV capacity materialized."""

    canonical_result = None

    def finish_lowering(value):
        if canonical_result is None:
            return value
        result = dict(canonical_result)
        for name in ("timing_model", "num_gpu_blocks", "tensor_parallel_size", "dp_size"):
            if name in value:
                result[name] = value[name]
        for name in (
            "gpu_memory_utilization",
            "mem_fraction_static",
            "free_gpu_memory_fraction",
            "cuda_graph_reserved_bytes",
            "systems_path",
        ):
            result.pop(name, None)
        return result

    lowered = dict(raw)
    timing = lowered.get("timing_model")
    timing_system_roots = None
    if isinstance(timing, dict) and timing.get("type") == "external" and timing.get("provider") == "aic":
        authored = timing.get("config")
        if not isinstance(authored, dict):
            raise ValueError("external AIC timing config must be a mapping")
        timing_system_roots = authored.get("systems_paths")
        if not timing_system_roots and authored.get("systems_path") is not None:
            timing_system_roots = [authored["systems_path"]]
        if "estimation_mode" in authored or "estimator_config" in authored:
            from aisimulate_core.sdk import RustForwardPassPerfModel

            memory_fields = {
                key: value
                for key, value in authored.items()
                if key
                in {
                    "gpu_memory_utilization",
                    "mem_fraction_static",
                    "free_gpu_memory_fraction",
                    "cuda_graph_reserved_bytes",
                }
            }
            for name in (
                "gpu_memory_utilization",
                "mem_fraction_static",
                "free_gpu_memory_fraction",
                "cuda_graph_reserved_bytes",
            ):
                if name in lowered:
                    if name in memory_fields and memory_fields[name] != lowered[name]:
                        raise ValueError(f"{name} conflicts with canonical timing configuration")
                    memory_fields[name] = lowered[name]
            request = {key: value for key, value in authored.items() if key not in memory_fields}
            model = RustForwardPassPerfModel.best_available(request)
            try:
                diagnostics = model.diagnostics()
            finally:
                model.close()
            if diagnostics.get("readiness") != "ready":
                raise ValueError("regression estimator is not ready; replay requires training observations")
            canonical_result = dict(raw)
            resolved = diagnostics["provenance"]["config"]
            lowered["timing_model"] = {**timing, "config": {**resolved, **memory_fields}}
            for names, expected in (
                (("tensor_parallel_size", "aic_tp_size"), resolved["tp"]),
                (("dp_size", "aic_attention_dp_size"), resolved["attention_dp"]),
            ):
                for name in names:
                    if name in lowered and lowered[name] != expected:
                        raise ValueError(f"{name} conflicts with canonical timing topology")
                lowered[names[0]] = expected
                lowered[names[1]] = expected
            lowered.update(
                {
                    key: value
                    for key, value in {
                        "aic_model_path": resolved["model"],
                        "aic_system": resolved["system"],
                        "aic_backend": resolved["backend"],
                        "aic_backend_version": resolved["backend_version"],
                        "aic_pp_size": resolved["pp"],
                        "aic_moe_tp_size": resolved["moe_tp_size"],
                        "aic_moe_ep_size": resolved["moe_ep_size"],
                    }.items()
                    if value is not None
                }
            )
            for target, source in (
                ("aic_gemm_dtype", "gemm_quant_mode"),
                ("aic_moe_dtype", "moe_quant_mode"),
                ("aic_fmha_dtype", "fmha_quant_mode"),
                ("aic_kv_cache_dtype", "kvcache_quant_mode"),
                ("aic_comm_dtype", "comm_quant_mode"),
            ):
                if resolved[source] is not None:
                    lowered[target] = resolved[source]
            for name in ("moe_backend", "attention_backend", "enable_eplb", "wideep_num_slots"):
                if resolved.get(name) is not None and not (name == "enable_eplb" and resolved[name] is False):
                    lowered[f"aic_{name}"] = resolved[name]
            if resolved["systems_paths"]:
                lowered["systems_path"] = resolved["systems_paths"][0]
            for name, value in memory_fields.items():
                if name in lowered and lowered[name] != value:
                    raise ValueError(f"{name} conflicts with canonical timing configuration")
                lowered.setdefault(name, value)

    attention_dp = lowered.get("aic_attention_dp_size")
    dp = attention_dp or 1
    configured_dp = lowered.get("dp_size") or 1
    has_aic_config = lowered.get("aic_backend") is not None or attention_dp is not None
    if has_aic_config and configured_dp > 1 and configured_dp != dp:
        raise ValueError(
            "dp_size must match aic_attention_dp_size for AIC-backed replay "
            f"(got dp_size={configured_dp}, aic_attention_dp_size={dp})"
        )
    if attention_dp is not None and dp > 1:
        lowered["dp_size"] = dp

    if lowered.get("num_gpu_blocks") is not None:
        return finish_lowering(lowered)
    backend = lowered.get("aic_backend")
    if backend is None:
        return finish_lowering(lowered)
    if not isinstance(backend, str) or backend not in DEFAULT_BACKEND_VERSIONS:
        supported = ", ".join(sorted(DEFAULT_BACKEND_VERSIONS))
        raise ValueError(
            f"AIC KV cache capacity estimation does not support {backend!r}; supported backends: {supported}"
        )
    model = lowered.get("aic_model_path")
    if not model:
        raise ValueError("AIC KV cache capacity estimation requires aic_model_path in engine args")

    capacity_systems_path = lowered.get("systems_path")
    if timing_system_roots and canonical_result is None:
        from .sweeper.forward_pass_estimator import resolve_systems_paths

        resolved_roots = resolve_systems_paths(timing_system_roots)
        if resolved_roots:
            # Native timing gives systems_paths precedence over systems_path;
            # canonical lowering has already pinned the selected root above.
            capacity_systems_path = resolved_roots[0]

    lowered["num_gpu_blocks"] = estimate_num_gpu_blocks(
        backend_name=backend,
        system=lowered.get("aic_system") or _DEFAULT_AIC_SYSTEM,
        model_path=model,
        tp_size=(lowered.get("aic_tp_size") if lowered.get("aic_tp_size") is not None else 1),
        block_size=_resolve_block_size(lowered, backend),
        max_num_batched_tokens=(
            lowered.get("max_num_batched_tokens")
            if lowered.get("max_num_batched_tokens") is not None
            else _DEFAULT_MAX_NUM_BATCHED_TOKENS
        ),
        max_num_sequences=(
            lowered.get("max_num_seqs") if lowered.get("max_num_seqs") is not None else _DEFAULT_MAX_NUM_SEQUENCES
        ),
        gpu_memory_utilization=lowered.get("gpu_memory_utilization"),
        mem_fraction_static=lowered.get("mem_fraction_static"),
        free_gpu_memory_fraction=lowered.get("free_gpu_memory_fraction"),
        backend_version=(
            lowered.get("aic_backend_version")
            if lowered.get("aic_backend_version") is not None
            else lowered.get("backend_version")
        ),
        pp_size=(lowered.get("aic_pp_size") if lowered.get("aic_pp_size") is not None else 1),
        moe_tp_size=lowered.get("aic_moe_tp_size"),
        moe_ep_size=lowered.get("aic_moe_ep_size"),
        attention_dp_size=attention_dp,
        gemm_dtype=lowered.get("aic_gemm_dtype"),
        moe_dtype=lowered.get("aic_moe_dtype"),
        fmha_dtype=lowered.get("aic_fmha_dtype"),
        kv_cache_dtype=lowered.get("aic_kv_cache_dtype"),
        comm_dtype=lowered.get("aic_comm_dtype"),
        **{
            name: lowered[f"aic_{name}"]
            for name in ("moe_backend", "attention_backend", "enable_eplb", "wideep_num_slots")
            if lowered.get(f"aic_{name}") is not None
        },
        systems_path=capacity_systems_path,
        cuda_graph_reserved_bytes=lowered.get("cuda_graph_reserved_bytes", 0),
        **({"diagnostics": memory_diagnostics} if memory_diagnostics is not None else {}),
    )
    return finish_lowering(lowered)


def estimate_num_gpu_blocks(
    *,
    backend_name: str,
    system: str,
    model_path: str,
    tp_size: int,
    block_size: int,
    max_num_batched_tokens: int,
    max_num_sequences: int = _DEFAULT_MAX_NUM_SEQUENCES,
    gpu_memory_utilization: float | None = None,
    mem_fraction_static: float | None = None,
    free_gpu_memory_fraction: float | None = None,
    backend_version: str | None = None,
    pp_size: int = 1,
    moe_tp_size: int | None = None,
    moe_ep_size: int | None = None,
    attention_dp_size: int | None = None,
    gemm_dtype: str | None = None,
    moe_dtype: str | None = None,
    fmha_dtype: str | None = None,
    kv_cache_dtype: str | None = None,
    comm_dtype: str | None = None,
    moe_backend: str | None = None,
    attention_backend: str | None = None,
    enable_eplb: bool = False,
    wideep_num_slots: int | None = None,
    systems_path: str | None = None,
    cuda_graph_reserved_bytes: int = 0,
    diagnostics: dict[str, Any] | None = None,
) -> int:
    """Estimate per-rank KV blocks using the replay-wide AIC contract.

    NextN is intentionally absent. AIC currently can return negative KV
    capacity for Eagle when speculative-decoding state is included. Timing
    compilation still receives NextN; only capacity estimation omits it.
    """

    if backend_name not in DEFAULT_BACKEND_VERSIONS:
        supported = ", ".join(sorted(DEFAULT_BACKEND_VERSIONS))
        raise ValueError(
            f"AIC KV cache capacity estimation does not support {backend_name!r}; supported backends: {supported}"
        )
    from aisimulate_core.sdk.config_builders import validate_moe_controls
    from aisimulate_core.sdk.memory import (
        estimate_num_gpu_blocks as aic_estimate_num_gpu_blocks,
    )

    if backend_version is None:
        from aisimulate_core.sdk.perf_database import get_latest_database_version

        # Use the maintained current slot (or the latest version in a custom
        # legacy root), matching the database used for this capacity estimate.
        backend_version = get_latest_database_version(system, backend_name, systems_paths=systems_path)
        if backend_version is None:
            raise ValueError(f"no perf database for system={system!r}, backend={backend_name!r}")

    validate_moe_controls(enable_eplb=enable_eplb, wideep_num_slots=wideep_num_slots)

    if backend_name == "trtllm":
        memory_fraction_kind = "of_free"
        memory_fraction_value = (
            free_gpu_memory_fraction if free_gpu_memory_fraction is not None else DEFAULT_FREE_GPU_MEMORY_FRACTION
        )
    elif backend_name == "sglang":
        memory_fraction_kind = "of_total"
        memory_fraction_value = mem_fraction_static if mem_fraction_static is not None else DEFAULT_MEM_FRACTION_STATIC
    else:
        memory_fraction_kind = "of_total"
        memory_fraction_value = (
            gpu_memory_utilization if gpu_memory_utilization is not None else DEFAULT_GPU_MEMORY_UTILIZATION
        )

    return int(
        aic_estimate_num_gpu_blocks(
            model_path,
            system,
            backend_name,
            backend_version=backend_version,
            scheduler_block_size=block_size,
            max_num_tokens=max_num_batched_tokens,
            max_batch_size=max_num_sequences,
            memory_fraction_kind=memory_fraction_kind,
            memory_fraction_value=memory_fraction_value,
            tp_size=tp_size,
            pp_size=pp_size,
            attention_dp_size=(attention_dp_size if attention_dp_size is not None else 1),
            moe_tp_size=moe_tp_size,
            moe_ep_size=moe_ep_size,
            gemm_quant_mode=_quant_mode_name("gemm", gemm_dtype),
            moe_quant_mode=_quant_mode_name("moe", moe_dtype),
            fmha_quant_mode=_quant_mode_name("fmha", fmha_dtype),
            kvcache_quant_mode=_quant_mode_name("kvcache", kv_cache_dtype),
            comm_quant_mode=_quant_mode_name("comm", comm_dtype),
            **{
                name: value
                for name, value in (
                    ("moe_backend", moe_backend),
                    ("attention_backend", attention_backend),
                    ("enable_eplb", enable_eplb),
                    ("wideep_num_slots", wideep_num_slots),
                )
                if value is not None and not (name == "enable_eplb" and value is False)
            },
            systems_path=systems_path,
            cuda_graph_reserved_bytes=cuda_graph_reserved_bytes,
            **({"diagnostics": diagnostics} if diagnostics is not None else {}),
        )
    )


def estimate_kv_bytes_per_token(
    model_name: str,
    *,
    tp_size: int,
    pp_size: int,
    moe_tp_size: int = 1,
    moe_ep_size: int = 1,
    kvcache_quant_mode: str | None = None,
) -> int:
    """Derive per-rank KV bytes/token from the resolved Hugging Face config."""

    from aisimulate_core.sdk.memory import NaiveKVCacheEstimator

    estimator = NaiveKVCacheEstimator.from_model_path(
        model_name,
        tp_size=tp_size,
        pp_size=pp_size,
        moe_tp_size=moe_tp_size,
        moe_ep_size=moe_ep_size,
        allow_hf_config_download=True,
    )
    kvcache_quant_mode = _quant_mode_name("kvcache", kvcache_quant_mode)
    if kvcache_quant_mode is not None:
        from aisimulate_core.sdk.common import KVCacheQuantMode

        try:
            estimator.dtype_bytes = int(KVCacheQuantMode[kvcache_quant_mode].value.memory)
        except KeyError as exc:
            raise ValueError(f"unsupported kvcache_quant_mode {kvcache_quant_mode!r}") from exc
    value = estimator.kv_bytes_per_token()
    if value is None or value <= 0:
        raise ValueError(f"could not derive KV bytes per token for model {model_name!r}")
    return int(value)


@cache
def resolve_model_context_length(model_name: str) -> int:
    """Resolve ``context_length: max`` from a local, cached, or HF config."""

    from aisimulate_core.sdk.memory import NaiveKVCacheEstimator

    # Both modules ship in the same distribution; reuse the canonical loader.
    config = NaiveKVCacheEstimator._load_config(model_name, allow_hf_config_download=True)
    if not isinstance(config, dict):
        raise ValueError(f"could not load Hugging Face config for {model_name!r}")
    text_config = config.get("text_config")
    mappings = [config]
    if isinstance(text_config, dict):
        mappings.insert(0, text_config)
    for mapping in mappings:
        for name in (
            "max_position_embeddings",
            "model_max_length",
            "max_sequence_length",
            "seq_length",
            "n_positions",
        ):
            value = mapping.get(name)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
    raise ValueError(f"Hugging Face config for {model_name!r} does not declare a maximum context length")


def _resolve_block_size(raw: dict[str, Any], backend: str) -> int:
    block_size = raw.get("block_size")
    if block_size is not None:
        return int(block_size)
    if backend == "sglang":
        sglang = raw.get("sglang")
        if isinstance(sglang, dict) and sglang.get("page_size") is not None:
            return int(sglang["page_size"])
    return _DEFAULT_BLOCK_SIZES[backend]


def _quant_mode_name(field: str, value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"AIC {field} quant mode must be a string when set")
    normalized = value.strip()
    if not normalized or normalized.lower() in {"auto", "none", "null"}:
        return None
    if normalized == "int4":
        normalized = "int4_wo"

    from aisimulate_core.sdk import common

    enum_cls = {
        "gemm": common.GEMMQuantMode,
        "moe": common.MoEQuantMode,
        "fmha": common.FMHAQuantMode,
        "kvcache": common.KVCacheQuantMode,
        "comm": common.CommQuantMode,
    }[field]
    try:
        return enum_cls[normalized].name
    except KeyError:
        allowed = ", ".join(member.name for member in enum_cls)
        raise ValueError(f"unsupported AIC {field} quant mode {value!r}; supported values: {allowed}") from None
