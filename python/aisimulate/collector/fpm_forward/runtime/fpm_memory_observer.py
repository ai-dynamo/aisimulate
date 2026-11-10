# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observe initialized vLLM memory without changing execution or profiling.

Original collector adapters against vLLM 0.27.0, commit
4bdc8a788d2e2ce9165d552b3d4d8b72604626bf: vllm/v1/worker/gpu_worker.py
(determine_available_memory, initialize_from_config, compile_or_warm_up_model),
vllm/v1/worker/gpu_model_runner.py (initialize_kv_cache_tensors), and
vllm/v1/core/{kv_cache_manager,block_pool}.py. Cache specs are read, not rebuilt.
https://github.com/vllm-project/vllm/tree/4bdc8a788d2e2ce9165d552b3d4d8b72604626bf/vllm/v1

The same reads were audited against vLLM 0.32.0, commit
5df63a7203cdb9328d824a27b682e77c7b9928be, and 0.33.0, commit
4ba787ec9959e566d3b587a796dff07673324da5. Lines are given as 0.32.0/0.33.0:
- vllm/v1/worker/gpu_worker.py: determine_available_memory 634/641,
  initialize_from_config 853/866, compile_or_warm_up_model 908/933, and the
  runner choice in _make_model_runner 563-579/570-586.
- vllm/v1/worker/gpu/model_runner.py (Model Runner V2): initialize_kv_cache
  605/623, kv_caches 791-793/812-814.
- vllm/v1/worker/gpu_model_runner.py (Model Runner V1):
  initialize_kv_cache_tensors 6979/6914, initialize_kv_cache 7055/6990.
- vllm/v1/worker/utils.py (identical at both tags): allocate_kv_cache 564-627,
  bind_kv_cache_to_layers 868-885.
- vllm/v1/worker/gpu/attn_utils.py: init_kv_cache 396-443/397-444.
- vllm/v1/kv_cache_interface.py: KVCacheTensor 1409-1429/1475-1495,
  KVCacheGroupSpec 1439-1456/1505-1522, create_kv_cache_views 377-441/418-482.
- Layout: vllm/v1/core/kv_cache_utils.py:1588-1687 at 0.32.0 and
  vllm/v1/core/kv_cache_planning.py:642-741 at 0.33.0.
- vllm/v1/core/kv_cache_manager.py: block_pool 197/199, watermark_blocks 208/210.
- vllm/v1/core/block_pool.py: null_block 187-188/191-192, get_num_free_blocks 868/912.
https://github.com/vllm-project/vllm/tree/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1
https://github.com/vllm-project/vllm/tree/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1

Unsupported observation APIs write unresolved evidence. Exceptions from the
runtime itself are never caught by these observation helpers or wrappers.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import os
import re
import socket
import threading
from enum import Enum
from pathlib import Path
from typing import Any

SCHEMA_NAME = "aisimulate_fpm_runtime_memory"
SCHEMA_VERSION = 1
MEMORY_SUPPORTED_VERSIONS = ("0.27.0", "0.32.0", "0.33.0")
EXECUTION_SUPPORTED_VERSIONS = ("0.27.0", "0.28.0", "0.32.0", "0.33.0")
# The one vLLM version whose probe records carry the hybrid cache evidence. The probe compares the exact string.
HYBRID_EVIDENCE_VERSION = "0.33.0"
RESULTS_DIR = Path("/results")
_SPEC_MODULE = "vllm.v1.kv_cache_interface."
_UNIFORM_SPEC = _SPEC_MODULE + "UniformTypeKVCacheSpecs"
_MAMBA_SPEC = _SPEC_MODULE + "MambaSpec"
_CONV_CLASS = "vllm.models.inkling.nvidia.sconv_swa_attn.InklingConvState"
# Cache spec classes of vllm/v1/kv_cache_interface.py at vLLM 0.33.0
# (4ba787ec9959e566d3b587a796dff07673324da5), keyed by class name. The table
# maps by class, not by get_kv_cache_spec_kind: CircularBufferSpec has no spec
# kind (1421-1459) but holds one block per request (982-989). RSWASpec is
# left out: its manager frees the blocks between the prompt and the decode
# window (v1/core/single_type_kv_cache_manager.py:900-943), which no kind models.
SPEC_KINDS: dict[str, str] = {
    **dict.fromkeys(
        (
            "FullAttentionSpec",
            "MLAAttentionSpec",
            "SinkFullAttentionSpec",
            "HiddenStateCacheSpec",
            "SlidingWindowSpec",
            "SlidingWindowMLASpec",
        ),
        "attention",
    ),
    **dict.fromkeys(("CircularBufferSpec", "MambaSpec"), "state"),
}
# SlidingWindowSpec and its subclass SlidingWindowMLASpec (901-902, 1008) declare a positive window.
_WINDOW_SPECS = frozenset(("SlidingWindowSpec", "SlidingWindowMLASpec"))
# The spec classes that records without hybrid evidence accept (vLLM 0.27.0, 0.28.0, 0.32.0 and 0.33.0).
_LEGACY_SPECS = frozenset((_SPEC_MODULE + "FullAttentionSpec", _SPEC_MODULE + "SlidingWindowSpec"))
# Cache settings that only hybrid evidence records (vLLM 0.33.0 config/cache.py:187-304).
_HYBRID_CACHE_FIELDS = (
    "block_size",
    "mamba_cache_mode",
    "mamba_block_size",
    "prefix_match_unit",
    "mamba_cache_dtype",
    "mamba_ssm_cache_dtype",
)
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


def cpu_affinity_snapshot(*, task_directory: Path = Path("/proc/self/task")) -> dict[str, Any]:
    """Observe Linux thread affinity once, without binding or running a model.

    Linux affinity is per thread. Querying PID explicitly observes the main
    thread even when a runtime hook executes on another Python/native thread.
    The task scan is not atomic; changing or unreadable threads stay partial.
    CPU IDs are logical CPUs, not a physical-core or NUMA placement claim.
    """
    pid = os.getpid()
    payload: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "pid": pid,
        "observer_thread_id": threading.get_native_id(),
        "main_thread_allowed_cpus": [],
        "threads": [],
        "thread_ids_before": [],
        "thread_ids_after": [],
        "thread_errors": [],
        "errors": [],
        "status": "unavailable",
    }
    try:
        payload["main_thread_allowed_cpus"] = sorted(os.sched_getaffinity(pid))
        if not payload["main_thread_allowed_cpus"]:
            raise ValueError("main thread has an empty CPU affinity mask")
    except (AttributeError, OSError, ValueError) as error:
        payload["errors"].append(f"main thread affinity: {type(error).__name__}: {error}")
        return payload
    try:
        before = payload["thread_ids_before"] = sorted(
            int(path.name) for path in task_directory.iterdir() if path.name.isdecimal()
        )
        for tid in before:
            try:
                cpus = sorted(os.sched_getaffinity(tid))
                if not cpus:
                    raise ValueError("thread has an empty CPU affinity mask")
                payload["threads"].append({"tid": tid, "allowed_cpus": cpus})
            except (OSError, ValueError) as error:
                payload["thread_errors"].append({"tid": tid, "error": f"{type(error).__name__}: {error}"})
        after = payload["thread_ids_after"] = sorted(
            int(path.name) for path in task_directory.iterdir() if path.name.isdecimal()
        )
        if before != after:
            payload["errors"].append("thread list changed during observation")
        if payload["observer_thread_id"] not in before:
            payload["errors"].append("observation thread is missing from the task snapshot")
        main = [thread for thread in payload["threads"] if thread["tid"] == pid]
        if not main or main[0]["allowed_cpus"] != payload["main_thread_allowed_cpus"]:
            payload["errors"].append("main thread was missing or its affinity changed during observation")
    except OSError as error:
        payload["errors"].append(f"thread enumeration: {type(error).__name__}: {error}")
    payload["status"] = "partial" if payload["errors"] or payload["thread_errors"] else "observed"
    return payload


def observe_cpu(
    kind: str,
    *,
    dp_rank: int | None = None,
    tp_rank: int | None = None,
    pp_rank: int | None = None,
    requested_cpus_per_task: int | None = None,
    cpu_bind: str | None = None,
    local_gpu_count: int | None = None,
    directory: Path | None = None,
) -> dict[str, Any]:
    """Save independent CPU evidence; unavailable observation never changes memory.

    The launcher uses the same container and srun step as the engine, before
    engine startup. Its caller must inspect this saved result before proceeding.
    This helper records affinity; it does not set it or claim it remains fixed.
    """
    measurements = {
        "launcher": "before_engine_start",
        "worker": "after_warmup",
        "scheduler": "scheduler_initialized",
    }
    if kind not in measurements:
        raise ValueError("CPU observation requires launcher, worker or scheduler role")
    ranks = {"dp_rank": dp_rank, "tp_rank": tp_rank, "pp_rank": pp_rank}
    required = set(ranks) if kind == "worker" else {"dp_rank"} if kind == "scheduler" else set()
    if any(
        (type(value) is not int or value < 0) if key in required else value is not None for key, value in ranks.items()
    ):
        raise ValueError("CPU observation has invalid role/rank coordinates")
    if directory is None:
        directory, binding = _execution_destination()
    else:
        source = directory / "collector-provenance.json"
        raw = source.read_bytes()
        provenance = json.loads(raw)
        if not isinstance(provenance, dict):
            raise ValueError("CPU observation collector provenance must be an object")
        binding = {
            "collector_provenance": provenance,
            "execution_provenance": provenance,
            "provenance_source": {
                "path": str(source),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size_bytes": len(raw),
            },
        }
    payload = {
        "schema_name": "aisimulate_fpm_cpu_affinity",
        "schema_version": 1,
        "kind": kind,
        "measurement": measurements[kind],
        **ranks,
        **binding,
        **cpu_affinity_snapshot(),
    }
    if kind == "launcher":
        payload.update(
            requested_cpus_per_task=requested_cpus_per_task, cpu_bind=cpu_bind, local_gpu_count=local_gpu_count
        )
    suffix = (
        f"-dp{dp_rank}-tp{tp_rank}-pp{pp_rank}" if kind == "worker" else f"-dp{dp_rank}" if kind == "scheduler" else ""
    )
    path = directory / f"fpm-cpu-{kind}{suffix}.json"
    try:
        _write_execution_file(path, payload)
    except FileExistsError:
        raise RuntimeError(f"refusing duplicate runtime CPU observation: {path}") from None
    return payload


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


def _write_execution_file(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Publish atomically without replacing an earlier observation, including
        # when two worker processes accidentally report the same rank.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _execution_destination() -> tuple[Path, dict[str, Any]]:
    output_value = os.environ.get("FPM_EXECUTION_OUTPUT_DIR")
    provenance_value = os.environ.get("FPM_EXECUTION_PROVENANCE_FILE")
    custom = output_value is not None or provenance_value is not None
    if custom:
        if not output_value or not provenance_value:
            raise ValueError("FPM_EXECUTION_OUTPUT_DIR and FPM_EXECUTION_PROVENANCE_FILE must be supplied together")
        for name, value in (("output directory", output_value), ("provenance file", provenance_value)):
            if not Path(value).is_absolute() or any(char in value for char in ("\n", "\r", "\x00")):
                raise ValueError(f"execution observation {name} must be an absolute path without control characters")
        directory, source = Path(output_value), Path(provenance_value)
    else:
        directory, source = RESULTS_DIR, RESULTS_DIR / "collector-provenance.json"
    raw = source.read_bytes()
    provenance = json.loads(raw)
    if not isinstance(provenance, dict):
        raise ValueError("execution provenance must be a JSON object")
    binding = {
        "execution_provenance": provenance,
        "provenance_source": {"path": str(source), "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)},
    }
    if not custom:
        binding["collector_provenance"] = provenance
        return directory, binding
    if (
        provenance.get("purpose") != "matched_serving_accuracy"
        or not isinstance(provenance.get("run_id"), str)
        or not provenance["run_id"].strip()
        or "collector_attempt_id" in provenance
        or provenance.get("schema_name") == "aic_fpm_collector_provenance"
    ):
        raise ValueError("custom execution provenance requires a serving run_id and purpose=matched_serving_accuracy")
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / ".execution-run.json"
    if (directory / "collector-provenance.json").exists() or (
        not marker.exists() and any(directory.glob("fpm-execution-worker-*.json"))
    ):
        raise ValueError("serving execution output requires a fresh directory, separate from collector attempts")
    try:
        _write_execution_file(marker, binding)
    except FileExistsError:
        if json.loads(marker.read_bytes()) != binding:
            raise ValueError("execution observation directory already belongs to a different serving run") from None
    return directory, binding


def observe_execution(worker: Any, *, dp_rank: int, tp_rank: int, pp_rank: int) -> None:
    """Record selected backend objects after initialization, outside timed forwards.

    This original adapter reads AttentionGroup.backend/layer_names populated by
    GPUModelRunner.initialize_attn_backend, including runtime backend wrappers:
    https://github.com/vllm-project/vllm/blob/4bdc8a788d2e2ce9165d552b3d4d8b72604626bf/vllm/v1/worker/gpu_model_runner.py#L7020
    https://github.com/vllm-project/vllm/blob/4bdc8a788d2e2ce9165d552b3d4d8b72604626bf/vllm/v1/worker/utils.py#L217
    The same read-only interfaces were verified for vLLM 0.28.0 at:
    https://github.com/vllm-project/vllm/blob/2cf0a6915ce544dc493a0990f2ea38d81601128a/vllm/v1/worker/gpu_model_runner.py#L7155
    https://github.com/vllm-project/vllm/blob/2cf0a6915ce544dc493a0990f2ea38d81601128a/vllm/v1/worker/utils.py#L241
    They were also verified for vLLM 0.32.0, commit 5df63a7203cdb9328d824a27b682e77c7b9928be,
    and 0.33.0, commit 4ba787ec9959e566d3b587a796dff07673324da5. AttentionGroup keeps these
    fields (vllm/v1/worker/utils.py:261-266, identical at both tags). Model Runner V1 fills
    attn_groups in initialize_attn_backend (vllm/v1/worker/gpu_model_runner.py 0.32.0:6688,
    0.33.0:6623). Model Runner V2 assigns them from init_attn_backend
    (vllm/v1/worker/gpu/model_runner.py 0.32.0:665-674, 0.33.0:689-698;
    vllm/v1/worker/gpu/attn_utils.py 0.32.0:196, 0.33.0:197), which appends an empty list
    for a cache group without layer views (attn_utils.py 0.32.0:220-222, 0.33.0:221-223).
    An empty list adds no group.
    It does not infer a backend from requested flags or trace per-point dispatch.

    Ordinary serving can use FpmExecutionWorker with absolute
    FPM_EXECUTION_OUTPUT_DIR and FPM_EXECUTION_PROVENANCE_FILE paths. Both are
    required together; the provenance identifies a fresh matched-serving run.
    Collector launches retain their default /results output and provenance.
    """
    directory, binding = _execution_destination()
    payload = {
        "schema_name": "aisimulate_fpm_runtime_execution",
        "schema_version": 1,
        **binding,
        "backend_version": importlib.metadata.version("vllm"),
        "dp_rank": dp_rank,
        "tp_rank": tp_rank,
        "pp_rank": pp_rank,
        "per_point_dispatch": "unreported",
    }
    try:
        if payload["backend_version"] not in EXECUTION_SUPPORTED_VERSIONS:
            raise ValueError(f"execution observation requires audited vLLM {EXECUTION_SUPPORTED_VERSIONS}")
        groups = []
        for cache_groups in worker.model_runner.attn_groups:
            for group in cache_groups:
                names = list(group.layer_names)
                if not names or any(not isinstance(name, str) or not name for name in names):
                    raise ValueError("runtime attention group has no explicit layer names")
                backend = group.backend
                if not isinstance(backend, type):
                    raise ValueError("runtime attention group backend is not a class")
                groups.append(
                    {
                        "kv_cache_group_id": group.kv_cache_group_id,
                        "backend_class": f"{backend.__module__}.{backend.__qualname__}",
                        "layer_names": names,
                    }
                )
        if not groups:
            raise ValueError("runtime reported no initialized attention groups")
        payload.update(
            attention_groups=groups,
            graph_config=compilation_config(worker.vllm_config),
            resolved_config=execution_config(worker.vllm_config),
            status="observed",
        )
    except Exception as error:
        payload.update(status="unresolved", error=f"{type(error).__name__}: {error}")
        logging.getLogger(__name__).warning("FPM execution evidence is incomplete: %s", payload["error"])
    path = directory / f"fpm-execution-worker-dp{dp_rank}-tp{tp_rank}-pp{pp_rank}.json"
    if path.exists():
        raise RuntimeError(f"refusing duplicate runtime execution observation: {path}")
    try:
        _write_execution_file(path, payload)
    except FileExistsError:
        raise RuntimeError(f"refusing duplicate runtime execution observation: {path}") from None


def _cached_config_commit(model_config: Any) -> str | None:
    """Return the Hub snapshot commit in the local path of the loaded config.json, else None.

    File-system lookups only; nothing here may reach the network.
    """
    try:
        source = getattr(model_config, "hf_config_path", None) or model_config.model
        if os.path.isdir(source):
            resolved = os.path.join(source, "config.json")
            if not os.path.isfile(resolved):
                return None
        else:
            from huggingface_hub import try_to_load_from_cache

            # Only a pinned commit is evidence: the current cache ref is not proof of the loaded commit
            # (REVIEW.md lines 131-133). vLLM's ResolvedRevision.resolved holds it (huggingface_hub/_revision.py;
            # vLLM v0.32.0 config/model.py lines 632-641, 672).
            effective = getattr(model_config, "_hf_config_revision", None) or model_config.revision
            pinned = getattr(effective, "resolved", None)
            if not (isinstance(pinned, str) and re.fullmatch(r"[0-9a-f]{40}", pinned)):
                pinned = effective
            if not (isinstance(pinned, str) and re.fullmatch(r"[0-9a-f]{40}", pinned)):
                return None
            revision = str(pinned)
            resolved = try_to_load_from_cache(source, "config.json", revision=revision)
        if not isinstance(resolved, str):
            return None
        # Same rule as transformers' extract_commit_hash: first snapshots/<segment>/, lowercase commit.
        match = re.search(r"snapshots/([^/]+)/", resolved.replace("\\", "/"))
        if match and re.fullmatch(r"[0-9a-f]{40}", match.group(1)):
            return match.group(1)
    except Exception as error:
        logging.getLogger(__name__).debug("loaded config commit lookup failed: %s", type(error).__name__)
    return None


def execution_config(config: Any, *, hybrid_evidence: bool = False) -> dict[str, Any]:
    """Read execution identity without requiring the memory-finalization APIs.

    ``hybrid_evidence`` adds the vLLM 0.33.0 cache settings; without it the
    record keeps the fields that 0.27.0 and 0.28.0 records have.
    """
    sections = (
        {**_CONFIG_FIELDS, "cache_config": _CONFIG_FIELDS["cache_config"] + _HYBRID_CACHE_FIELDS}
        if hybrid_evidence
        else _CONFIG_FIELDS
    )
    result = {
        section: {
            key: _json_value(getattr(getattr(config, section), key))
            for key in fields
            if hasattr(getattr(config, section), key)
        }
        for section, fields in sections.items()
        if getattr(config, section, None) is not None
    }
    hf_config = getattr(config.model_config, "hf_config", None)
    if hf_config is not None and hasattr(hf_config, "_commit_hash"):
        if hf_config._commit_hash is not None:
            result["model_config"]["loaded_config_commit_hash"] = hf_config._commit_hash
    elif hf_config is not None:
        # transformers 5.19.0 (Dynamo 1.7.0-rc.1 image, configuration_utils.py line 366) discards
        # `_commit_hash`; 5.17.0 and earlier keep it. Read the commit from the local cache path instead.
        commit = _cached_config_commit(config.model_config)
        if commit is not None:
            result["model_config"]["loaded_config_commit_hash"] = commit
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


def resolved_config(config: Any, *, hybrid_evidence: bool = False) -> dict[str, Any]:
    result = execution_config(config, hybrid_evidence=hybrid_evidence)
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
    return result


def _offloader_type() -> str:
    # vLLM 0.27.0 model_executor/offloader/base.py:get_offloader is read-only.
    from vllm.model_executor.offloader import get_offloader

    return _class_name(get_offloader())


def _positive(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer, got {value!r}")
    return value


def _table_kind(spec: Any) -> str:
    spec_type = _class_name(spec)
    kind = SPEC_KINDS.get(spec_type.removeprefix(_SPEC_MODULE))
    if kind is None:
        raise ValueError(f"unsupported runtime cache spec: {spec_type}")
    return kind


def spec_kind(spec: Any) -> str:
    """Map a cache spec class to its FPM cache kind; report every other class.

    A UniformTypeKVCacheSpecs group takes the one kind of all its inner specs.
    """
    if _class_name(spec) != _UNIFORM_SPEC:
        return _table_kind(spec)
    inner = list(spec.kv_cache_specs.values())
    kinds = {_table_kind(item) for item in inner}
    if len(kinds) != 1:
        classes = sorted({_class_name(item) for item in inner})
        raise ValueError(f"unsupported runtime cache spec: {_UNIFORM_SPEC} of {classes}")
    return kinds.pop()


def _window(spec: Any) -> int | None:
    window = getattr(spec, "sliding_window", None)
    if _class_name(spec).removeprefix(_SPEC_MODULE) in _WINDOW_SPECS:
        return _positive(window, "sliding_window")
    if window is not None or getattr(spec, "attention_chunk_size", None) is not None:
        raise ValueError("full-attention cache with alternate retention is unsupported")
    return None


def _spec_dtype(spec: Any) -> str | None:
    # MambaSpec has one dtype per state tensor (`dtypes`), recorded in the group's state.
    return None if _class_name(spec) == _MAMBA_SPEC else str(spec.dtype)


def _layer_spec(spec: Any) -> dict[str, Any]:
    return {
        "spec_type": _class_name(spec),
        "page_size_bytes": _positive(spec.page_size_bytes, "cache spec page size"),
        "dtype": _spec_dtype(spec),
        **{
            key: _json_value(getattr(spec, key, None))
            for key in ("tokens_per_state", "cache_role", "storage_block_size")
        },
    }


def _state(spec: Any) -> dict[str, Any]:
    # A CircularBufferSpec holds one block per request (982-989).
    if _class_name(spec) != _MAMBA_SPEC:
        return {"blocks_per_request": 1}
    fields = (
        "mamba_cache_mode",
        "num_speculative_blocks",
        "num_prefill_checkpoint_blocks",
        "shapes",
        "dtypes",
        "real_page_size_bytes",
    )
    # The steady count: one state block plus the speculative blocks, held between block boundaries in both modes
    # (1140-1141). In align mode, vLLM's startup sizing counts 2 + k + c blocks (1135-1139). After a boundary, the
    # group also holds the previous state block for one or two steps: two with async scheduling, the vLLM 0.33.0
    # default (config/vllm.py:1685-1719, v1/core/single_type_kv_cache_manager.py:1640-1657).
    return {
        **{key: _json_value(getattr(spec, key)) for key in fields},
        "blocks_per_request": 1 + spec.num_speculative_blocks,
    }


def _shared(values: list[Any], label: str) -> Any:
    if any(value != values[0] for value in values):
        raise ValueError(f"cache group layers disagree on {label}")
    return values[0]


def cache_groups(cache_config: Any, *, hybrid_evidence: bool = False) -> list[dict[str, Any]]:
    """Record every cache group with its spec class, block, page, window and dtype.

    ``hybrid_evidence`` maps the vLLM 0.33.0 spec classes of SPEC_KINDS and adds
    each group's kind, the spec of each layer and state evidence. A
    UniformTypeKVCacheSpecs group keeps one inner spec per layer; its layers share
    one window and one state, and the group dtype is None when layers differ.
    Without it, only FullAttentionSpec and SlidingWindowSpec groups are supported
    and the record is the one vLLM 0.27.0, 0.28.0, 0.32.0 and 0.33.0 runtimes have.
    """
    groups = []
    all_names: set[str] = set()
    for group in cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        spec_type = _class_name(spec)
        kind = spec_kind(spec) if hybrid_evidence else None
        if not hybrid_evidence and spec_type not in _LEGACY_SPECS:
            raise ValueError(f"unsupported runtime cache spec: {spec_type}")
        if getattr(group, "is_eagle_group", False):
            raise ValueError("speculative cache groups are unsupported")
        # vLLM 0.32.0 KVCacheGroupSpec adds host_resident and role (v1/kv_cache_interface.py:1451, :1456;
        # 0.33.0 :1517, :1522). A missing field means the 0.27.0 group: device-resident, role DEFAULT.
        if getattr(group, "host_resident", False):
            raise ValueError("host-resident cache groups are unsupported")
        role = _json_value(getattr(group, "role", "DEFAULT"))
        if role != "DEFAULT":
            raise ValueError(f"unsupported cache group role: {role}")
        names = list(group.layer_names)
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("runtime cache groups require explicit layer names")
        if len(names) != len(set(names)) or all_names.intersection(names):
            raise ValueError("runtime cache group layer aliases are unsupported")
        all_names.update(names)
        if spec_type == _UNIFORM_SPEC:
            specs = dict(spec.kv_cache_specs)
            if list(specs) != names:
                raise ValueError("the inner specs of a uniform cache group do not match its layer names")
        else:
            specs = dict.fromkeys(names, spec)
        window = _shared([_window(item) for item in specs.values()], "sliding window")
        block_size = _positive(spec.block_size, "cache block size")
        page_size = _positive(spec.page_size_bytes, "cache spec page size")
        dtypes = {_spec_dtype(item) for item in specs.values()}
        record = {
            "layer_names": names,
            "spec_type": spec_type,
            "block_size_tokens": block_size,
            "spec_page_size_bytes": page_size,
            "sliding_window": window,
            # Uniform group layers can differ: an MLA indexer is uint8 at every KV precision.
            "dtype": dtypes.pop() if len(dtypes) == 1 else None,
        }
        if hybrid_evidence:
            record.update(kind=kind, layer_specs={name: _layer_spec(item) for name, item in specs.items()})
            if kind == "state":
                # A state group keeps blocks_per_request blocks, not a window of tokens.
                record.update(
                    sliding_window=None, state=_shared([_state(item) for item in specs.values()], "state evidence")
                )
        groups.append(record)
    if not groups:
        raise ValueError("runtime emitted no cache groups")
    return groups


def _storage(tensor: Any) -> tuple[tuple[str, int], int]:
    storage = tensor.untyped_storage()
    device = str(storage.device)
    if not device.startswith("cuda:"):
        raise ValueError(f"cache storage is not on a CUDA device: {device}")
    return (device, _positive(storage.data_ptr(), "storage pointer")), _positive(storage.nbytes(), "storage bytes")


def _layer_storages(kv_cache: Any) -> set[tuple[str, int]]:
    """Storage keys of one cache tensor, or of every tensor of a tuple or list.

    A vLLM 0.33.0 Mamba layer binds a tuple of state views of one page (model_executor/layers/mamba/abstract.py:28-42).
    """
    tensors = kv_cache if isinstance(kv_cache, (tuple, list)) else (kv_cache,)
    if not tensors:
        raise ValueError("cache layer binds no tensors")
    return {_storage(tensor)[0] for tensor in tensors}


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


def _shared_by_allocations(
    config: Any, layers: dict[str, Any], storages: dict[tuple[str, int], int]
) -> list[dict[str, Any]]:
    """Check a vLLM 0.27.0 or 0.28.0 plan, where ``shared_by`` names the aliases of each tensor."""
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
    return allocations


def _layer_strided_allocations(
    config: Any, groups: list[dict[str, Any]], layers: dict[str, Any], storages: dict[tuple[str, int], int]
) -> list[dict[str, Any]]:
    """Check a vLLM 0.32.0 or 0.33.0 plan, where every tensor places its ``layers`` in one buffer.

    The page of layer l for block b starts at offset + l * layer_stride + b * block_stride
    (v1/kv_cache_interface.py 0.32.0:1413-1415, 0.33.0:1479-1481), so the last page of the last
    layer bounds the region in layer-outer and block-outer layouts. A layer's page is the spec
    page of its group, or the page of its own spec when the group holds ``layer_specs``.
    Tensors whose regions overlap alias the same bytes: cache groups overlay each other
    (0.32.0:1419-1421, 0.33.0:1485-1487); the importer checks how they overlap.
    """
    num_blocks = _positive(config.num_blocks, "cache block count")
    layer_storages = {name: _layer_storages(layer.kv_cache) for name, layer in layers.items()}
    if set().union(*layer_storages.values()) != set(storages):
        raise ValueError("cache group layers do not account for every model-runner storage")
    pages = {
        name: group["layer_specs"][name]["page_size_bytes"] if "layer_specs" in group else group["spec_page_size_bytes"]
        for group in groups
        for name in group["layer_names"]
    }
    expected: dict[tuple[str, int], int] = {}
    covered: set[str] = set()
    allocations = []
    for tensor in config.kv_cache_tensors:
        if getattr(tensor, "host_resident", False):
            raise ValueError("host-resident cache tensors are unsupported")
        names = list(tensor.layers)
        if not names or covered.intersection(names) or any(name not in layers for name in names):
            raise ValueError("runtime tensor allocation has missing or duplicate cache layers")
        keys = set().union(*(layer_storages[name] for name in names))
        if len(keys) != 1:
            raise ValueError("runtime-declared shared tensor uses different physical storages")
        key = next(iter(keys))
        if expected and key not in expected:
            raise ValueError("cache tensors with layer strides use multiple backing allocations")
        size = _positive(tensor.size, "configured cache tensor bytes")
        if key in expected and expected[key] != size:
            raise ValueError("packed allocation sizes disagree")
        layer_stride, block_stride, offset = (
            getattr(tensor, field, None) for field in ("layer_stride", "block_stride", "offset")
        )
        if any(type(value) is not int or value < 0 for value in (layer_stride, block_stride, offset)):
            raise ValueError("invalid cache tensor layer stride, block stride or offset")
        page = max(pages[name] for name in names)
        if page > block_stride:
            raise ValueError("cache tensor page exceeds its block stride")
        if offset + (len(names) - 1) * layer_stride + (num_blocks - 1) * block_stride + page > size:
            raise ValueError("cache tensor layers exceed their backing allocation")
        expected[key] = size
        covered.update(names)
        allocations.append(
            {
                "size": size,
                "layers": list(names),
                "layer_stride": layer_stride,
                "block_stride": block_stride,
                "offset": offset,
            }
        )
    if covered != set(layers) or expected != storages:
        raise ValueError("runtime allocation plan disagrees with physical cache storage")
    return allocations


def worker_memory(worker: Any, *, hybrid_evidence: bool = False) -> dict[str, Any]:
    """Read the initialized cache groups, storage and allocation plan of one worker.

    ``hybrid_evidence`` records the vLLM 0.33.0 evidence of ``cache_groups``;
    without it the groups are the ones vLLM 0.27.0, 0.28.0, 0.32.0 and 0.33.0
    runtimes have. The shape of the allocation record picks its reader.
    """
    if not getattr(worker, "_fpm_cache_initialized", False):
        raise ValueError("worker cache initialization was not observed")
    runner = worker.model_runner
    if getattr(runner, "shared_kv_cache_layers", None):
        raise ValueError("cross-layer KV sharing is not yet supported by memory finalization")
    config = runner.kv_cache_config
    groups = cache_groups(config, hybrid_evidence=hybrid_evidence)
    storages = _tensor_storages(runner.kv_caches)
    context = worker.vllm_config.compilation_config.static_forward_context
    layers = {name: context[name] for group in groups for name in group["layer_names"]}
    # The record shape selects the reader. vLLM 0.27.0 names the aliases of each tensor in shared_by
    # (v1/kv_cache_interface.py:952); 0.32.0 and 0.33.0 place layers by stride and have no shared_by
    # (0.32.0:1424-1429, 0.33.0:1490-1495). Every other shape stays unresolved.
    tensors = config.kv_cache_tensors
    if all(
        hasattr(item, "layers") and hasattr(item, "layer_stride") and not hasattr(item, "shared_by") for item in tensors
    ):
        allocations = _layer_strided_allocations(config, groups, layers, storages)
    elif all(hasattr(item, "shared_by") and not hasattr(item, "layers") for item in tensors):
        allocations = _shared_by_allocations(config, layers, storages)
    else:
        raise ValueError("unsupported runtime cache tensor record")
    for group in groups:
        classes = {name: _class_name(layers[name]) for name in group["layer_names"]}
        conv = {value == _CONV_CLASS for value in classes.values()}
        if len(conv) != 1:
            raise ValueError("mixed convolution and attention cache group is unsupported")
        # Layers of the Inkling convolution class make a convolution group; hybrid
        # evidence keeps the kind of the spec class otherwise.
        group["kind"] = "convolution" if conv == {True} else group.get("kind", "attention")
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


def scheduler_memory(scheduler: Any, config: Any, *, hybrid_evidence: bool = False) -> dict[str, Any]:
    """Read the initialized block pool of one scheduler.

    ``hybrid_evidence`` adds the vLLM 0.33.0 block sizes, Mamba cache mode and
    startup capacity, and records the ``cache_groups`` evidence without the kind;
    without it the record is the one vLLM 0.27.0, 0.28.0, 0.32.0 and 0.33.0 runtimes have.
    """
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
    record = {
        "num_blocks": num_blocks,
        "initial_free_blocks": free,
        "reserved_blocks": num_blocks - free,
        "null_block_id": pool.null_block.block_id,
        "watermark_blocks": manager.watermark_blocks,
        "pool_count": 1,
        "groups": cache_groups(config, hybrid_evidence=hybrid_evidence),
    }
    if hybrid_evidence:
        # vLLM 0.33.0 Scheduler.block_size and hash_block_size (core/sched/scheduler.py:218, :350);
        # its BlockPool keeps the hash block size too (core/block_pool.py:173).
        hash_block_size = getattr(scheduler, "hash_block_size", None)
        if hasattr(pool, "hash_block_size") and pool.hash_block_size != hash_block_size:
            raise ValueError("scheduler hash block size differs from its block pool")
        cache = scheduler.vllm_config.cache_config
        record.update(
            block_size=_json_value(getattr(scheduler, "block_size", None)),
            hash_block_size=_json_value(hash_block_size),
            mamba_cache_mode=_json_value(getattr(cache, "mamba_cache_mode", None)),
            # The engine core sets the startup capacity before it builds the scheduler (engine/core.py:351).
            kv_cache_size_tokens=_json_value(getattr(cache, "kv_cache_size_tokens", None)),
            kv_cache_max_concurrency=_json_value(getattr(cache, "kv_cache_max_concurrency", None)),
            # The importer compares these groups with the worker groups without their kind.
            groups=[{key: value for key, value in group.items() if key != "kind"} for group in record["groups"]],
        )
    return record


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
        if payload["backend_version"] not in MEMORY_SUPPORTED_VERSIONS:
            raise ValueError(
                f"memory observation currently supports vLLM {', '.join(MEMORY_SUPPORTED_VERSIONS)}; "
                f"found {payload['backend_version']}"
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
