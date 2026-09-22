# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate runtime initialization evidence and resolve physical cache capacity."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aisimulate_core.sdk.fpm_identity import EXECUTION_COLUMNS, LEGACY_EXECUTION_IDENTITY

from .native_artifact import COLLECTOR_PROVENANCE_FILENAME, validate_native_collection
from .planner import BackendPolicy, FPMCell, _canonical_hash
from .runtime.fpm_memory_observer import SCHEMA_NAME, SCHEMA_VERSION, SUPPORTED_VERSION
from .types import ParallelTopology

_CELL_EXECUTION_FIELDS = frozenset({"execution_identity", "input_text_sha256"})


def _positive(value: Any, label: str) -> int:
    if type(value) is not int or not 0 < value <= 2**53:
        raise ValueError(f"{label} must be a positive integer no greater than 2**53")
    return value


def cell_from_dict(payload: dict[str, Any]) -> FPMCell:
    """Read an exact saved cell without resolving current models or capabilities."""
    if not isinstance(payload, dict):
        raise ValueError("saved FPM cell must be an object")
    execution_fields = _CELL_EXECUTION_FIELDS.intersection(payload)
    if execution_fields and execution_fields != _CELL_EXECUTION_FIELDS:
        raise ValueError("saved FPM cell has an incomplete execution identity")
    execution = LEGACY_EXECUTION_IDENTITY
    input_text_sha256 = ""
    if execution_fields:
        declared = payload["execution_identity"]
        input_text_sha256 = payload["input_text_sha256"]
        if (
            not isinstance(declared, dict)
            or set(declared) != set(EXECUTION_COLUMNS)
            or any(not isinstance(value, str) for value in declared.values())
            or not isinstance(input_text_sha256, str)
        ):
            raise ValueError("saved FPM cell has an invalid execution identity")
        execution = tuple(declared[name] for name in EXECUTION_COLUMNS)
    try:
        topology = ParallelTopology(**payload["topology"])
        for name, value in topology.to_dict().items():
            _positive(value, f"topology.{name}")
        dtypes = payload["resolved_dtypes"]
        policy = payload["backend_policy"]
        cell = FPMCell(
            cell_id=payload["cell_id"],
            workload_kind=payload["workload_kind"],
            topology=topology,
            weight_quantization=payload["weight_quantization"],
            kv_cache_dtype=payload["kv_cache_dtype"],
            backend_policy=BackendPolicy(**policy),
            parallel_strategy=payload["parallel_strategy"],
            gemm_quant_mode=dtypes["gemm_quant_mode"],
            moe_quant_mode=dtypes["moe_quant_mode"],
            fmha_quant_mode=dtypes["fmha_quant_mode"],
            comm_quant_mode=dtypes["comm_quant_mode"],
            fmha_resolution=dtypes["fmha_resolution"],
            execution_identity=execution,
            input_text_sha256=input_text_sha256,
        )
    except (KeyError, TypeError) as error:
        raise ValueError(f"invalid saved FPM cell: {error}") from error
    serialized = cell.to_dict()
    if not execution_fields:
        # Old cells retain their exact stored shape and identity when hashed.
        for name in _CELL_EXECUTION_FIELDS:
            serialized.pop(name)
    if serialized != payload or cell.workload_kind not in {"prefill", "decode"}:
        raise ValueError("saved FPM cell does not match the native collection contract")
    if not isinstance(cell.cell_id, str) or not cell.cell_id:
        raise ValueError("saved FPM cell requires a nonempty identity")
    return cell


def validate_saved_plan(payload: dict[str, Any]) -> None:
    """Verify current and historical producer hashes without rewriting identity."""
    if (
        not isinstance(payload, dict)
        or payload.get("schema_name") != "aic_fpm_collection_plan"
        or payload.get("schema_version") not in (10, 11)
    ):
        raise ValueError("runtime memory finalization requires a schema-v10 or schema-v11 collection plan")
    try:
        canonical = {
            key: payload[key]
            for key in (
                "backend",
                "model_path",
                "system",
                "aic_revision",
                "generator_config_sha256",
                "options",
                "capability",
                "dtype_profile",
                "cells",
            )
        }
        canonical["point_generation"] = "dynamo_native_self_benchmark"
        canonical["policies"] = payload["backend_policies"]
        canonical["topologies"] = [
            {name: item[name] for name in ("tp", "pp", "dp", "moe_tp", "moe_ep", "cp")}
            for item in payload["topologies"]
        ]
        admissions = copy.deepcopy(payload["topology_memory_admission"])
        for admission in admissions:
            admission.pop("reason", None)
            for estimate in admission.get("estimates", []):
                estimate.pop("reason", None)
        canonical["topology_memory_admission"] = admissions
        if "fpm_profile" in payload:
            canonical["fpm_profile"] = payload["fpm_profile"]
        if "runtime_memory_policy" in payload:
            canonical["runtime_memory_policy"] = payload["runtime_memory_policy"]
        for item in payload["cells"]:
            present = _CELL_EXECUTION_FIELDS.intersection(item)
            required = _CELL_EXECUTION_FIELDS if payload["schema_version"] == 11 else frozenset()
            if present != required:
                raise ValueError("saved collection plan cell execution identity does not match its schema version")
        cells = [cell_from_dict(item) for item in payload["cells"]]
        if not cells or len({cell.cell_id for cell in cells}) != len(cells):
            raise ValueError("saved collection plan requires unique cells")
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError(f"invalid saved collection plan: {error}") from error
    if _canonical_hash(canonical) != payload.get("sha256"):
        raise ValueError("saved collection plan SHA-256 does not match its immutable inputs")


@dataclass(frozen=True, slots=True)
class _SavedOptions:
    warmup_iterations: int
    benchmark_points_json: str | None


@dataclass(frozen=True, slots=True)
class _SavedCapability:
    support_level: str
    template_id: str | None
    template_version: int | None
    aic_database_version: str


@dataclass(frozen=True, slots=True)
class SavedCollectionIdentity:
    """The verified plan fields used by native aggregation and commit validation."""

    backend: str
    model_path: str
    system: str
    sha256: str
    options: _SavedOptions
    capability: _SavedCapability
    cells: tuple[FPMCell, ...]


def saved_plan_identity(payload: dict[str, Any]) -> SavedCollectionIdentity:
    validate_saved_plan(payload)
    capability = payload["capability"]
    points = payload["options"].get("benchmark_points")
    return SavedCollectionIdentity(
        backend=payload["backend"],
        model_path=payload["model_path"],
        system=payload["system"],
        sha256=payload["sha256"],
        options=_SavedOptions(
            warmup_iterations=payload["options"]["global_warmup_iterations"],
            benchmark_points_json=json.dumps(points["payload"]) if points is not None else None,
        ),
        cells=tuple(cell_from_dict(item) for item in payload["cells"]),
        capability=_SavedCapability(
            **{
                key: capability[key]
                for key in (
                    "support_level",
                    "template_id",
                    "template_version",
                    "aic_database_version",
                )
            }
        ),
    )


def _dtype(value: Any) -> str:
    return {
        "torch.bfloat16": "bfloat16",
        "torch.float16": "half",
        "float16": "half",
        "fp8_e4m3": "fp8",
        "fp8_e4m3fn": "fp8",
    }.get(value, value)


def _validate_precision(config: dict[str, Any], cell: FPMCell) -> None:
    """Check audited quantization families without assigning every layer a dtype.

    vLLM 4bdc8a788d2e2ce9165d552b3d4d8b72604626bf:
    model_executor/layers/quantization/{fp8,modelopt}.py config classes and
    ModelOptQuantConfigBase.get_quant_method preserve unquantized exclusions.
    """
    model = config["model_config"]
    base = _dtype(model.get("dtype"))
    method = model.get("quantization")
    quant = config.get("quantization_config")
    if "quantization" not in model or "quantization_config" not in config or base not in {"bfloat16", "half"}:
        raise ValueError("runtime memory weight precision evidence is incomplete or unsupported")
    gemm, moe = None, None
    if method is None and quant is None:
        gemm = moe = base
    elif isinstance(quant, dict):
        prefix = "vllm.model_executor.layers.quantization."
        if method == "modelopt_fp4" and quant.get("type") == prefix + "modelopt.ModelOptNvFp4Config":
            if quant.get("quant_method") == "NVFP4":
                gemm = moe = "nvfp4"
        elif method == "fp8" and quant.get("type") == prefix + "fp8.Fp8Config":
            block = quant.get("weight_block_size")
            scheme = quant.get("activation_scheme")
            if block == [128, 128] and scheme == "dynamic":
                gemm = moe = "fp8_block"
            elif block is None and scheme in {"static", "dynamic"}:
                gemm, moe = ("fp8_static" if scheme == "static" else "fp8"), "fp8"
        elif (
            method == "modelopt"
            and quant.get("type") == prefix + "modelopt.ModelOptFp8Config"
            and quant.get("quant_method") == "FP8"
        ):
            gemm, moe = "fp8_static", "fp8"
    if gemm is None:
        raise ValueError("runtime memory weight precision mapping is unsupported or contradicts quantization evidence")
    # A mixed checkpoint may leave attention/shared experts in the base dtype.
    # The global quantization method establishes a family, not every layer's mode.
    if cell.gemm_quant_mode not in {base, gemm} or cell.moe_quant_mode not in {None, base, moe}:
        raise ValueError("runtime memory weight precision differs from the planned GEMM/MoE identity")


def _validate_graph(
    config: dict[str, Any], *, eager: Any, tokens: int, allow_unused_captures: bool = False
) -> dict[str, Any]:
    # vLLM 0.27.0 config/vllm.py:_set_cudagraph_sizes and
    # config/compilation.py:resolve_cudagraph_mode_and_sizes. A worker may retain
    # configured sizes after disabling graphs for its initialized attention backend.
    fields = {"mode", "cudagraph_mode", "cudagraph_capture_sizes", "max_cudagraph_capture_size"}
    if not isinstance(config, dict) or not fields.issubset(config) or type(eager) is not bool:
        raise ValueError("runtime memory evidence lacks complete graph configuration")
    result = copy.deepcopy(config)
    modes = ("NONE", "STOCK_TORCH_COMPILE", "DYNAMO_TRACE_ONCE", "VLLM_COMPILE")
    mode = result["mode"]
    if type(mode) is int and 0 <= mode < len(modes):
        mode = modes[mode]
    elif isinstance(mode, str) and mode in {"0", "1", "2", "3"}:
        mode = modes[int(mode)]
    if not isinstance(mode, str) or mode not in modes:
        raise ValueError("runtime memory graph compilation mode is unsupported")
    result["mode"] = mode
    graph = result["cudagraph_mode"]
    sizes, maximum = result["cudagraph_capture_sizes"], result["max_cudagraph_capture_size"]
    if (
        not isinstance(graph, str)
        or graph not in {"NONE", "PIECEWISE", "FULL", "FULL_DECODE_ONLY", "FULL_AND_PIECEWISE"}
        or not isinstance(sizes, list)
        or any(type(size) is not int or not 0 < size <= tokens for size in sizes)
        or sizes != sorted(set(sizes))
        or type(maximum) is not int
        or maximum != max(sizes, default=0)
        or (graph != "NONE" and not sizes)
        or (graph == "NONE" and sizes and not allow_unused_captures)
        or (eager and (mode != "NONE" or graph != "NONE" or sizes))
    ):
        raise ValueError("runtime memory graph configuration is incomplete or inconsistent")
    return result


def _validate_no_offload(payload: dict[str, Any], config: dict[str, Any]) -> None:
    offload = config.get("offload_config")
    if (
        not isinstance(offload, dict)
        or not isinstance(offload.get("uva"), dict)
        or not isinstance(offload.get("prefetch"), dict)
    ):
        raise ValueError("runtime memory evidence lacks offload configuration")
    uva, prefetch = offload["uva"], offload["prefetch"]
    amount = uva.get("cpu_offload_gb")
    if (
        offload.get("offload_backend") != "auto"
        or type(amount) not in {int, float}
        or amount != 0
        or type(prefetch.get("offload_group_size")) is not int
        or prefetch["offload_group_size"] != 0
        or type(prefetch.get("offload_num_in_group")) is not int
        or prefetch["offload_num_in_group"] < 1
        or type(prefetch.get("offload_prefetch_step")) is not int
        or prefetch["offload_prefetch_step"] < 0
        or config["cache_config"].get("kv_cache_memory_bytes") is not None
        or config["cache_config"].get("num_gpu_blocks_override") is not None
        or (
            payload.get("kind") == "worker"
            and payload.get("effective_offloader") != "vllm.model_executor.offloader.base.NoopOffloader"
        )
    ):
        raise ValueError("runtime memory requires the supported automatic HBM-only no-offload policy")


def _validate_scheduler_config(worker: dict[str, Any], scheduler: dict[str, Any]) -> None:
    actual, scheduled = (copy.deepcopy(item["resolved_config"]) for item in (worker, scheduler))
    for config, is_worker in ((actual, True), (scheduled, False)):
        config["compilation_config"] = _validate_graph(
            config["compilation_config"],
            eager=config["model_config"]["enforce_eager"],
            tokens=config["scheduler_config"]["max_num_batched_tokens"],
            allow_unused_captures=is_worker and "initial_compilation_config" in worker,
        )
    if actual == scheduled:
        return
    initial = worker.get("initial_compilation_config")
    if not isinstance(initial, dict):
        raise ValueError("runtime memory launch configurations differ between scheduler and worker")
    initial = _validate_graph(
        initial,
        eager=actual["model_config"]["enforce_eager"],
        tokens=actual["scheduler_config"]["max_num_batched_tokens"],
    )
    original = {**actual, "compilation_config": initial}
    graph = actual["compilation_config"]
    # Only backend-dependent graph downgrades in the audited non-speculative
    # MRV1 resolver are process-local. Compilation/capture sizes remain unchanged.
    fallbacks = {
        "FULL": {"FULL_AND_PIECEWISE", "FULL_DECODE_ONLY"},
        "FULL_AND_PIECEWISE": {"PIECEWISE", "NONE"},
        "FULL_DECODE_ONLY": {"PIECEWISE", "NONE"},
    }
    if (
        original != scheduled
        or graph["cudagraph_mode"] not in fallbacks.get(initial["cudagraph_mode"], set())
        or {**graph, "cudagraph_mode": initial["cudagraph_mode"]} != initial
    ):
        raise ValueError("runtime memory launch configurations differ between scheduler and worker")


def _validate_config(
    payload: dict[str, Any], cell: FPMCell, *, context: int, tokens: int, sequences: int, fraction: float
) -> dict[str, Any]:
    config = copy.deepcopy(payload.get("resolved_config"))
    if not isinstance(config, dict):
        raise ValueError("runtime memory evidence lacks resolved launch configuration")
    expected = {
        "model_config": {"max_model_len": context},
        "cache_config": {"gpu_memory_utilization": fraction},
        "scheduler_config": {"max_num_batched_tokens": tokens, "max_num_seqs": sequences, "async_scheduling": False},
        "parallel_config": {
            "tensor_parallel_size": cell.topology.tp,
            "pipeline_parallel_size": cell.topology.pp,
            "data_parallel_size": cell.topology.dp,
            "decode_context_parallel_size": cell.topology.cp,
        },
    }
    for section, fields in expected.items():
        actual = config.get(section)
        if not isinstance(actual, dict):
            raise ValueError(f"runtime memory evidence lacks {section}")
        for key, value in fields.items():
            if type(actual.get(key)) is not type(value) or actual.get(key) != value:
                raise ValueError(
                    f"runtime memory launch mismatch: {section}.{key}={actual.get(key)!r}, expected {value!r}"
                )
    parallel = config["parallel_config"]
    if parallel.get("prefill_context_parallel_size", 1) != 1:
        raise ValueError("runtime memory with prefill context parallelism is unsupported")
    if parallel.get("enable_expert_parallel") != (cell.topology.moe_ep > 1):
        raise ValueError("runtime memory expert parallel setting differs from the planned topology")
    if parallel.get("enable_eplb", False) is not False:
        raise ValueError("runtime memory with expert load balancing is unsupported")
    expected_backend = cell.backend_policy.aic_fields.get("moe_backend")
    if (
        expected_backend not in (None, "auto")
        and (config.get("kernel_config") or {}).get("moe_backend") != expected_backend
    ):
        raise ValueError("runtime memory MoE backend differs from the planned identity")
    cache = config["cache_config"]
    actual_dtype = cache.get("cache_dtype")
    if actual_dtype == "auto":
        actual_dtype = config["model_config"].get("dtype")
    if _dtype(actual_dtype) != cell.kv_cache_dtype:
        raise ValueError("runtime memory KV precision differs from the planned identity")
    _validate_precision(config, cell)
    _validate_no_offload(payload, config)
    config["compilation_config"] = _validate_graph(
        config.get("compilation_config"),
        eager=config["model_config"].get("enforce_eager"),
        tokens=tokens,
        allow_unused_captures=payload.get("kind") == "worker" and "initial_compilation_config" in payload,
    )
    return config


def _validate_revision(config: dict[str, Any], expected: str | None) -> None:
    # A local PVC can omit revision or retain a symbolic ref. Only explicit
    # immutable commits establish contradictions; this is not a weight hash.
    if expected is None or re.fullmatch(r"[0-9a-fA-F]{40}", expected) is None:
        return
    model = config["model_config"]
    for key in ("revision", "loaded_config_commit_hash"):
        observed = model.get(key)
        if (
            isinstance(observed, str)
            and re.fullmatch(r"[0-9a-fA-F]{40}", observed)
            and observed.lower() != expected.lower()
        ):
            raise ValueError(f"runtime memory model {key} contradicts the reviewed immutable revision")


def _normalized_groups(cache: dict[str, Any], scheduler_cache: dict[str, Any], page_bytes: int) -> list[dict[str, Any]]:
    raw_groups = cache.get("groups")
    scheduler_groups = scheduler_cache.get("groups")
    if not isinstance(raw_groups, list) or not raw_groups or not isinstance(scheduler_groups, list):
        raise ValueError("runtime memory evidence lacks cache groups")
    if len(raw_groups) != len(scheduler_groups):
        raise ValueError("worker and scheduler cache group counts disagree")
    groups = []
    layer_names: set[str] = set()
    for index, (group, scheduled) in enumerate(zip(raw_groups, scheduler_groups, strict=True)):
        if not isinstance(group, dict) or not isinstance(scheduled, dict):
            raise ValueError("runtime memory cache groups must be objects")
        if {key: value for key, value in group.items() if key not in {"kind", "layer_classes"}} != scheduled:
            raise ValueError("worker and scheduler cache group geometry disagrees")
        names = group.get("layer_names")
        if (
            not isinstance(names, list)
            or not names
            or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)
            or layer_names.intersection(names)
        ):
            raise ValueError("runtime memory cache layer names must be nonempty and unique")
        layer_names.update(names)
        spec = group.get("spec_type")
        if spec not in {"vllm.v1.kv_cache_interface.FullAttentionSpec", "vllm.v1.kv_cache_interface.SlidingWindowSpec"}:
            raise ValueError(f"unsupported runtime cache spec: {spec!r}")
        window = group.get("sliding_window")
        if spec.endswith(".SlidingWindowSpec"):
            _positive(window, "sliding window")
        elif window is not None:
            raise ValueError("full-attention cache cannot declare sliding retention")
        kind = group.get("kind")
        if kind not in {"attention", "convolution"} or (kind == "convolution" and window != 4):
            raise ValueError("runtime memory cache kind is unsupported")
        classes = group.get("layer_classes")
        if (
            not isinstance(classes, dict)
            or set(classes) != set(names)
            or any(not isinstance(value, str) or not value for value in classes.values())
        ):
            raise ValueError("runtime memory cache lacks complete layer class evidence")
        convolution_class = "vllm.models.inkling.nvidia.sconv_swa_attn.InklingConvState"
        if any((value == convolution_class) != (kind == "convolution") for value in classes.values()):
            raise ValueError("runtime memory cache kind contradicts actual layer classes")
        groups.append(
            {
                "name": f"runtime_group_{index}",
                "kind": kind,
                "num_layers": len(names),
                "block_size_tokens": _positive(group.get("block_size_tokens"), "cache block size"),
                "page_size_bytes": page_bytes,
                "sliding_window": window,
            }
        )
    return groups


def resolve_runtime_resources(
    cell: FPMCell,
    raw_root: Path,
    *,
    expected_plan_sha256: str,
    expected_attempt_id: str,
    expected_backend_version: str,
    expected_context_length: int,
    expected_max_num_tokens: int,
    expected_max_batch_size: int,
    expected_gpu_memory_utilization: float,
    expected_model_revision: str | None = None,
) -> dict[str, Any]:
    """Require complete matching native timings and initialized rank/pool evidence.

    Every group draws pages from one scheduler block pool. Therefore each page
    is charged the complete physical cost of one pool block, including layer
    padding and shared buffers. Free pool blocks exclude permanent reservations.
    No legacy activation/overhead decomposition is inferred from cache capacity.
    """
    for name, value in (
        ("context", expected_context_length),
        ("tokens", expected_max_num_tokens),
        ("sequences", expected_max_batch_size),
    ):
        _positive(value, name)
    fraction = expected_gpu_memory_utilization
    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not math.isfinite(fraction)
        or not 0 < fraction <= 1
    ):
        raise ValueError("runtime memory GPU memory utilization must be in (0, 1]")
    if not expected_plan_sha256 or not expected_attempt_id:
        raise ValueError("runtime memory resolution requires exact plan and attempt identities")
    if expected_backend_version != SUPPORTED_VERSION:
        raise ValueError(f"runtime memory conversion currently supports vLLM {SUPPORTED_VERSION}")
    collection = validate_native_collection(
        cell, raw_root, expected_plan_sha256=expected_plan_sha256, expected_attempt_id=expected_attempt_id
    )
    if collection.backend_version != expected_backend_version:
        raise ValueError("native timing runtime version differs from the memory profile")
    if cell.topology.pp != 1 or cell.topology.cp != 1:
        raise ValueError("runtime memory supports PP1 and CP1")
    workers: dict[tuple[int, int], dict[str, Any]] = {}
    schedulers: dict[int, dict[str, Any]] = {}
    files = []
    canonical_config = None
    for path in sorted(raw_root.glob("**/fpm-memory-*.json")):
        if len(path.relative_to(raw_root).parts) != 2:
            raise ValueError(f"runtime memory artifact must belong to one collected pod: {path}")
        payload = json.loads(path.read_text())
        if (
            not isinstance(payload, dict)
            or payload.get("schema_name") != SCHEMA_NAME
            or payload.get("schema_version") != SCHEMA_VERSION
        ):
            raise ValueError(f"unsupported runtime memory evidence schema: {path}")
        provenance_path = path.parent / COLLECTOR_PROVENANCE_FILENAME
        provenance = json.loads(provenance_path.read_text())
        expected_provenance = {
            "schema_name": "aic_fpm_collector_provenance",
            "schema_version": 1,
            "cell_id": cell.cell_id,
            "plan_sha256": expected_plan_sha256,
            "attempt_id": expected_attempt_id,
            "runtime": {"backend": "vllm", "backend_version": expected_backend_version},
        }
        if provenance != expected_provenance or payload.get("collector_provenance") != provenance:
            raise ValueError(f"runtime memory evidence belongs to a different plan, attempt, cell, or runtime: {path}")
        if payload.get("backend_version") != expected_backend_version:
            raise ValueError(f"runtime memory backend version mismatch: {path}")
        if payload.get("status") != "resolved":
            raise ValueError(f"runtime memory remains unresolved: {path}: {payload.get('error')}")
        config = _validate_config(
            payload,
            cell,
            context=expected_context_length,
            tokens=expected_max_num_tokens,
            sequences=expected_max_batch_size,
            fraction=float(fraction),
        )
        _validate_revision(config, expected_model_revision)
        dp = payload.get("dp_rank")
        if type(dp) is not int or not 0 <= dp < cell.topology.dp:
            raise ValueError(f"runtime memory has an invalid DP rank: {path}")
        if payload.get("kind") == "worker":
            if canonical_config is None:
                canonical_config = config
            elif config != canonical_config:
                raise ValueError("runtime memory launch configurations differ across worker ranks")
            tp, pp = payload.get("tp_rank"), payload.get("pp_rank")
            if type(tp) is not int or not 0 <= tp < cell.topology.tp or type(pp) is not int or pp != 0:
                raise ValueError(f"runtime memory has an invalid TP/PP rank: {path}")
            key = (dp, tp)
            if key in workers:
                raise ValueError(f"duplicate runtime memory worker rank: {key}")
            workers[key] = payload
        elif payload.get("kind") == "scheduler":
            if dp in schedulers or payload.get("tp_rank") is not None or payload.get("pp_rank") is not None:
                raise ValueError(f"duplicate or invalid runtime memory scheduler rank: {dp}")
            schedulers[dp] = payload
        else:
            raise ValueError(f"unknown runtime memory observation kind: {path}")
        files.append(
            {
                "path": str(path.relative_to(raw_root)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "evidence": payload,
            }
        )
    if set(workers) != {(dp, tp) for dp in range(cell.topology.dp) for tp in range(cell.topology.tp)}:
        raise ValueError("runtime memory worker rank evidence is incomplete")
    if set(schedulers) != set(range(cell.topology.dp)):
        raise ValueError("runtime memory scheduler rank evidence is incomplete")
    canonical_groups = None
    canonical_geometry = None
    capacities = []
    for (dp, tp), payload in sorted(workers.items()):
        _validate_scheduler_config(payload, schedulers[dp])
        cache = payload.get("cache")
        scheduled = schedulers[dp].get("cache")
        if not isinstance(cache, dict) or not isinstance(scheduled, dict):
            raise ValueError("runtime memory evidence lacks cache allocation")
        count = _positive(cache.get("num_blocks"), "worker cache block count")
        free = _positive(scheduled.get("initial_free_blocks"), "initial free block count")
        if (
            type(scheduled.get("num_blocks")) is not int
            or scheduled["num_blocks"] != count
            or scheduled.get("pool_count") != 1
            or type(scheduled.get("pool_count")) is not int
            or scheduled.get("watermark_blocks") != 0
            or type(scheduled.get("watermark_blocks")) is not int
            or free >= count
            or scheduled.get("reserved_blocks") != count - free
            or type(scheduled.get("reserved_blocks")) is not int
            or type(scheduled.get("null_block_id")) is not int
            or not 0 <= scheduled["null_block_id"] < count
        ):
            raise ValueError("scheduler block pool capacity/reservations are unsupported or inconsistent")
        allocated = _positive(cache.get("allocated_cache_bytes"), "physical cache bytes")
        available = _positive(cache.get("available_cache_bytes"), "profiled cache budget")
        storages = cache.get("storages")
        if not isinstance(storages, list) or not storages:
            raise ValueError("runtime memory evidence lacks physical storage accounting")
        keys = [(item.get("device"), item.get("pointer")) for item in storages]
        if (
            len(keys) != len(set(keys))
            or len({device for device, _pointer in keys}) != 1
            or any(
                not isinstance(device, str)
                or not device.startswith("cuda:")
                or type(pointer) is not int
                or pointer <= 0
                for device, pointer in keys
            )
            or sum(_positive(item.get("size_bytes"), "storage bytes") for item in storages) != allocated
        ):
            raise ValueError("physical runtime cache storage accounting is inconsistent")
        if allocated > available or allocated % count:
            raise ValueError("physical cache allocation exceeds its budget or has a non-integral pool page size")
        groups = _normalized_groups(cache, scheduled, allocated // count)
        geometry = [{key: value for key, value in group.items() if key != "layer_classes"} for group in cache["groups"]]
        if canonical_groups is None:
            canonical_groups, canonical_geometry = groups, geometry
        elif groups != canonical_groups or geometry != canonical_geometry:
            raise ValueError("cache group geometry or physical page cost differs across ranks")
        capacities.append({"dp_rank": dp, "tp_rank": tp, "kv_cache_bytes": free * (allocated // count)})
    runtime_settings = copy.deepcopy(canonical_config)
    runtime_settings["cache_config"].pop("enable_prefix_caching", None)
    provenance = {
        "source": "vllm_initialized_cache",
        "observer_schema_version": SCHEMA_VERSION,
        "plan_sha256": expected_plan_sha256,
        "attempt_id": expected_attempt_id,
        "cell_id": cell.cell_id,
        "runtime_run_id": collection.runtime_run_id,
        "runtime_grid_digest": collection.runtime_grid_digest,
        "backend_version": expected_backend_version,
        "resolved_config": canonical_config,
        "runtime_settings": runtime_settings,
        "rank_capacities": capacities,
        "artifacts": files,
        "scope": "physical memory allocated by the complete collection worker; "
        "text-only timing does not subtract other loaded components",
    }
    return {
        "max_num_tokens": expected_max_num_tokens,
        "max_batch_size": expected_max_batch_size,
        "cache_layout": "grouped",
        "cache_groups": canonical_groups,
        "runtime_memory": {
            "kv_cache_bytes": min(item["kv_cache_bytes"] for item in capacities),
            "gpu_memory_utilization": float(fraction),
            "max_model_len": expected_context_length,
            "provenance": json.dumps(provenance, sort_keys=True, separators=(",", ":")),
        },
        "provenance": "Observed initialized vLLM shared cache pool after successful warmup; "
        "minimum usable capacity across all ranks, with padding and permanent reservations included.",
    }
