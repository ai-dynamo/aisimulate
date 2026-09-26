# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only validation of saved runtime observations, independent of timing data.

The supported allocation representation is one HBM pool shared by full-history
or sliding-window groups. Runtime mappings are evidence: the importer rechecks
raw geometry, configuration, physical views, and pool accounting itself.
"""

from __future__ import annotations

import copy
import math
import re
from importlib.resources import files
from pathlib import Path, PurePosixPath
from typing import Any

from aisimulate_core.fpm_profile import FpmResourceProfile
from aisimulate_core.sdk.perf_database import load_system_spec

from . import runtime_memory
from .config import PrefillSamplingProfile
from .planner import BackendPolicy, FPMCell
from .runtime_instrumentation import (
    OBSERVATION_SCHEMA,
    InstrumentationBundle,
    canonical_json,
    contained_file,
    load_instrumentation,
    read_json,
    sha256_bytes,
    validate_sha256,
)
from .types import ParallelTopology

INDEX_SCHEMA = "aisimulate-runtime-observations/v1"
LAUNCH_SCHEMA = "aisimulate-runtime-probe-launch/v1"
_SEMANTICS = {
    "allocation": "shared_block_pool",
    "storage": "hbm",
    "retention": "full_or_window",
    "prefix_reuse": False,
    "offload": False,
    "speculative": False,
}


def _same(actual: Any, expected: Any, label: str) -> None:
    if canonical_json(actual) != canonical_json(expected):
        raise ValueError(f"runtime observation {label} mismatch")


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"runtime observation requires {label}")
    return value


def _nonnegative(value: Any, label: str) -> int:
    if type(value) is not int or not 0 <= value <= 2**53:
        raise ValueError(f"{label} must be a nonnegative integer no greater than 2**53")
    return value


def _positive(value: Any, label: str) -> int:
    return runtime_memory._positive(value, label)


def expected_ranks(topology: dict[str, int]) -> dict[str, list[dict[str, int]]]:
    """Derive rank coverage from explicit topology, never configuration labels."""
    if not isinstance(topology, dict) or set(topology) != {"tp", "pp", "dp", "moe_tp", "moe_ep", "cp"}:
        raise ValueError("runtime observations require complete explicit topology")
    for key, value in topology.items():
        _positive(value, f"topology.{key}")
    if topology["pp"] != 1 or topology["cp"] != 1:
        raise ValueError("runtime observation allocation supports PP1 and CP1")
    return {
        "workers": [
            {"dp_rank": dp, "tp_rank": tp, "pp_rank": 0} for dp in range(topology["dp"]) for tp in range(topology["tp"])
        ],
        "schedulers": [{"dp_rank": dp} for dp in range(topology["dp"])],
    }


def _launch_cell(launch: dict[str, Any], phase: str) -> FPMCell:
    identity, collection, precision = (launch[key] for key in ("identity", "collection", "precision"))
    topology = launch["topology"]
    kind = identity.get("model_kind")
    if kind == "moe":
        if topology["moe_tp"] * topology["moe_ep"] != topology["tp"] * topology["dp"]:
            raise ValueError("runtime observation attention and MoE topology sizes disagree")
    elif kind != "dense" or any(topology[name] != 1 for name in ("dp", "moe_tp", "moe_ep")):
        raise ValueError("dense runtime observations require DP1 and MoE TP1/EP1")
    if identity.get("framework") != "vllm":
        raise ValueError("runtime observation currently requires a vLLM source mapping")
    for field in ("model", "model_revision", "framework_version", "gpu", "interconnect"):
        _text(identity.get(field), f"identity.{field}")
    for field in ("max_model_len", "max_num_batched_tokens", "max_num_seqs"):
        _positive(collection.get(field), f"collection.{field}")
    fraction = collection.get("gpu_memory_utilization")
    if type(fraction) not in (int, float) or not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("GPU memory utilization must be finite and in (0, 1]")
    policy, size = collection.get("prefill_cudagraph_policy"), collection.get("max_prefill_cudagraph_size")
    if policy not in {"runtime", "explicit"} or (policy == "runtime" and size is not None):
        raise ValueError("runtime observation has an inconsistent prefill graph policy")
    if policy == "explicit":
        _positive(size, "explicit prefill graph size")
    if collection.get("async_scheduling", False) is not False:
        raise ValueError("runtime observation requires synchronous scheduling")
    if precision.get("enable_eplb", False) is not False or precision.get("enable_wideep", False) is not False:
        raise ValueError("runtime observation does not support EPLB or wide-EP allocations")
    if precision.get("comm_quant_mode") != "half":
        raise ValueError("runtime observation requires the collector half communication identity")
    for field in ("gemm_quant_mode", "fmha_quant_mode", "kvcache_quant_mode"):
        _text(precision.get(field), f"precision.{field}")
    validate_sha256(launch["model_config"]["sha256"], "model configuration")
    _text(launch["deployment"].get("image"), "exact runtime image")
    return FPMCell(
        cell_id="runtime-observation",
        workload_kind=phase,
        topology=ParallelTopology(**launch["topology"]),
        weight_quantization=precision["gemm_quant_mode"],
        kv_cache_dtype=precision["kvcache_quant_mode"],
        backend_policy=BackendPolicy("runtime-observation", {}, {}, aic_fields=precision),
        gemm_quant_mode=precision["gemm_quant_mode"],
        moe_quant_mode=precision.get("moe_quant_mode"),
        fmha_quant_mode=precision["fmha_quant_mode"],
        comm_quant_mode=precision["comm_quant_mode"],
    )


def _artifact(root: Path, reference: dict[str, Any]) -> tuple[Path, str]:
    path = contained_file(root, reference["path"])
    digest = validate_sha256(reference["sha256"], "runtime artifact")
    if sha256_bytes(path.read_bytes()) != digest:
        raise ValueError(f"runtime artifact hash mismatch: {reference['path']}")
    return path, digest


def _validate_hardware(hardware: dict[str, Any], identity: dict[str, Any]) -> None:
    system = identity["gpu"]
    family = system.split("_", 1)[0]
    if system == "rtx_pro_6000_server":
        family = "rtx pro 6000"
    elif not re.fullmatch(r"(?:a100|h100|h200|b200|b300|gh200|gb200|gb300|l40s|l4)(?:_sxm|_pcie)?", system):
        raise ValueError("runtime hardware lacks a supported device-family mapping")
    spec = load_system_spec(system, systems_paths=str(files("aisimulate_core") / "systems"))
    expected_sm = spec.get("gpu", {}).get("sm_version")
    actual_name = _text(hardware.get("device_name"), "hardware device name")
    if (
        hardware.get("gpu") != system
        or hardware.get("sm") != expected_sm
        or not re.search(r"(?<![a-z0-9])" + re.escape(family) + r"(?![a-z0-9])", actual_name.lower())
    ):
        raise ValueError("observed hardware device family or SM differs from the packaged system")
    if identity.get("sm") is not None and hardware["sm"] != identity["sm"]:
        raise ValueError("observed GPU SM differs from the selected hardware")


def _deployed_model_path(launch: dict[str, Any]) -> str:
    deployment = launch["deployment"]
    model_cache = deployment.get("model_cache")
    if not model_cache:
        return launch["identity"]["model"]
    if deployment.get("executor") != "kubernetes":
        raise ValueError("model_cache checkpoint mapping requires the Kubernetes executor")
    parts = _text(model_cache, "deployment.model_cache").split(":")
    if len(parts) > 3 or not parts[0]:
        raise ValueError("model_cache must be NAME[:MOUNT[:SUBPATH]] with a non-empty PVC name")
    # Match entry._load_generator_overrides and runner._cell_generator_overrides:
    # a cache-only PVC preserves the public model; mount + subpath select the
    # actual --model path while the served model name keeps its public identity.
    mount = parts[1] if len(parts) > 1 else ""
    subpath = parts[2] if len(parts) > 2 else ""
    if not mount and not subpath:
        return launch["identity"]["model"]
    if not mount or not subpath:
        raise ValueError("model_cache checkpoint mapping requires both mount and subpath")
    relative = PurePosixPath(subpath)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("model_cache checkpoint subpath must be relative without '..'")
    return str(PurePosixPath(mount) / relative)


def _validate_record(
    record: dict[str, Any],
    *,
    launch: dict[str, Any],
    bundle: InstrumentationBundle,
    binding: dict[str, str],
    cell: FPMCell,
) -> dict[str, Any]:
    if record.get("schema_version") != OBSERVATION_SCHEMA:
        raise ValueError("unsupported runtime observation schema")
    for name, value in binding.items():
        _same(record.get(name), value, name)
    _same(record.get("identity"), launch["identity"], "model/hardware identity")
    _same(record.get("launch"), launch, "effective launch settings")
    validate_sha256(record.get("model_config_sha256"), "observed model configuration")
    _same(record["model_config_sha256"], launch["model_config"]["sha256"], "loaded model configuration hash")
    source_files = launch["model_config"].get("source_files", {})
    if not isinstance(source_files, dict) or source_files.keys() - {"hf_quant_config.json"}:
        raise ValueError("model configuration source files must name the adjacent hf_quant_config.json")
    for name, digest in source_files.items():
        validate_sha256(digest, f"model configuration source {name}")
    _same(record.get("model_config_source_files", {}), source_files, "loaded model configuration source hashes")
    _same(
        record.get("runtime"),
        {**bundle.manifest["runtime"], "image": launch["deployment"]["image"]},
        "runtime build/source",
    )
    if record.get("unresolved_fields") != [] or record.get("error"):
        raise ValueError(
            f"runtime observation has unresolved fields: {record.get('unresolved_fields')}: {record.get('error')}"
        )
    kind = record.get("kind")
    if kind not in {"worker", "scheduler"}:
        raise ValueError("runtime observation requires worker or scheduler role")
    lifecycle = record.get("lifecycle", {})
    if lifecycle.get("cache_initialized") is not True:
        raise ValueError("cache initialization lifecycle evidence is missing")
    if kind == "worker":
        if (
            lifecycle.get("measurement") != "after_warmup"
            or lifecycle.get("warmup_completed") is not True
            or lifecycle.get("capture_completed") is not True
        ):
            raise ValueError("worker warmup and graph capture completion evidence is missing")
        hardware = record.get("hardware", {})
        _validate_hardware(hardware, launch["identity"])
        total = _positive(hardware.get("total_memory_bytes"), "hardware memory bytes")
        if _positive(record["cache"]["available_cache_bytes"], "profiled cache allowance") > total:
            raise ValueError("profiled cache allowance exceeds physical device memory")
    elif lifecycle.get("measurement") != "scheduler_initialized":
        raise ValueError("scheduler initialization lifecycle evidence is missing")
    collection = launch["collection"]
    config = runtime_memory._validate_config(
        record,
        cell,
        context=collection["max_model_len"],
        tokens=collection["max_num_batched_tokens"],
        sequences=collection["max_num_seqs"],
        fraction=float(collection["gpu_memory_utilization"]),
    )
    if launch["identity"]["model_kind"] == "moe":
        parallel = config["parallel_config"]
        enabled = parallel["enable_expert_parallel"]
        if type(enabled) is not bool:
            raise ValueError("observed expert topology requires a Boolean expert-parallel setting")
        # vLLM .27/.28 fused_moe/config.py:FusedMoEParallelConfig.make flattens
        # attention TP and DP (supported PCP1). EP owns whole experts; partial
        # simultaneous expert TP/EP is not represented by this source mapping.
        width = parallel["tensor_parallel_size"] * parallel["data_parallel_size"]
        expert_axes = (1, width) if enabled else (width, 1)
        if (cell.topology.moe_tp, cell.topology.moe_ep) != expert_axes:
            raise ValueError("observed expert topology differs from the selected MoE TP/EP axes")
    if config.get("speculative_config") is not None or config.get("kv_transfer_config") is not None:
        raise ValueError("speculative or transferred cache semantics are unsupported")
    model = config["model_config"]
    if model.get("model") != _deployed_model_path(launch):
        raise ValueError("observed model path differs from the selected deployment")
    revision = launch["identity"]["model_revision"]
    if revision not in (model.get("revision"), model.get("loaded_config_commit_hash")):
        raise ValueError("observed model revision differs from the selected model")
    runtime_memory._validate_revision(config, revision)
    if (
        cell.fmha_quant_mode not in {"half", "bfloat16"}
        or runtime_memory._dtype(model.get("dtype")) != cell.fmha_quant_mode
    ):
        raise ValueError("FMHA precision lacks a supported observed compute-dtype mapping")
    # The raw launch config is independent corroboration of the envelope.
    if model.get("enforce_eager") is not collection.get("enforce_eager", False):
        raise ValueError("runtime graph eager setting differs from the reviewed launch")
    if (
        binding["phase"] == "prefill"
        and collection["prefill_cudagraph_policy"] == "explicit"
        and not model["enforce_eager"]
    ):
        sampling = PrefillSamplingProfile.build(
            max_isl=collection["max_num_batched_tokens"],
            max_batch_size=collection["max_num_seqs"],
            max_cudagraph_capture_size=collection["max_prefill_cudagraph_size"],
        )
        # The runner applies this override to prefill only. Backend-dependent
        # worker mode downgrades are checked against the scheduler separately;
        # they do not authorize a different initial capture list or maximum.
        initial = runtime_memory._validate_graph(
            record.get("initial_compilation_config", config["compilation_config"])
            if kind == "worker"
            else config["compilation_config"],
            eager=model["enforce_eager"],
            tokens=collection["max_num_batched_tokens"],
        )
        if (
            initial["cudagraph_capture_sizes"] != list(sampling.cudagraph_capture_sizes)
            or initial["max_cudagraph_capture_size"] != sampling.max_cudagraph_capture_size
        ):
            raise ValueError("initialized explicit prefill graph captures differ from the reviewed launch")
    requested_backend = launch["precision"].get("attention_backend", "auto")
    if requested_backend not in (None, "auto") and record.get("attention_backend") != requested_backend:
        raise ValueError("runtime observation lacks the selected attention backend evidence")
    return config


def _validate_alias_ownership(views: list[tuple[dict[str, Any], int]], count: int) -> None:
    """Prove disjoint pool-block ownership for the union of aliased views."""
    ownership, block_span = None, 0
    for view, ratio in views:
        axis = view["block_axis"]
        strides = view["stride_bytes"]
        step = ratio * strides[axis]
        within = view["element_size_bytes"] + (ratio - 1) * strides[axis]
        planes: list[tuple[int, int]] = []
        for stride, dimension in sorted(
            (stride, dimension)
            for index, (stride, dimension) in enumerate(zip(strides, view["shape"], strict=True))
            if index != axis and dimension > 1
        ):
            if stride < step:
                within += (dimension - 1) * stride
            elif planes and stride == planes[-1][0] * planes[-1][1]:
                previous_stride, previous_dimension = planes.pop()
                planes.append((previous_stride, previous_dimension * dimension))
            else:
                planes.append((stride, dimension))
        # One pool block may contain several kernel blocks, and K/V planes may
        # enclose the block axis. Normalize those dimensions, including physical
        # permutations, before comparing which pool ID owns a byte. Matching
        # tensor-local bounds alone cannot establish this cross-layer property.
        mapping = (step, planes)
        if ownership is not None and mapping != ownership:
            raise ValueError("aliased tensors have incompatible pool block ownership")
        ownership = mapping
        block_span = max(block_span, within)
    step, planes = ownership
    if block_span > step:
        raise ValueError("aliased pool block byte intervals overlap")
    span = (count - 1) * step + block_span
    for stride, dimension in planes:
        if stride < span:
            raise ValueError("aliased pool block planes overlap")
        span += (dimension - 1) * stride


def _physical_views(
    cache: dict[str, Any], scheduled: dict[str, Any], precision: str
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    _same(cache.get("semantics"), _SEMANTICS, "supported allocation semantics")
    accounting = runtime_memory.validate_shared_pool_accounting(cache, scheduled)
    count, page = accounting["num_blocks"], accounting["page_size_bytes"]
    pool = _text(cache.get("pool_id"), "pool identity")
    if scheduled.get("pool_id") != pool or scheduled.get("group_pool_ids") != [pool] * len(cache["groups"]):
        raise ValueError("worker and scheduler group-to-pool mapping disagrees")
    reserved = scheduled.get("permanent_reserved_block_ids")
    if (
        not isinstance(reserved, list)
        or any(type(value) is not int or not 0 <= value < count for value in reserved)
        or len(set(reserved)) != len(reserved)
        or len(reserved) != scheduled["reserved_blocks"]
        or scheduled["null_block_id"] not in reserved
    ):
        raise ValueError("scheduler permanent reservation IDs do not account for usable blocks")
    groups = runtime_memory._normalized_groups(cache, scheduled, page)
    layer_groups = {}
    for group_id, group in enumerate(cache["groups"]):
        if group.get("pool_id") != pool:
            raise ValueError("cache group references another pool")
        if (
            group.get("extra_retained_tokens") != 0
            or type(group.get("extra_retained_tokens")) is not int
            or group.get("attention_chunk_size") is not None
        ):
            raise ValueError("unsupported or unresolved extra/chunk cache retention")
        if runtime_memory._dtype(group.get("dtype")) != precision:
            raise ValueError("cache group dtype contradicts requested KV precision")
        _positive(group.get("spec_page_size_bytes"), "cache spec page bytes")
        for name in group["layer_names"]:
            layer_groups[name] = (group_id, group)
    storages = {}
    for storage in cache["storages"]:
        key = _text(storage.get("storage_id"), "backing storage identity")
        if key in storages or storage["size_bytes"] % count:
            raise ValueError("duplicate storage identity or non-integral storage slot size")
        storages[key] = storage
    tensors = cache.get("layer_tensors")
    allocations = cache.get("tensor_allocations")
    if (
        not isinstance(tensors, dict)
        or set(tensors) != set(layer_groups)
        or not isinstance(allocations, list)
        or not allocations
    ):
        raise ValueError("physical layer tensor/allocation coverage is incomplete")
    covered, used_storage = set(), set()
    intervals: dict[str, list[tuple[int, int]]] = {}
    packing: dict[str, int] = {}
    for allocation in allocations:
        storage_id = allocation.get("storage_id")
        if (
            storage_id not in storages
            or allocation.get("size") != storages[storage_id]["size_bytes"]
            or type(allocation.get("size")) is not int
        ):
            raise ValueError("allocation plan differs from actual backing storage")
        names = allocation.get("shared_by")
        if (
            not isinstance(names, list)
            or not names
            or any(not isinstance(name, str) or name not in layer_groups for name in names)
            or len(set(names)) != len(names)
            or covered.intersection(names)
        ):
            raise ValueError("allocation plan has missing, duplicate or unknown layers")
        if len({layer_groups[name][0] for name in names}) != len(names):
            raise ValueError("layers in one cache group cannot alias the same allocation slot")
        stride = _nonnegative(allocation.get("block_stride"), "allocation block stride")
        offset = _nonnegative(allocation.get("offset"), "allocation byte offset")
        slot = storages[storage_id]["size_bytes"] // count
        if (stride and stride != slot) or (not stride and (offset or storage_id in used_storage)):
            raise ValueError("allocation has an unsupported packed stride or undeclared storage alias")
        if storage_id in packing and packing[storage_id] != stride:
            raise ValueError("packed and unpacked allocations alias one storage")
        packing[storage_id] = stride
        width = max(layer_groups[name][1]["spec_page_size_bytes"] for name in names)
        if offset + width > slot:
            raise ValueError("allocation page bytes exceed the physical storage slot")
        for lower, upper in intervals.setdefault(storage_id, []):
            if offset < upper and lower < offset + width:
                raise ValueError("packed allocation page intervals overlap")
        intervals[storage_id].append((offset, offset + width))
        views = []
        for name in names:
            view = tensors[name]
            group = layer_groups[name][1]
            observed_offset = _nonnegative(view.get("storage_offset_bytes"), "tensor storage byte offset")
            if view.get("storage_id") != storage_id or observed_offset != offset:
                raise ValueError("layer tensor aliases or offsets contradict the allocation plan")
            shape, strides = view.get("shape"), view.get("stride_bytes")
            item_size = _positive(view.get("element_size_bytes"), "tensor element bytes")
            if not isinstance(shape, list) or not shape or not isinstance(strides, list) or len(shape) != len(strides):
                raise ValueError("physical tensor shape and strides are incomplete")
            for dimension in shape:
                _positive(dimension, "tensor dimension")
            for value in strides:
                _nonnegative(value, "tensor byte stride")
                if value % item_size:
                    raise ValueError("tensor byte strides must align to its element size")
            axis = _nonnegative(view.get("block_axis"), "tensor block axis")
            kernel = _positive(view.get("kernel_block_size_tokens"), "kernel block tokens")
            if (
                axis >= len(shape)
                or group["block_size_tokens"] % kernel
                or shape[axis] != count * (group["block_size_tokens"] // kernel)
            ):
                raise ValueError("kernel tensor block coverage contradicts pool block geometry")
            ratio = group["block_size_tokens"] // kernel
            if stride and (ratio != 1 or strides[axis] != stride):
                raise ValueError("packed kernel tensor block stride is unsupported")
            # Sufficient affine non-overlap check for the dense/permuted/strided
            # views produced by the audited runtime; no enumeration of blocks.
            span = item_size
            for byte_stride, dimension in sorted(zip(strides, shape, strict=True)):
                if dimension > 1:
                    if byte_stride < span:
                        raise ValueError("physical tensor dimensions overlap")
                    span += (dimension - 1) * byte_stride
            if offset + span > storages[storage_id]["size_bytes"]:
                raise ValueError("physical tensor byte bounds exceed its backing storage")
            logical = math.prod(shape) * item_size
            if logical > count * group["spec_page_size_bytes"]:
                raise ValueError("tensor logical bytes exceed source cache spec page bytes")
            if stride:
                within_page = item_size + sum(
                    (d - 1) * s for index, (d, s) in enumerate(zip(shape, strides, strict=True)) if index != axis
                )
                if within_page > group["spec_page_size_bytes"]:
                    raise ValueError("packed tensor exceeds its declared per-slot page interval")
            if runtime_memory._dtype(group["dtype"]) in {"bfloat16", "half"} and item_size != 2:
                raise ValueError("tensor element size contradicts cache dtype")
            views.append((view, ratio))
        _validate_alias_ownership(views, count)
        covered.update(names)
        used_storage.add(storage_id)
    if covered != set(layer_groups) or used_storage != set(storages):
        raise ValueError("allocation plan does not account for every layer and backing storage")
    return groups, accounting


def _configuration(root: Path, label: str, saved: dict[str, Any], launch: dict[str, Any]) -> dict[str, Any]:
    _same(saved.get("launch"), launch, "reviewed configuration launch")
    ranks = expected_ranks(launch["topology"])
    attempts = saved.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise ValueError("configuration has no runtime probe attempts")
    ids = [_text(attempt.get("attempt_id"), "attempt identity") for attempt in attempts]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate runtime probe attempt identity")
    active = saved.get("active_attempt_id")
    if active not in ids:
        raise ValueError("configuration requires an explicit active attempt identity")
    attempt = attempts[ids.index(active)]
    bundle_path = contained_file(root, attempt["bundle"]["manifest"])
    bundle = load_instrumentation(bundle_path, expected_version=launch["identity"]["framework_version"])
    _same(bundle.sha256, attempt["bundle"].get("sha256"), "instrumentation bundle hash")
    if not bundle.manifest.get("source_notes") or not bundle.manifest["runtime"].get("source_files"):
        raise ValueError("runtime source mapping requires hashed source_notes and runtime.source_files")
    phases = attempt.get("phases", {})
    if set(phases) != {"prefill", "decode"}:
        raise ValueError("runtime evidence requires both prefill and decode phases")
    canonical_settings, canonical_geometry, canonical_hardware, normalized = None, None, None, None
    capacities, artifacts, launch_artifacts, runtime_artifacts = [], [], [], []
    used_paths: set[str] = set()
    for phase in ("prefill", "decode"):
        result = phases[phase]
        binding = {"attempt_id": active, "configuration": label, "phase": phase, "bundle_sha256": bundle.sha256}
        launch_path, digest = _artifact(root, result["launch_manifest"])
        context = read_json(launch_path)
        if context.get("schema_version") != LAUNCH_SCHEMA:
            raise ValueError("unsupported runtime launch manifest schema")
        for name, value in {**binding, "launch": launch, "expected_ranks": ranks}.items():
            _same(context.get(name), value, f"launch manifest {name}")
        launch_artifacts.append({**result["launch_manifest"], "sha256": digest})
        workers, schedulers = {}, {}
        cell = _launch_cell(launch, phase)
        references = result.get("artifacts")
        if not isinstance(references, list):
            raise ValueError("runtime phase is missing its artifact manifest")
        for reference in references:
            path, digest = _artifact(root, reference)
            if str(path.resolve()) in used_paths:
                raise ValueError("duplicate runtime artifact path across ranks or phases")
            used_paths.add(str(path.resolve()))
            if reference.get("kind") != "observation":
                runtime_artifacts.append(copy.deepcopy(reference))
                continue
            record = read_json(path)
            config = _validate_record(record, launch=launch, bundle=bundle, binding=binding, cell=cell)
            dp = record.get("dp_rank")
            if type(dp) is not int or not 0 <= dp < cell.topology.dp:
                raise ValueError("invalid observed DP rank")
            if record["kind"] == "worker":
                tp, pp = record.get("tp_rank"), record.get("pp_rank")
                if type(tp) is not int or not 0 <= tp < cell.topology.tp or type(pp) is not int or pp != 0:
                    raise ValueError("invalid observed TP/PP rank")
                if (dp, tp) in workers:
                    raise ValueError("duplicate observed worker rank")
                workers[(dp, tp)] = record
                if canonical_hardware is None:
                    canonical_hardware = {key: record["hardware"][key] for key in ("gpu", "device_name", "sm")}
                else:
                    _same(
                        {key: record["hardware"][key] for key in ("gpu", "device_name", "sm")},
                        canonical_hardware,
                        "hardware across phases/ranks",
                    )
                settings = copy.deepcopy(config)
                settings["cache_config"].pop("enable_prefix_caching", None)
                if canonical_settings is None:
                    canonical_settings = settings
                else:
                    _same(settings, canonical_settings, "effective runtime/graph settings across phases/ranks")
            else:
                if dp in schedulers or record.get("tp_rank") is not None or record.get("pp_rank") is not None:
                    raise ValueError("duplicate or invalid observed scheduler rank")
                schedulers[dp] = record
            artifacts.append({"path": reference["path"], "sha256": digest, "evidence": record})
        if set(workers) != {(r["dp_rank"], r["tp_rank"]) for r in ranks["workers"]} or set(schedulers) != {
            r["dp_rank"] for r in ranks["schedulers"]
        }:
            raise ValueError(f"{phase} worker/scheduler rank evidence is incomplete")
        for (dp, tp), worker in sorted(workers.items()):
            scheduler = schedulers[dp]
            runtime_memory._validate_scheduler_config(worker, scheduler)
            groups, accounting = _physical_views(worker["cache"], scheduler["cache"], cell.kv_cache_dtype)
            geometry = [
                {key: value for key, value in group.items() if key != "pool_id"} for group in worker["cache"]["groups"]
            ]
            if canonical_geometry is None:
                canonical_geometry, normalized = geometry, groups
            else:
                _same(geometry, canonical_geometry, "cache geometry across phases/ranks")
                _same(groups, normalized, "physical page cost across phases/ranks")
            capacities.append(
                {
                    "phase": phase,
                    "dp_rank": dp,
                    "tp_rank": tp,
                    **accounting,
                    "kv_cache_bytes": accounting["initial_free_blocks"] * accounting["page_size_bytes"],
                }
            )
    provenance = {
        "source": "runtime_observations",
        "observation_schema": OBSERVATION_SCHEMA,
        "configuration": label,
        "attempt_id": active,
        "bundle_sha256": bundle.sha256,
        "runtime": bundle.manifest["runtime"],
        "launch": launch,
        "runtime_settings": canonical_settings,
        "rank_capacities": capacities,
        "launch_artifacts": launch_artifacts,
        "artifacts": artifacts,
        "runtime_artifacts": runtime_artifacts,
        "instrumentation": {"manifest": str(bundle_path.relative_to(root)), "files": bundle.files},
        "scope": "Physical memory of every loaded worker component. The loaded model configuration hash "
        "does not attest weight file contents. Source and synthetic checks alone do not establish "
        "live runtime qualification.",
    }
    collection = launch["collection"]
    resources = {
        "cache_layout": "grouped",
        "cache_groups": normalized,
        "max_num_tokens": collection["max_num_batched_tokens"],
        "max_batch_size": collection["max_num_seqs"],
        "runtime_memory": {
            "kv_cache_bytes": min(value["kv_cache_bytes"] for value in capacities),
            "gpu_memory_utilization": float(collection["gpu_memory_utilization"]),
            "max_model_len": collection["max_model_len"],
            "provenance": canonical_json(provenance),
        },
        "provenance": "Validated runtime shared-pool observations; minimum usable capacity across compatible phases "
        "and ranks, including padding and permanent reservations.",
    }
    FpmResourceProfile.model_validate(resources)
    return {"status": "complete", "resources": resources, "diagnostics": [], "provenance": provenance}


def validate_observations(
    index_path: str | Path, expected_configurations: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Validate every requested configuration while preserving independent success.

    Only AISimulate code runs. No observer verdict is authoritative, no timing
    rows are required, and historical attempts never substitute for the active
    attempt. Returned resources remain drafts for the existing review workflow.
    """
    index_path = Path(index_path).absolute()
    if index_path.is_symlink():
        raise ValueError("runtime observation index cannot be a symlink")
    index = read_json(index_path)
    if index.get("schema_version") != INDEX_SCHEMA or not isinstance(index.get("configurations"), dict):
        raise ValueError("unsupported runtime observation index schema")
    if not isinstance(expected_configurations, dict) or not expected_configurations:
        raise ValueError("runtime import requires selected configurations with reviewed launch facts")
    results = {}
    for label, launch in expected_configurations.items():
        try:
            if label not in index["configurations"]:
                raise ValueError("selected configuration has no runtime observation evidence")
            results[label] = _configuration(index_path.parent, label, index["configurations"][label], launch)
            results[label]["provenance"]["observations_index"] = {
                "path": str(index_path),
                "sha256": sha256_bytes(index_path.read_bytes()),
            }
            results[label]["resources"]["runtime_memory"]["provenance"] = canonical_json(results[label]["provenance"])
        except (ValueError, TypeError, KeyError, AttributeError, IndexError, OSError) as error:
            results[label] = {
                "status": "incomplete",
                "resources": None,
                "diagnostics": [f"{label}: {error}"],
                "provenance": {"observations_index": str(index_path)},
            }
    return results
