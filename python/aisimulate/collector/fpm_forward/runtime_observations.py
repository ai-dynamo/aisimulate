# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only validation of saved runtime observations, independent of timing data.

The supported allocation representation is one HBM pool shared by full-history,
sliding-window or fixed-state groups. Runtime mappings are evidence: the importer
rechecks raw geometry, configuration, physical views, and pool accounting itself.
"""

from __future__ import annotations

import copy
import itertools
import math
import re
from importlib.resources import files
from pathlib import Path, PurePosixPath
from typing import Any

from aisimulate_core.fpm_profile import FpmResourceProfile
from aisimulate_core.sdk.perf_database import load_system_spec

from . import runtime_memory
from .config import PrefillSamplingProfile, collection_phases
from .planner import BackendPolicy, FPMCell
from .runtime.fpm_memory_observer import (
    _LEGACY_SPECS,
    _MAMBA_SPEC,
    _SPEC_MODULE,
    _WINDOW_SPECS,
    HYBRID_EVIDENCE_VERSION,
)
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


def _matches_model_path(observed: Any, launch: dict[str, Any]) -> bool:
    expected = _deployed_model_path(launch)
    if observed == expected:
        return True
    revision = launch["identity"]["model_revision"]
    # vLLM 0.27.0 engine/arg_utils.py:EngineArgs.__post_init__ resolves HF
    # identifiers to local snapshots in offline mode. Bind that spelling to
    # the exact repo and immutable revision; never alias explicit local paths.
    if (
        not isinstance(observed, str)
        or expected != launch["identity"]["model"]
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*(?:/[A-Za-z0-9][A-Za-z0-9_.-]*)?", expected) is None
        or "--" in expected
        or ".." in expected
        or re.fullmatch(r"[0-9a-fA-F]{40}", revision) is None
    ):
        return False
    path = PurePosixPath(observed)
    return (
        path.is_absolute()
        and ".." not in path.parts
        and path.parts[-3:] == ("models--" + expected.replace("/", "--"), "snapshots", revision)
    )


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
    # The hybrid rules of this import follow vLLM 0.33.0 exactly; a patch release or another version gets none.
    version = record["runtime"]["version"]
    if runtime_memory._hybrid_evidence(record.get("cache")) and version != HYBRID_EVIDENCE_VERSION:
        raise ValueError(
            f"hybrid cache evidence requires vLLM {HYBRID_EVIDENCE_VERSION}; the record names vLLM {version}"
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
    # The observer hashes the loaded config.json only for the auto and hf formats
    # (runtime/instrumentation/observer.py:78-80). Each scheduler's resolved model settings must equal the worker's
    # (runtime_memory._validate_scheduler_config).
    if (
        kind == "worker"
        and runtime_memory._hybrid_evidence(record["cache"])
        and model.get("config_format", "auto") not in ("auto", "hf")
    ):
        raise ValueError(f"loaded config format requires another source mapping: {model['config_format']}")
    if not _matches_model_path(model.get("model"), launch):
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
        (binding["phase"] == "prefill" or launch.get("worker_type") is not None)
        and collection["prefill_cudagraph_policy"] == "explicit"
        and not model["enforce_eager"]
    ):
        sampling = PrefillSamplingProfile.build(
            max_isl=collection["max_num_batched_tokens"],
            max_batch_size=collection["max_num_seqs"],
            max_cudagraph_capture_size=collection["max_prefill_cudagraph_size"],
        )
        # Historical launches apply this to prefill only; explicit serving
        # roles apply their accepted configuration to every exercised phase.
        # Backend-dependent
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
    graph_settings = {
        name: collection[name]
        for name in ("cudagraph_mode", "cudagraph_capture_sizes", "max_cudagraph_capture_size")
        if collection.get(name) is not None
    }
    if graph_settings:
        if launch.get("worker_type") is None:
            raise ValueError("reviewed CUDA graph settings require an explicit serving role")
        if "cudagraph_capture_sizes" in graph_settings:
            graph_settings.setdefault("max_cudagraph_capture_size", max(graph_settings["cudagraph_capture_sizes"]))
        runtime_memory.validate_requested_graph(record, config, graph_settings)
    requested_backend = launch["precision"].get("attention_backend", "auto")
    if requested_backend not in (None, "auto") and record.get("attention_backend") != requested_backend:
        raise ValueError("runtime observation lacks the selected attention backend evidence")
    return config


def _validate_import_binding(record: dict, expected: dict, bundle: InstrumentationBundle) -> None:
    """Recheck actual runtime imports without importing any campaign code."""
    evidence = record.get("instrumentation_binding")
    if not isinstance(evidence, dict) or evidence.get("status") != "verified" or evidence.get("error"):
        raise ValueError("runtime observation lacks verified instrumentation imports")
    if (
        not isinstance(expected, dict)
        or set(expected) != {"schema_version", "loader_sha256"}
        or expected.get("schema_version") != "aisimulate-runtime-instrumentation-imports/v1"
    ):
        raise ValueError("unsupported runtime instrumentation import binding")
    validate_sha256(expected.get("loader_sha256"), "runtime instrumentation loader")
    for name, value in expected.items():
        _same(evidence.get(name), value, f"instrumentation import {name}")
    for name in ("attempt_id", "configuration", "phase", "bundle_sha256"):
        _same(evidence.get(name), record.get(name), f"instrumentation import {name}")
    _same(evidence.get("expected_bundle_sha256"), bundle.sha256, "instrumentation expected bundle")
    _same(evidence.get("files"), bundle.files, "runtime instrumentation file hashes")
    _positive(evidence.get("pid"), "instrumentation import process ID")
    _text(evidence.get("hostname"), "instrumentation import hostname")
    cpu = record.get("cpu_affinity_observation")
    if isinstance(cpu, dict):
        for name in ("pid", "hostname"):
            if cpu.get(name) is not None:
                _same(evidence[name], cpu[name], f"instrumentation import process {name}")
    root = PurePosixPath(_text(evidence.get("root"), "runtime instrumentation root"))
    if not root.is_absolute() or ".." in root.parts:
        raise ValueError("runtime instrumentation root must be an absolute normalized path")
    imports = evidence.get("imports")
    if not isinstance(imports, dict) or not imports:
        raise ValueError("runtime instrumentation import evidence is empty")
    for module, imported in imports.items():
        if not isinstance(module, str) or not isinstance(imported, dict):
            raise ValueError("runtime instrumentation module evidence must be an object")
        filename = imported.get("file")
        if (
            not isinstance(filename, str)
            or filename not in bundle.files
            or filename
            not in (
                module.replace(".", "/") + ".py",
                module.replace(".", "/") + "/__init__.py",
            )
        ):
            raise ValueError("runtime instrumentation module is not a declared bundle source")
        _same(imported.get("sha256"), bundle.files[filename], "imported instrumentation source")
        path = PurePosixPath(_text(imported.get("path"), "imported instrumentation path"))
        if path != root / filename:
            raise ValueError("runtime instrumentation module origin contradicts its declared source")
    field = "worker_class" if record["kind"] == "worker" else "scheduler_class"
    classes = evidence.get("classes")
    if not isinstance(classes, dict):
        raise ValueError("runtime instrumentation classes must be an object")
    _same(classes.get(field), bundle.manifest[field], "imported instrumentation class")
    if bundle.manifest[field].rsplit(".", 1)[0] not in imports:
        raise ValueError("runtime instrumentation class module was not observed")


def _validate_alias_ownership(views: list[tuple[dict[str, Any], int, int]], count: int) -> None:
    """Prove disjoint pool-block ownership for the union of aliased views.

    Each view comes with its kernel blocks per pool block and the kernel blocks from one pool block to the next:
    the same number, unless a vLLM 0.33.0 kernel-block view has a larger stride.
    """
    ownership, block_span = None, 0
    for view, ratio, pages in views:
        axis = view["block_axis"]
        strides = view["stride_bytes"]
        step = pages * strides[axis]
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


def _view_shape(view: dict[str, Any]) -> tuple[list[int], list[int], int]:
    """Validate the shape, byte strides and element size of one physical view."""
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
    return shape, strides, item_size


def _view_extent(shape: list[int], strides: list[int], item_size: int, start: int, size: int) -> int:
    """The byte span of a view that starts at byte ``start`` of a ``size``-byte backing storage.

    Sufficient affine non-overlap check for the dense/permuted/strided views produced by the
    audited runtime; no enumeration of blocks.
    """
    span = item_size
    for byte_stride, dimension in sorted(zip(strides, shape, strict=True)):
        if dimension > 1:
            if byte_stride < span:
                raise ValueError("physical tensor dimensions overlap")
            span += (dimension - 1) * byte_stride
    if start + span > size:
        raise ValueError("physical tensor byte bounds exceed its backing storage")
    return span


def _check_element_size(item_size: int, precision: str) -> None:
    expected_item_size = {"bfloat16": 2, "half": 2, "fp8": 1}.get(precision)
    if expected_item_size is not None and item_size != expected_item_size:
        raise ValueError("tensor element size contradicts cache dtype")


# The element bytes of the vLLM 0.33.0 Mamba state dtypes. MambaStateDtypeCalculator gives every state the model
# dtype, a MambaDType choice (config/cache.py:175) or float32 (model_executor/layers/mamba/mamba_utils.py:53-150).
_STATE_DTYPE_BYTES = {"torch.float32": 4, "torch.float16": 2, "torch.bfloat16": 2}


def _indexer(spec: dict[str, Any]) -> bool:
    """The sparse MLA indexer, an MLAAttentionSpec with role INDEXER (vLLM 0.33.0 kv_cache_interface.py:737).

    It stores uint8 keys at every KV precision (models/glm5next/common/attention.py:260-262).
    """
    return (
        spec.get("spec_type") == "vllm.v1.kv_cache_interface.MLAAttentionSpec" and spec.get("cache_role") == "INDEXER"
    )


def _layer_views(
    name: str,
    entries: Any,
    storage_id: str,
    region: int,
    block_stride: int,
    count: int,
    size: int,
    group: dict[str, Any],
    precision: str,
    group_kernel: int,
) -> list[tuple[dict[str, Any], int, int]]:
    """Check the views of one vLLM 0.33.0 layer, of any rank with the block axis first.

    A layer views its pages as one tensor, or as one tensor per state that a Mamba layer
    cuts from its page (kv_cache_interface.py:418-482, mamba/abstract.py:28-42). Each view
    starts and ends inside the layer's page of a block, the views of one layer do not
    overlap there, and together they hold at most that page, because pages share bytes only
    for the same block ID in different groups. A block-outer block also holds the pages of
    the tensor's other layers.

    A layer of an attention group whose backend sets a kernel_block_size is bound to a view in kernel
    blocks (v1/worker/utils.py:460-473): kernel block j of block b is b * kernel_block_stride + j, so the
    view holds (count - 1) * stride + block // kernel of them, one kernel page apart (:419-430). With
    stride 0, vLLM binds the manager view, which holds count * (block // kernel) of them (:402-403).
    """
    spec = group["layer_specs"][name]
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"physical layer views are missing: {name}")
    if any(not isinstance(view, dict) or view.get("storage_id") != storage_id for view in entries):
        raise ValueError("layer tensor aliases or offsets contradict the allocation plan")
    if any("kernel_block_stride" not in view for view in entries):
        raise ValueError(f"layer view lacks its kernel block stride: {name}")
    block = group["block_size_tokens"]
    # A Mamba layer keeps the group block, which both native runners also give its group (utils.py:677-679). vLLM
    # maps only a cache with more than one token per state (utils.py:404-411), so any other layer keeps the
    # runner's kernel block of its group and stride 0. Every view of one layer has one kernel block and stride.
    kernel = (entries[0].get("kernel_block_size_tokens"), entries[0]["kernel_block_stride"])
    tokens = spec.get("tokens_per_state")
    if spec.get("spec_type") == _MAMBA_SPEC:
        expected = (block, 0) if group_kernel == block else None
    elif type(tokens) is not int or tokens <= 1:
        expected = (group_kernel, 0)
    else:
        expected = kernel
    if expected != kernel or any(
        (view.get("kernel_block_size_tokens"), view["kernel_block_stride"]) != kernel for view in entries
    ):
        raise ValueError(f"layer view kernel block contradicts its cache spec: {name}")
    views, intervals, dense = [], [], 0
    for view in entries:
        start = _nonnegative(view.get("storage_offset_bytes"), "tensor storage byte offset") - region
        shape, strides, item_size = _view_shape(view)
        kernel = _positive(view.get("kernel_block_size_tokens"), "kernel block tokens")
        stride = _nonnegative(view["kernel_block_stride"], "kernel block stride")
        ratio = block // kernel
        if view.get("block_axis") != 0 or block % kernel:
            raise ValueError("kernel tensor block coverage contradicts pool block geometry")
        if stride and stride < ratio:
            raise ValueError("kernel block stride is below the kernel blocks of one block")
        # The kernel blocks from one pool block to the next.
        pages = stride or ratio
        if shape[0] != (count - 1) * pages + ratio or strides[0] * pages != block_stride:
            raise ValueError("kernel tensor block coverage contradicts pool block geometry")
        _view_extent(shape, strides, item_size, region + start, size)
        within = item_size + (ratio - 1) * strides[0]
        within += sum((dimension - 1) * stride for dimension, stride in zip(shape[1:], strides[1:], strict=True))
        if not 0 <= start < spec["page_size_bytes"] or start + within > spec["page_size_bytes"]:
            raise ValueError(f"layer view lies outside the page of its layer: {name}")
        if group["kind"] == "attention" and not _indexer(spec):
            _check_element_size(item_size, precision)
        intervals.append((start, start + within))
        dense += ratio * math.prod(shape[1:]) * item_size
        views.append((view, ratio, pages))
    intervals.sort()
    if any(end > following for (_, end), (following, _) in itertools.pairwise(intervals)):
        raise ValueError(f"views of one layer overlap inside the block: {name}")
    if spec.get("spec_type") == _MAMBA_SPEC:
        # bind_kv_cache cuts one view per state from the page, in the order of the spec's shapes and dtypes: the
        # block axis, then the state's shape (mamba/abstract.py:28-42, :76-78). The states fill the spec's real
        # page, and the page may pad it (kv_cache_interface.py:1116-1129).
        state = group["state"]
        shapes, dtypes = state.get("shapes"), state.get("dtypes")
        if (
            not isinstance(shapes, list)
            or not isinstance(dtypes, list)
            or not len(entries) == len(shapes) == len(dtypes)
            or any(
                view["shape"][1:] != shape
                or not isinstance(dtype, str)
                or view["element_size_bytes"] != _STATE_DTYPE_BYTES.get(dtype)
                for view, shape, dtype in zip(entries, shapes, dtypes, strict=True)
            )
            or type(state.get("real_page_size_bytes")) is not int
            or dense != state["real_page_size_bytes"]
        ):
            raise ValueError(f"layer views do not match the recorded state tensors: {name}")
        # Each state starts where the one before it ends, from the start of the layer's page, and is viewed
        # contiguously (mamba/abstract.py:34-41).
        offset = region
        for view, shape in zip(entries, shapes, strict=True):
            item = view["element_size_bytes"]
            if view["storage_offset_bytes"] != offset or view["stride_bytes"][1:] != [
                item * math.prod(shape[axis + 1 :]) for axis in range(len(shape))
            ]:
                raise ValueError(f"layer views do not match the recorded state tensors: {name}")
            offset += item * math.prod(shape)
    return views


def _layer_strided_views(
    allocations: list[Any],
    tensors: dict[str, Any],
    storages: dict[str, dict[str, Any]],
    layer_groups: dict[str, tuple[int, dict[str, Any]]],
    count: int,
    precision: str,
    kernels: list[int],
) -> None:
    """Check the vLLM 0.33.0 placement of every cache layer in one backing buffer.

    Block ``b`` of layer ``layers[l]`` starts at ``offset + l * layer_stride + b * block_stride``
    (kv_cache_interface.py:1475-1495). The last page of the last layer bounds the region in
    layer-outer and block-outer layouts. Tensors whose regions intersect overlay
    cache groups, which is sound because a block ID belongs to one group at a time (:1485-1487).
    The generic layouts start every group's tensors at byte 0 (v1/core/kv_cache_planning.py:706-732), so
    overlapping tensors may differ in offset and layer count. The overlap check therefore compares pages:
    two pages may share bytes only when they hold one block ID for layers of different groups.
    """
    if len(storages) != 1:
        raise ValueError("layer-strided cache allocations require one backing storage")
    ((storage_id, storage),) = storages.items()
    size = storage["size_bytes"]
    placements: list[tuple[list[str], int, int, int, int]] = []
    views: dict[str, list[tuple[dict[str, Any], int, int]]] = {}
    for allocation in allocations:
        if (
            not isinstance(allocation, dict)
            or allocation.get("storage_id") != storage_id
            or type(allocation.get("size")) is not int
            or allocation["size"] != size
        ):
            raise ValueError("allocation plan differs from actual backing storage")
        names = allocation.get("layers")
        # The observer writes a layer-strided allocation without shared_by (runtime/fpm_memory_observer.py,
        # _layer_strided_allocations).
        if (
            not isinstance(names, list)
            or not names
            or any(not isinstance(name, str) or name not in layer_groups for name in names)
            or len(set(names)) != len(names)
            or views.keys() & set(names)
            or "shared_by" in allocation
        ):
            raise ValueError("allocation plan has missing, duplicate or unknown layers")
        offset = _nonnegative(allocation.get("offset"), "allocation byte offset")
        layer_stride = _nonnegative(allocation.get("layer_stride"), "allocation layer stride")
        block_stride = _positive(allocation.get("block_stride"), "allocation block stride")
        page = max(layer_groups[name][1]["layer_specs"][name]["page_size_bytes"] for name in names)
        end = offset + (len(names) - 1) * layer_stride + (count - 1) * block_stride + page
        if page > block_stride or end > size:
            raise ValueError("allocation layers exceed their block stride or the backing storage")
        placements.append((names, offset, layer_stride, block_stride, end))
        for position, name in enumerate(names):
            region = offset + position * layer_stride
            group_id, group = layer_groups[name]
            views[name] = _layer_views(
                name, tensors[name], storage_id, region, block_stride, count, size, group, precision, kernels[group_id]
            )
    if views.keys() != layer_groups.keys():
        raise ValueError("allocation plan does not account for every layer and backing storage")
    pages = {name: layer_groups[name][1]["layer_specs"][name]["page_size_bytes"] for name in views}
    for first, second in itertools.combinations_with_replacement(range(len(placements)), 2):
        names, offset, layer_stride, block_stride, end = placements[first]
        others, other_offset, other_layer_stride, other_block_stride, other_end = placements[second]
        if not (offset < other_end and other_offset < end):
            continue
        if block_stride != other_block_stride:
            raise ValueError("aliased cache allocations must share one block stride")
        for position, name in enumerate(names):
            for other_position, other in enumerate(others):
                if first == second and other_position <= position:
                    continue
                # Page b of name and page b + d of other share bytes iff
                # -page(other) < delta + d * block_stride < page(name), for d in a contiguous range.
                delta = other_offset + other_position * other_layer_stride - offset - position * layer_stride
                low = max((-pages[other] - delta) // block_stride + 1, 1 - count)
                high = min(-((delta - pages[name]) // block_stride) - 1, count - 1)
                if low > high:
                    continue
                if (low, high) != (0, 0):
                    raise ValueError(f"cache pages of different block IDs overlap: {name}, {other}")
                if layer_groups[name][0] == layer_groups[other][0]:
                    raise ValueError(f"cache pages of one block overlap for layers of one group: {name}, {other}")
        _validate_alias_ownership([item for name in dict.fromkeys([*names, *others]) for item in views[name]], count)


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
    hybrid = runtime_memory._hybrid_evidence(cache)
    # A vLLM 0.33.0 scheduler reserves only the null block (runtime/instrumentation/observer.py _reservations).
    if hybrid and reserved != [scheduled["null_block_id"]]:
        raise ValueError("additional initialized reservations lack an audited permanent-reservation mapping")
    groups = runtime_memory._normalized_groups(cache, scheduled, page)
    layer_groups = {}
    for group_id, group in enumerate(cache["groups"]):
        # _normalized_groups accepts every table class, but vLLM 0.27.0 and 0.28.0 records hold only the legacy ones.
        if not hybrid and group.get("spec_type") not in _LEGACY_SPECS:
            raise ValueError(f"unsupported runtime cache spec: {group.get('spec_type')!r}")
        if group.get("pool_id") != pool:
            raise ValueError("cache group references another pool")
        if (
            group.get("extra_retained_tokens") != 0
            or type(group.get("extra_retained_tokens")) is not int
            or group.get("attention_chunk_size") is not None
        ):
            raise ValueError("unsupported or unresolved extra/chunk cache retention")
        # A vLLM 0.33.0 uniform group can mix dtypes, so its layers are checked one by one. Only
        # attention layers store KV at the requested precision; the sparse MLA indexer is exempt.
        dtypes = (
            [
                group["layer_specs"][name].get("dtype")
                for name in group["layer_names"]
                if group["kind"] == "attention" and not _indexer(group["layer_specs"][name])
            ]
            if hybrid
            else [group.get("dtype")]
        )
        for dtype in dtypes:
            storage_dtype = runtime_memory._dtype(dtype)
            # vLLM 4bdc8a788d2e2ce9165d552b3d4d8b72604626bf
            # utils/torch_utils.py:STR_DTYPE_TO_TORCH_DTYPE stores FP8 as uint8,
            # also used by other packed formats. Only the independently validated
            # resolved cache precision establishes FP8; retain raw storage evidence.
            if storage_dtype != precision and not (storage_dtype == "torch.uint8" and precision == "fp8"):
                raise ValueError("cache group dtype contradicts requested KV precision")
        _positive(group.get("spec_page_size_bytes"), "cache spec page bytes")
        # The page of a vLLM 0.33.0 uniform group is the sum of its layer pages (kv_cache_interface.py:1319-1320). In
        # any other group, the spec of each layer is the group spec (runtime/fpm_memory_observer.py cache_groups).
        if hybrid:
            pages = [spec["page_size_bytes"] for spec in group["layer_specs"].values()]
            if any(
                page != group["spec_page_size_bytes"]
                for page in ([sum(pages)] if runtime_memory._uniform(group) else pages)
            ):
                raise ValueError(f"cache group page contradicts its layer specs: {group['layer_names']}")
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
    if hybrid:
        # The runner's kernel block of every group, which the worker records before any view
        # (runtime/instrumentation/observer.py, _layer_views).
        runner = cache.get("model_runner")
        kernels = runner.get("kernel_block_sizes") if isinstance(runner, dict) else None
        if not isinstance(kernels, list) or len(kernels) != len(cache["groups"]):
            raise ValueError("initialized kernel block sizes do not cover every cache group")
        for kernel in kernels:
            _positive(kernel, "kernel block size")
        # The observer reads that list only from the two audited native runners
        # (runtime/instrumentation/observer.py, _layer_views).
        if runner.get("class") not in (
            "vllm.v1.worker.gpu_model_runner.GPUModelRunner",
            "vllm.v1.worker.gpu.model_runner.GPUModelRunner",
        ):
            raise ValueError(f"model runner has no audited cache mapping: {runner.get('class')}")
        _layer_strided_views(allocations, tensors, storages, layer_groups, count, precision, kernels)
        return groups, accounting
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
            shape, strides, item_size = _view_shape(view)
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
            _view_extent(shape, strides, item_size, offset, storages[storage_id]["size_bytes"])
            logical = math.prod(shape) * item_size
            if logical > count * group["spec_page_size_bytes"]:
                raise ValueError("tensor logical bytes exceed source cache spec page bytes")
            if stride:
                within_page = item_size + sum(
                    (d - 1) * s for index, (d, s) in enumerate(zip(shape, strides, strict=True)) if index != axis
                )
                if within_page > group["spec_page_size_bytes"]:
                    raise ValueError("packed tensor exceeds its declared per-slot page interval")
            _check_element_size(item_size, precision)
            views.append((view, ratio, ratio))
        _validate_alias_ownership(views, count)
        covered.update(names)
        used_storage.add(storage_id)
    if covered != set(layer_groups) or used_storage != set(storages):
        raise ValueError("allocation plan does not account for every layer and backing storage")
    return groups, accounting


def _prefix_caching_settings(cache: dict[str, Any], *, scheduler: bool, hybrid: bool) -> dict[str, Any] | None:
    """The prefix-caching settings that one vLLM 0.33.0 process recorded; None for other records.

    vLLM 0.27.0 and 0.28.0 records have no ``prefix_caching`` key. A campaign-local adapter can add one;
    the import reads it only when ``hybrid`` says that the phase holds vLLM 0.33.0 records. The scheduler
    adds its block size and hash block size (core/sched/scheduler.py:218, :348-350).
    """
    if not hybrid or "prefix_caching" not in cache:
        return None
    value = cache["prefix_caching"]
    keys = {"enable_prefix_caching", "mamba_cache_mode"} | ({"block_size", "hash_block_size"} if scheduler else set())
    if (
        not isinstance(value, dict)
        or set(value) != keys
        or type(value["enable_prefix_caching"]) is not bool
        or value["mamba_cache_mode"] not in {"none", "align", "all"}
    ):
        raise ValueError("runtime observation requires complete prefix-caching evidence")
    if value["mamba_cache_mode"] == "all":
        raise ValueError(runtime_memory._MAMBA_ALL_UNMODELED)
    if value["mamba_cache_mode"] == "align" and not value["enable_prefix_caching"]:
        raise ValueError("mamba_cache_mode align requires enable_prefix_caching=true")
    if scheduler:
        _positive(value["block_size"], "scheduler block size")
        _positive(value["hash_block_size"], "scheduler hash block size")
    return value


# The vLLM 0.33.0 spec classes whose groups take no part in prefix caching (kv_cache_interface.py:998-1000). A
# SlidingWindowMLASpec group takes part unless it replays a bounded window (:1015, :1027-1029), which no record holds.
_NO_PREFIX_CACHING_SPECS = frozenset({_SPEC_MODULE + "CircularBufferSpec"})


def _scheduler_hash_block_sizes(record: dict[str, Any]) -> list[Any]:
    """The hash block sizes that vLLM 0.33.0 can derive for a scheduler from its cache groups and cache settings.

    The engine core resolves the hash block size from the scheduler copies of the cache groups (engine/core.py:176-188,
    v1/core/kv_cache_utils.py:816-891). One group hashes in its own block size. Several groups hash in the scheduler
    block size, the LCM of their block sizes, when prefix caching is off or a Mamba group is outside align mode. A KV
    connector would give them finer hashes without prefix caching, but every accepted record has no
    kv_transfer_config. Otherwise several groups hash in prefix_match_unit tokens, or in the GCD of the block sizes of
    the groups that take part in prefix caching, or of every group when none does. Each of these cases gives one size.
    Each SlidingWindowMLASpec group takes part unless its spec replays a bounded window (kv_cache_interface.py:1015,
    :1027-1029), which no record holds. With such groups, each choice of the groups that take part gives one size.
    """
    groups, settings = record["cache"]["groups"], record["resolved_config"]["cache_config"]
    blocks = [_positive(group["block_size_tokens"], "cache block size") for group in groups]
    if (
        len(groups) == 1
        or not settings.get("enable_prefix_caching")
        or any(
            group["spec_type"] == _MAMBA_SPEC and group["state"].get("mamba_cache_mode") != "align" for group in groups
        )
    ):
        return [math.lcm(*blocks)]
    if settings.get("prefix_match_unit") is not None:
        return [settings["prefix_match_unit"]]
    unknown = _SPEC_MODULE + "SlidingWindowMLASpec"
    caching = [
        block
        for block, group in zip(blocks, groups, strict=True)
        if group["spec_type"] not in {*_NO_PREFIX_CACHING_SPECS, unknown}
    ]
    # Start from the GCD of the blocks of the groups that surely take part, which is 0 without such a group. Each
    # SlidingWindowMLASpec group may add its block to every GCD so far, so the set holds the GCD of every choice of
    # those groups, and it never holds more values than there are distinct GCDs.
    hashes = {math.gcd(*caching)}
    for block, group in zip(blocks, groups, strict=True):
        if group["spec_type"] == unknown:
            hashes |= {math.gcd(value, block) for value in hashes}
    # Without a group in prefix caching, vLLM hashes in the GCD of every block.
    return sorted({math.gcd(*blocks) if value == 0 else value for value in hashes})


def _scheduler_hash_block_size_fits(record: dict[str, Any]) -> bool:
    """Whether vLLM 0.33.0 starts with the hash block size of a scheduler and its cache groups.

    The engine core resolves the block sizes before it builds the scheduler (engine/core.py:176-188). With several
    groups that hash in prefix_match_unit tokens or in a GCD, resolve_kv_cache_block_sizes raises unless two rules hold
    (v1/core/kv_cache_utils.py:885-917). The hash block size divides the block size of every group that takes part in
    prefix caching, or of every group when none does (:885-898). A GCD always does, so only a unit can break this rule.
    The cache-hit alignment is a multiple of every int tokens_per_state above 1 of a spec in prefix caching
    (:899-917). At the decode context parallel size of 1 that every accepted record has, the alignment is the hash
    block size when a Mamba group has a larger block, else the scheduler block size. The scheduler record holds the
    groups that vLLM reads, in which a uniform group keeps only its first inner spec (:1110-1116).

    A SlidingWindowMLASpec group may or may not take part, so one choice of the groups that take part must give the
    recorded hash block size and meet both rules. With that hash block size, the alignment no longer depends on the
    choice. So every group that takes part must pass the same test: the hash block size divides its block, and the
    alignment is a multiple of its tokens per state. A SlidingWindowMLASpec group that passes it may take part whatever
    the other groups do. Every such group keeps the GCD a multiple of the hash block size, so the choice with all of
    them gives the smallest GCD.
    """
    cache, settings = record["cache"], record["resolved_config"]["cache_config"]
    groups, hash_block = cache["groups"], cache["hash_block_size"]
    if (
        len(groups) == 1
        or not settings.get("enable_prefix_caching")
        or any(
            group["spec_type"] == _MAMBA_SPEC and group["state"].get("mamba_cache_mode") != "align" for group in groups
        )
    ):
        return True
    partial = any(group["spec_type"] == _MAMBA_SPEC and group["block_size_tokens"] > hash_block for group in groups)
    alignment = hash_block if partial else cache["block_size"]

    def can_take_part(group: dict[str, Any]) -> bool:
        return group["block_size_tokens"] % hash_block == 0 and all(
            not isinstance(tokens, int) or tokens <= 1 or alignment % tokens == 0
            for tokens in (spec.get("tokens_per_state") for spec in group["layer_specs"].values())
        )

    unknown = _SPEC_MODULE + "SlidingWindowMLASpec"
    caching = [group for group in groups if group["spec_type"] not in {*_NO_PREFIX_CACHING_SPECS, unknown}]
    sliding = [group["block_size_tokens"] for group in groups if group["spec_type"] == unknown and can_take_part(group)]
    unit = settings.get("prefix_match_unit") is not None
    if caching:
        # The groups that surely take part meet both rules under every choice or under none. A unit is the hash block
        # size of every choice. Without one, the smallest GCD must be the hash block size.
        return all(can_take_part(group) for group in caching) and (
            unit or math.gcd(*(group["block_size_tokens"] for group in caching), *sliding) == hash_block
        )
    # Otherwise some of those SlidingWindowMLASpec groups take part, or no group does and vLLM hashes every block.
    blocks = [group["block_size_tokens"] for group in groups]
    if unit:
        return bool(sliding) or all(block % hash_block == 0 for block in blocks)
    return math.gcd(*sliding) == hash_block or math.gcd(*blocks) == hash_block


def _phase_prefix_caching(
    phase: str, workers: dict[tuple[int, int], dict[str, Any]], schedulers: dict[int, dict[str, Any]]
) -> dict[str, Any] | None:
    """The prefix-caching evidence of one phase, on which every rank of the phase agrees.

    The hash block size comes from the scheduler. Every MambaSpec takes the mode of its cache
    config (mamba/abstract.py:83), so each Mamba state group reports the phase's mode. A vLLM
    0.33.0 record always carries the evidence. Every other record gives None.
    """
    label = f"{phase} prefix-caching settings across ranks"
    hybrid = any(runtime_memory._hybrid_evidence(record["cache"]) for record in workers.values())
    if hybrid and any("prefix_caching" not in record["cache"] for record in (*schedulers.values(), *workers.values())):
        raise ValueError("runtime observation requires complete prefix-caching evidence")
    settings = [
        _prefix_caching_settings(record["cache"], scheduler=True, hybrid=hybrid)
        for _, record in sorted(schedulers.items())
    ]
    for value in settings[1:]:
        _same(value, settings[0], label)
    scheduled = settings[0]
    shared = (
        None if scheduled is None else {key: scheduled[key] for key in ("enable_prefix_caching", "mamba_cache_mode")}
    )
    for _, record in sorted(workers.items()):
        _same(_prefix_caching_settings(record["cache"], scheduler=False, hybrid=hybrid), shared, label)
    if scheduled is None:
        return None
    for record in workers.values():
        for group in record["cache"]["groups"]:
            state = group.get("state")
            # A vLLM 0.33.0 record holds the mode of every MambaSpec, also inside a uniform group
            # (runtime/fpm_memory_observer.py _state). The scheduler copy of each group equals the worker group
            # (runtime_memory._normalized_groups), so it holds the mode too.
            if (
                hybrid
                and _MAMBA_SPEC in runtime_memory._spec_classes(group, group["layer_names"], hybrid)
                and "mamba_cache_mode" not in state
            ):
                raise ValueError(f"runtime state cache group lacks Mamba cache mode evidence: {group['layer_names']}")
            if group.get("kind") == "state" and isinstance(state, dict) and "mamba_cache_mode" in state:
                _same(state["mamba_cache_mode"], scheduled["mamba_cache_mode"], f"{phase} Mamba cache mode")
            # A MambaSpec takes the Mamba block size of its worker's cache config, which the worker resolves before it
            # builds the specs (mamba/abstract.py:72-91). Grouping keeps that block: the packed path never scales it
            # (v1/core/kv_cache_planning.py:914-1041), the other path pads a Mamba page rather than scale its block
            # (:380-389), and only HiddenStateCacheSpec layers scale every group block there (:1081-1110, :1179-1182).
            # Those layers serve hidden-state extraction through a KV connector (:1089-1090), and the probe refuses a
            # run with a KV transfer config.
            if (
                hybrid
                and _MAMBA_SPEC in runtime_memory._spec_classes(group, group["layer_names"], hybrid)
                and canonical_json(group["block_size_tokens"])
                != canonical_json(record["resolved_config"]["cache_config"].get("mamba_block_size"))
            ):
                raise ValueError(
                    f"cache group block size contradicts the resolved Mamba block size: {group['layer_names']}"
                )
    # Each process reads this evidence from the cache config that it resolved
    # (runtime/instrumentation/observer.py, _prefix_caching). The launch-setting comparisons leave those resolved
    # settings out, so the evidence must equal them.
    if hybrid:
        names = ("enable_prefix_caching", "mamba_cache_mode")
        for record in (*schedulers.values(), *workers.values()):
            resolved = record["resolved_config"]["cache_config"]
            if canonical_json([record["cache"]["prefix_caching"][name] for name in names]) != canonical_json(
                [resolved.get(name) for name in names]
            ):
                raise ValueError("runtime prefix-caching evidence contradicts the resolved cache settings")
        # A scheduler copies its own block size and hash block size into the evidence
        # (runtime/instrumentation/observer.py, _prefix_caching).
        sizes = ("block_size", "hash_block_size")
        for record in schedulers.values():
            cache = record["cache"]
            if canonical_json([cache["prefix_caching"][name] for name in sizes]) != canonical_json(
                [cache.get(name) for name in sizes]
            ):
                raise ValueError("runtime prefix-caching evidence contradicts the scheduler block sizes")
            # The scheduler also records the mode of its resolved cache config
            # (runtime/fpm_memory_observer.py, scheduler_memory).
            if canonical_json(cache["prefix_caching"]["mamba_cache_mode"]) != canonical_json(
                cache.get("mamba_cache_mode")
            ):
                raise ValueError("runtime prefix-caching evidence contradicts the resolved cache settings")
            # The scheduler block size is the LCM of the block sizes of the scheduler's cache groups, which a decode
            # context parallel size of 1 leaves unscaled (v1/core/kv_cache_utils.py:860-863, engine/core.py:176-188).
            # The hash block size is one that vLLM can derive from those groups, and vLLM also refuses to start unless
            # it fits them (v1/core/kv_cache_utils.py:885-917).
            hash_blocks = {canonical_json(value) for value in _scheduler_hash_block_sizes(record)}
            if (
                cache["block_size"] != math.lcm(*(group["block_size_tokens"] for group in cache["groups"]))
                or canonical_json(cache["hash_block_size"]) not in hash_blocks
                or not _scheduler_hash_block_size_fits(record)
            ):
                raise ValueError("scheduler block sizes contradict its cache groups")
    return {
        "enable_prefix_caching": scheduled["enable_prefix_caching"],
        "mamba_cache_mode": scheduled["mamba_cache_mode"],
        "hash_block_size_tokens": scheduled["hash_block_size"],
    }


# The vLLM 0.33.0 spec classes that vLLM's startup sizing counts as one block per request: the ring
# (kv_cache_interface.py:982-985).
_ONE_BLOCK_SPECS = frozenset({_SPEC_MODULE + "CircularBufferSpec"})


def _validate_startup_concurrency(record: dict[str, Any]) -> None:
    """Check the kv_cache_max_concurrency of a vLLM 0.33.0 scheduler against its cache groups.

    vLLM divides the pool blocks by the blocks that its startup sizing counts for one request at max_model_len in the
    scheduler's groups (v1/core/kv_cache_utils.py:1076-1087, v1/engine/core.py:335-352). It counts 2 + k + c blocks
    for a MambaSpec group in align mode and 1 + k in mode none (kv_cache_interface.py:1135-1141), so this check sees
    the align count that the steady blocks_per_request leaves out. It counts one block for the ring (:982-985).
    Every other class of SPEC_KINDS but the windowed ones is FullAttentionSpec or a subclass of it. For these it counts
    cdiv(max_model_len, block_size) blocks at the decode context parallel size of 1 that every accepted record has
    (:643-648). vLLM's count for a windowed group also covers the tokens in flight (:918-944), which this check does
    not model, so a layout with one keeps the other checks only.
    """
    cache = record["cache"]
    context = record["resolved_config"]["model_config"]["max_model_len"]
    blocks = 0
    for group in cache["groups"]:
        spec = group["spec_type"]
        if spec == _MAMBA_SPEC:
            state = group["state"]
            blocks += state["num_speculative_blocks"] + (
                2 + state["num_prefill_checkpoint_blocks"] if cache["mamba_cache_mode"] == "align" else 1
            )
        elif spec in _ONE_BLOCK_SPECS:
            blocks += 1
        elif spec.removeprefix(_SPEC_MODULE) in _WINDOW_SPECS:
            return
        else:
            blocks += -(-context // group["block_size_tokens"])
    label = f"{record['phase']} scheduler of DP rank {record['dp_rank']}"
    concurrency = cache.get("kv_cache_max_concurrency")
    if isinstance(concurrency, bool) or not isinstance(concurrency, (int, float)) or not 0 < concurrency < math.inf:
        raise ValueError(f"{label} lacks a positive kv_cache_max_concurrency")
    # vLLM keeps the quotient as a float. The tolerance covers its rounding, far below the change of one block.
    if not math.isclose(concurrency, cache["num_blocks"] / blocks, rel_tol=1e-9):
        raise ValueError(
            f"{label} records kv_cache_max_concurrency {concurrency!r}, which gives "
            f"{cache['num_blocks'] / concurrency:.3f} blocks per request, but its cache groups reserve {blocks}"
        )


def _mode_free(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Cache groups without the parts of a state group that follow the prefix-caching mode.

    A Mamba block holds the hybrid block in align mode and max_model_len in none mode
    (models/config.py:763-787), and every MambaSpec records the mode (kv_cache_interface.py:1099).
    No layer spec field follows the block size, so the layer specs stay compared.
    """
    result = copy.deepcopy(groups)
    for group in result:
        if group.get("kind") == "state":
            group.pop("block_size_tokens", None)
            if isinstance(group.get("state"), dict):
                group["state"].pop("mamba_cache_mode", None)
    return result


def _configuration(
    root: Path,
    label: str,
    saved: dict[str, Any],
    launch: dict[str, Any],
    *,
    phase: str | None = None,
    attempt_id: str | None = None,
) -> dict[str, Any]:
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
    if phase is not None:
        _same(active, attempt_id, "selected phase attempt identity")
    attempt = attempts[ids.index(active)]
    bundle_path = contained_file(root, attempt["bundle"]["manifest"])
    identity = launch["identity"]
    bundle = load_instrumentation(
        bundle_path,
        expected_version=identity.get("runtime_framework_version"),
    )
    _same(bundle.sha256, attempt["bundle"].get("sha256"), "instrumentation bundle hash")
    if not bundle.manifest.get("source_notes") or not bundle.manifest["runtime"].get("source_files"):
        raise ValueError("runtime source mapping requires hashed source_notes and runtime.source_files")
    phases = attempt.get("phases", {})
    required_phases = collection_phases(launch.get("worker_type"))
    if phase is not None:
        if phase not in required_phases or phase not in phases or set(phases) - set(required_phases):
            raise ValueError("runtime evidence does not contain the selected serving-role phase")
        required_phases = (phase,)
    elif set(phases) != set(required_phases):
        raise ValueError("runtime evidence requires every phase of the selected serving role")
    canonical_settings, canonical_geometry, canonical_hardware, normalized = None, None, None, None
    capacities, artifacts, launch_artifacts, runtime_artifacts, import_bindings = [], [], [], [], []
    prefix_caching: dict[str, dict[str, Any] | None] = {}
    geometry_differences: list[str] = []
    used_paths: set[str] = set()
    for phase in required_phases:
        result = phases[phase]
        binding = {"attempt_id": active, "configuration": label, "phase": phase, "bundle_sha256": bundle.sha256}
        launch_path, digest = _artifact(root, result["launch_manifest"])
        context = read_json(launch_path)
        if context.get("schema_version") != LAUNCH_SCHEMA:
            raise ValueError("unsupported runtime launch manifest schema")
        for name, value in {**binding, "launch": launch, "expected_ranks": ranks}.items():
            _same(context.get(name), value, f"launch manifest {name}")
        launch_artifacts.append({**result["launch_manifest"], "sha256": digest})
        import_binding = context.get("instrumentation_binding")
        import_bindings.append(import_binding is not None)
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
            if import_binding is not None:
                _validate_import_binding(record, import_binding, bundle)
            elif "instrumentation_binding" in record:
                raise ValueError("observed instrumentation imports lack their frozen launch binding")
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
                # In a vLLM 0.33.0 record these settings follow the phase's prefix-caching mode or the engine core,
                # and provenance prefix_caching records the mode of every phase instead. Another record can hold
                # them only from a campaign-local adapter, and its provenance keeps them, as finalized campaigns
                # compare that provenance byte for byte.
                removed = (
                    runtime_memory._PHASE_CACHE_SETTINGS
                    if runtime_memory._hybrid_evidence(record["cache"])
                    else ("enable_prefix_caching",)
                )
                for name in removed:
                    settings["cache_config"].pop(name, None)
                if canonical_settings is None:
                    canonical_settings = settings
                else:
                    if settings["compilation_config"] != canonical_settings["compilation_config"]:
                        raise ValueError(
                            "effective runtime/graph settings across phases/ranks differ: a shared memory "
                            "profile requires matching observed compilation and CUDA-graph captures. "
                            "use separate prefill/decode serving-role configurations when the intended engines differ"
                        )
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
        phase_geometry = None
        for (dp, tp), worker in sorted(workers.items()):
            scheduler = schedulers[dp]
            runtime_memory._validate_scheduler_config(worker, scheduler)
            groups, accounting = _physical_views(
                worker["cache"], scheduler["cache"], runtime_memory._resolved_cache_dtype(worker["resolved_config"])
            )
            geometry = [
                {key: value for key, value in group.items() if key != "pool_id"} for group in worker["cache"]["groups"]
            ]
            # The ranks of one phase agree exactly. Another phase may differ only in the parts of a
            # state group that follow the prefix-caching mode; the first phase's groups are kept.
            if phase_geometry is None:
                phase_geometry = geometry, groups
            else:
                _same(geometry, phase_geometry[0], "cache geometry across phases/ranks")
                _same(groups, phase_geometry[1], "physical page cost across phases/ranks")
            if canonical_geometry is None:
                canonical_geometry, normalized = geometry, groups
            else:
                _same(_mode_free(geometry), _mode_free(canonical_geometry), "cache geometry across phases/ranks")
                _same(_mode_free(groups), _mode_free(normalized), "physical page cost across phases/ranks")
                for group, first in zip(groups, normalized, strict=True):
                    if group["block_size_tokens"] != first["block_size_tokens"]:
                        difference = (
                            f"{phase}: {group['name']} block_size_tokens "
                            f"{group['block_size_tokens']} != {first['block_size_tokens']}"
                        )
                        if difference not in geometry_differences:
                            geometry_differences.append(difference)
            capacities.append(
                {
                    "phase": phase,
                    "dp_rank": dp,
                    "tp_rank": tp,
                    **accounting,
                    "kv_cache_bytes": accounting["initial_free_blocks"] * accounting["page_size_bytes"],
                }
            )
        prefix_caching[phase] = _phase_prefix_caching(phase, workers, schedulers)
        # Only vLLM 0.33.0 records hold prefix-caching evidence and the startup concurrency.
        if prefix_caching[phase] is not None:
            for _, scheduler in sorted(schedulers.items()):
                _validate_startup_concurrency(scheduler)
    # The record holds the prefix-caching values of the role's first collection phase that this validation covers.
    record_phase = next(item for item in collection_phases(launch.get("worker_type")) if item in required_phases)
    # Only vLLM 0.33.0 evidence adds these keys. Its records carry prefix caching, which every
    # hybrid record must. The cache geometry comparison across phases includes layer_specs, so every phase
    # agrees on hybrid evidence. The provenance of a vLLM 0.27.0 or 0.28.0 import stays as it was, because
    # finalized campaigns compare it byte for byte.
    hybrid_evidence = (
        {
            "prefix_caching": {item: value for item, value in prefix_caching.items() if value is not None},
            "phase_geometry_differences": geometry_differences,
        }
        if any(value is not None for value in prefix_caching.values())
        else {}
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
        **hybrid_evidence,
        "rank_capacities": capacities,
        "launch_artifacts": launch_artifacts,
        "artifacts": artifacts,
        "runtime_artifacts": runtime_artifacts,
        "instrumentation": {
            "manifest": str(bundle_path.relative_to(root)),
            "files": bundle.files,
            "import_binding": "verified" if all(import_bindings) else "unverified_legacy",
        },
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
            **(prefix_caching[record_phase] or {}),
            "provenance": canonical_json(provenance),
        },
        "provenance": "Validated runtime shared-pool observations; minimum usable capacity across compatible phases "
        "and ranks, including padding and permanent reservations.",
    }
    FpmResourceProfile.model_validate(resources)
    return {"status": "complete", "resources": resources, "diagnostics": [], "provenance": provenance}


def validate_observation_phase(
    index_path: str | Path,
    configuration: str,
    launch: dict[str, Any],
    *,
    phase: str,
    attempt_id: str,
) -> dict[str, Any]:
    """Validate one explicitly bound active phase; does not qualify a serving role.

    The onboarding collection verifier binds this selection to its frozen plan
    and passed native attempt, then requires complete role coverage separately.
    """
    index_path = Path(index_path).absolute()
    if index_path.is_symlink():
        raise ValueError("runtime observation index cannot be a symlink")
    index = read_json(index_path)
    if index.get("schema_version") != INDEX_SCHEMA or not isinstance(index.get("configurations"), dict):
        raise ValueError("unsupported runtime observation index schema")
    result = _configuration(
        index_path.parent,
        configuration,
        index["configurations"][configuration],
        launch,
        phase=phase,
        attempt_id=_text(attempt_id, "selected phase attempt identity"),
    )
    result["provenance"]["observations_index"] = {
        "path": str(index_path),
        "sha256": sha256_bytes(index_path.read_bytes()),
    }
    result["resources"]["runtime_memory"]["provenance"] = canonical_json(result["provenance"])
    return result


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
