# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inspect the initialized native V2 dispatch policy before real requests.

Original inspection/validation code. Native contract: vllm-project/vllm at
ced6857afa0ea7b2e3f0846a62e1394e90f15607, v1/worker/gpu/cudagraph_utils.py
(_init_candidates, dispatch) and compilation/breakable_cudagraph.py. No native
model, state, capture or dispatch implementation is replaced. This inventory
alone provides neither measured graph operations nor performance admission.
"""

from __future__ import annotations

import bisect
import dataclasses
import hashlib
import re
from importlib.metadata import version
from pathlib import Path

from collector.glm53flash_contract import BACKENDS, CHECKPOINTS, canonical_json, sha256_json
from collector.glm53flash_runtime_identity import validate_backend_version

SOURCE_PINS = {
    "v1/worker/gpu/cudagraph_utils.py": "6e9c042890603535e300a40df8ee159dbed1058a64a83ae50ff0329e332e05ff",
    "v1/worker/gpu/model_runner.py": "174c93db921c23cf0396eee4764be25b2bd2d4b6a06e9fa41ce3598b884ce8ce",
    "config/compilation.py": "c9cec5c7200e8e559810ec8c30113ad61dab6780f9f7fb3c116bd0d9b5a43065",
    "compilation/breakable_cudagraph.py": "3cc427612a08e2b9b3fee47548026400c1d0776e2d4747535e59ef5512bdf1e8",
}
FLAGS = (
    "compiled_model",
    "varlen_decode",
    "microbatch_runner",
    "speculative",
    "lora",
    "encoder_decoder",
    "async_scheduling",
    "expert_parallel",
    "prefix_caching",
    "kda_recoverssm",
)


def descriptor(value):
    """Serialize the original native dataclass, preserving every dispatch axis."""
    if not dataclasses.is_dataclass(value):
        raise ValueError("native graph descriptor must be its actual dataclass")
    result = dataclasses.asdict(value)
    result["cg_mode"] = value.cg_mode.name
    return result


def _expected(mode, tokens):
    return {
        "cg_mode": mode,
        "num_tokens": tokens,
        "num_reqs": tokens if mode == "FULL" else None,
        "uniform_token_count": 1 if mode == "FULL" else None,
        "max_query_len": None,
        "num_active_loras": 0,
        "num_ubatches": 1,
    }


def _ceil(sizes, count):
    index = bisect.bisect_left(sizes, count)
    return sizes[index] if index < len(sizes) else None


def validate_snapshot(value):
    if value.get("backend") != "vllm" or value.get("source_pins") != SOURCE_PINS:
        raise ValueError("unqualified native V2 dispatch source")
    validate_backend_version("vllm", value.get("backend_version"))
    if value.get("backend_revision") != BACKENDS["vllm"][1]:
        raise ValueError("unqualified native V2 dispatch revision")
    flags = value.get("native_flags", {})
    if set(flags) != set(FLAGS) or any(flag is not False for flag in flags.values()):
        raise ValueError("native V2 dispatch predicates exceed the ordinary homogeneous profile")
    sizes = value.get("capture_sizes")
    if (
        not isinstance(sizes, list)
        or not sizes
        or any(type(size) is not int or size < 1 for size in sizes)
        or sizes != sorted(set(sizes))
        or type(value.get("max_num_reqs")) is not int
        or value["max_num_reqs"] < 1
        or type(value.get("max_capture_tokens")) is not int
        or value["max_capture_tokens"] != sizes[-1]
        or type(value.get("decode_query_len")) is not int
        or value["decode_query_len"] != 1
        or value.get("graphs_captured") is not True
        or value.get("lora_capture_cases") != [0]
        or any(type(item) is not int for item in value.get("lora_capture_cases", []))
        or value.get("dp_size") != 1
        or type(value.get("dp_size")) is not int
        or value.get("tp_size") not in (2, 4)
        or type(value.get("tp_size")) is not int
        or type(value.get("tp_rank")) is not int
        or not 0 <= value["tp_rank"] < value["tp_size"]
    ):
        raise ValueError("native V2 capture geometry or topology is incomplete")
    mode = value.get("resolved_mode")
    if mode not in ("FULL_AND_PIECEWISE", "FULL_DECODE_ONLY") or value.get("use_breakable_cg") is not (
        mode == "FULL_AND_PIECEWISE"
    ):
        raise ValueError("native V2 execution policy is neither reviewed FULL nor native breakable PIECEWISE")
    full = [size for size in sizes if size <= value["max_num_reqs"]]
    if not full:
        raise ValueError("native V2 capture lacks ordinary decode descriptors")
    pw = sizes if mode == "FULL_AND_PIECEWISE" else []
    expected = {"FULL": [_expected("FULL", size) for size in reversed(full)]}
    if pw:
        expected["PIECEWISE"] = [_expected("PIECEWISE", size) for size in reversed(pw)]
    if canonical_json(value.get("capture_descriptors")) != canonical_json(expected) or canonical_json(
        value.get("full_graphs")
    ) != canonical_json(list(reversed(expected["FULL"]))):
        raise ValueError("native V2 initialized captures differ from actual configured descriptors")
    candidates = []
    for tokens in range(sizes[-1] + 1):
        options = []
        for kind, buckets in (("FULL", full), ("PIECEWISE", pw)):
            padded = _ceil(buckets, tokens)
            if padded is not None:
                options.append(_expected(kind, padded))
        if options:
            candidates.append({"num_tokens": tokens, "num_active_loras": 0, "descriptors": options})
    if canonical_json(value.get("candidates")) != canonical_json(candidates):
        raise ValueError("native V2 priority candidates differ from source/config-derived capture ranges")
    entries = value.get("piecewise_entries", [])
    if len(entries) != len(pw) or [entry.get("num_tokens") for entry in entries] != pw:
        raise ValueError("native PIECEWISE initialized entry coverage is incomplete")
    for entry in entries:
        if (
            set(entry)
            != {
                "num_tokens",
                "num_reqs",
                "uniform",
                "has_lora",
                "num_active_loras",
                "completed",
                "num_graphs",
                "num_eager_breaks",
            }
            or entry.get("num_reqs") is not None
            or entry.get("uniform") is not False
            or entry.get("has_lora") is not False
            or type(entry.get("num_active_loras")) is not int
            or entry["num_active_loras"] != 0
            or type(entry.get("num_graphs")) is not int
            or entry["num_graphs"] < 1
            or type(entry.get("num_eager_breaks")) is not int
            or entry["num_eager_breaks"] < 0
            or entry.get("completed") is not True
        ):
            raise ValueError("native PIECEWISE entry lacks completed segment inventory")
    return value


def select_descriptor(snapshot, *, batch, query, is_context):
    """Select from pre-request native capture regions, never holdout answers.

    PIECEWISE selection is an execution-identity result only; no measured unit
    or table is implied. A downstream consumer must reject missing coverage.
    """
    validate_snapshot(snapshot)
    if (
        type(batch) is not int
        or not 1 <= batch <= snapshot["max_num_reqs"]
        or type(query) is not int
        or query < 1
        or type(is_context) is not bool
        or (not is_context and query != 1)
    ):
        raise ValueError("native V2 policy requires ordinary homogeneous real query geometry")
    tokens = batch * query
    if not is_context:
        full = [row["num_tokens"] for row in snapshot["full_graphs"]]
        padded = _ceil(full, tokens)
        if padded is not None:
            return _expected("FULL", padded)
    if snapshot["use_breakable_cg"]:
        padded = _ceil(snapshot["capture_sizes"], tokens)
        if padded is not None:
            return _expected("PIECEWISE", padded)
    return {**_expected("NONE", tokens), "num_reqs": batch}


def snapshot_native(manager, model, tp_rank):
    """Read the actual initialized manager once, outside all measured windows."""
    import vllm
    from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager, has_compiled_submodule

    if type(manager) is not ModelCudaGraphManager:
        raise ValueError("policy snapshot requires the actual native V2 manager")
    config = manager.vllm_config
    package = Path(vllm.__file__).resolve().parent
    entries = []
    if manager.breakable_cg_runner is not None:
        for key, entry in manager.breakable_cg_runner.entries.items():
            capture = entry.capture
            entries.append(
                {
                    **dataclasses.asdict(key),
                    "completed": capture is not None and not capture._capturing,
                    "num_graphs": capture.num_graphs if capture is not None else None,
                    "num_eager_breaks": capture.num_eager_breaks if capture is not None else None,
                }
            )
    value = {
        "backend": "vllm",
        "backend_version": version("vllm"),
        "backend_revision": BACKENDS["vllm"][1],
        "source_pins": {name: hashlib.sha256((package / name).read_bytes()).hexdigest() for name in SOURCE_PINS},
        "native_flags": {
            "compiled_model": has_compiled_submodule(model),
            "varlen_decode": manager.varlen_decode,
            "microbatch_runner": manager.ubatch_runner is not None,
            "speculative": config.speculative_config is not None,
            "lora": config.lora_config is not None,
            "encoder_decoder": config.model_config.is_encoder_decoder,
            "async_scheduling": config.scheduler_config.async_scheduling,
            "expert_parallel": config.parallel_config.enable_expert_parallel,
            "prefix_caching": config.cache_config.enable_prefix_caching,
            "kda_recoverssm": config.cache_config.use_kda_recoverssm,
        },
        "capture_sizes": sorted(manager.compilation_config.cudagraph_capture_sizes),
        "max_num_reqs": manager.max_num_reqs,
        "max_capture_tokens": manager.compilation_config.max_cudagraph_capture_size,
        "decode_query_len": manager.decode_query_len,
        "graphs_captured": manager._graphs_captured,
        "lora_capture_cases": manager.lora_capture_cases,
        "dp_size": manager.dp_size,
        "tp_size": manager.tp_size,
        "tp_rank": tp_rank,
        "resolved_mode": manager.cudagraph_mode.name,
        "use_breakable_cg": manager.use_breakable_cg,
        "capture_descriptors": {
            kind.name: list(map(descriptor, rows)) for kind, rows in manager._capture_descs.items()
        },
        "full_graphs": sorted(map(descriptor, manager.graphs), key=lambda row: row["num_tokens"]),
        "candidates": [
            {"num_tokens": tokens, "num_active_loras": loras, "descriptors": list(map(descriptor, rows))}
            for (tokens, loras), rows in sorted(manager._candidates.items())
        ],
        "piecewise_entries": sorted(entries, key=lambda row: row["num_tokens"]),
    }
    return validate_snapshot(value)


def persist_snapshot(manager, model, rank, output):
    value = snapshot_native(manager, model, rank)
    path = Path(output) / f"vllm-graph-policy-rank-{rank}.json"
    encoded = canonical_json(value).encode()
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError("native V2 capture policy changed after its immutable snapshot")
    else:
        with path.open("xb") as stream:
            stream.write(encoded)
    return value


def full_policy_fields(snapshot):
    """Reduce a validated native inventory to its ordinary FULL query region.

    PIECEWISE buckets outside max_num_reqs are never decode FULL coverage.
    The complete original inventory remains in its hashed native receipt.
    """
    validate_snapshot(snapshot)
    return {
        "backend": "vllm",
        "backend_version": snapshot["backend_version"],
        "backend_revision": snapshot["backend_revision"],
        "capture_sizes": [row["num_tokens"] for row in snapshot["full_graphs"]],
        "disable_padding": False,
        "captured_req_width": 1,
        "native_flags": snapshot["native_flags"],
        "source_pins": snapshot["source_pins"],
    }


def build_full_policy(
    snapshots,
    *,
    checkpoint_format,
    tp_size,
    provenance,
    resolved_config_sha256,
    state_layout_sha256,
    capture_registry_sha256,
):
    from collector.glm53flash_runtime_identity import vllm_source_pins

    if checkpoint_format not in CHECKPOINTS or type(tp_size) is not int or tp_size not in (2, 4):
        raise ValueError("native V2 graph policy has an unqualified checkpoint/topology")
    if set(snapshots) != set(range(tp_size)):
        raise ValueError("native V2 graph policy requires every actual TP worker")
    common = None
    for rank, value in snapshots.items():
        validate_snapshot(value)
        if value["tp_rank"] != rank or value["tp_size"] != tp_size:
            raise ValueError("native V2 graph worker belongs to another topology")
        candidate = {key: item for key, item in value.items() if key != "tp_rank"}
        if common is not None and common != candidate:
            raise ValueError("native V2 graph workers disagree on their initialized policy")
        common = candidate
    first = snapshots[0]
    manifest = Path(__file__).parent / "fpm_forward/runtime/glm53flash/runtime-source-sha256.json"
    pins = vllm_source_pins(first["backend_version"], manifest)
    if (
        provenance.get("backend") != "vllm"
        or provenance.get("backend_version") != first["backend_version"]
        or provenance.get("backend_revision") != first["backend_revision"]
        or provenance.get("checkpoint_revision") != CHECKPOINTS[checkpoint_format][1]
        or provenance.get("source_sha256") != sha256_json(pins)
        or not re.fullmatch(r"[0-9a-f]{64}", provenance.get("config_sha256", ""))
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", provenance.get("runtime_digest", ""))
        or not re.fullmatch(r"[0-9a-f]{64}", resolved_config_sha256)
    ):
        raise ValueError("native V2 graph policy differs from exact measured runtime/source/config")
    fields = full_policy_fields(first)
    for values in (state_layout_sha256, capture_registry_sha256):
        if set(values) != set(range(tp_size)):
            raise ValueError("native V2 graph policy omits worker state/capture evidence")
    for rank in range(tp_size):
        hashes = [state_layout_sha256[rank], *capture_registry_sha256[rank]]
        if len(capture_registry_sha256[rank]) != len(fields["capture_sizes"]) or any(
            not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes
        ):
            raise ValueError("native V2 graph policy lacks exact FULL capture/state hashes")
    return {
        "schema_version": 2,
        **fields,
        "checkpoint_format": checkpoint_format,
        "checkpoint_revision": CHECKPOINTS[checkpoint_format][1],
        "tp_size": tp_size,
        "phase": "generation",
        "runtime_mode": "FULL",
        "source_sha256": provenance["source_sha256"],
        "config_sha256": provenance["config_sha256"],
        "runtime_digest": provenance["runtime_digest"],
        "resolved_config_sha256": resolved_config_sha256,
        "native_policy_receipt_sha256": sha256_json({str(rank): value for rank, value in sorted(snapshots.items())}),
        "state_layout_sha256": {str(rank): value for rank, value in sorted(state_layout_sha256.items())},
        "capture_registry_sha256": {str(rank): value for rank, value in sorted(capture_registry_sha256.items())},
    }
