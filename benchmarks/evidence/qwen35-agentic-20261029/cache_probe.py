#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read resolved vLLM worker geometry and qualify the baseline cache mapping.

Launch vLLM 0.31.0 with this directory on PYTHONPATH,
--worker-extension-cls cache_probe.AlignmentCacheProbe and
VLLM_SERVER_DEV_MODE=1, bound to localhost. This extension only reads metadata.
The null-block convention is checked against vLLM tag v0.31.0,
vllm/v1/core/block_pool.py; no upstream allocator code is copied here.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import urllib.request
from pathlib import Path
from typing import Any

SCHEMA = "qwen35-cache-geometry/v1"
VLLM_REVISION = "db9527a46873454610df6dbedf79a36d6bf1a7f6"


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def serializable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if dataclasses.is_dataclass(value):
        return {
            field.name: serializable(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    return str(value)


def fields(obj: Any, names: str) -> dict[str, Any]:
    return {name: serializable(getattr(obj, name, None)) for name in names.split()}


class AlignmentCacheProbe:
    """vLLM worker extension; invoked after startup through collective_rpc."""

    def alignment_cache_geometry(self) -> dict[str, Any]:
        config = self.vllm_config
        runner = self.model_runner
        cache = runner.kv_cache_config
        groups = []
        for group in cache.kv_cache_groups:
            spec = group.kv_cache_spec
            groups.append(
                {
                    "layer_names": list(group.layer_names),
                    "spec_type": type(spec).__name__,
                    "page_size_bytes": int(spec.page_size_bytes),
                    "spec": serializable(spec),
                }
            )
        return {
            "vllm_version": importlib.metadata.version("vllm"),
            "rank": self.rank,
            "pid": os.getpid(),
            "model": fields(
                config.model_config,
                "model revision dtype max_model_len quantization enforce_eager",
            ),
            "compilation": fields(
                config.compilation_config,
                "mode cudagraph_mode cudagraph_capture_sizes max_cudagraph_capture_size",
            ),
            "parallelism": fields(
                config.parallel_config,
                "tensor_parallel_size pipeline_parallel_size "
                "data_parallel_size prefill_context_parallel_size decode_context_parallel_size "
                "enable_expert_parallel",
            ),
            "scheduler": fields(
                config.scheduler_config,
                "max_num_batched_tokens max_num_seqs "
                "enable_chunked_prefill async_scheduling disable_hybrid_kv_cache_manager",
            ),
            "speculative_config": serializable(config.speculative_config),
            "kv_transfer_config": serializable(config.kv_transfer_config),
            "cache_config": fields(
                config.cache_config,
                "block_size prefix_match_unit cache_dtype "
                "mamba_block_size mamba_cache_mode mamba_cache_dtype mamba_ssm_cache_dtype "
                "enable_prefix_caching prefix_cache_retention_interval use_replayssm "
                "kv_cache_memory_bytes gpu_memory_utilization kv_cache_layout",
            ),
            "num_blocks": int(cache.num_blocks),
            "prefix_cache_retention_interval": cache.prefix_cache_retention_interval,
            "groups": groups,
            "tensors": serializable(cache.kv_cache_tensors),
            "memory": {
                **fields(
                    self,
                    "requested_memory available_kv_cache_memory_bytes total_consumed "
                    "peak_activation_memory cudagraph_memory_estimate",
                ),
                "model_weights_bytes": getattr(runner, "model_memory_usage", None),
                "initial_snapshot": serializable(getattr(self, "init_snapshot", None)),
                "actual_graph_bytes": None,
                "actual_graph_bytes_note": "vLLM logs capture size but does not retain it on this worker; see server.log",
            },
        }


def derive_manual_geometry(worker: dict[str, Any]) -> dict[str, Any]:
    """Accept only the inspected, equal-page, 15:45 baseline; fail on drift."""
    if worker["vllm_version"].split("+")[0] != "0.31.0":
        raise ValueError("cache mapping requires the inspected vLLM 0.31.0 runtime")
    if (
        worker["speculative_config"] is not None
        or worker["kv_transfer_config"] is not None
    ):
        raise ValueError(
            "baseline requires speculation and KV transfer/offload disabled"
        )
    parallel = worker["parallelism"]
    for name in (
        "pipeline_parallel_size",
        "data_parallel_size",
        "prefill_context_parallel_size",
        "decode_context_parallel_size",
    ):
        if parallel[name] != 1:
            raise ValueError(f"baseline requires {name}=1")
    scheduler = worker["scheduler"]
    if (
        scheduler["async_scheduling"] is not False
        or scheduler["enable_chunked_prefill"] is not True
    ):
        raise ValueError(
            "baseline requires synchronous scheduling with chunked prefill"
        )
    if scheduler["disable_hybrid_kv_cache_manager"]:
        raise ValueError("hybrid cache manager must be enabled")
    cache = worker["cache_config"]
    block_size = cache["block_size"]
    if type(block_size) is not int or block_size < 2:
        raise ValueError("resolved physical block size is missing")
    if (
        cache["mamba_cache_mode"] != "align"
        or not cache["enable_prefix_caching"]
        or cache["prefix_match_unit"] not in (None, block_size)
        or cache["use_replayssm"]
    ):
        raise ValueError(
            "baseline requires ordinary align state caching and full-block prefix matching"
        )
    if worker["prefix_cache_retention_interval"] != 0:
        raise ValueError("baseline requires vLLM semantic-checkpoint retention=0")
    groups = worker["groups"]
    if len(groups) != 4 or sorted(group["spec_type"] for group in groups) != [
        "FullAttentionSpec",
        "MambaSpec",
        "MambaSpec",
        "MambaSpec",
    ]:
        raise ValueError("expected one full-attention group and three GDN state groups")
    if any(len(group["layer_names"]) != 15 for group in groups):
        raise ValueError("expected four groups of 15 layers without grouping padding")
    names = [name for group in groups for name in group["layer_names"]]
    if len(set(names)) != 60:
        raise ValueError("cache groups must partition the 60 target layers")
    pages = {group["page_size_bytes"] for group in groups}
    if len(pages) != 1 or next(iter(pages)) <= 0:
        raise ValueError("all cache groups must share one positive physical page size")
    for group in groups:
        spec = group["spec"]
        if spec["block_size"] != block_size:
            raise ValueError("token and state groups must use the same block size")
        if group["spec_type"] == "MambaSpec":
            if (
                spec["mamba_cache_mode"] != "align"
                or spec.get("num_speculative_blocks", 0) != 0
                or spec.get("num_prefill_checkpoint_blocks", 0) != 0
            ):
                raise ValueError("unexpected state scratch/checkpoint allocation")
    state_specs = [
        group["spec"] for group in groups if group["spec_type"] == "MambaSpec"
    ]
    if any(spec != state_specs[0] for spec in state_specs[1:]):
        raise ValueError("all three state groups must have identical specs")
    pool_block_bytes = next(iter(pages)) * 15
    blocks = worker["num_blocks"]
    if type(blocks) is not int or blocks < 6 or pool_block_bytes % block_size:
        raise ValueError("invalid shared block pool geometry")
    # v0.31 tensor descriptors alias one backing allocation across groups.
    # Each descriptor's size is the complete backing size, not an additive size.
    tensors = worker["tensors"]
    if len(tensors) != 4:
        raise ValueError("expected one aliased tensor descriptor per cache group")
    for group, tensor in zip(groups, tensors):
        if (
            tensor["size"] != blocks * pool_block_bytes
            or tensor["offset"] != 0
            or tensor.get("host_resident", False)
            or set(tensor["layers"]) != set(group["layer_names"])
        ):
            raise ValueError("tensor allocation disagrees with the shared-pool mapping")
        end = (
            14 * tensor["layer_stride"]
            + (blocks - 1) * tensor["block_stride"]
            + group["page_size_bytes"]
        )
        if end != tensor["size"]:
            raise ValueError("unexpected cache tensor padding or strides")
    return {
        "block_size": block_size,
        "prefix_match_unit": block_size,
        "bytes_per_token": pool_block_bytes // block_size,
        "prefix_caching": True,
        "capacity": {"type": "fixed", "blocks": blocks - 1},
        "state_cache": {"bytes_per_request": 3 * pool_block_bytes},
    }


def make_receipt(workers: list[dict[str, Any]]) -> dict[str, Any]:
    if not workers:
        raise ValueError("probe returned no workers")
    mapping = derive_manual_geometry(workers[0])
    tensor_parallel = workers[0]["parallelism"]["tensor_parallel_size"]
    if (
        len(workers) != tensor_parallel
        or len({worker["rank"] for worker in workers}) != tensor_parallel
    ):
        raise ValueError("probe must contain every tensor-parallel rank exactly once")
    if any(derive_manual_geometry(worker) != mapping for worker in workers[1:]):
        raise ValueError(
            "rank cache geometry differs; a uniform AISim worker cannot represent it"
        )
    for key in ("parallelism", "scheduler", "cache_config", "model"):
        if any(worker[key] != workers[0][key] for worker in workers[1:]):
            raise ValueError(f"rank configuration differs: {key}")
    receipt = {
        "schema": SCHEMA,
        "vllm_source_revision": VLLM_REVISION,
        "workers": workers,
        "manual_kv_cache": mapping,
        "reserved_null_blocks": 1,
        "qualification": "geometry_only; requires observed-versus-engine cache gate",
    }
    receipt["geometry_id"] = digest({"workers": workers, "manual_kv_cache": mapping})
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8800")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/collective_rpc",
        data=json.dumps({"method": "alignment_cache_geometry", "timeout": 60}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=70) as response:
        workers = json.load(response)["results"]
    # Preserve the raw observation even when a new runtime layout fails validation.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".raw.json").write_text(
        json.dumps({"workers": workers}, indent=2) + "\n"
    )
    receipt = make_receipt(workers)
    with args.output.open("x") as output:
        json.dump(receipt, output, indent=2)
        output.write("\n")
    print(
        json.dumps(
            {
                "geometry_id": receipt["geometry_id"],
                "manual_kv_cache": receipt["manual_kv_cache"],
            }
        )
    )


if __name__ == "__main__":
    main()
