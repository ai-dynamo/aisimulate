# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixed workloads: never resize a case during a base/head comparison."""

from copy import deepcopy

VERSIONS = {"vllm": "0.24.0", "sglang": "0.5.14", "trtllm": "1.3.0rc20"}
AGENTX_SHA256 = "d65b573413396bb689cf7e1d5c85c50ea970ad7ad5714c0a78d4f75102e5d86d"

# Calibrated once on the qualification host; never adjusted during comparison.
WORKLOAD_COUNTS = {
    "dense-vllm": 131072,
    "dense-sglang": 16384,
    "dense-trtllm": 131072,
    "moe-long-prefill": 16384,
    "moe-long-decode": 4096,
    "cache-pressure-vllm": 4096,
    "cache-pressure-sglang": 2048,
    "mla-multiworker-dp": 32768,
    "pd-vllm": 65536,
    "pd-sglang": 8192,
}


def worker(backend: str, *, tp: int = 4, ep: int = 1, dp: int = 1, replicas: int = 1) -> dict:
    block_size = {"vllm": 64, "sglang": 1, "trtllm": 32}[backend]
    return {
        "parallelism": {
            "replicas": replicas,
            "tensor": tp,
            "pipeline": 1,
            "attention_data": dp,
            "moe_tensor": 1,
            "moe_expert": ep,
        },
        "scheduler": {"max_batched_tokens": 8192, "max_sequences": 256},
        "kv_cache": {
            "block_size": block_size,
            "prefix_caching": True,
            "capacity": {"type": "fixed", "blocks": 1_048_576 // block_size},
        },
        "timing": {"type": "default"},
    }


def case(
    case_id: str,
    backend: str,
    *,
    model: str = "Qwen/Qwen3-32B",
    tp: int = 4,
    ep: int = 1,
    dp: int = 1,
    replicas: int = 1,
    mode: str = "aggregated",
    isl: int = 1024,
    osl: int = 128,
    concurrency: int = 64,
    requests: int = 256,
) -> dict:
    role = worker(backend, tp=tp, ep=ep, dp=dp, replicas=replicas)
    roles = {"aggregated": role}
    if mode == "disaggregated":
        roles = {"prefill": deepcopy(role), "decode": deepcopy(role)}
        roles["prefill"]["parallelism"]["replicas"] = 1
        roles["prefill"]["scheduler"]["max_sequences"] = 16
        roles["decode"]["parallelism"]["replicas"] = 2
    engine = {
        "mode": mode,
        "backend": backend,
        "backend_version": VERSIONS[backend],
        "model": model,
        "hardware": "b200_sxm",
        "context_length": "max",
        "estimation_mode": "op_level",
        "fallback_policy": "deny",
        "database_mode": "SILICON",
        "enable_shared_layer": True,
        "workers": roles,
    }
    if mode == "disaggregated":
        engine["kv_transfer"] = {"bandwidth_gb_per_second": 400.0, "timing_mode": "destination_missing"}
    return {
        "case_id": case_id,
        "determinism": "canonical_v1",
        "config": {
            "engine": engine,
            "traffic": {
                "source": {"type": "synthetic", "input_tokens": isl, "output_tokens": osl},
                "load": {"type": "concurrency", "concurrency": concurrency},
                "stop": {"requests": requests},
            },
        },
        "expected_requests": requests,
        "expected_output_tokens": requests * osl,
    }


def expand_cases() -> list[dict]:
    result = [case(f"dense-{backend}", backend) for backend in VERSIONS]
    result += [
        case("moe-long-prefill", "vllm", model="Qwen/Qwen3-235B-A22B", tp=8, ep=8, isl=32768, osl=32, concurrency=16),
        case("moe-long-decode", "sglang", model="Qwen/Qwen3-235B-A22B", tp=8, ep=8, osl=2048, concurrency=128),
    ]
    for backend in ("vllm", "sglang"):
        item = case(f"cache-pressure-{backend}", backend)
        item["config"]["traffic"] = {
            "source": {
                "type": "synthetic-session",
                "new_input_tokens_per_turn": 1024,
                "output_tokens_per_turn": 128,
                "session": {"turns": 4, "shared_prefix_ratio": 0.5, "prefix_groups": 16, "inter_turn_delay_ms": 0.0},
            },
            "load": {"type": "concurrency", "concurrency": 16},
            "stop": {"sessions": 64},
        }
        cache = item["config"]["engine"]["workers"]["aggregated"]["kv_cache"]
        cache["capacity"]["blocks"] = 32768 // cache["block_size"]
        item["require_cache_pressure"] = True
        result.append(item)
    result.append(
        case(
            "mla-multiworker-dp",
            "vllm",
            model="deepseek-ai/DeepSeek-V3.2",
            tp=1,
            ep=8,
            dp=8,
            replicas=2,
            concurrency=128,
        )
    )
    result += [case(f"pd-{backend}", backend, mode="disaggregated") for backend in ("vllm", "sglang")]
    for backend, mode in (("vllm", "aggregated"), ("sglang", "disaggregated")):
        item = case(f"agentx-{backend}-{mode}", backend, mode=mode, model="Qwen/Qwen3.5-397B-A17B", tp=8, ep=8)
        item["config"]["traffic"] = {
            "source": {
                "type": "trace",
                "format": "weka",
                "paths": ["fixtures/agentx.jsonl"],
                "nested_timestamp_basis": "absolute",
            },
            "load": {"type": "trace_timestamps", "speedup": 1.0, "agentic_lanes": 1},
        }
        item.update(trace_sha256=AGENTX_SHA256, expected_requests=129, expected_output_tokens=114540)
        result.append(item)
    for item in result:
        if item["case_id"] not in WORKLOAD_COUNTS:
            continue
        traffic = item["config"]["traffic"]
        sessions = traffic["source"]["type"] == "synthetic-session"
        count = WORKLOAD_COUNTS[item["case_id"]]
        traffic["stop"]["sessions" if sessions else "requests"] = count
        item["expected_requests"] = count * (4 if sessions else 1)
        item["expected_output_tokens"] = (
            item["expected_requests"] * traffic["source"]["output_tokens_per_turn" if sessions else "output_tokens"]
        )
    return result
