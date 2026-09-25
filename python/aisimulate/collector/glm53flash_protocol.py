# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared real-request protocol names; mathematical geometry lives in the SDK."""

PROTOCOL = "glm53flash_same_request_real_hybrid_v1"
MAX_MEASURED_CONTEXT = 131072
# SGLang 0.5.20 computes max_req_input_len = min(context - 1, capacity - 1) - 5,
# then rejects input_length >= max_req_input_len. These seven internal slots
# admit an inclusive measured boundary; actual allocator capacity is separate.
SGLANG_CONTEXT_HEADROOM = 7


def sglang_runtime_context_length(measured_limit: int) -> int:
    if type(measured_limit) is not int or not 1 <= measured_limit <= MAX_MEASURED_CONTEXT:
        raise ValueError("GLM measured context limit must be between 1 and 131072")
    return measured_limit + SGLANG_CONTEXT_HEADROOM


# vLLM requests reserve output positions beyond the final measured forward.
# Match the explicitly bounded internal context used by the SG campaign.
VLLM_CONTEXT_HEADROOM = 7
VLLM_CONTEXT_POLICY_VERSION = 1


def vllm_context_policy(measured_limit: int) -> dict[str, int]:
    if measured_limit == -1 and type(measured_limit) is int:
        measured_limit = MAX_MEASURED_CONTEXT
    if type(measured_limit) is not int or not 1 <= measured_limit <= MAX_MEASURED_CONTEXT:
        raise ValueError("GLM measured context limit must be between 1 and 131072")
    return {
        "measured_context_limit": measured_limit,
        "runtime_context_length": measured_limit + VLLM_CONTEXT_HEADROOM,
        "native_admission_headroom": VLLM_CONTEXT_HEADROOM,
    }


TIMING_BOUNDARIES = {
    "vllm": "vllm_native_scheduler_output_interval",
    "sglang": "sglang_native_forward_device_timer",
}


def native_gpu_identity(torch_module) -> dict:
    """Read this worker's selected CUDA device, without inference from flags."""
    index = torch_module.cuda.current_device()
    properties = torch_module.cuda.get_device_properties(index)
    identity = {
        "schema": "glm53flash_gpu_identity_v1",
        "name": torch_module.cuda.get_device_name(index),
        "compute_capability": list(torch_module.cuda.get_device_capability(index)),
        "total_memory_bytes": properties.total_memory,
        "cuda_device_index": index,
    }
    uuid = getattr(properties, "uuid", None)
    if uuid is not None:
        identity["uuid"] = str(uuid)
    return identity


def validate_gb300_identity(identity: dict) -> None:
    """Admit actual GB300/sm103 receipts; caller separately binds TP rank."""
    import re

    if not isinstance(identity, dict) or identity.get("schema") != "glm53flash_gpu_identity_v1":
        raise ValueError("native GPU identity receipt is missing or unknown")
    name = identity.get("name")
    capability = identity.get("compute_capability")
    if (
        not isinstance(name, str)
        or re.search(r"\bGB300\b", name, re.IGNORECASE) is None
        or not isinstance(capability, list)
        or len(capability) != 2
        or any(type(value) is not int for value in capability)
        or capability != [10, 3]
        or type(identity.get("total_memory_bytes")) is not int
        or identity["total_memory_bytes"] <= 0
        or type(identity.get("cuda_device_index")) is not int
        or identity["cuda_device_index"] < 0
        or ("uuid" in identity and (not isinstance(identity["uuid"], str) or not identity["uuid"].strip()))
    ):
        raise ValueError("native GPU identity does not attest GB300 with sm103 and valid device properties")
