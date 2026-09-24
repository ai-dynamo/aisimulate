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


TIMING_BOUNDARIES = {
    "vllm": "vllm_native_scheduler_output_interval",
    "sglang": "sglang_native_forward_device_timer",
}
