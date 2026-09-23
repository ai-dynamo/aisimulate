# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared real-request protocol names; mathematical geometry lives in the SDK."""

PROTOCOL = "glm53flash_same_request_real_hybrid_v1"
TIMING_BOUNDARIES = {
    "vllm": "vllm_native_scheduler_output_interval",
    "sglang": "sglang_native_forward_device_timer",
}
