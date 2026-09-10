# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit verification bridge between the two qualified native KV axes.

This transforms a copy used for prediction only. Retain the original native
telemetry and receipt. Qualify the exact producer source before choosing its
semantics; backend names alone are not enough after a framework upgrade.
"""

from copy import deepcopy

PRODUCERS = {"sglang_inclusive_query", "vllm_past_kv"}
TARGETS = {"op_level_inclusive_query", "whole_forward_past_kv"}


def prediction_input(native, *, producer_semantics, target_axis):
    if producer_semantics not in PRODUCERS or target_axis not in TARGETS:
        raise ValueError("unknown producer semantics or prediction axis")
    result = deepcopy(native)
    scheduled = result["scheduled_requests"]
    count = scheduled["num_decode_requests"]
    raw = scheduled["sum_decode_kv_tokens"]
    if any(type(value) is not int or value < 0 for value in (count, raw)) or (count == 0 and raw != 0):
        raise ValueError("invalid decode count or KV sum")
    past = raw - count if producer_semantics == "sglang_inclusive_query" else raw
    if past < 0:
        raise ValueError("inclusive KV sum is smaller than its decode query count")
    adjusted = past + count if target_axis == "op_level_inclusive_query" else past
    scheduled["sum_decode_kv_tokens"] = adjusted
    return result, {
        "producer_semantics": producer_semantics,
        "target_axis": target_axis,
        "decode_requests": count,
        "native_decode_kv_sum": raw,
        "canonical_past_kv_sum": past,
        "prediction_decode_kv_sum": adjusted,
        "delta": adjusted - raw,
    }
