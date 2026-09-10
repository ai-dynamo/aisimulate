# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import pytest
from normalize_fpm import prediction_input


@pytest.mark.parametrize("past", [127, 128, 129, 768])
@pytest.mark.parametrize("batch", [1, 2])
def test_both_producers_map_to_same_prediction_workload_at_boundaries(past, batch):
    outputs = []
    for source, raw in [("vllm_past_kv", past * batch), ("sglang_inclusive_query", (past + 1) * batch)]:
        original = {
            "wall_time": 0.1,
            "scheduled_requests": {
                "num_decode_requests": batch,
                "sum_decode_kv_tokens": raw,
                "sum_prefill_tokens": 128,
                "sum_prefill_kv_tokens": 512,
            },
        }
        snapshot = deepcopy(original)
        for axis, expected in [
            ("op_level_inclusive_query", (past + 1) * batch),
            ("whole_forward_past_kv", past * batch),
        ]:
            result, receipt = prediction_input(original, producer_semantics=source, target_axis=axis)
            assert result["scheduled_requests"]["sum_decode_kv_tokens"] == expected
            assert result["scheduled_requests"]["sum_prefill_kv_tokens"] == 512
            assert receipt["canonical_past_kv_sum"] == past * batch
            assert original == snapshot
            outputs.append(result)
    assert outputs[0] == outputs[2]
    assert outputs[1] == outputs[3]


def test_bad_context_cannot_underflow_into_a_plausible_prediction():
    with pytest.raises(ValueError, match="smaller"):
        prediction_input(
            {"scheduled_requests": {"num_decode_requests": 2, "sum_decode_kv_tokens": 1}},
            producer_semantics="sglang_inclusive_query",
            target_axis="whole_forward_past_kv",
        )
