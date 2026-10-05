# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks of the SGLang native-FPM client: request plans and measured-step attribution."""

import pytest

from collector.fpm_forward import sglang_native_fpm as drv

pytestmark = pytest.mark.unit


def message(sequence, *, prefill=(0, 0, 0), decode=(0, 0), wall=0.01):
    return {
        "sequence": sequence,
        "wall_time": wall,
        "scheduled_requests": {
            "num_prefill_requests": prefill[0],
            "sum_prefill_tokens": prefill[1],
            "sum_prefill_kv_tokens": prefill[2],
            "num_decode_requests": decode[0],
            "sum_decode_kv_tokens": decode[1],
        },
    }


def test_decode_attribution_ignores_previous_round_stragglers():
    point = {"phase": "decode", "batch_size": 2, "total_kv_read_tokens": 2 * 122}
    # A straggling trailing step of the previous round (seq_len sum 2*125)
    # arrives first; the measured fourth step has sum K + B.
    messages = [message(1, decode=(2, 250)), message(2, prefill=(2, 4, 2 * 116))]
    messages += [message(3 + j, decode=(2, 2 * (120 + j)), wall=0.001 * (j + 1)) for j in range(6)]
    messages[3:] = [dict(m, wall_time=0.004) for m in messages[3:]]
    measured, reason = drv.select_measured(point, messages)
    assert reason is None and measured["sequence"] == 6
    # A missing first decode emission (folded into the next one) does not
    # matter, but a merged measured interval is rejected.
    assert drv.select_measured(point, messages[:2] + messages[3:])[1] is None
    merged = [dict(m) for m in messages]
    merged[5]["wall_time"] = 0.008
    assert "merged" in drv.select_measured(point, merged)[1]


def test_decode_attribution_rejects_out_of_lockstep_batches():
    point = {"phase": "decode", "batch_size": 2, "total_kv_read_tokens": 2 * 122}
    messages = [message(3 + j, decode=(1 if j == 3 else 2, 2 * (120 + j))) for j in range(6)]
    assert drv.select_measured(point, messages)[0] is None


def test_prefill_attribution_requires_one_exact_batched_forward():
    point = {"phase": "prefill", "batch_size": 4, "total_prefill_tokens": 128, "total_kv_read_tokens": 512}
    ok = [message(1, prefill=(4, 128, 512))]
    assert drv.select_measured(point, ok)[1] is None
    split = [message(1, prefill=(2, 64, 256)), message(2, prefill=(2, 64, 256))]
    assert drv.select_measured(point, split)[0] is None
    miss = [message(1, prefill=(4, 128, 448))]
    assert "differs" in drv.select_measured(point, miss)[1]


def test_request_plans_match_the_clean_truth_construction():
    pool = list(range(1000, 1841))
    prefill = {"phase": "prefill", "benchmark_id": 7, "prefix_tokens": [128, 128], "new_tokens": [32, 32]}
    plan = drv.plan_rounds(prefill, pool)
    assert plan["expected_cached"] == [128, 128] and plan["max_tokens"] == 1
    firsts = {tuple(prompts[0][128:129]) for prompts in plan["rounds"]}
    assert len(firsts) == len(plan["rounds"])  # round-unique first suffix token
    decode = {"phase": "decode", "benchmark_id": 3, "contexts": [200, 199]}
    plan = drv.plan_rounds(decode, pool)
    assert [len(p) for p in plan["rounds"][0]] == [197, 196]
    assert plan["expected_cached"] == [192, 192] and plan["max_tokens"] == 6
