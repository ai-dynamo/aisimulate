# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright 2023-2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
# Independent synthetic contract checks for the pinned SGLang source in README.md.

"""Validate the comparison against hand-derived scheduler boundaries."""

import argparse
import json
import math
from pathlib import Path


def verify(base, head):
    assert base["pd_admission"] == {"reserved": [True] * 4, "free_tokens": 96}
    # Each held prompt consumes 1000 tokens and protects a 512-token decode reserve.
    admitted = min(4, 4096 // (1000 + 512))
    assert head["pd_admission"] == {
        "reserved": [i < admitted for i in range(4)],
        "free_tokens": 4096 - admitted * 1000,
    }
    input_limit = min(1008 - 1 - 5, 1008 - 16)
    for case in head["boundaries"]:
        rejected = case["prompt_tokens"] >= input_limit
        assert case["outcome"]["rejected"] == ([1] if rejected else [])
        assert len(case["outcome"]["admissions"]) == int(not rejected)
        assert case["outcome"]["end_ms"] == (0 if rejected else 10)
    assert [case["outcome"]["drained"] for case in base["boundaries"]] == [True, True, False]
    assert all(not case["outcome"]["rejected"] for case in base["boundaries"])

    clipped_outputs = min(
        100,
        (1008 - 1) - 900 - 1,
        1008 - math.ceil(900 / 16) * 16 - 16 - 1,
    )
    assert head["generation_clip"]["outputs"] == {"1": clipped_outputs}
    assert head["generation_clip"]["end_ms"] == 10 + (clipped_outputs - 1) * 2
    assert base["generation_clip"]["passes"] == 128
    assert not base["generation_clip"]["admissions"]
    assert not base["generation_clip"]["drained"]

    for case in head["completions"]:
        outputs = case["requested_outputs"]
        materialized = case["prompt_tokens"] + outputs - 1
        cached = materialized // 16 * 16
        assert case["cached_tokens"] == cached
        assert case["follower"]["admissions"] == [{"id": 2, "reused_tokens": cached}]
        assert case["outcome"]["outputs"] == {"1": outputs}
        burst = 3 if case["speculative"] else 1
        forwards = math.ceil((outputs - 1) / burst)
        observed_tail = int(case["overlap"] and case["speculative"])
        assert case["outcome"]["end_ms"] == 10 + 2 * (forwards + observed_tail)
    for before, after in zip(base["completions"], head["completions"], strict=True):
        assert before["outcome"]["outputs"] == after["outcome"]["outputs"]
        assert before["cached_tokens"] >= after["cached_tokens"]
        assert before["outcome"]["end_ms"] - after["outcome"]["end_ms"] == (
            2 if after["overlap"] and not after["speculative"] else 0
        )

    assert base["default_decode_cache"]["cached_tokens"] == 112
    assert head["default_decode_cache"]["cached_tokens"] == 0
    assert head["default_decode_cache"]["outcome"]["free_tokens"] == 2048
    assert base["queue_order"]["terminal_order"] == [3, 2]
    assert head["queue_order"]["terminal_order"] == [2, 3]
    assert head["queue_order"]["completion_ms"] == {"2": 70, "3": 80}
    assert head["ratio_lifecycle"] == {"blocked": [0.3] * 3, "drained": 0.7}
    decay = (0.7 - 0.7 * 0.14) / 600
    for index, ratio in enumerate(base["ratio_lifecycle"]["blocked"], 1):
        assert math.isclose(ratio, 0.3 - index * decay)
    assert math.isclose(base["ratio_lifecycle"]["drained"], 0.3 - 4 * decay)

    outcomes = [case["outcome"] for case in head["boundaries"]]
    outcomes += [head["generation_clip"], head["default_decode_cache"]["outcome"], head["queue_order"]]
    outcomes += [case[key] for case in head["completions"] for key in ("outcome", "follower")]
    for outcome in outcomes:
        assert outcome["drained"]
        assert outcome["terminals"]
        assert outcome["outputs"].keys() == outcome["terminals"].keys() == outcome["completion_ms"].keys()
        assert all(count == 1 for count in outcome["terminals"].values())
        assert len(outcome["terminal_order"]) == len(outcome["terminals"])
        assert set(outcome["terminal_order"]) == {int(key) for key in outcome["terminals"]}
        assert outcome["pressure_events"] == 0
        assert math.isfinite(outcome["end_ms"]) and outcome["end_ms"] >= 0
        assert math.isfinite(outcome["ratio"]) and 0 <= outcome["ratio"] <= 1
        assert all(0 <= time <= outcome["end_ms"] for time in outcome["completion_ms"].values())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path(__file__).with_name("base.json"))
    parser.add_argument("--head", type=Path, default=Path(__file__).with_name("head.json"))
    args = parser.parse_args()
    verify(json.loads(args.base.read_text()), json.loads(args.head.read_text()))
    print("Semantic boundaries and anomaly checks passed.")
