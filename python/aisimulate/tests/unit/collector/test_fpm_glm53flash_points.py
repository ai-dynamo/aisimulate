# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json

import pytest
from collector.fpm_forward.config import _freeze_benchmark_points
from collector.fpm_forward.glm53flash_points import glm_point_manifest, glm_smoke_subset

pytestmark = pytest.mark.unit

# GLM-5.3-Flash at TP 4 as the runtime probe measured it: 2,176-token hash blocks, a 180,218,332,800-byte cache of
# 13,045,120-byte pool blocks and five state groups, with the onboarding limits of 8,192 tokens and 256 sequences.
_TP4 = {
    "hash_block_size": 2176,
    "token_budget": 8192,
    "prefill_batch_limit": 256,
    "decode_batch_limit": 256,
    "measured_context": 131072,
    "kv_cache_bytes": 180_218_332_800,
    "page_size_bytes": 13_045_120,
    "state_groups": 5,
}


def test_tp4_point_list_follows_the_producer_rules(tmp_path):
    payload, canonical, digest = glm_point_manifest(**_TP4)
    prefill, decode = payload["prefill"], payload["decode"]

    assert payload["schema_version"] == 3
    assert (len(prefill), len(decode)) == (496, 96)
    assert sorted({point["total_prefill_tokens"] // point["batch_size"] for point in prefill}) == [
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
        2176,
        4096,
        4352,
        6528,
        8192,
    ]
    assert sorted({point["total_kv_read_tokens"] // point["batch_size"] for point in decode}) == [
        2**exponent for exponent in range(2, 18)
    ]
    assert prefill[0] == {"batch_size": 1, "total_prefill_tokens": 4, "total_kv_read_tokens": 0}
    assert prefill[-1] == {"batch_size": 1, "total_prefill_tokens": 8192, "total_kv_read_tokens": 121856}
    assert decode[0] == {"batch_size": 1, "total_kv_read_tokens": 4}
    assert decode[-1] == {"batch_size": 32, "total_kv_read_tokens": 4194304}
    # Prefill points ascend by per-request new tokens, batch size and prefix; decode points by context and batch size.
    prefill_keys = [
        (point["total_prefill_tokens"] // point["batch_size"], point["batch_size"], point["total_kv_read_tokens"])
        for point in prefill
    ]
    decode_keys = [(point["total_kv_read_tokens"] // point["batch_size"], point["batch_size"]) for point in decode]
    assert prefill_keys == sorted(set(prefill_keys))
    assert decode_keys == sorted(set(decode_keys))
    for point in prefill:
        batch, new, prefix = point["batch_size"], point["total_prefill_tokens"], point["total_kv_read_tokens"]
        assert new % batch == 0 and new // batch % 4 == 0
        assert new <= 8192 and batch <= 32
        assert prefix % (batch * 2176) == 0
        assert (prefix + new) // batch <= 131072
    for point in decode:
        batch, kv = point["batch_size"], point["total_kv_read_tokens"]
        assert kv % batch == 0 and 4 <= kv // batch <= 131072
        assert batch <= 32
    path = tmp_path / "points.json"
    path.write_text(json.dumps(payload))
    assert _freeze_benchmark_points(str(path)) == (canonical, digest)


def test_point_list_is_deterministic():
    _, first, digest = glm_point_manifest(**_TP4)
    _, second, _ = glm_point_manifest(**_TP4)

    assert first == second
    assert digest == hashlib.sha256(first.encode("utf-8")).hexdigest()


def test_smoke_subset_is_four_points_of_the_list(tmp_path):
    payload, _, _ = glm_point_manifest(**_TP4)
    subset, canonical, digest = glm_smoke_subset(payload, 2176)

    assert subset == {
        "schema_version": 3,
        "prefill": [
            {"batch_size": 1, "total_prefill_tokens": 64, "total_kv_read_tokens": 0},
            {"batch_size": 1, "total_prefill_tokens": 64, "total_kv_read_tokens": 2176},
        ],
        "decode": [
            {"batch_size": 1, "total_kv_read_tokens": 4096},
            {"batch_size": 2, "total_kv_read_tokens": 8192},
        ],
    }
    for phase in ("prefill", "decode"):
        assert all(point in payload[phase] for point in subset[phase])
    path = tmp_path / "smoke-points.json"
    path.write_text(json.dumps(subset))
    assert _freeze_benchmark_points(str(path)) == (canonical, digest)
    without = {**payload, "decode": [point for point in payload["decode"] if point != subset["decode"][1]]}
    with pytest.raises(ValueError) as error:
        glm_smoke_subset(without, 2176)
    assert str(error.value) == "GLM-5.3-Flash smoke subset requires the generated list to hold its four points"


def test_capacity_rule_drops_points_beyond_the_accepted_capacity():
    # 99 usable blocks: one request of 131,072 tokens needs 61 attention blocks and one block in each state group.
    payload, _, _ = glm_point_manifest(**{**_TP4, "kv_cache_bytes": 100 * 13_045_120})
    decode = payload["decode"]

    assert len(decode) == 67
    assert {"batch_size": 1, "total_kv_read_tokens": 131072} in decode
    assert {"batch_size": 2, "total_kv_read_tokens": 262144} not in decode


@pytest.mark.parametrize("field,value", [("token_budget", 8190), ("hash_block_size", 2178)])
def test_budget_must_be_a_multiple_of_four(field, value):
    with pytest.raises(ValueError) as error:
        glm_point_manifest(**{**_TP4, field: value})
    assert str(error.value) == "GLM-5.3-Flash collection requires a token budget that is a multiple of 4"
