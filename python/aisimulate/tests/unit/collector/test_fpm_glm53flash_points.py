# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import logging
from types import SimpleNamespace

import pytest
from collector.fpm_forward.config import _freeze_benchmark_points
from collector.fpm_forward.glm53flash_points import (
    CHUNK_ALIGN,
    MAX_BATCH,
    MAX_CONTEXT,
    MAX_NEW,
    glm_measured_context,
    glm_point_manifest,
    glm_smoke_subset,
)

from .test_fpm_glm53flash_prefix_seed import load_producer

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
_GENERATOR_LOGGER = "collector.fpm_forward.glm53flash_points"


def _logged_lines(caplog) -> list[tuple[int, str]]:
    """The level and the text of each line that the point generator logged."""
    return [(record.levelno, record.getMessage()) for record in caplog.records if record.name == _GENERATOR_LOGGER]


def test_tp4_point_list_follows_the_producer_rules(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=_GENERATOR_LOGGER)
    payload, canonical, digest = glm_point_manifest(**_TP4)
    prefill, decode = payload["prefill"], payload["decode"]

    assert payload["schema_version"] == 3
    assert (len(prefill), len(decode)) == (496, 96)
    # The count is logged when the cache holds every candidate, too.
    assert _logged_lines(caplog) == [
        (
            logging.INFO,
            "GLM-5.3-Flash points: dropped 0/496 prefill and 0/96 decode cases (cache capacity, usable_blocks=13814)",
        )
    ]
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
    # The live smoke of an aggregated TP4 request stages these bytes, so their digest stays pinned.
    assert digest == "60e85d447ea36458a4d6a7f584d6709e7c02f7b1c1f306c4af18f353c93d14ce"
    assert digest == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    without = {**payload, "decode": [point for point in payload["decode"] if point != subset["decode"][1]]}
    with pytest.raises(ValueError) as error:
        glm_smoke_subset(without, 2176)
    assert str(error.value) == "GLM-5.3-Flash smoke subset requires the generated list to hold its four points"


_PREFILL_PAIR = [
    {"batch_size": 1, "total_prefill_tokens": 64, "total_kv_read_tokens": 0},
    {"batch_size": 1, "total_prefill_tokens": 64, "total_kv_read_tokens": 2176},
]
_TWO_REQUEST_DECODE_PAIR = [
    {"batch_size": 1, "total_kv_read_tokens": 4096},
    {"batch_size": 2, "total_kv_read_tokens": 8192},
]
_FOUR_POINTS = "GLM-5.3-Flash smoke subset requires the generated list to hold its four points"


def test_smoke_subset_keeps_the_points_of_the_collected_phases_only():
    payload, _, _ = glm_point_manifest(**_TP4)

    for phases, expected in (
        (("prefill", "decode"), {"prefill": _PREFILL_PAIR, "decode": _TWO_REQUEST_DECODE_PAIR}),
        (("prefill",), {"prefill": _PREFILL_PAIR, "decode": []}),
        (("decode",), {"prefill": [], "decode": _TWO_REQUEST_DECODE_PAIR}),
    ):
        subset, canonical, digest = glm_smoke_subset(payload, 2176, phases)
        assert subset == {"schema_version": 3, **expected}
        assert digest == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    # Both phases are the default, and they give the same bytes as before the argument existed.
    assert glm_smoke_subset(payload, 2176) == glm_smoke_subset(payload, 2176, ("prefill", "decode"))


def test_smoke_subset_of_a_prefill_role_needs_no_second_decode_context(tmp_path):
    # A prefill role limited to one sequence, 4,096 tokens and a 4,096-token context measures contexts up to 4,089.
    # Its list holds both smoke prefill points, but only one decode context above the hash block of 2,176 tokens.
    measured = glm_measured_context(SimpleNamespace(vllm_max_model_len=4096))
    limits = {"token_budget": 4096, "prefill_batch_limit": 1, "decode_batch_limit": 1, "measured_context": measured}
    payload, _, _ = glm_point_manifest(**{**_TP4, **limits})
    above = {point["total_kv_read_tokens"] for point in payload["decode"] if point["total_kv_read_tokens"] > 2176}
    assert above == {4089}
    assert all(point in payload["prefill"] for point in _PREFILL_PAIR)

    # The role never measures a decode point, so the list has no decode point to lack.
    subset, canonical, digest = glm_smoke_subset(payload, 2176, ("prefill",))
    assert subset == {"schema_version": 3, "prefill": _PREFILL_PAIR, "decode": []}
    path = tmp_path / "smoke-points.json"
    path.write_text(json.dumps(subset))
    assert _freeze_benchmark_points(str(path)) == (canonical, digest)
    # An aggregated plan measures both phases, so the same list is refused for it, with or without the argument.
    for arguments in ((), (("prefill", "decode"),)):
        with pytest.raises(ValueError) as error:
            glm_smoke_subset(payload, 2176, *arguments)
        assert str(error.value) == _FOUR_POINTS


def test_smoke_subset_of_a_decode_role_needs_no_prefill_points():
    payload, _, _ = glm_point_manifest(**_TP4)
    without = {**payload, "prefill": [point for point in payload["prefill"] if point["total_prefill_tokens"] != 64]}

    # A decode role never measures a prefill point, so a list without the 64-token prefill points still serves it.
    subset, _, _ = glm_smoke_subset(without, 2176, ("decode",))
    assert subset == {"schema_version": 3, "prefill": [], "decode": _TWO_REQUEST_DECODE_PAIR}
    with pytest.raises(ValueError) as error:
        glm_smoke_subset(without, 2176)
    assert str(error.value) == _FOUR_POINTS


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_smoke_subset_of_one_phase_refuses_a_list_without_its_points(phase):
    payload, _, _ = glm_point_manifest(**_TP4)
    smoke = glm_smoke_subset(payload, 2176)[0][phase]
    without = {**payload, phase: [point for point in payload[phase] if point not in smoke]}

    with pytest.raises(ValueError) as error:
        glm_smoke_subset(without, 2176, (phase,))
    # A role that collects one phase needs the two points of that phase, not the four of an aggregated plan.
    assert str(error.value) == f"GLM-5.3-Flash smoke subset requires the generated list to hold its two {phase} points"


@pytest.mark.parametrize(
    "limits",
    [{"decode_batch_limit": 1}, {"prefill_batch_limit": 1, "decode_batch_limit": 1}],
    ids=["decode-limit", "one-sequence-request"],
)
def test_smoke_subset_of_a_list_without_two_request_decode_points_takes_two_contexts(tmp_path, limits):
    payload, _, _ = glm_point_manifest(**{**_TP4, **limits})
    assert {point["batch_size"] for point in payload["decode"]} == {1}
    subset, canonical, digest = glm_smoke_subset(payload, 2176)

    # The prefill points are those of the TP4 smoke. The decode points are one request at each of the two smallest
    # contexts of the list above one hash block.
    assert subset == {
        "schema_version": 3,
        "prefill": [
            {"batch_size": 1, "total_prefill_tokens": 64, "total_kv_read_tokens": 0},
            {"batch_size": 1, "total_prefill_tokens": 64, "total_kv_read_tokens": 2176},
        ],
        "decode": [
            {"batch_size": 1, "total_kv_read_tokens": 4096},
            {"batch_size": 1, "total_kv_read_tokens": 8192},
        ],
    }
    for phase in ("prefill", "decode"):
        assert all(point in payload[phase] for point in subset[phase])
    path = tmp_path / "smoke-points.json"
    path.write_text(json.dumps(subset))
    assert _freeze_benchmark_points(str(path)) == (canonical, digest)


def test_smoke_subset_follows_a_cache_capacity_that_excludes_two_requests():
    # 13 usable blocks: two requests fit only below one hash block, because one request of 4,096 tokens needs 2
    # attention blocks and 5 state blocks and two such requests need 14.
    payload, _, _ = glm_point_manifest(**{**_TP4, "kv_cache_bytes": 14 * 13_045_120})
    two_request_contexts = {
        point["total_kv_read_tokens"] // 2 for point in payload["decode"] if point["batch_size"] == 2
    }
    assert two_request_contexts and max(two_request_contexts) < 2176

    subset, _, _ = glm_smoke_subset(payload, 2176)
    assert subset["decode"] == [
        {"batch_size": 1, "total_kv_read_tokens": 4096},
        {"batch_size": 1, "total_kv_read_tokens": 8192},
    ]


def test_smoke_subset_of_a_one_request_list_with_one_context_above_the_hash_block_is_refused():
    # The measured context of 4,096 tokens leaves one decode context above one hash block, so there is no second one.
    payload, _, _ = glm_point_manifest(**{**_TP4, "decode_batch_limit": 1, "measured_context": 4096})
    above = {point["total_kv_read_tokens"] for point in payload["decode"] if point["total_kv_read_tokens"] > 2176}
    assert above == {4096}
    with pytest.raises(ValueError) as error:
        glm_smoke_subset(payload, 2176)
    assert str(error.value) == "GLM-5.3-Flash smoke subset requires the generated list to hold its four points"


def test_capacity_rule_drops_points_beyond_the_accepted_capacity(caplog):
    caplog.set_level(logging.INFO, logger=_GENERATOR_LOGGER)
    # 99 usable blocks: one request of 131,072 tokens needs 61 attention blocks and one block in each state group.
    payload, _, _ = glm_point_manifest(**{**_TP4, "kv_cache_bytes": 100 * 13_045_120})
    decode = payload["decode"]

    assert len(decode) == 67
    assert {"batch_size": 1, "total_kv_read_tokens": 131072} in decode
    assert {"batch_size": 2, "total_kv_read_tokens": 262144} not in decode
    # The candidates are the 496 prefill and 96 decode points of the TP4 list. The rule keeps 314 and 67 of them.
    assert len(payload["prefill"]) == 314
    assert _logged_lines(caplog) == [
        (
            logging.INFO,
            "GLM-5.3-Flash points: dropped 182/496 prefill and 29/96 decode cases (cache capacity, usable_blocks=99)",
        )
    ]


def test_capacity_without_a_feasible_point_is_refused(caplog):
    caplog.set_level(logging.INFO, logger=_GENERATOR_LOGGER)
    # Five usable blocks: every request needs one attention block and one block in each of the five state groups.
    with pytest.raises(ValueError) as error:
        glm_point_manifest(**{**_TP4, "kv_cache_bytes": 6 * 13_045_120})
    assert str(error.value) == "GLM-5.3-Flash collection has no feasible point for the accepted cache capacity"
    # The logged count explains the refusal.
    assert _logged_lines(caplog) == [
        (
            logging.INFO,
            "GLM-5.3-Flash points: dropped 496/496 prefill and 96/96 decode cases (cache capacity, usable_blocks=5)",
        )
    ]


def test_copied_limits_match_the_producer(monkeypatch):
    producer = load_producer(monkeypatch)

    assert (CHUNK_ALIGN, MAX_BATCH, MAX_NEW, MAX_CONTEXT) == (
        producer.CHUNK_ALIGN,
        producer.MAX_BATCH,
        producer.MAX_NEW,
        producer.MAX_CONTEXT,
    )
    # The measured context keeps the producer's headroom below the runtime context.
    assert glm_measured_context(SimpleNamespace(vllm_max_model_len=4096)) == 4096 - producer.CONTEXT_HEADROOM


@pytest.mark.parametrize("field,value", [("token_budget", 8190), ("hash_block_size", 2178)])
def test_budget_must_be_a_multiple_of_four(field, value):
    with pytest.raises(ValueError) as error:
        glm_point_manifest(**{**_TP4, field: value})
    assert str(error.value) == "GLM-5.3-Flash collection requires a token budget that is a multiple of 4"
