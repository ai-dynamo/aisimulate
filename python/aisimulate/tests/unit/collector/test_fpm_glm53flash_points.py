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
    assert (len(prefill), len(decode)) == (472, 81)
    # The count is logged when the cache holds every candidate, too.
    assert _logged_lines(caplog) == [
        (
            logging.INFO,
            "GLM-5.3-Flash points: dropped 0/472 prefill and 0/81 decode cases (cache capacity, usable_blocks=13814)",
        )
    ]
    # The new tokens above one hash block are whole hash blocks, so 4,096 and 8,192 are not on the axis.
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
        4352,
        6528,
    ]
    assert sorted({point["total_kv_read_tokens"] // point["batch_size"] for point in decode}) == [
        2**exponent for exponent in range(2, 18)
    ]
    assert prefill[0] == {"batch_size": 1, "total_prefill_tokens": 4, "total_kv_read_tokens": 0}
    assert prefill[-1] == {"batch_size": 1, "total_prefill_tokens": 6528, "total_kv_read_tokens": 124032}
    # The formal cell of an aggregated TP4 request stages these bytes.
    assert digest == "130d476feeb3f2d83ff698a24e802a1d07803088b35291335a018c3e37f1e7dc"
    assert decode[0] == {"batch_size": 1, "total_kv_read_tokens": 4}
    assert decode[-1] == {"batch_size": 16, "total_kv_read_tokens": 2097152}
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


def test_tp4_prefill_points_run_in_one_align_mode_step():
    payload, _, _ = glm_point_manifest(**_TP4)
    # With prefix caching, vLLM v0.31.0 runs GLM-5.3-Flash in Mamba align mode, on 2,176-token blocks at TP 4. It stops
    # a prefill chunk at the last block boundary of its prompt (scheduler.py:444, :526). A point whose chunk holds that
    # boundary runs in two steps, and the producer accepts one step per repetition.
    split = []
    for point in payload["prefill"]:
        batch = point["batch_size"]
        start = point["total_kv_read_tokens"] // batch
        end = start + point["total_prefill_tokens"] // batch
        if start < end - end % 2176 < end:
            split.append(point)
    assert split == []


def test_point_list_is_deterministic():
    _, first, digest = glm_point_manifest(**_TP4)
    _, second, _ = glm_point_manifest(**_TP4)

    assert first == second
    assert digest == hashlib.sha256(first.encode("utf-8")).hexdigest()


# The decode batches of the TP4 list at each context. Before a batch decodes, each request prefills its prompt beyond
# its cached seed, and the 8,192-token budget must hold these suffixes of the whole batch in one step.
_TP4_DECODE_BATCHES = {
    **{2**exponent: [1, 2, 4, 8, 16, 32] for exponent in range(2, 9)},
    512: [1, 2, 4, 8, 16],
    1024: [1, 2, 4, 8],
    2048: [1, 2, 4],
    4096: [1, 2, 4],
    8192: [1, 2, 4],
    16384: [1, 2, 4, 7],
    32768: [1, 2, 4, 8, 16, 32],
    65536: [1, 2, 4, 8, 16, 32],
    131072: [1, 2, 4, 8, 16],
}


def _producer_suffix(producer, context: int, hash_block_size: int) -> int:
    """The tokens that one request of a decode point prefills after its seed, from the producer's own functions."""
    prompt = producer._glm_decode_prompt(context)
    return prompt - producer._glm_decode_seed(prompt, hash_block_size)


def _decode_batches(payload: dict) -> dict[int, list[int]]:
    """The batch sizes of the decode points at each per-request context."""
    batches: dict[int, list[int]] = {}
    for point in payload["decode"]:
        batches.setdefault(point["total_kv_read_tokens"] // point["batch_size"], []).append(point["batch_size"])
    return batches


def test_tp4_decode_batches_fit_the_one_step_suffix_prefill(monkeypatch):
    producer = load_producer(monkeypatch).Glm53FlashPrefixSeedScheduler
    payload, _, _ = glm_point_manifest(**_TP4)
    batches = _decode_batches(payload)

    assert batches == _TP4_DECODE_BATCHES
    assert sum(len(sizes) for sizes in batches.values()) == 81
    # vLLM prefills the suffixes of two or more requests in one step only if they fit the token budget together.
    # Otherwise the first decode step never holds every request, and the producer rejects every repetition of the point.
    for point in payload["decode"]:
        context = point["total_kv_read_tokens"] // point["batch_size"]
        assert point["batch_size"] * _producer_suffix(producer, context, 2176) <= 8192
    # Each context takes the largest batch that the budget holds, up to the producer's limit of 32.
    assert {context: max(sizes) for context, sizes in batches.items()} == {
        context: min(32, 8192 // _producer_suffix(producer, context, 2176)) for context in batches
    }


@pytest.mark.parametrize(
    ("hash_block_size", "token_budget", "measured_context"),
    [
        # The prompt at the measured context fills one hash block, so the seed is empty and the suffix is the block.
        (2176, 8192, 2177),
        # One position more: the seed is the whole block and the suffix is one token.
        (2176, 8192, 2178),
        # A hash block that does not divide the powers of two.
        (96, 1024, 1000),
        # One request's suffix at the measured context exceeds the budget. It prefills over several steps, and no other
        # request can decode before it, so that context keeps the point of one request.
        (2176, 2048, 2100),
    ],
)
def test_decode_batches_follow_the_producer_suffix_rule(monkeypatch, hash_block_size, token_budget, measured_context):
    module = load_producer(monkeypatch)
    producer = module.Glm53FlashPrefixSeedScheduler
    # The cache holds every candidate, so only the token budget bounds the batches.
    limits = {
        "hash_block_size": hash_block_size,
        "token_budget": token_budget,
        "measured_context": measured_context,
        "kv_cache_bytes": 10**8 * 13_045_120,
    }
    payload, _, _ = glm_point_manifest(**{**_TP4, **limits})
    budget = min(token_budget, module.MAX_NEW)

    contexts = {measured_context, *(2**exponent for exponent in range(2, 18) if 2**exponent <= measured_context)}
    expected = {}
    for context in sorted(contexts):
        suffix = _producer_suffix(producer, context, hash_block_size)
        limit = min(module.MAX_BATCH, _TP4["decode_batch_limit"], max(1, budget // suffix))
        expected[context] = sorted({*(2**exponent for exponent in range(6) if 2**exponent <= limit), limit})
    assert _decode_batches(payload) == expected
    # A point of two or more requests prefills all its suffixes in one step.
    for point in payload["decode"]:
        context = point["total_kv_read_tokens"] // point["batch_size"]
        suffix = _producer_suffix(producer, context, hash_block_size)
        assert point["batch_size"] == 1 or point["batch_size"] * suffix <= budget


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
    # Without the point of two requests at 4,096, the pair comes from 8,192, the next context that holds both.
    without = {**payload, "decode": [point for point in payload["decode"] if point != subset["decode"][1]]}
    assert glm_smoke_subset(without, 2176)[0]["decode"] == [
        {"batch_size": 1, "total_kv_read_tokens": 8192},
        {"batch_size": 2, "total_kv_read_tokens": 16384},
    ]
    # The generator puts one request on the batch axis of every context, so a list whose points of two requests above
    # one hash block have no point of one request at their context is incomplete.
    unmatched = {
        **payload,
        "decode": [
            point for point in payload["decode"] if point["batch_size"] != 1 or point["total_kv_read_tokens"] <= 2176
        ],
    }
    with pytest.raises(ValueError) as error:
        glm_smoke_subset(unmatched, 2176)
    assert str(error.value) == "GLM-5.3-Flash smoke subset requires the generated list to hold its four points"


@pytest.mark.parametrize(
    ("token_budget", "decode", "digest"),
    [
        # Two suffixes of 1,919 tokens at 4,096 fit the default budget in one step, so the pair stays at 4,096.
        (
            8192,
            [{"batch_size": 1, "total_kv_read_tokens": 4096}, {"batch_size": 2, "total_kv_read_tokens": 8192}],
            "60e85d447ea36458a4d6a7f584d6709e7c02f7b1c1f306c4af18f353c93d14ce",
        ),
        # 3,836 tokens hold one suffix of 1,919 tokens at 4,096 but not two, so 4,096 holds one request only. Two
        # suffixes of 1,663 tokens at 8,192 fit, so the pair comes from 8,192.
        (
            3836,
            [{"batch_size": 1, "total_kv_read_tokens": 8192}, {"batch_size": 2, "total_kv_read_tokens": 16384}],
            "b817a7f76c7aea73992af154bee98fed24a0bd1e417e714e8654b62a8d075a70",
        ),
    ],
    ids=["default-budget", "budget-3836"],
)
def test_smoke_decode_pair_is_at_the_smallest_context_that_holds_both(token_budget, decode, digest):
    payload, _, _ = glm_point_manifest(**{**_TP4, "token_budget": token_budget})
    subset, canonical, subset_digest = glm_smoke_subset(payload, 2176)

    assert subset == {"schema_version": 3, "prefill": _PREFILL_PAIR, "decode": decode}
    assert all(point in payload[phase] for phase in ("prefill", "decode") for point in subset[phase])
    assert subset_digest == digest == hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def test_smoke_subset_serves_every_tp4_token_budget():
    # 256 sequences need a budget of at least 256 tokens. Every budget up to 8,192 gives four smoke points of its list.
    for token_budget in range(256, 8193, 4):
        payload, _, _ = glm_point_manifest(**{**_TP4, "token_budget": token_budget})
        subset, _, _ = glm_smoke_subset(payload, 2176)
        assert [len(subset[phase]) for phase in ("prefill", "decode")] == [2, 2], token_budget
        assert all(point in payload[phase] for phase in ("prefill", "decode") for point in subset[phase])


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
    smoke = glm_smoke_subset(payload, 2176)[0]["prefill"]
    # The prefill smoke takes these two points only. The decode smoke can take any context above one hash block, so
    # only a list without a decode point above the block lacks its points.
    keep = {
        "prefill": lambda point: point not in smoke,
        "decode": lambda point: point["total_kv_read_tokens"] // point["batch_size"] <= 2176,
    }[phase]
    without = {**payload, phase: [point for point in payload[phase] if keep(point)]}

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

    assert len(decode) == 63
    assert {"batch_size": 1, "total_kv_read_tokens": 131072} in decode
    assert {"batch_size": 2, "total_kv_read_tokens": 262144} not in decode
    # The candidates are the 472 prefill and 81 decode points of the TP4 list. The rule keeps 291 and 63 of them.
    assert len(payload["prefill"]) == 291
    assert _logged_lines(caplog) == [
        (
            logging.INFO,
            "GLM-5.3-Flash points: dropped 181/472 prefill and 18/81 decode cases (cache capacity, usable_blocks=99)",
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
            "GLM-5.3-Flash points: dropped 472/472 prefill and 81/81 decode cases (cache capacity, usable_blocks=5)",
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


@pytest.mark.parametrize(
    "field,value,text",
    [
        ("token_budget", 8190, "GLM-5.3-Flash collection requires a token budget that is a multiple of 4"),
        ("hash_block_size", 2178, "GLM-5.3-Flash collection requires a hash block size that is a multiple of 4"),
    ],
)
def test_budget_and_hash_block_must_be_multiples_of_four(field, value, text):
    with pytest.raises(ValueError) as error:
        glm_point_manifest(**{**_TP4, field: value})
    assert str(error.value) == text
