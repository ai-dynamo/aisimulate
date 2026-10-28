# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from copy import deepcopy

import pytest

from aisimulate import resources
from aisimulate.compiler import prediction_to_replay_spec
from aisimulate.config import CorePredictionConfig
from aisimulate.config.common import ResourceConfig
from aisimulate.resources import (
    GB,
    MIB,
    HostResources,
    ResourceLimitError,
    build_plan,
    estimate_workload,
    require_plan,
    workload_bounds,
)
from aisimulate.runner import EngineReplayRunnerFactory
from aisimulate.sweeper.replay import ReplayOutputRequirements


def _request(input_length=64, output_length=1, *, hashes=None, timestamp=0.0):
    return {
        "type": "s",
        "model": "model",
        "t": timestamp,
        "in": input_length,
        "out": output_length,
        "hash_ids": [1] if hashes is None else hashes,
    }


def _play(requests=None, *, block_size=64, play_id="play"):
    return {
        "id": play_id,
        "models": ["model"],
        "block_size": block_size,
        "hash_id_scope": "local",
        "requests": [_request()] if requests is None else requests,
    }


def _write_trace(tmp_path, *plays, name="trace.jsonl"):
    path = tmp_path / name
    path.write_text("".join(json.dumps(play) + "\n" for play in plays))
    return path


def _workload(path, **options):
    return {"source_type": "trace", "trace_format": "weka", "trace_path": str(path), **options}


def _peak(workload):
    estimate = estimate_workload(workload, stack="engine")
    assert estimate.estimated_peak_bytes is not None, estimate.reason
    return estimate.estimated_peak_bytes


def test_weka_source_block_size_does_not_inherit_the_generic_default(tmp_path):
    trace = _write_trace(tmp_path, _play([_request(4096, 128, hashes=list(range(64)))]))
    implicit = _workload(trace)
    explicit = _workload(trace, trace_block_size=64)
    assert _peak(implicit) == _peak(explicit)


@pytest.mark.parametrize(
    "timings,expected_tokens",
    [
        ([(0, 10, 10), (2, 1, 100), (5, 10, 1000)], 1010),
        ([(0, 2, 10), (2, 1, 100)], 100),
        ([(0, None, 10), (0, 0, 20), (1, None, 7)], 30),
    ],
)
def test_weka_active_inputs_follow_completion_dependencies(tmp_path, timings, expected_tokens):
    requests = []
    for start, duration, tokens in timings:
        request = _request(tokens, timestamp=start)
        if duration is not None:
            request["api_time"] = duration
        requests.append(request)
    trace = _write_trace(tmp_path, _play(requests))
    estimate = estimate_workload(_workload(trace), stack="engine")
    assert estimate.input_token_bytes == 4 * expected_tokens


@pytest.mark.parametrize("corner", ["hashless", "preamble", "epsilon"])
def test_weka_input_allowance_covers_native_nonmonotone_main_stream(tmp_path, corner):
    # These source intervals hide a main-stream request behind an earlier
    # request with a later recorded end. Native completion frontiers can then
    # release a different stream while that main-stream request is still live.
    if corner == "hashless":
        rows = [(0, 1, 1, 2, [1]), (1, 99, 1, 2, []), (2, 1, 100, 100, [1]), (101, 1, 100, 100, [2])]
    elif corner == "preamble":
        rows = [(0, 100, 1, 2, [99]), (1, 1, 100, 100, [1]), (101, 1, 100, 100, [2])]
    else:
        rows = [(0, 1, 1, 2, [1]), (0.9999995, 0, 100, 100, [1]), (1.01, 1, 100, 100, [2])]
    requests = [
        {**_request(tokens, output, hashes=hashes, timestamp=start), "api_time": duration}
        for start, duration, tokens, output, hashes in rows
    ]
    trace = _write_trace(tmp_path, _play(requests))
    config = CorePredictionConfig.model_validate(
        {
            "engine": {
                "mode": "aggregated",
                "model": "example/model",
                "hardware": "h200_sxm",
                "context_length": 1024,
                "workers": {
                    "aggregated": {
                        "kv_cache": {"capacity": {"type": "fixed", "blocks": 128}},
                        "timing": {"type": "fixed", "prefill_ms": 1100, "decode_ms": 1100},
                    }
                },
            },
            "traffic": {
                "source": {"type": "trace", "format": "weka", "paths": [str(trace)]},
                "load": {"type": "trace_timestamps", "agentic_lanes": 1},
            },
        }
    )
    result = (
        EngineReplayRunnerFactory()
        .create(0)
        .run(
            prediction_to_replay_spec(config),
            output_requirements=ReplayOutputRequirements(include_raw_report=True, capture_per_request=True),
        )
    )
    observed = result.metadata["native_report"]["per_request"]
    assert len(observed) == len(requests)
    peak_live_inputs = max(
        sum(r["input_length"] for r in observed if r["arrival_time_ms"] <= start < r["terminal_time_ms"])
        for start in (r["arrival_time_ms"] for r in observed)
    )
    assert peak_live_inputs == 200
    estimate = estimate_workload(workload_bounds(config), stack="engine")
    assert estimate.input_token_bytes >= 4 * peak_live_inputs


def test_weka_header_order_does_not_change_allocations(tmp_path):
    play = _play([_request(4096, 128, hashes=[])])
    original = _write_trace(tmp_path, play)
    reordered = _write_trace(tmp_path, dict(reversed(list(play.items()))), name="reordered.jsonl")
    assert _peak(_workload(original)) == _peak(_workload(reordered))


def test_weka_nested_scopes_and_files_contribute_active_inputs(tmp_path):
    child = _request(100, timestamp=10)
    play = _play(
        [
            _request(200),
            {"type": "subagent", "t": 5, "requests": [child]},
        ]
    )
    first = _write_trace(tmp_path, play, name="first.jsonl")
    second = _write_trace(tmp_path, _play([_request(300)], play_id="second"), name="second.jsonl")
    explicit = {"trace_format": "weka", "trace_paths": [str(first), str(second)]}
    estimate = estimate_workload(explicit, stack="engine")
    assert estimate.input_token_bytes == 4 * 600
    assert estimate == estimate_workload(_workload(tmp_path), stack="engine")


def test_weka_bounds_preserve_source_and_snapshot_configuration(tmp_path):
    trace = _write_trace(tmp_path, _play())
    config = CorePredictionConfig.model_validate(
        {
            "engine": {
                "mode": "aggregated",
                "model": "example/model",
                "hardware": "h200_sxm",
                "context_length": 1024,
                "workers": {"aggregated": {}},
            },
            "traffic": {
                "source": {"type": "trace", "format": "weka", "paths": [str(trace)]},
                "load": {
                    "type": "trace_timestamps",
                    "agentic_lanes": 12,
                    "agentic_snapshot": {"seed": 42},
                    "agentic_warmup": True,
                },
            },
        }
    )
    bounds = workload_bounds(config)
    assert bounds.get("trace_block_size") is None
    assert bounds["agentic_snapshot"] == {"seed": 42}
    assert bounds["agentic_warmup"] is True
    assert _peak(bounds) == _peak(
        _workload(trace, agentic_lanes=12, agentic_snapshot={"seed": 42}, agentic_warmup=True)
    )


@pytest.mark.parametrize("allocation", ["planned_outputs", "normalized_hashes"])
def test_weka_rejects_large_native_allocations_hidden_in_tiny_files(tmp_path, allocation):
    if allocation == "planned_outputs":
        play = _play([_request(1, 10**9)])
    else:
        # The importer synthesizes missing hashes; an empty source array does
        # not mean the normalized graph has no per-block allocations.
        play = _play([_request(10**9, 1, hashes=[])], block_size=1)
    trace = _write_trace(tmp_path, play)
    assert trace.stat().st_size < 1024
    plan = build_plan(
        _workload(trace),
        stack="engine",
        host=HostResources(8 * GB, 4 * GB, 4),
        policy=ResourceConfig(memory_limit_gb=1.0),
    )
    assert plan["status"] == "resource_limited"
    # Both paths allocate at least one four-byte native value per item. Only
    # inspect metadata here; never materialize these billion-element arrays.
    assert plan["estimate"]["estimated_peak_bytes"] >= 4 * GB
    with pytest.raises(ResourceLimitError, match="estimated peak="):
        require_plan(plan)


def test_weka_finite_lanes_do_not_duplicate_the_complete_corpus(tmp_path):
    hashes = list(range(1024))
    play = _play([_request(65536, 128, hashes=hashes, timestamp=float(i)) for i in range(64)])
    trace = _write_trace(tmp_path, play)
    one_lane = _peak(_workload(trace, agentic_lanes=1))
    twelve_lanes = _peak(_workload(trace, agentic_lanes=12))
    # A single finite play is executed once even when more lanes are available.
    # Allow conservative scheduling headroom, but not twelve corpus copies.
    assert twelve_lanes <= 2 * one_lane
    plan = build_plan(
        _workload(trace, agentic_lanes=12),
        stack="engine",
        host=HostResources(16 * GB, 12 * GB, 8),
        policy=ResourceConfig(memory_limit_gb=4.0),
    )
    assert plan["status"] == "admitted"


def test_weka_import_reserves_the_graph_while_lowering_later_plays(tmp_path):
    # Large metadata makes import dominate replay, so graph growth must still
    # increase the estimate when another play has the same scratch requirement.
    play = {**_play([_request()]), "metadata": "x" * 100000}
    one = _write_trace(tmp_path, play, name="one.jsonl")
    two = tmp_path / "two"
    two.mkdir()
    _write_trace(two, play, name="a.jsonl")
    _write_trace(two, play, name="b.jsonl")
    one_peak = _peak(_workload(one))
    two_peak = _peak(_workload(two))
    assert two_peak > one_peak
    assert one_peak > 256 * MIB + 32 * one.stat().st_size + 128 + 32768


def test_weka_snapshot_lanes_account_for_repeated_play_payloads(tmp_path):
    trace = _write_trace(tmp_path, _play([_request(64, 10**6)]))
    options = {"agentic_snapshot": {"seed": 42}}
    one_lane = _peak(_workload(trace, agentic_lanes=1, **options))
    twelve_lanes = _peak(_workload(trace, agentic_lanes=12, **options))
    assert twelve_lanes > one_lane
    assert _peak(_workload(trace, agentic_lanes=12, agentic_warmup=True, **options)) >= twelve_lanes


@pytest.mark.parametrize("field,value", [("in", -1), ("in", 1.5), ("in", True), ("out", -1), ("out", 1.5)])
def test_weka_invalid_lengths_have_no_qualified_estimate(tmp_path, field, value):
    request = _request()
    request[field] = value
    trace = _write_trace(tmp_path, _play([request]))
    estimate = estimate_workload(_workload(trace), stack="engine")
    assert estimate.estimated_peak_bytes is None
    assert estimate.reason


@pytest.mark.parametrize("case", ["zero", "mixed", "assertion_mismatch"])
def test_weka_invalid_source_block_sizes_have_no_qualified_estimate(tmp_path, case):
    options = {}
    if case == "zero":
        plays = [_play(block_size=0)]
    elif case == "mixed":
        plays = [_play(play_id="first"), _play(block_size=128, play_id="second")]
    else:
        plays = [_play()]
        options["trace_block_size"] = 512
    trace = _write_trace(tmp_path, *plays)
    estimate = estimate_workload(_workload(trace, **options), stack="engine")
    assert estimate.estimated_peak_bytes is None
    assert estimate.reason


def test_weka_summary_lengths_are_not_replayed_request_allocations(tmp_path):
    play = _play()
    ordinary = _write_trace(tmp_path, play, name="ordinary.jsonl")
    annotated = deepcopy(play)
    annotated.update(tool_tokens=10**12, system_tokens=10**12, totals={"in": 10**12, "out": 10**12})
    with_summary = _write_trace(tmp_path, annotated, name="summary.jsonl")
    # Native Weka keeps these fields as metadata. A few extra JSON scalars
    # cannot require terabytes of request state or another replayed request.
    assert abs(_peak(_workload(with_summary)) - _peak(_workload(ordinary))) < 1024**2


def test_weka_nested_subagent_outputs_are_included_in_admission(tmp_path):
    play = _play(
        [
            _request(timestamp=0.0),
            {
                "t": 0.1,
                "type": "subagent",
                "agent_id": "worker",
                "subagent_type": "Explore",
                "duration_ms": 10,
                "status": "completed",
                "models": ["model"],
                "requests": [_request(1, 10**9, timestamp=0.1)],
            },
        ]
    )
    trace = _write_trace(tmp_path, play)
    plan = build_plan(
        _workload(trace),
        stack="engine",
        host=HostResources(8 * GB, 4 * GB, 4),
        policy=ResourceConfig(memory_limit_gb=1.0),
    )
    assert plan["status"] == "resource_limited"
    assert plan["estimate"]["estimated_peak_bytes"] >= 4 * GB


def test_weka_hash_arrays_cross_parser_buffers_without_changing_allocations(tmp_path):
    hashes = [2**64 - 1] * 10000
    trace = _write_trace(tmp_path, _play([_request(64, 3, hashes=hashes)]))
    assert trace.stat().st_size > 2 * 64 * 1024
    estimate = estimate_workload(_workload(trace), stack="engine")
    assert estimate.allocation_model == "weka-materialized-v2"
    assert estimate.input_token_bytes == 4 * 64
    assert estimate.lower_bound_bytes == max(8, 4 * 3)
    assert (
        estimate.estimated_peak_bytes == 256 * MIB + 32 * trace.stat().st_size + 128 * len(hashes) + 32768 + 32 + 2048
    )


@pytest.mark.parametrize("value", [-1, 2**64, 1.5, True, None, "1", [1], {"nested": 1}])
def test_weka_invalid_hash_after_parser_buffer_has_no_qualified_estimate(tmp_path, value):
    trace = _write_trace(tmp_path, _play([_request(hashes=[2**64 - 1] * 4000 + [value])]))
    assert trace.stat().st_size > 64 * 1024
    estimate = estimate_workload(_workload(trace), stack="engine")
    assert estimate.estimated_peak_bytes is None
    assert estimate.reason == "cannot inspect Weka metadata: Weka hash_ids must contain unsigned 64-bit integers"


def test_weka_truncated_hash_array_has_no_qualified_estimate(tmp_path):
    trace = _write_trace(tmp_path, _play([_request(hashes=[2**64 - 1] * 4000)]))
    trace.write_text(trace.read_text()[:-20])
    estimate = estimate_workload(_workload(trace), stack="engine")
    assert estimate.estimated_peak_bytes is None
    assert estimate.reason.startswith("cannot inspect Weka metadata:")


@pytest.fixture
def weka_parses(monkeypatch):
    resources._clear_trace_scan_cache()
    calls = []
    original = resources.ijson.parse

    def parse(*args, **kwargs):
        calls.append(None)
        return original(*args, **kwargs)

    monkeypatch.setattr(resources.ijson, "parse", parse)
    yield calls
    resources._clear_trace_scan_cache()


def test_weka_cached_inspection_keeps_live_memory_admission(tmp_path, weka_parses):
    workload = _workload(_write_trace(tmp_path, _play()))
    first = estimate_workload(workload, stack="engine")
    host = HostResources(8 * GB, 4 * GB, 4)
    admitted = build_plan(workload, stack="engine", host=host, policy=ResourceConfig(memory_limit_gb=1))
    limited = build_plan(workload, stack="engine", host=host, policy=ResourceConfig(memory_limit_gb=0.25))
    assert admitted["status"] == "admitted"
    assert limited["status"] == "resource_limited"
    assert admitted["estimate"] == limited["estimate"]
    assert first.estimated_peak_bytes == limited["estimate"]["estimated_peak_bytes"]
    with pytest.raises(ResourceLimitError):
        require_plan(limited)
    assert len(weka_parses) == 1


def test_weka_cached_profile_inspection_rechecks_live_headroom(tmp_path, weka_parses):
    workload = _workload(_write_trace(tmp_path, _play()), agentic_profile={})
    complete = estimate_workload(workload, stack="engine")
    limited = estimate_workload(workload, stack="engine", inspection_budget_bytes=1)
    assert complete.allocation_model == "agentic-profile-unqualified-v1"
    assert complete.estimated_peak_bytes is None
    assert limited.allocation_model == "agentic-profile-materialization-v1"
    assert limited.estimated_peak_bytes > 1
    assert len(weka_parses) == 1


def test_weka_budget_limited_scans_are_not_cached(tmp_path, weka_parses):
    workload = _workload(_write_trace(tmp_path, _play([_request(output_length=10**8)])))
    early = estimate_workload(workload, stack="engine", inspection_budget_bytes=1)
    assert early.lower_bound_bytes == 0
    assert len(weka_parses) == 0
    limited = estimate_workload(workload, stack="engine", inspection_budget_bytes=GB)
    assert limited.estimated_peak_bytes > GB
    assert len(weka_parses) == 1
    assert estimate_workload(workload, stack="engine") == limited
    assert len(weka_parses) == 2


def test_weka_cache_invalidates_changed_files_and_rejects_malformed_tails(tmp_path, weka_parses):
    path = _write_trace(tmp_path, _play())
    workload = _workload(path)
    first = estimate_workload(workload, stack="engine")
    original = path.read_text()
    path.write_text(original + "not JSON\n")
    for _ in range(2):
        assert estimate_workload(workload, stack="engine").estimated_peak_bytes is None
    assert len(weka_parses) == 3
    replacement = _write_trace(tmp_path, _play([_request(6400)]), name="replacement.jsonl")
    replacement.replace(path)
    assert estimate_workload(workload, stack="engine").input_token_bytes > first.input_token_bytes
    assert len(weka_parses) == 4


def test_weka_source_changed_during_inspection_is_not_cached(tmp_path, monkeypatch):
    resources._clear_trace_scan_cache()
    path = _write_trace(tmp_path, _play())
    workload = _workload(path)
    original = resources.ijson.parse
    calls = []

    def parse(*args, **kwargs):
        calls.append(None)
        yield from original(*args, **kwargs)
        if len(calls) == 1:
            with path.open("a") as stream:
                stream.write(json.dumps(_play(play_id="added")) + "\n")

    monkeypatch.setattr(resources.ijson, "parse", parse)
    changed = estimate_workload(workload, stack="engine")
    assert changed.estimated_peak_bytes is None
    assert changed.reason == "Weka source changed during metadata inspection"
    assert estimate_workload(workload, stack="engine").input_token_bytes == 4 * 128
    assert len(calls) == 2


@pytest.mark.parametrize(
    "options", [{"agentic_lanes": 2}, {"agentic_snapshot": {}}, {"agentic_warmup": True}, {"trace_block_size": 128}]
)
def test_weka_cache_keys_cover_allocation_configuration(tmp_path, weka_parses, options):
    path = _write_trace(tmp_path, _play())
    estimate_workload(_workload(path), stack="engine")
    workload = _workload(path, **options)
    actual = estimate_workload(workload, stack="engine")
    expected = resources._inspect_weka_trace(workload, inspection_budget_bytes=None)
    assert actual == expected
    assert len(weka_parses) == 3


def test_weka_cache_is_bounded_and_directory_changes_invalidate_it(tmp_path, monkeypatch, weka_parses):
    monkeypatch.setattr(resources, "_TRACE_SCAN_CACHE_SIZE", 2)
    paths = [_write_trace(tmp_path, _play(play_id=str(i)), name=f"{i}.jsonl") for i in range(3)]
    for path in paths:
        estimate_workload(_workload(path), stack="engine")
    estimate_workload(_workload(paths[-1]), stack="engine")
    estimate_workload(_workload(paths[0]), stack="engine")
    assert len(weka_parses) == 4
    assert len(resources._WEKA_SCAN_CACHE) == 2
    before = estimate_workload(_workload(tmp_path), stack="engine")
    _write_trace(tmp_path, _play(play_id="added"), name="added.jsonl")
    after = estimate_workload(_workload(tmp_path), stack="engine")
    assert after.input_token_bytes > before.input_token_bytes
    assert len(weka_parses) == 4 + 3 + 4
